"""
Caching Strategies for Real-time Fraud Detection

This module provides comprehensive caching strategies to minimize computation
and improve response times for fraud detection systems.

Key Features:
- Multi-level caching architecture
- LRU, LFU, and TTL-based cache policies
- Distributed caching support
- Cache warming and prefetching
- Cache invalidation strategies
- Performance-aware cache sizing

Author: IntegratedML Pluggable Models Team
"""

import numpy as np
import pandas as pd
import logging
import time
import threading
import hashlib
import pickle
import json
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import weakref
import warnings
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"         # Least Recently Used
    LFU = "lfu"         # Least Frequently Used
    TTL = "ttl"         # Time To Live
    FIFO = "fifo"       # First In First Out
    RANDOM = "random"   # Random eviction


class CacheLevel(Enum):
    """Cache levels in the hierarchy."""
    L1_MEMORY = "l1_memory"
    L2_DISK = "l2_disk"
    L3_DISTRIBUTED = "l3_distributed"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: datetime
    access_count: int = 0
    last_access: Optional[datetime] = None
    ttl_seconds: Optional[float] = None
    size_bytes: int = 0
    
    def __post_init__(self):
        if self.last_access is None:
            self.last_access = self.timestamp
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds is None:
            return False
        
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl_seconds
    
    def access(self) -> None:
        """Record access to this entry."""
        self.access_count += 1
        self.last_access = datetime.now()


class CacheInterface(ABC):
    """Abstract interface for cache implementations."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> bool:
        """Put value in cache."""
        pass
    
    @abstractmethod
    def remove(self, key: str) -> bool:
        """Remove value from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get number of entries in cache."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCache(CacheInterface):
    """
    High-performance in-memory cache with multiple eviction policies.
    """
    
    def __init__(self,
                 max_size: int = 1000,
                 policy: CachePolicy = CachePolicy.LRU,
                 default_ttl_seconds: Optional[float] = None,
                 enable_stats: bool = True):
        """
        Initialize memory cache.
        
        Parameters:
        -----------
        max_size : int, default=1000
            Maximum number of entries
        policy : CachePolicy, default=CachePolicy.LRU
            Cache eviction policy
        default_ttl_seconds : float, optional
            Default TTL for entries
        enable_stats : bool, default=True
            Whether to collect statistics
        """
        self.max_size = max_size
        self.policy = policy
        self.default_ttl_seconds = default_ttl_seconds
        self.enable_stats = enable_stats
        
        # Cache storage
        if policy == CachePolicy.LRU:
            self.cache = OrderedDict()
        else:
            self.cache = {}
        
        # Entry metadata
        self.entries = {}
        
        # Statistics
        if self.enable_stats:
            self.stats = CacheStatistics()
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Initialized MemoryCache with policy {policy.value}, max_size {max_size}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Parameters:
        -----------
        key : str
            Cache key
            
        Returns:
        --------
        value : any or None
            Cached value if found and not expired
        """
        with self.lock:
            if self.enable_stats:
                self.stats.record_access()
            
            if key not in self.cache:
                if self.enable_stats:
                    self.stats.record_miss()
                return None
            
            # Check if entry exists in metadata
            if key not in self.entries:
                # Inconsistent state, remove from cache
                del self.cache[key]
                if self.enable_stats:
                    self.stats.record_miss()
                return None
            
            entry = self.entries[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                if self.enable_stats:
                    self.stats.record_miss()
                    self.stats.record_expiration()
                return None
            
            # Update access information
            entry.access()
            
            # Handle LRU policy
            if self.policy == CachePolicy.LRU:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
            
            if self.enable_stats:
                self.stats.record_hit()
            
            return self.cache[key]
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> bool:
        """
        Put value in cache.
        
        Parameters:
        -----------
        key : str
            Cache key
        value : any
            Value to cache
        ttl_seconds : float, optional
            TTL for this entry
            
        Returns:
        --------
        success : bool
            Whether value was successfully cached
        """
        with self.lock:
            # Use default TTL if not specified
            if ttl_seconds is None:
                ttl_seconds = self.default_ttl_seconds
            
            # Calculate entry size
            size_bytes = self._calculate_size(value)
            
            # Check if we need to evict entries
            while len(self.cache) >= self.max_size and key not in self.cache:
                evicted = self._evict_entry()
                if not evicted:
                    # Unable to evict, cache might be full of non-evictable entries
                    logger.warning("Unable to evict entries from cache")
                    return False
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=datetime.now(),
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes
            )
            
            # Store in cache and metadata
            self.cache[key] = value
            self.entries[key] = entry
            
            # Handle LRU policy
            if self.policy == CachePolicy.LRU:
                self.cache.move_to_end(key)
            
            if self.enable_stats:
                self.stats.record_put(size_bytes)
            
            return True
    
    def remove(self, key: str) -> bool:
        """
        Remove value from cache.
        
        Parameters:
        -----------
        key : str
            Cache key
            
        Returns:
        --------
        success : bool
            Whether value was removed
        """
        with self.lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.entries.clear()
            
            if self.enable_stats:
                self.stats.record_clear()
    
    def size(self) -> int:
        """Get number of entries in cache."""
        with self.lock:
            return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            stats = {
                'size': len(self.cache),
                'max_size': self.max_size,
                'policy': self.policy.value
            }
            
            if self.enable_stats:
                stats.update(self.stats.get_stats())
            
            return stats
    
    def _evict_entry(self) -> bool:
        """
        Evict an entry based on the cache policy.
        
        Returns:
        --------
        success : bool
            Whether an entry was evicted
        """
        if not self.cache:
            return False
        
        if self.policy == CachePolicy.LRU:
            # Remove least recently used (first item in OrderedDict)
            key, _ = self.cache.popitem(last=False)
            if key in self.entries:
                del self.entries[key]
        
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used
            if not self.entries:
                return False
            
            min_access_count = min(entry.access_count for entry in self.entries.values())
            key_to_remove = None
            
            for key, entry in self.entries.items():
                if entry.access_count == min_access_count:
                    key_to_remove = key
                    break
            
            if key_to_remove:
                self._remove_entry(key_to_remove)
        
        elif self.policy == CachePolicy.TTL:
            # Remove expired entries first
            expired_keys = [
                key for key, entry in self.entries.items()
                if entry.is_expired()
            ]
            
            if expired_keys:
                self._remove_entry(expired_keys[0])
            else:
                # No expired entries, fall back to LRU
                if self.cache:
                    key = next(iter(self.cache))
                    self._remove_entry(key)
        
        elif self.policy == CachePolicy.FIFO:
            # Remove first inserted (first item)
            if self.cache:
                key = next(iter(self.cache))
                self._remove_entry(key)
        
        elif self.policy == CachePolicy.RANDOM:
            # Remove random entry
            import random
            if self.cache:
                key = random.choice(list(self.cache.keys()))
                self._remove_entry(key)
        
        if self.enable_stats:
            self.stats.record_eviction()
        
        return True
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from both cache and metadata."""
        if key in self.cache:
            del self.cache[key]
        if key in self.entries:
            del self.entries[key]
    
    def _calculate_size(self, value: Any) -> int:
        """
        Calculate approximate size of value in bytes.
        
        Parameters:
        -----------
        value : any
            Value to measure
            
        Returns:
        --------
        size_bytes : int
            Approximate size in bytes
        """
        try:
            return len(pickle.dumps(value))
        except:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) 
                          for k, v in value.items())
            else:
                return 100  # Default estimate


class CacheStatistics:
    """Cache statistics collection."""
    
    def __init__(self):
        """Initialize statistics."""
        self.hits = 0
        self.misses = 0
        self.puts = 0
        self.evictions = 0
        self.expirations = 0
        self.clears = 0
        self.total_size_bytes = 0
        self.start_time = time.time()
        
        # Performance metrics
        self.access_times = deque(maxlen=1000)
        self.put_times = deque(maxlen=1000)
    
    def record_hit(self) -> None:
        """Record cache hit."""
        self.hits += 1
    
    def record_miss(self) -> None:
        """Record cache miss."""
        self.misses += 1
    
    def record_put(self, size_bytes: int) -> None:
        """Record cache put."""
        self.puts += 1
        self.total_size_bytes += size_bytes
    
    def record_eviction(self) -> None:
        """Record cache eviction."""
        self.evictions += 1
    
    def record_expiration(self) -> None:
        """Record cache expiration."""
        self.expirations += 1
    
    def record_clear(self) -> None:
        """Record cache clear."""
        self.clears += 1
        self.total_size_bytes = 0
    
    def record_access(self) -> None:
        """Record access timing."""
        self.access_times.append(time.time())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics summary."""
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / max(total_accesses, 1)
        
        elapsed_time = time.time() - self.start_time
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'puts': self.puts,
            'evictions': self.evictions,
            'expirations': self.expirations,
            'clears': self.clears,
            'total_size_bytes': self.total_size_bytes,
            'elapsed_time_seconds': elapsed_time,
            'accesses_per_second': total_accesses / max(elapsed_time, 1)
        }


class MultiLevelCache(CacheInterface):
    """
    Multi-level cache hierarchy with automatic promotion/demotion.
    """
    
    def __init__(self,
                 l1_cache: Optional[CacheInterface] = None,
                 l2_cache: Optional[CacheInterface] = None,
                 l3_cache: Optional[CacheInterface] = None,
                 auto_promote: bool = True):
        """
        Initialize multi-level cache.
        
        Parameters:
        -----------
        l1_cache : CacheInterface, optional
            Level 1 (fastest) cache
        l2_cache : CacheInterface, optional
            Level 2 (medium) cache
        l3_cache : CacheInterface, optional
            Level 3 (slowest) cache
        auto_promote : bool, default=True
            Whether to auto-promote frequently accessed items
        """
        self.l1_cache = l1_cache or MemoryCache(max_size=100, policy=CachePolicy.LRU)
        self.l2_cache = l2_cache
        self.l3_cache = l3_cache
        self.auto_promote = auto_promote
        
        # Statistics
        self.stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'l3_hits': 0,
            'total_misses': 0,
            'promotions': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("Initialized MultiLevelCache")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache hierarchy.
        
        Parameters:
        -----------
        key : str
            Cache key
            
        Returns:
        --------
        value : any or None
            Cached value if found
        """
        with self.lock:
            # Try L1 cache first
            value = self.l1_cache.get(key)
            if value is not None:
                self.stats['l1_hits'] += 1
                return value
            
            # Try L2 cache
            if self.l2_cache:
                value = self.l2_cache.get(key)
                if value is not None:
                    self.stats['l2_hits'] += 1
                    
                    # Promote to L1 if auto-promote enabled
                    if self.auto_promote:
                        self.l1_cache.put(key, value)
                        self.stats['promotions'] += 1
                    
                    return value
            
            # Try L3 cache
            if self.l3_cache:
                value = self.l3_cache.get(key)
                if value is not None:
                    self.stats['l3_hits'] += 1
                    
                    # Promote to L1 and L2 if auto-promote enabled
                    if self.auto_promote:
                        self.l1_cache.put(key, value)
                        if self.l2_cache:
                            self.l2_cache.put(key, value)
                        self.stats['promotions'] += 1
                    
                    return value
            
            # Not found in any cache
            self.stats['total_misses'] += 1
            return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> bool:
        """
        Put value in cache hierarchy.
        
        Parameters:
        -----------
        key : str
            Cache key
        value : any
            Value to cache
        ttl_seconds : float, optional
            TTL for this entry
            
        Returns:
        --------
        success : bool
            Whether value was successfully cached
        """
        with self.lock:
            # Always put in L1
            success = self.l1_cache.put(key, value, ttl_seconds)
            
            # Optionally put in L2 and L3
            if self.l2_cache:
                self.l2_cache.put(key, value, ttl_seconds)
            
            if self.l3_cache:
                self.l3_cache.put(key, value, ttl_seconds)
            
            return success
    
    def remove(self, key: str) -> bool:
        """
        Remove value from all cache levels.
        
        Parameters:
        -----------
        key : str
            Cache key
            
        Returns:
        --------
        success : bool
            Whether value was removed from any level
        """
        with self.lock:
            removed = False
            
            if self.l1_cache.remove(key):
                removed = True
            
            if self.l2_cache and self.l2_cache.remove(key):
                removed = True
            
            if self.l3_cache and self.l3_cache.remove(key):
                removed = True
            
            return removed
    
    def clear(self) -> None:
        """Clear all cache levels."""
        with self.lock:
            self.l1_cache.clear()
            
            if self.l2_cache:
                self.l2_cache.clear()
            
            if self.l3_cache:
                self.l3_cache.clear()
    
    def size(self) -> int:
        """Get total number of unique entries across all levels."""
        with self.lock:
            # This is approximate since entries might be duplicated across levels
            total = self.l1_cache.size()
            
            if self.l2_cache:
                total += self.l2_cache.size()
            
            if self.l3_cache:
                total += self.l3_cache.size()
            
            return total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.lock:
            stats = dict(self.stats)
            
            stats['l1_stats'] = self.l1_cache.get_stats()
            
            if self.l2_cache:
                stats['l2_stats'] = self.l2_cache.get_stats()
            
            if self.l3_cache:
                stats['l3_stats'] = self.l3_cache.get_stats()
            
            # Calculate overall hit rate
            total_hits = self.stats['l1_hits'] + self.stats['l2_hits'] + self.stats['l3_hits']
            total_accesses = total_hits + self.stats['total_misses']
            stats['overall_hit_rate'] = total_hits / max(total_accesses, 1)
            
            return stats


class FraudDetectionCacheManager:
    """
    Specialized cache manager for fraud detection system.
    """
    
    def __init__(self):
        """Initialize fraud detection cache manager."""
        # Feature caches with different policies
        self.feature_cache = MemoryCache(
            max_size=1000,
            policy=CachePolicy.LRU,
            default_ttl_seconds=300  # 5 minutes
        )
        
        # Model prediction cache
        self.prediction_cache = MemoryCache(
            max_size=5000,
            policy=CachePolicy.LRU,
            default_ttl_seconds=60   # 1 minute
        )
        
        # Customer profile cache
        self.customer_cache = MemoryCache(
            max_size=10000,
            policy=CachePolicy.LFU,
            default_ttl_seconds=3600  # 1 hour
        )
        
        # Rule evaluation cache
        self.rule_cache = MemoryCache(
            max_size=2000,
            policy=CachePolicy.TTL,
            default_ttl_seconds=120   # 2 minutes
        )
        
        # Multi-level cache for expensive computations
        self.computation_cache = MultiLevelCache(
            l1_cache=MemoryCache(max_size=500, policy=CachePolicy.LRU),
            l2_cache=MemoryCache(max_size=2000, policy=CachePolicy.LFU)
        )
        
        # Cache warming
        self.cache_warmer = CacheWarmer(self)
        
        logger.info("Initialized FraudDetectionCacheManager")
    
    def get_features(self, feature_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached features.
        
        Parameters:
        -----------
        feature_key : str
            Feature cache key
            
        Returns:
        --------
        features : dict or None
            Cached features if available
        """
        return self.feature_cache.get(feature_key)
    
    def cache_features(self, feature_key: str, features: Dict[str, Any],
                      ttl_seconds: Optional[float] = None) -> bool:
        """
        Cache computed features.
        
        Parameters:
        -----------
        feature_key : str
            Feature cache key
        features : dict
            Features to cache
        ttl_seconds : float, optional
            TTL for features
            
        Returns:
        --------
        success : bool
            Whether features were cached
        """
        return self.feature_cache.put(feature_key, features, ttl_seconds)
    
    def get_prediction(self, prediction_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached prediction.
        
        Parameters:
        -----------
        prediction_key : str
            Prediction cache key
            
        Returns:
        --------
        prediction : dict or None
            Cached prediction if available
        """
        return self.prediction_cache.get(prediction_key)
    
    def cache_prediction(self, prediction_key: str, prediction: Dict[str, Any],
                        ttl_seconds: Optional[float] = None) -> bool:
        """
        Cache model prediction.
        
        Parameters:
        -----------
        prediction_key : str
            Prediction cache key
        prediction : dict
            Prediction to cache
        ttl_seconds : float, optional
            TTL for prediction
            
        Returns:
        --------
        success : bool
            Whether prediction was cached
        """
        return self.prediction_cache.put(prediction_key, prediction, ttl_seconds)
    
    def get_customer_profile(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached customer profile.
        
        Parameters:
        -----------
        customer_id : str
            Customer identifier
            
        Returns:
        --------
        profile : dict or None
            Cached customer profile if available
        """
        return self.customer_cache.get(customer_id)
    
    def cache_customer_profile(self, customer_id: str, profile: Dict[str, Any],
                              ttl_seconds: Optional[float] = None) -> bool:
        """
        Cache customer profile.
        
        Parameters:
        -----------
        customer_id : str
            Customer identifier
        profile : dict
            Customer profile to cache
        ttl_seconds : float, optional
            TTL for profile
            
        Returns:
        --------
        success : bool
            Whether profile was cached
        """
        return self.customer_cache.put(customer_id, profile, ttl_seconds)
    
    def generate_cache_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from arguments.
        
        Parameters:
        -----------
        *args, **kwargs : various
            Arguments to include in key
            
        Returns:
        --------
        cache_key : str
            Generated cache key
        """
        # Create deterministic key from arguments
        key_data = {
            'args': args,
            'kwargs': kwargs
        }
        
        # Serialize and hash
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def warm_cache(self, warming_data: Dict[str, Any]) -> None:
        """
        Warm caches with common data.
        
        Parameters:
        -----------
        warming_data : dict
            Data for cache warming
        """
        self.cache_warmer.warm_caches(warming_data)
    
    def invalidate_customer_cache(self, customer_id: str) -> None:
        """
        Invalidate customer-related cache entries.
        
        Parameters:
        -----------
        customer_id : str
            Customer identifier
        """
        # Remove customer profile
        self.customer_cache.remove(customer_id)
        
        # Could also remove related feature and prediction caches
        # This would require more sophisticated key tracking
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
        --------
        stats : dict
            Comprehensive cache statistics
        """
        return {
            'feature_cache': self.feature_cache.get_stats(),
            'prediction_cache': self.prediction_cache.get_stats(),
            'customer_cache': self.customer_cache.get_stats(),
            'rule_cache': self.rule_cache.get_stats(),
            'computation_cache': self.computation_cache.get_stats(),
            'cache_warmer': self.cache_warmer.get_stats()
        }


class CacheWarmer:
    """
    Cache warming system for preloading common data.
    """
    
    def __init__(self, cache_manager: FraudDetectionCacheManager):
        """
        Initialize cache warmer.
        
        Parameters:
        -----------
        cache_manager : FraudDetectionCacheManager
            Cache manager to warm
        """
        self.cache_manager = cache_manager
        
        # Warming statistics
        self.warming_runs = 0
        self.items_warmed = 0
        self.last_warming_time = None
        
        logger.info("Initialized CacheWarmer")
    
    def warm_caches(self, warming_data: Dict[str, Any]) -> None:
        """
        Warm caches with provided data.
        
        Parameters:
        -----------
        warming_data : dict
            Data for warming caches
        """
        start_time = time.time()
        items_warmed = 0
        
        try:
            # Warm customer profiles
            if 'customer_profiles' in warming_data:
                for customer_id, profile in warming_data['customer_profiles'].items():
                    self.cache_manager.cache_customer_profile(customer_id, profile)
                    items_warmed += 1
            
            # Warm common features
            if 'common_features' in warming_data:
                for feature_key, features in warming_data['common_features'].items():
                    self.cache_manager.cache_features(feature_key, features)
                    items_warmed += 1
            
            # Warm rule evaluations
            if 'rule_evaluations' in warming_data:
                for rule_key, result in warming_data['rule_evaluations'].items():
                    self.cache_manager.rule_cache.put(rule_key, result)
                    items_warmed += 1
            
            # Update statistics
            self.warming_runs += 1
            self.items_warmed += items_warmed
            self.last_warming_time = datetime.now()
            
            warming_time = (time.time() - start_time) * 1000
            logger.info(f"Cache warming completed: {items_warmed} items in {warming_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")
    
    def warm_popular_customers(self, customer_ids: List[str],
                              profile_generator: Callable[[str], Dict[str, Any]]) -> None:
        """
        Warm cache with popular customer profiles.
        
        Parameters:
        -----------
        customer_ids : list of str
            Customer IDs to warm
        profile_generator : callable
            Function to generate profile for customer ID
        """
        for customer_id in customer_ids:
            try:
                profile = profile_generator(customer_id)
                self.cache_manager.cache_customer_profile(customer_id, profile)
                self.items_warmed += 1
            except Exception as e:
                logger.warning(f"Failed to warm profile for customer {customer_id}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache warming statistics.
        
        Returns:
        --------
        stats : dict
            Warming statistics
        """
        return {
            'warming_runs': self.warming_runs,
            'items_warmed': self.items_warmed,
            'last_warming_time': (
                self.last_warming_time.isoformat() 
                if self.last_warming_time else None
            )
        }


def cached_computation(cache_manager: FraudDetectionCacheManager,
                      cache_type: str = 'computation',
                      ttl_seconds: Optional[float] = None,
                      key_generator: Optional[Callable] = None):
    """
    Decorator for caching expensive computations.
    
    Parameters:
    -----------
    cache_manager : FraudDetectionCacheManager
        Cache manager to use
    cache_type : str, default='computation'
        Type of cache to use
    ttl_seconds : float, optional
        TTL for cached results
    key_generator : callable, optional
        Custom key generation function
        
    Returns:
    --------
    decorator : callable
        Caching decorator
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                cache_key = cache_manager.generate_cache_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            if cache_type == 'computation':
                cached_result = cache_manager.computation_cache.get(cache_key)
            elif cache_type == 'feature':
                cached_result = cache_manager.feature_cache.get(cache_key)
            elif cache_type == 'prediction':
                cached_result = cache_manager.prediction_cache.get(cache_key)
            else:
                cached_result = None
            
            if cached_result is not None:
                return cached_result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache result
            if cache_type == 'computation':
                cache_manager.computation_cache.put(cache_key, result, ttl_seconds)
            elif cache_type == 'feature':
                cache_manager.feature_cache.put(cache_key, result, ttl_seconds)
            elif cache_type == 'prediction':
                cache_manager.prediction_cache.put(cache_key, result, ttl_seconds)
            
            return result
        
        return wrapper
    return decorator


# Example usage functions
def example_fraud_detection_with_caching():
    """Example of fraud detection with comprehensive caching."""
    
    # Initialize cache manager
    cache_manager = FraudDetectionCacheManager()
    
    # Example cached feature computation
    @cached_computation(cache_manager, cache_type='feature', ttl_seconds=300)
    def compute_customer_features(customer_id: str, transaction_data: Dict) -> Dict[str, Any]:
        """Compute customer features with caching."""
        # Expensive feature computation here
        features = {
            'avg_transaction_amount': 150.0,
            'transaction_frequency': 5.2,
            'risk_score': 0.3
        }
        return features
    
    # Example cached prediction
    @cached_computation(cache_manager, cache_type='prediction', ttl_seconds=60)
    def predict_fraud(features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict fraud with caching."""
        # Expensive model prediction here
        prediction = {
            'fraud_probability': 0.15,
            'risk_level': 'medium',
            'confidence': 0.85
        }
        return prediction
    
    # Warm cache with common data
    warming_data = {
        'customer_profiles': {
            'customer_123': {'tier': 'premium', 'account_age_days': 365},
            'customer_456': {'tier': 'standard', 'account_age_days': 180}
        },
        'common_features': {
            'default_features': {'baseline_risk': 0.1}
        }
    }
    cache_manager.warm_cache(warming_data)
    
    # Example usage
    transaction_data = {'amount': 100.0, 'merchant': 'amazon'}
    features = compute_customer_features('customer_123', transaction_data)
    prediction = predict_fraud(features)
    
    # Get comprehensive statistics
    stats = cache_manager.get_comprehensive_stats()
    
    return cache_manager, stats


if __name__ == "__main__":
    # Run example
    cache_manager, stats = example_fraud_detection_with_caching()
    print("Cache Statistics:")
    print(json.dumps(stats, indent=2, default=str))