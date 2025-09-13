"""
Real-time Feature Processing

This module provides real-time feature processing orchestration for fraud detection,
combining all feature engineering components into a unified, high-performance pipeline.

Key Features:
- Unified feature processing pipeline
- Sub-100ms real-time processing optimization
- Feature caching and pre-computation
- Parallel feature computation
- Memory-efficient processing
- Real-time performance monitoring

Author: IntegratedML Pluggable Models Team
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict
import warnings

from .transaction_features import TransactionFeatureEngineer
from .velocity_features import VelocityFeatureEngineer
from .location_features import LocationFeatureEngineer
from .behavioral_features import BehavioralFeatureEngineer
from .risk_features import RiskFeatureEngineer

logger = logging.getLogger(__name__)


class RealTimeFeatureProcessor:
    """
    Real-time feature processing orchestrator for fraud detection.
    
    This class combines all feature engineering components into a unified
    pipeline optimized for real-time processing with sub-100ms latency.
    """
    
    def __init__(self,
                 enable_transaction_features: bool = True,
                 enable_velocity_features: bool = True,
                 enable_location_features: bool = True,
                 enable_behavioral_features: bool = True,
                 enable_risk_features: bool = True,
                 parallel_processing: bool = True,
                 cache_size: int = 10000,
                 performance_target_ms: float = 100.0,
                 **kwargs):
        """
        Initialize real-time feature processor.
        
        Parameters:
        -----------
        enable_transaction_features : bool, default=True
            Whether to compute transaction-level features
        enable_velocity_features : bool, default=True
            Whether to compute velocity features
        enable_location_features : bool, default=True
            Whether to compute location features
        enable_behavioral_features : bool, default=True
            Whether to compute behavioral features
        enable_risk_features : bool, default=True
            Whether to compute risk scoring features
        parallel_processing : bool, default=True
            Whether to use parallel processing for feature computation
        cache_size : int, default=10000
            Size of feature cache for performance optimization
        performance_target_ms : float, default=100.0
            Target processing time in milliseconds
        """
        self.enable_transaction_features = enable_transaction_features
        self.enable_velocity_features = enable_velocity_features
        self.enable_location_features = enable_location_features
        self.enable_behavioral_features = enable_behavioral_features
        self.enable_risk_features = enable_risk_features
        self.parallel_processing = parallel_processing
        self.cache_size = cache_size
        self.performance_target_ms = performance_target_ms

        # Backward-compatibility aliases (from notebooks/tests)
        # - enable_parallel_processing: alias for parallel_processing
        # - enable_caching: boolean toggle for feature cache (accepted but optional)
        # - cache_ttl_seconds: TTL for cache entries (accepted but optional)
        if 'enable_parallel_processing' in kwargs and kwargs['enable_parallel_processing'] is not None:
            self.parallel_processing = bool(kwargs['enable_parallel_processing'])
        self.enable_caching = bool(kwargs.get('enable_caching', False))
        self.cache_ttl_seconds = int(kwargs.get('cache_ttl_seconds', 300))
        
        # Initialize feature engineers
        self.feature_engineers = {}
        
        if self.enable_transaction_features:
            self.feature_engineers['transaction'] = TransactionFeatureEngineer()
        
        if self.enable_velocity_features:
            self.feature_engineers['velocity'] = VelocityFeatureEngineer()
        
        if self.enable_location_features:
            self.feature_engineers['location'] = LocationFeatureEngineer()
        
        if self.enable_behavioral_features:
            self.feature_engineers['behavioral'] = BehavioralFeatureEngineer()
        
        if self.enable_risk_features:
            self.feature_engineers['risk'] = RiskFeatureEngineer()
        
        # Feature caches
        self.feature_cache = {}
        self.cache_timestamps = {}
        self.cache_lock = threading.Lock()
        
        # Performance monitoring
        self.processing_times = []
        self.performance_stats = {
            'total_processed': 0,
            'avg_processing_time_ms': 0.0,
            'target_violations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Threading for parallel processing
        self.thread_pool = None
        if self.parallel_processing:
            self.thread_pool = ThreadPoolExecutor(max_workers=len(self.feature_engineers))
        
        # Fitted status
        self.is_fitted = False
        
        logger.info("Initialized RealTimeFeatureProcessor")
    
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'RealTimeFeatureProcessor':
        """
        Fit all feature engineers on training data.
        
        Parameters:
        -----------
        X : DataFrame
            Training transaction data
        y : array-like, optional
            Fraud labels
            
        Returns:
        --------
        self : RealTimeFeatureProcessor
            Fitted feature processor
        """
        logger.info("Fitting RealTimeFeatureProcessor...")
        start_time = time.time()
        
        # Fit feature engineers in sequence to avoid memory issues
        for name, engineer in self.feature_engineers.items():
            logger.info(f"Fitting {name} feature engineer...")
            
            try:
                if name == 'risk':
                    # Risk engineer needs features from other engineers
                    # First compute other features
                    X_with_features = X.copy()
                    
                    for other_name, other_engineer in self.feature_engineers.items():
                        if other_name != 'risk' and hasattr(other_engineer, 'is_fitted') and other_engineer.is_fitted:
                            X_with_features = other_engineer.transform(X_with_features)
                    
                    engineer.fit(X_with_features, y)
                else:
                    engineer.fit(X, y)
                
                logger.info(f"Successfully fitted {name} feature engineer")
                
            except Exception as e:
                logger.error(f"Failed to fit {name} feature engineer: {e}")
                # Continue with other engineers
                continue
        
        self.is_fitted = True
        
        fit_time = (time.time() - start_time) * 1000
        logger.info(f"RealTimeFeatureProcessor fitted successfully in {fit_time:.2f}ms")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform transaction data into features using all feature engineers.
        
        Parameters:
        -----------
        X : DataFrame
            Transaction data
            
        Returns:
        --------
        X_features : DataFrame
            Data with all features added
        """
        if not self.is_fitted:
            raise ValueError("RealTimeFeatureProcessor must be fitted before transform")
        
        start_time = time.time()
        
        logger.debug(f"Processing {len(X)} transactions with RealTimeFeatureProcessor...")
        
        # Start with copy of original data
        X_features = X.copy()
        
        # Apply feature engineers in sequence
        for name, engineer in self.feature_engineers.items():
            if hasattr(engineer, 'is_fitted') and engineer.is_fitted:
                try:
                    engineer_start = time.time()
                    X_features = engineer.transform(X_features)
                    engineer_time = (time.time() - engineer_start) * 1000
                    
                    logger.debug(f"{name} features computed in {engineer_time:.2f}ms")
                    
                except Exception as e:
                    logger.error(f"Failed to compute {name} features: {e}")
                    continue
        
        processing_time = (time.time() - start_time) * 1000
        self._update_performance_stats(processing_time, len(X))
        
        logger.debug(f"Total feature processing completed in {processing_time:.2f}ms")
        
        return X_features
    
    def transform_single(self, transaction: Dict[str, Any], 
                        use_cache: bool = True) -> Dict[str, float]:
        """
        Transform a single transaction into features for real-time processing.
        
        Parameters:
        -----------
        transaction : dict
            Single transaction data
        use_cache : bool, default=True
            Whether to use feature caching
            
        Returns:
        --------
        features : dict
            Computed features for the transaction
        """
        if not self.is_fitted:
            raise ValueError("RealTimeFeatureProcessor must be fitted before transform")
        
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(transaction) if use_cache else None
        
        # Check cache first
        if use_cache and cache_key:
            cached_features = self._get_from_cache(cache_key)
            if cached_features is not None:
                self.performance_stats['cache_hits'] += 1
                return cached_features
            else:
                self.performance_stats['cache_misses'] += 1
        
        # Compute features
        if self.parallel_processing and self.thread_pool:
            features = self._compute_features_parallel(transaction)
        else:
            features = self._compute_features_sequential(transaction)
        
        # Cache results
        if use_cache and cache_key:
            self._add_to_cache(cache_key, features)
        
        processing_time = (time.time() - start_time) * 1000
        self._update_performance_stats(processing_time, 1)
        
        # Log performance warning if target exceeded
        if processing_time > self.performance_target_ms:
            logger.warning(f"Real-time processing took {processing_time:.2f}ms, "
                         f"exceeding target of {self.performance_target_ms}ms")
            self.performance_stats['target_violations'] += 1
        
        return features
    
    def _compute_features_parallel(self, transaction: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute features using parallel processing.
        
        Parameters:
        -----------
        transaction : dict
            Transaction data
            
        Returns:
        --------
        features : dict
            Computed features
        """
        features = {}
        
        # Submit feature computation tasks
        future_to_engineer = {}
        
        for name, engineer in self.feature_engineers.items():
            if hasattr(engineer, 'is_fitted') and engineer.is_fitted:
                if name == 'velocity' and hasattr(engineer, 'transform_realtime'):
                    future = self.thread_pool.submit(engineer.transform_realtime, transaction)
                elif name == 'risk' and hasattr(engineer, 'compute_realtime_risk'):
                    # Risk features need other features as input
                    future = self.thread_pool.submit(self._compute_risk_features_delayed, 
                                                   engineer, transaction, features)
                else:
                    # Convert to DataFrame for other engineers
                    df = pd.DataFrame([transaction])
                    future = self.thread_pool.submit(engineer.transform, df)
                
                future_to_engineer[future] = name
        
        # Collect results
        for future in as_completed(future_to_engineer):
            engineer_name = future_to_engineer[future]
            
            try:
                result = future.result(timeout=0.05)  # 50ms timeout per engineer
                
                if isinstance(result, dict):
                    features.update(result)
                elif isinstance(result, pd.DataFrame) and len(result) > 0:
                    # Convert DataFrame row to dict
                    row_dict = result.iloc[0].to_dict()
                    features.update(row_dict)
                
            except Exception as e:
                logger.warning(f"Failed to compute {engineer_name} features: {e}")
                continue
        
        return features
    
    def _compute_features_sequential(self, transaction: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute features using sequential processing.
        
        Parameters:
        -----------
        transaction : dict
            Transaction data
            
        Returns:
        --------
        features : dict
            Computed features
        """
        features = {}
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction])
        
        # Apply each feature engineer
        for name, engineer in self.feature_engineers.items():
            if not (hasattr(engineer, 'is_fitted') and engineer.is_fitted):
                continue
            
            try:
                if name == 'velocity' and hasattr(engineer, 'transform_realtime'):
                    # Use real-time method for velocity features
                    velocity_features = engineer.transform_realtime(transaction)
                    features.update(velocity_features)
                    
                elif name == 'risk' and hasattr(engineer, 'compute_realtime_risk'):
                    # Risk features need other features as input
                    risk_features = engineer.compute_realtime_risk(features)
                    features.update(risk_features)
                    
                else:
                    # Use standard transform method
                    df_with_features = engineer.transform(df)
                    
                    # Extract new features (columns not in original df)
                    new_columns = set(df_with_features.columns) - set(df.columns)
                    if new_columns:
                        new_features = df_with_features[list(new_columns)].iloc[0].to_dict()
                        features.update(new_features)
                
            except Exception as e:
                logger.warning(f"Failed to compute {name} features: {e}")
                continue
        
        return features
    
    def _compute_risk_features_delayed(self, risk_engineer, transaction: Dict[str, Any], 
                                     current_features: Dict[str, float]) -> Dict[str, float]:
        """
        Compute risk features after other features are available.
        
        Parameters:
        -----------
        risk_engineer : RiskFeatureEngineer
            Risk feature engineer
        transaction : dict
            Transaction data
        current_features : dict
            Features computed so far
            
        Returns:
        --------
        risk_features : dict
            Risk features
        """
        # Wait a bit for other features to be computed
        time.sleep(0.01)  # 10ms delay
        
        # Combine transaction data with computed features
        combined_features = {**transaction, **current_features}
        
        try:
            return risk_engineer.compute_realtime_risk(combined_features)
        except Exception as e:
            logger.warning(f"Failed to compute risk features: {e}")
            return {}
    
    def _generate_cache_key(self, transaction: Dict[str, Any]) -> str:
        """
        Generate cache key for a transaction.
        
        Parameters:
        -----------
        transaction : dict
            Transaction data
            
        Returns:
        --------
        cache_key : str
            Cache key
        """
        # Use relevant fields for cache key
        key_fields = ['customer_id', 'merchant_name', 'amount', 'payment_method']
        
        key_parts = []
        for field in key_fields:
            value = transaction.get(field, 'unknown')
            key_parts.append(f"{field}:{value}")
        
        # Add timestamp bucket (round to nearest minute for caching)
        if 'timestamp' in transaction:
            timestamp = pd.to_datetime(transaction['timestamp'])
            minute_bucket = timestamp.replace(second=0, microsecond=0)
            key_parts.append(f"time:{minute_bucket}")
        
        return "|".join(key_parts)
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, float]]:
        """
        Get features from cache.
        
        Parameters:
        -----------
        cache_key : str
            Cache key
            
        Returns:
        --------
        features : dict or None
            Cached features if available
        """
        with self.cache_lock:
            if cache_key in self.feature_cache:
                # Check if cache entry is still valid (within 5 minutes)
                cache_time = self.cache_timestamps.get(cache_key)
                if cache_time and (datetime.now() - cache_time).total_seconds() < 300:
                    return self.feature_cache[cache_key].copy()
                else:
                    # Remove stale cache entry
                    del self.feature_cache[cache_key]
                    if cache_key in self.cache_timestamps:
                        del self.cache_timestamps[cache_key]
        
        return None
    
    def _add_to_cache(self, cache_key: str, features: Dict[str, float]) -> None:
        """
        Add features to cache.
        
        Parameters:
        -----------
        cache_key : str
            Cache key
        features : dict
            Features to cache
        """
        with self.cache_lock:
            # Manage cache size
            if len(self.feature_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = min(self.cache_timestamps.keys(), 
                               key=lambda k: self.cache_timestamps[k])
                del self.feature_cache[oldest_key]
                del self.cache_timestamps[oldest_key]
            
            # Add new entry
            self.feature_cache[cache_key] = features.copy()
            self.cache_timestamps[cache_key] = datetime.now()
    
    def _update_performance_stats(self, processing_time_ms: float, num_transactions: int) -> None:
        """
        Update performance statistics.
        
        Parameters:
        -----------
        processing_time_ms : float
            Processing time in milliseconds
        num_transactions : int
            Number of transactions processed
        """
        self.processing_times.append(processing_time_ms)
        
        # Keep only recent processing times (last 1000)
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-1000:]
        
        # Update stats
        self.performance_stats['total_processed'] += num_transactions
        self.performance_stats['avg_processing_time_ms'] = np.mean(self.processing_times)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
        --------
        stats : dict
            Performance statistics
        """
        stats = self.performance_stats.copy()
        
        if self.processing_times:
            stats.update({
                'median_processing_time_ms': np.median(self.processing_times),
                'p95_processing_time_ms': np.percentile(self.processing_times, 95),
                'p99_processing_time_ms': np.percentile(self.processing_times, 99),
                'max_processing_time_ms': np.max(self.processing_times),
                'min_processing_time_ms': np.min(self.processing_times)
            })
        
        # Cache statistics
        with self.cache_lock:
            stats['cache_size'] = len(self.feature_cache)
            stats['cache_hit_rate'] = (
                self.performance_stats['cache_hits'] / 
                max(self.performance_stats['cache_hits'] + self.performance_stats['cache_misses'], 1)
            ) * 100
        
        # Performance compliance
        target_compliance = 100 - (
            self.performance_stats['target_violations'] / 
            max(self.performance_stats['total_processed'], 1) * 100
        )
        stats['target_compliance_pct'] = target_compliance
        
        return stats
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all features that will be generated.
        
        Returns:
        --------
        feature_names : list of str
            List of all feature names
        """
        all_features = []
        
        for name, engineer in self.feature_engineers.items():
            if hasattr(engineer, 'get_feature_names'):
                try:
                    engineer_features = engineer.get_feature_names()
                    all_features.extend(engineer_features)
                except Exception as e:
                    logger.warning(f"Could not get feature names from {name} engineer: {e}")
        
        return all_features
    
    def clear_cache(self) -> None:
        """
        Clear the feature cache.
        """
        with self.cache_lock:
            self.feature_cache.clear()
            self.cache_timestamps.clear()
        
        logger.info("Feature cache cleared")
    
    def warm_up_cache(self, sample_transactions: List[Dict[str, Any]]) -> None:
        """
        Warm up the cache with sample transactions.
        
        Parameters:
        -----------
        sample_transactions : list of dict
            Sample transactions for cache warming
        """
        logger.info(f"Warming up cache with {len(sample_transactions)} sample transactions...")
        
        for transaction in sample_transactions:
            try:
                self.transform_single(transaction, use_cache=True)
            except Exception as e:
                logger.warning(f"Failed to warm up cache for transaction: {e}")
        
        logger.info("Cache warm-up completed")
    
    def benchmark_performance(self, test_transactions: List[Dict[str, Any]], 
                            iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark processing performance.
        
        Parameters:
        -----------
        test_transactions : list of dict
            Test transactions for benchmarking
        iterations : int, default=100
            Number of iterations to run
            
        Returns:
        --------
        benchmark_results : dict
            Benchmark results
        """
        logger.info(f"Running performance benchmark with {iterations} iterations...")
        
        processing_times = []
        
        for i in range(iterations):
            transaction = test_transactions[i % len(test_transactions)]
            
            start_time = time.time()
            self.transform_single(transaction, use_cache=False)  # No cache for fair benchmark
            processing_time = (time.time() - start_time) * 1000
            
            processing_times.append(processing_time)
        
        results = {
            'mean_time_ms': np.mean(processing_times),
            'median_time_ms': np.median(processing_times),
            'p95_time_ms': np.percentile(processing_times, 95),
            'p99_time_ms': np.percentile(processing_times, 99),
            'max_time_ms': np.max(processing_times),
            'min_time_ms': np.min(processing_times),
            'target_compliance_pct': (np.array(processing_times) <= self.performance_target_ms).mean() * 100,
            'transactions_per_second': 1000 / np.mean(processing_times)
        }
        
        logger.info(f"Benchmark completed. Average processing time: {results['mean_time_ms']:.2f}ms")
        
        return results
    
    def __del__(self):
        """Cleanup resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)


class FeaturePipelineOptimizer:
    """
    Optimizer for feature processing pipeline performance.
    
    This class provides tools to optimize the feature processing pipeline
    for real-time performance requirements.
    """
    
    def __init__(self, processor: RealTimeFeatureProcessor):
        """
        Initialize pipeline optimizer.
        
        Parameters:
        -----------
        processor : RealTimeFeatureProcessor
            Feature processor to optimize
        """
        self.processor = processor
        
    def optimize_for_latency(self, sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize pipeline for minimum latency.
        
        Parameters:
        -----------
        sample_data : list of dict
            Sample transactions for optimization
            
        Returns:
        --------
        optimization_results : dict
            Optimization results and recommendations
        """
        logger.info("Optimizing pipeline for latency...")
        
        # Test different configurations
        configurations = [
            {'parallel_processing': True, 'cache_size': 1000},
            {'parallel_processing': True, 'cache_size': 5000},
            {'parallel_processing': True, 'cache_size': 10000},
            {'parallel_processing': False, 'cache_size': 5000}
        ]
        
        results = {}
        
        for i, config in enumerate(configurations):
            logger.info(f"Testing configuration {i+1}: {config}")
            
            # Create test processor with configuration
            test_processor = RealTimeFeatureProcessor(**config)
            
            # Fit on sample data (using DataFrame)
            if sample_data:
                df_sample = pd.DataFrame(sample_data[:100])  # Use subset for fitting
                test_processor.fit(df_sample)
                
                # Benchmark performance
                benchmark_results = test_processor.benchmark_performance(sample_data[:10], iterations=20)
                results[f"config_{i+1}"] = {
                    'configuration': config,
                    'performance': benchmark_results
                }
        
        # Find best configuration
        best_config = None
        best_time = float('inf')
        
        for config_name, config_data in results.items():
            mean_time = config_data['performance']['mean_time_ms']
            if mean_time < best_time:
                best_time = mean_time
                best_config = config_name
        
        optimization_results = {
            'all_results': results,
            'best_configuration': best_config,
            'best_performance': results[best_config] if best_config else None,
            'recommendations': self._generate_recommendations(results)
        }
        
        return optimization_results
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate optimization recommendations.
        
        Parameters:
        -----------
        results : dict
            Optimization results
            
        Returns:
        --------
        recommendations : list of str
            Optimization recommendations
        """
        recommendations = []
        
        # Analyze parallel processing benefit
        parallel_times = []
        sequential_times = []
        
        for config_data in results.values():
            config = config_data['configuration']
            mean_time = config_data['performance']['mean_time_ms']
            
            if config['parallel_processing']:
                parallel_times.append(mean_time)
            else:
                sequential_times.append(mean_time)
        
        if parallel_times and sequential_times:
            avg_parallel = np.mean(parallel_times)
            avg_sequential = np.mean(sequential_times)
            
            if avg_parallel < avg_sequential:
                recommendations.append("Enable parallel processing for better performance")
            else:
                recommendations.append("Sequential processing may be sufficient")
        
        # Analyze cache size impact
        cache_performance = {}
        for config_data in results.values():
            cache_size = config_data['configuration']['cache_size']
            mean_time = config_data['performance']['mean_time_ms']
            cache_performance[cache_size] = mean_time
        
        if len(cache_performance) > 1:
            best_cache_size = min(cache_performance.keys(), key=lambda k: cache_performance[k])
            recommendations.append(f"Optimal cache size appears to be {best_cache_size}")
        
        return recommendations