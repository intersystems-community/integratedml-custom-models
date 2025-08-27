"""
Memory Optimization for Real-time Fraud Detection

This module provides comprehensive memory optimization techniques including
memory pooling, garbage collection optimization, and memory-efficient data structures.

Key Features:
- Memory pooling for frequent allocations
- Garbage collection optimization
- Memory-efficient data structures
- Memory usage monitoring and alerts
- Object recycling and reuse
- Memory leak detection

Author: IntegratedML Pluggable Models Team
"""

import numpy as np
import pandas as pd
import logging
import gc
import sys
import threading
import time
import weakref
from typing import Dict, List, Any, Optional, Union, Tuple, Type
from datetime import datetime, timedelta
from collections import defaultdict, deque
import psutil
import warnings
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """
    Comprehensive memory optimization for fraud detection system.
    
    This class provides memory management strategies to minimize memory usage
    and prevent memory-related performance degradation.
    """
    
    def __init__(self,
                 enable_pooling: bool = True,
                 enable_gc_optimization: bool = True,
                 memory_threshold_mb: float = 1000.0,
                 gc_frequency_seconds: int = 30,
                 enable_monitoring: bool = True):
        """
        Initialize memory optimizer.
        
        Parameters:
        -----------
        enable_pooling : bool, default=True
            Whether to enable memory pooling
        enable_gc_optimization : bool, default=True
            Whether to enable garbage collection optimization
        memory_threshold_mb : float, default=1000.0
            Memory usage threshold in MB for alerts
        gc_frequency_seconds : int, default=30
            Garbage collection frequency in seconds
        enable_monitoring : bool, default=True
            Whether to enable memory monitoring
        """
        self.enable_pooling = enable_pooling
        self.enable_gc_optimization = enable_gc_optimization
        self.memory_threshold_mb = memory_threshold_mb
        self.gc_frequency_seconds = gc_frequency_seconds
        self.enable_monitoring = enable_monitoring
        
        # Memory pools
        self.memory_pools = {}
        self.pool_stats = defaultdict(lambda: {'allocations': 0, 'deallocations': 0, 'peak_size': 0})
        
        # Memory monitoring
        if self.enable_monitoring:
            self.memory_monitor = MemoryMonitor(self.memory_threshold_mb)
        
        # Garbage collection optimization
        if self.enable_gc_optimization:
            self.gc_optimizer = GarbageCollectionOptimizer(self.gc_frequency_seconds)
        
        # Object recycling
        self.object_recycler = ObjectRecycler()
        
        # Memory leak detection
        self.leak_detector = MemoryLeakDetector()
        
        logger.info("Initialized MemoryOptimizer")
    
    def create_memory_pool(self, pool_name: str, 
                          object_type: Type, 
                          initial_size: int = 100,
                          max_size: int = 1000) -> 'MemoryPool':
        """
        Create a memory pool for specific object type.
        
        Parameters:
        -----------
        pool_name : str
            Name of the memory pool
        object_type : type
            Type of objects to pool
        initial_size : int, default=100
            Initial pool size
        max_size : int, default=1000
            Maximum pool size
            
        Returns:
        --------
        pool : MemoryPool
            Created memory pool
        """
        if not self.enable_pooling:
            return None
        
        pool = MemoryPool(pool_name, object_type, initial_size, max_size)
        self.memory_pools[pool_name] = pool
        
        logger.info(f"Created memory pool '{pool_name}' for {object_type.__name__}")
        return pool
    
    def optimize_dataframe(self, df: pd.DataFrame, 
                          aggressive: bool = False) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame to optimize
        aggressive : bool, default=False
            Whether to use aggressive optimization
            
        Returns:
        --------
        optimized_df : DataFrame
            Memory-optimized DataFrame
        """
        original_memory = df.memory_usage(deep=True).sum()
        optimized = df.copy()
        
        for column in optimized.columns:
            col_data = optimized[column]
            
            # Skip if column is empty
            if col_data.empty:
                continue
            
            # Optimize numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                optimized[column] = self._optimize_numeric_column(col_data, aggressive)
            
            # Optimize string columns
            elif pd.api.types.is_object_dtype(col_data):
                optimized[column] = self._optimize_string_column(col_data, aggressive)
            
            # Optimize datetime columns
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                optimized[column] = self._optimize_datetime_column(col_data, aggressive)
        
        # Calculate memory savings
        final_memory = optimized.memory_usage(deep=True).sum()
        savings_bytes = original_memory - final_memory
        savings_pct = (savings_bytes / original_memory) * 100 if original_memory > 0 else 0
        
        logger.info(f"DataFrame memory optimization: {original_memory/1024/1024:.2f}MB -> "
                   f"{final_memory/1024/1024:.2f}MB ({savings_pct:.1f}% reduction)")
        
        return optimized
    
    def optimize_numpy_arrays(self, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Optimize NumPy arrays for memory usage.
        
        Parameters:
        -----------
        arrays : dict
            Dictionary of arrays to optimize
            
        Returns:
        --------
        optimized_arrays : dict
            Memory-optimized arrays
        """
        optimized = {}
        total_original_size = 0
        total_optimized_size = 0
        
        for name, array in arrays.items():
            original_size = array.nbytes
            total_original_size += original_size
            
            # Optimize array dtype
            if array.dtype == np.float64:
                # Check if values fit in float32
                if np.all(np.isfinite(array)) and np.all(np.abs(array) <= np.finfo(np.float32).max):
                    optimized_array = array.astype(np.float32)
                else:
                    optimized_array = array
            elif array.dtype == np.int64:
                # Check if values fit in smaller integer types
                min_val, max_val = array.min(), array.max()
                if min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                    optimized_array = array.astype(np.int32)
                elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
                    optimized_array = array.astype(np.int16)
                elif min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
                    optimized_array = array.astype(np.int8)
                else:
                    optimized_array = array
            else:
                optimized_array = array
            
            optimized[name] = optimized_array
            total_optimized_size += optimized_array.nbytes
        
        # Log optimization results
        if total_original_size > 0:
            savings_pct = ((total_original_size - total_optimized_size) / total_original_size) * 100
            logger.info(f"NumPy arrays optimization: {total_original_size/1024/1024:.2f}MB -> "
                       f"{total_optimized_size/1024/1024:.2f}MB ({savings_pct:.1f}% reduction)")
        
        return optimized
    
    @contextmanager
    def memory_efficient_processing(self, operation_name: str = "operation"):
        """
        Context manager for memory-efficient processing.
        
        Parameters:
        -----------
        operation_name : str, default="operation"
            Name of the operation for monitoring
        """
        # Record initial memory state
        initial_memory = self._get_memory_usage()
        
        try:
            # Force garbage collection before operation
            if self.enable_gc_optimization:
                gc.collect()
            
            logger.debug(f"Starting memory-efficient {operation_name}")
            yield
            
        finally:
            # Cleanup after operation
            final_memory = self._get_memory_usage()
            memory_change = final_memory - initial_memory
            
            # Force garbage collection after operation
            if self.enable_gc_optimization:
                gc.collect()
                post_gc_memory = self._get_memory_usage()
                gc_savings = final_memory - post_gc_memory
                
                logger.debug(f"Memory change for {operation_name}: {memory_change:.2f}MB, "
                           f"GC freed: {gc_savings:.2f}MB")
            
            # Check for memory threshold violations
            if self.enable_monitoring and final_memory > self.memory_threshold_mb:
                logger.warning(f"Memory usage ({final_memory:.2f}MB) exceeds threshold "
                             f"({self.memory_threshold_mb}MB) after {operation_name}")
    
    def recycle_object(self, obj: Any, object_type: str) -> None:
        """
        Recycle object for reuse.
        
        Parameters:
        -----------
        obj : any
            Object to recycle
        object_type : str
            Type identifier for the object
        """
        if self.enable_pooling:
            self.object_recycler.recycle(obj, object_type)
    
    def get_recycled_object(self, object_type: str) -> Any:
        """
        Get recycled object if available.
        
        Parameters:
        -----------
        object_type : str
            Type of object to retrieve
            
        Returns:
        --------
        obj : any or None
            Recycled object if available
        """
        if self.enable_pooling:
            return self.object_recycler.get_object(object_type)
        return None
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """
        Force garbage collection and return statistics.
        
        Returns:
        --------
        stats : dict
            Garbage collection statistics
        """
        if not self.enable_gc_optimization:
            return {}
        
        # Get initial counts
        initial_counts = [len(gc.get_objects(generation)) for generation in range(3)]
        
        # Force collection
        collected = gc.collect()
        
        # Get final counts
        final_counts = [len(gc.get_objects(generation)) for generation in range(3)]
        
        stats = {
            'objects_collected': collected,
            'generation_0_freed': initial_counts[0] - final_counts[0],
            'generation_1_freed': initial_counts[1] - final_counts[1],
            'generation_2_freed': initial_counts[2] - final_counts[2],
        }
        
        logger.info(f"Garbage collection freed {collected} objects")
        return stats
    
    def detect_memory_leaks(self) -> Dict[str, Any]:
        """
        Detect potential memory leaks.
        
        Returns:
        --------
        leak_report : dict
            Memory leak detection report
        """
        return self.leak_detector.check_for_leaks()
    
    def get_memory_report(self) -> Dict[str, Any]:
        """
        Get comprehensive memory usage report.
        
        Returns:
        --------
        report : dict
            Memory usage report
        """
        report = {
            'current_memory_mb': self._get_memory_usage(),
            'memory_threshold_mb': self.memory_threshold_mb,
            'gc_stats': gc.get_stats() if hasattr(gc, 'get_stats') else {},
            'pool_stats': dict(self.pool_stats),
            'recycler_stats': self.object_recycler.get_stats(),
        }
        
        if self.enable_monitoring:
            report['monitoring_stats'] = self.memory_monitor.get_stats()
        
        if hasattr(self, 'leak_detector'):
            report['leak_detection'] = self.leak_detector.get_summary()
        
        return report
    
    def _optimize_numeric_column(self, col: pd.Series, aggressive: bool = False) -> pd.Series:
        """
        Optimize numeric column memory usage.
        
        Parameters:
        -----------
        col : Series
            Numeric column to optimize
        aggressive : bool, default=False
            Whether to use aggressive optimization
            
        Returns:
        --------
        optimized_col : Series
            Optimized column
        """
        if pd.api.types.is_integer_dtype(col):
            return self._optimize_integer_column(col, aggressive)
        elif pd.api.types.is_float_dtype(col):
            return self._optimize_float_column(col, aggressive)
        else:
            return col
    
    def _optimize_integer_column(self, col: pd.Series, aggressive: bool = False) -> pd.Series:
        """Optimize integer column."""
        min_val = col.min()
        max_val = col.max()
        
        # Check for unsigned integers
        if min_val >= 0:
            if max_val <= np.iinfo(np.uint8).max:
                return col.astype('uint8')
            elif max_val <= np.iinfo(np.uint16).max:
                return col.astype('uint16')
            elif max_val <= np.iinfo(np.uint32).max:
                return col.astype('uint32')
        
        # Check for signed integers
        if (min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max):
            return col.astype('int8')
        elif (min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max):
            return col.astype('int16')
        elif (min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max):
            return col.astype('int32')
        
        return col
    
    def _optimize_float_column(self, col: pd.Series, aggressive: bool = False) -> pd.Series:
        """Optimize float column."""
        # Check if all values are integers
        if aggressive and col.dropna().apply(lambda x: x.is_integer()).all():
            return self._optimize_integer_column(col.astype('int64'), aggressive)
        
        # Check if can downcast to float32
        if (col.max() <= np.finfo(np.float32).max and 
            col.min() >= np.finfo(np.float32).min):
            return col.astype('float32')
        
        return col
    
    def _optimize_string_column(self, col: pd.Series, aggressive: bool = False) -> pd.Series:
        """Optimize string column."""
        # Convert to category if low cardinality
        nunique = col.nunique()
        total_len = len(col)
        
        if nunique / total_len < 0.5:  # Less than 50% unique values
            return col.astype('category')
        
        return col
    
    def _optimize_datetime_column(self, col: pd.Series, aggressive: bool = False) -> pd.Series:
        """Optimize datetime column."""
        # For now, keep as-is
        # Could potentially downcast datetime resolution if needed
        return col
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
        --------
        memory_mb : float
            Current memory usage in MB
        """
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0


class MemoryPool:
    """
    Memory pool for efficient object allocation and reuse.
    """
    
    def __init__(self, name: str, object_type: Type, 
                 initial_size: int = 100, max_size: int = 1000):
        """
        Initialize memory pool.
        
        Parameters:
        -----------
        name : str
            Pool name
        object_type : type
            Type of objects to pool
        initial_size : int, default=100
            Initial pool size
        max_size : int, default=1000
            Maximum pool size
        """
        self.name = name
        self.object_type = object_type
        self.max_size = max_size
        
        # Create initial pool
        self.available_objects = deque()
        for _ in range(initial_size):
            try:
                obj = object_type()
                self.available_objects.append(obj)
            except Exception as e:
                logger.warning(f"Failed to create object for pool {name}: {e}")
                break
        
        # Statistics
        self.allocations = 0
        self.deallocations = 0
        self.peak_size = len(self.available_objects)
        
        # Thread safety
        self.lock = threading.RLock()
    
    def get_object(self) -> Any:
        """
        Get object from pool.
        
        Returns:
        --------
        obj : any
            Object from pool or newly created
        """
        with self.lock:
            if self.available_objects:
                obj = self.available_objects.popleft()
                self.allocations += 1
                return obj
            else:
                # Pool empty, create new object
                try:
                    obj = self.object_type()
                    self.allocations += 1
                    return obj
                except Exception as e:
                    logger.error(f"Failed to create object for pool {self.name}: {e}")
                    return None
    
    def return_object(self, obj: Any) -> bool:
        """
        Return object to pool.
        
        Parameters:
        -----------
        obj : any
            Object to return
            
        Returns:
        --------
        success : bool
            Whether object was successfully returned
        """
        if not isinstance(obj, self.object_type):
            return False
        
        with self.lock:
            if len(self.available_objects) < self.max_size:
                # Reset object state if possible
                if hasattr(obj, 'reset'):
                    try:
                        obj.reset()
                    except:
                        pass
                
                self.available_objects.append(obj)
                self.deallocations += 1
                
                # Update peak size
                current_size = len(self.available_objects)
                if current_size > self.peak_size:
                    self.peak_size = current_size
                
                return True
            else:
                # Pool full, let object be garbage collected
                return False
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get pool statistics.
        
        Returns:
        --------
        stats : dict
            Pool statistics
        """
        with self.lock:
            return {
                'name': self.name,
                'available_objects': len(self.available_objects),
                'allocations': self.allocations,
                'deallocations': self.deallocations,
                'peak_size': self.peak_size,
                'max_size': self.max_size
            }


class ObjectRecycler:
    """
    Object recycling system for memory efficiency.
    """
    
    def __init__(self, max_objects_per_type: int = 100):
        """
        Initialize object recycler.
        
        Parameters:
        -----------
        max_objects_per_type : int, default=100
            Maximum objects to keep per type
        """
        self.max_objects_per_type = max_objects_per_type
        self.recycled_objects = defaultdict(deque)
        self.stats = defaultdict(lambda: {'recycled': 0, 'reused': 0})
        self.lock = threading.RLock()
    
    def recycle(self, obj: Any, object_type: str) -> bool:
        """
        Recycle object for reuse.
        
        Parameters:
        -----------
        obj : any
            Object to recycle
        object_type : str
            Type identifier
            
        Returns:
        --------
        success : bool
            Whether object was recycled
        """
        with self.lock:
            if len(self.recycled_objects[object_type]) < self.max_objects_per_type:
                # Clear object state if possible
                if hasattr(obj, 'clear'):
                    try:
                        obj.clear()
                    except:
                        pass
                
                self.recycled_objects[object_type].append(obj)
                self.stats[object_type]['recycled'] += 1
                return True
            
            return False
    
    def get_object(self, object_type: str) -> Any:
        """
        Get recycled object.
        
        Parameters:
        -----------
        object_type : str
            Type of object to get
            
        Returns:
        --------
        obj : any or None
            Recycled object if available
        """
        with self.lock:
            if self.recycled_objects[object_type]:
                obj = self.recycled_objects[object_type].popleft()
                self.stats[object_type]['reused'] += 1
                return obj
            
            return None
    
    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get recycling statistics.
        
        Returns:
        --------
        stats : dict
            Recycling statistics
        """
        with self.lock:
            return dict(self.stats)


class MemoryMonitor:
    """
    Real-time memory usage monitoring.
    """
    
    def __init__(self, threshold_mb: float = 1000.0, monitoring_interval: int = 5):
        """
        Initialize memory monitor.
        
        Parameters:
        -----------
        threshold_mb : float, default=1000.0
            Memory threshold in MB
        monitoring_interval : int, default=5
            Monitoring interval in seconds
        """
        self.threshold_mb = threshold_mb
        self.monitoring_interval = monitoring_interval
        
        # Monitoring data
        self.memory_history = deque(maxlen=1000)
        self.threshold_violations = []
        self.peak_memory = 0.0
        
        # Monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_memory(self) -> None:
        """Background memory monitoring."""
        while self.monitoring_active:
            try:
                current_memory = self._get_memory_usage()
                timestamp = datetime.now()
                
                # Record memory usage
                self.memory_history.append((timestamp, current_memory))
                
                # Update peak memory
                if current_memory > self.peak_memory:
                    self.peak_memory = current_memory
                
                # Check threshold
                if current_memory > self.threshold_mb:
                    self.threshold_violations.append((timestamp, current_memory))
                    logger.warning(f"Memory threshold exceeded: {current_memory:.2f}MB > {self.threshold_mb}MB")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(self.monitoring_interval * 2)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get monitoring statistics.
        
        Returns:
        --------
        stats : dict
            Monitoring statistics
        """
        current_memory = self._get_memory_usage()
        
        # Calculate averages
        recent_memory = [mem for _, mem in list(self.memory_history)[-60:]]  # Last 60 readings
        avg_memory = np.mean(recent_memory) if recent_memory else 0
        
        return {
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.peak_memory,
            'avg_memory_mb': avg_memory,
            'threshold_mb': self.threshold_mb,
            'threshold_violations': len(self.threshold_violations),
            'monitoring_duration_minutes': len(self.memory_history) * self.monitoring_interval / 60
        }
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)


class GarbageCollectionOptimizer:
    """
    Garbage collection optimization.
    """
    
    def __init__(self, frequency_seconds: int = 30):
        """
        Initialize GC optimizer.
        
        Parameters:
        -----------
        frequency_seconds : int, default=30
            GC frequency in seconds
        """
        self.frequency_seconds = frequency_seconds
        
        # GC statistics
        self.gc_runs = 0
        self.total_collected = 0
        self.last_gc_time = None
        
        # Configure GC thresholds
        self._optimize_gc_thresholds()
        
        # Start periodic GC
        self.gc_active = True
        self.gc_thread = threading.Thread(target=self._periodic_gc, daemon=True)
        self.gc_thread.start()
    
    def _optimize_gc_thresholds(self) -> None:
        """Optimize garbage collection thresholds."""
        # Get current thresholds
        current_thresholds = gc.get_threshold()
        
        # Set more aggressive thresholds for real-time systems
        # Reduce generation 0 threshold to collect more frequently
        new_thresholds = (
            int(current_thresholds[0] * 0.7),  # More frequent gen 0 collection
            int(current_thresholds[1] * 0.8),  # Slightly more frequent gen 1
            current_thresholds[2]              # Keep gen 2 as-is
        )
        
        gc.set_threshold(*new_thresholds)
        logger.info(f"Optimized GC thresholds: {current_thresholds} -> {new_thresholds}")
    
    def _periodic_gc(self) -> None:
        """Periodic garbage collection."""
        while self.gc_active:
            try:
                # Run garbage collection
                collected = gc.collect()
                
                # Update statistics
                self.gc_runs += 1
                self.total_collected += collected
                self.last_gc_time = datetime.now()
                
                if collected > 0:
                    logger.debug(f"Periodic GC collected {collected} objects")
                
                time.sleep(self.frequency_seconds)
                
            except Exception as e:
                logger.error(f"Periodic GC error: {e}")
                time.sleep(self.frequency_seconds * 2)
    
    def force_full_gc(self) -> Dict[str, int]:
        """
        Force full garbage collection.
        
        Returns:
        --------
        stats : dict
            GC statistics
        """
        # Collect all generations
        collected_gen0 = gc.collect(0)
        collected_gen1 = gc.collect(1)
        collected_gen2 = gc.collect(2)
        
        total_collected = collected_gen0 + collected_gen1 + collected_gen2
        self.total_collected += total_collected
        
        stats = {
            'generation_0': collected_gen0,
            'generation_1': collected_gen1,
            'generation_2': collected_gen2,
            'total_collected': total_collected
        }
        
        logger.info(f"Full GC collected {total_collected} objects")
        return stats
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get GC statistics.
        
        Returns:
        --------
        stats : dict
            GC statistics
        """
        return {
            'gc_runs': self.gc_runs,
            'total_collected': self.total_collected,
            'last_gc_time': self.last_gc_time.isoformat() if self.last_gc_time else None,
            'gc_thresholds': gc.get_threshold(),
            'gc_counts': gc.get_count()
        }
    
    def stop_gc(self) -> None:
        """Stop periodic garbage collection."""
        self.gc_active = False
        if self.gc_thread.is_alive():
            self.gc_thread.join(timeout=2)


class MemoryLeakDetector:
    """
    Memory leak detection system.
    """
    
    def __init__(self, check_interval_minutes: int = 5):
        """
        Initialize memory leak detector.
        
        Parameters:
        -----------
        check_interval_minutes : int, default=5
            Check interval in minutes
        """
        self.check_interval_minutes = check_interval_minutes
        
        # Tracking data
        self.object_counts = defaultdict(list)
        self.memory_snapshots = []
        
        # Weak references to track object lifecycle
        self.tracked_objects = weakref.WeakSet()
        
        # Detection thread
        self.detection_active = True
        self.detection_thread = threading.Thread(target=self._periodic_check, daemon=True)
        self.detection_thread.start()
    
    def track_object(self, obj: Any) -> None:
        """
        Track object for leak detection.
        
        Parameters:
        -----------
        obj : any
            Object to track
        """
        self.tracked_objects.add(obj)
    
    def _periodic_check(self) -> None:
        """Periodic leak detection check."""
        while self.detection_active:
            try:
                self._check_for_leaks()
                time.sleep(self.check_interval_minutes * 60)
            except Exception as e:
                logger.error(f"Leak detection error: {e}")
                time.sleep(self.check_interval_minutes * 60 * 2)
    
    def check_for_leaks(self) -> Dict[str, Any]:
        """
        Check for potential memory leaks.
        
        Returns:
        --------
        leak_report : dict
            Leak detection report
        """
        return self._check_for_leaks()
    
    def _check_for_leaks(self) -> Dict[str, Any]:
        """Internal leak detection implementation."""
        # Get current object counts by type
        current_objects = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            current_objects[obj_type] = current_objects.get(obj_type, 0) + 1
        
        # Record snapshot
        timestamp = datetime.now()
        self.memory_snapshots.append((timestamp, dict(current_objects)))
        
        # Keep only recent snapshots
        if len(self.memory_snapshots) > 100:
            self.memory_snapshots = self.memory_snapshots[-100:]
        
        # Analyze trends
        potential_leaks = []
        if len(self.memory_snapshots) >= 3:
            # Compare with earlier snapshots
            old_snapshot = self.memory_snapshots[-3][1]
            current_snapshot = current_objects
            
            for obj_type, current_count in current_snapshot.items():
                old_count = old_snapshot.get(obj_type, 0)
                if current_count > old_count * 1.5 and current_count > 100:
                    potential_leaks.append({
                        'type': obj_type,
                        'old_count': old_count,
                        'current_count': current_count,
                        'growth_factor': current_count / max(old_count, 1)
                    })
        
        leak_report = {
            'timestamp': timestamp.isoformat(),
            'total_objects': sum(current_objects.values()),
            'tracked_objects': len(self.tracked_objects),
            'potential_leaks': potential_leaks,
            'top_object_types': sorted(current_objects.items(), 
                                     key=lambda x: x[1], reverse=True)[:10]
        }
        
        if potential_leaks:
            logger.warning(f"Potential memory leaks detected: {len(potential_leaks)} object types")
        
        return leak_report
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get leak detection summary.
        
        Returns:
        --------
        summary : dict
            Leak detection summary
        """
        return {
            'snapshots_taken': len(self.memory_snapshots),
            'tracking_objects': len(self.tracked_objects),
            'check_interval_minutes': self.check_interval_minutes,
            'detection_active': self.detection_active
        }
    
    def stop_detection(self) -> None:
        """Stop leak detection."""
        self.detection_active = False
        if self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2)


def memory_efficient_decorator(func):
    """
    Decorator for memory-efficient function execution.
    
    Parameters:
    -----------
    func : callable
        Function to decorate
        
    Returns:
    --------
    decorated_func : callable
        Memory-efficient wrapper
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Force GC before execution
        gc.collect()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Force GC after execution
            gc.collect()
    
    return wrapper