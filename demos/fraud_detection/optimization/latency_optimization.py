"""
Latency Optimization for Real-time Fraud Detection

This module provides comprehensive latency optimization techniques to achieve
sub-100ms prediction latency including profiling, monitoring, and optimization.

Key Features:
- Real-time latency monitoring
- Performance profiling and bottleneck detection
- Latency optimization strategies
- Asynchronous processing optimization
- Memory pool management
- Critical path optimization

Author: IntegratedML Pluggable Models Team
"""

import numpy as np
import pandas as pd
import logging
import time
import threading
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
import concurrent.futures
import psutil
import warnings

logger = logging.getLogger(__name__)


class LatencyOptimizer:
    """
    Comprehensive latency optimization for fraud detection system.
    
    This class provides tools and techniques to minimize prediction latency
    and achieve sub-100ms performance targets.
    """
    
    def __init__(self,
                 target_latency_ms: float = 100.0,
                 enable_async_processing: bool = True,
                 enable_precomputation: bool = True,
                 enable_memory_pooling: bool = True,
                 monitoring_window_seconds: int = 60):
        """
        Initialize latency optimizer.
        
        Parameters:
        -----------
        target_latency_ms : float, default=100.0
            Target latency in milliseconds
        enable_async_processing : bool, default=True
            Whether to enable asynchronous processing optimizations
        enable_precomputation : bool, default=True
            Whether to enable precomputation optimizations
        enable_memory_pooling : bool, default=True
            Whether to enable memory pooling
        monitoring_window_seconds : int, default=60
            Performance monitoring window in seconds
        """
        self.target_latency_ms = target_latency_ms
        self.enable_async_processing = enable_async_processing
        self.enable_precomputation = enable_precomputation
        self.enable_memory_pooling = enable_memory_pooling
        self.monitoring_window_seconds = monitoring_window_seconds
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor(monitoring_window_seconds)
        
        # Async processing components
        self.async_executor = None
        if self.enable_async_processing:
            self.async_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # Memory pooling
        self.memory_pools = {}
        
        # Precomputation cache
        self.precomputed_features = {}
        
        # Optimization strategies
        self.optimization_strategies = []
        
        # Critical path tracking
        self.critical_path_timings = defaultdict(list)
        
        logger.info("Initialized LatencyOptimizer")
    
    def optimize_pipeline(self, pipeline_func: Callable) -> Callable:
        """
        Optimize a processing pipeline for latency.
        
        Parameters:
        -----------
        pipeline_func : callable
            Pipeline function to optimize
            
        Returns:
        --------
        optimized_func : callable
            Optimized pipeline function
        """
        def optimized_pipeline(*args, **kwargs):
            start_time = time.time()
            
            # Apply optimization strategies
            optimized_args, optimized_kwargs = self._apply_optimizations(args, kwargs)
            
            # Execute with monitoring
            with self.performance_monitor.measure_operation("pipeline_execution"):
                result = pipeline_func(*optimized_args, **optimized_kwargs)
            
            # Record latency
            latency_ms = (time.time() - start_time) * 1000
            self.performance_monitor.record_latency(latency_ms)
            
            # Check if target met
            if latency_ms > self.target_latency_ms:
                logger.warning(f"Latency target exceeded: {latency_ms:.2f}ms > {self.target_latency_ms}ms")
            
            return result
        
        return optimized_pipeline
    
    def profile_critical_path(self, func: Callable, *args, **kwargs) -> Dict[str, float]:
        """
        Profile critical path execution times.
        
        Parameters:
        -----------
        func : callable
            Function to profile
        *args, **kwargs : arguments
            Function arguments
            
        Returns:
        --------
        timings : dict
            Critical path timings
        """
        timings = {}
        
        # Measure total execution time
        total_start = time.time()
        
        # Profile individual components if available
        if hasattr(func, '__name__'):
            func_name = func.__name__
        else:
            func_name = 'anonymous_function'
        
        try:
            result = func(*args, **kwargs)
            total_time = (time.time() - total_start) * 1000
            
            timings[func_name] = total_time
            
            # Store for analysis
            self.critical_path_timings[func_name].append(total_time)
            
            # Keep only recent timings
            if len(self.critical_path_timings[func_name]) > 1000:
                self.critical_path_timings[func_name] = self.critical_path_timings[func_name][-1000:]
            
        except Exception as e:
            logger.error(f"Profiling failed for {func_name}: {e}")
            timings[func_name] = -1
        
        return timings
    
    def optimize_memory_usage(self, data_size_mb: float) -> Dict[str, Any]:
        """
        Optimize memory usage for given data size.
        
        Parameters:
        -----------
        data_size_mb : float
            Expected data size in MB
            
        Returns:
        --------
        memory_config : dict
            Optimized memory configuration
        """
        # Get current memory usage
        process = psutil.Process()
        current_memory_mb = process.memory_info().rss / 1024 / 1024
        available_memory_mb = psutil.virtual_memory().available / 1024 / 1024
        
        memory_config = {
            'current_memory_mb': current_memory_mb,
            'available_memory_mb': available_memory_mb,
            'target_data_size_mb': data_size_mb,
            'recommendations': []
        }
        
        # Calculate optimal configuration
        if data_size_mb > available_memory_mb * 0.8:
            memory_config['recommendations'].append("Consider batch processing or data streaming")
            memory_config['suggested_batch_size'] = int(available_memory_mb * 0.5 / data_size_mb * 1000)
        
        if current_memory_mb > available_memory_mb * 0.6:
            memory_config['recommendations'].append("High memory usage detected, consider garbage collection")
        
        # Memory pooling recommendations
        if self.enable_memory_pooling:
            memory_config['enable_pooling'] = True
            memory_config['pool_size_mb'] = min(data_size_mb * 2, available_memory_mb * 0.1)
        
        return memory_config
    
    def create_async_pipeline(self, sync_pipeline: Callable) -> Callable:
        """
        Create asynchronous version of pipeline.
        
        Parameters:
        -----------
        sync_pipeline : callable
            Synchronous pipeline function
            
        Returns:
        --------
        async_pipeline : callable
            Asynchronous pipeline function
        """
        if not self.enable_async_processing:
            return sync_pipeline
        
        async def async_pipeline(*args, **kwargs):
            loop = asyncio.get_event_loop()
            
            # Run in thread pool to avoid blocking
            result = await loop.run_in_executor(
                self.async_executor,
                sync_pipeline,
                *args,
                **kwargs
            )
            
            return result
        
        return async_pipeline
    
    def precompute_features(self, feature_func: Callable, 
                           input_ranges: Dict[str, List[Any]]) -> None:
        """
        Precompute features for common input ranges.
        
        Parameters:
        -----------
        feature_func : callable
            Feature computation function
        input_ranges : dict
            Ranges of input values to precompute
        """
        if not self.enable_precomputation:
            return
        
        logger.info("Precomputing features for common inputs...")
        
        precompute_start = time.time()
        computed_count = 0
        
        # Generate combinations of input values
        import itertools
        
        # Create input combinations (limit to avoid explosion)
        max_combinations = 1000
        input_keys = list(input_ranges.keys())
        
        if len(input_keys) <= 3:  # Only precompute for small number of inputs
            input_values = [input_ranges[key] for key in input_keys]
            
            for combination in itertools.product(*input_values):
                if computed_count >= max_combinations:
                    break
                
                try:
                    # Create input dict
                    input_dict = dict(zip(input_keys, combination))
                    
                    # Compute features
                    features = feature_func(input_dict)
                    
                    # Store in cache
                    cache_key = self._generate_precompute_key(input_dict)
                    self.precomputed_features[cache_key] = features
                    
                    computed_count += 1
                    
                except Exception as e:
                    logger.warning(f"Precomputation failed for {combination}: {e}")
                    continue
        
        precompute_time = (time.time() - precompute_start) * 1000
        logger.info(f"Precomputed {computed_count} feature sets in {precompute_time:.2f}ms")
    
    def get_precomputed_features(self, input_dict: Dict[str, Any]) -> Optional[Any]:
        """
        Get precomputed features if available.
        
        Parameters:
        -----------
        input_dict : dict
            Input parameters
            
        Returns:
        --------
        features : any or None
            Precomputed features if available
        """
        if not self.enable_precomputation:
            return None
        
        cache_key = self._generate_precompute_key(input_dict)
        return self.precomputed_features.get(cache_key)
    
    def optimize_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize data types for memory and performance.
        
        Parameters:
        -----------
        data : DataFrame
            Input data
            
        Returns:
        --------
        optimized_data : DataFrame
            Data with optimized types
        """
        optimized = data.copy()
        
        for column in optimized.columns:
            col_data = optimized[column]
            
            # Optimize numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                # Check if can be converted to smaller integer type
                if pd.api.types.is_integer_dtype(col_data):
                    min_val = col_data.min()
                    max_val = col_data.max()
                    
                    if min_val >= 0:  # Unsigned integers
                        if max_val <= 255:
                            optimized[column] = col_data.astype('uint8')
                        elif max_val <= 65535:
                            optimized[column] = col_data.astype('uint16')
                        elif max_val <= 4294967295:
                            optimized[column] = col_data.astype('uint32')
                    else:  # Signed integers
                        if min_val >= -128 and max_val <= 127:
                            optimized[column] = col_data.astype('int8')
                        elif min_val >= -32768 and max_val <= 32767:
                            optimized[column] = col_data.astype('int16')
                        elif min_val >= -2147483648 and max_val <= 2147483647:
                            optimized[column] = col_data.astype('int32')
                
                # Check if float can be converted to float32
                elif pd.api.types.is_float_dtype(col_data):
                    if col_data.max() <= np.finfo(np.float32).max and col_data.min() >= np.finfo(np.float32).min:
                        optimized[column] = col_data.astype('float32')
            
            # Optimize string columns
            elif pd.api.types.is_object_dtype(col_data):
                # Convert to category if low cardinality
                if col_data.nunique() / len(col_data) < 0.5:
                    optimized[column] = col_data.astype('category')
        
        # Log memory savings
        original_memory = data.memory_usage(deep=True).sum() / 1024 / 1024
        optimized_memory = optimized.memory_usage(deep=True).sum() / 1024 / 1024
        savings_pct = (1 - optimized_memory / original_memory) * 100
        
        logger.info(f"Data type optimization: {original_memory:.2f}MB -> {optimized_memory:.2f}MB "
                   f"({savings_pct:.1f}% reduction)")
        
        return optimized
    
    def _apply_optimizations(self, args: Tuple, kwargs: Dict) -> Tuple[Tuple, Dict]:
        """
        Apply optimization strategies to function arguments.
        
        Parameters:
        -----------
        args : tuple
            Function arguments
        kwargs : dict
            Function keyword arguments
            
        Returns:
        --------
        optimized_args, optimized_kwargs : tuple, dict
            Optimized arguments
        """
        # For now, return as-is
        # In production, you might apply specific optimizations
        return args, kwargs
    
    def _generate_precompute_key(self, input_dict: Dict[str, Any]) -> str:
        """
        Generate key for precomputed features.
        
        Parameters:
        -----------
        input_dict : dict
            Input parameters
            
        Returns:
        --------
        key : str
            Cache key
        """
        # Create a sorted string representation
        sorted_items = sorted(input_dict.items())
        key_parts = [f"{k}:{v}" for k, v in sorted_items]
        return "|".join(key_parts)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization report.
        
        Returns:
        --------
        report : dict
            Optimization report
        """
        report = {
            'target_latency_ms': self.target_latency_ms,
            'performance_summary': self.performance_monitor.get_summary(),
            'critical_path_analysis': self._analyze_critical_paths(),
            'memory_analysis': self._analyze_memory_usage(),
            'optimization_recommendations': self._generate_optimization_recommendations()
        }
        
        return report
    
    def _analyze_critical_paths(self) -> Dict[str, Any]:
        """
        Analyze critical path performance.
        
        Returns:
        --------
        analysis : dict
            Critical path analysis
        """
        analysis = {}
        
        for func_name, timings in self.critical_path_timings.items():
            if timings:
                analysis[func_name] = {
                    'avg_time_ms': np.mean(timings),
                    'median_time_ms': np.median(timings),
                    'p95_time_ms': np.percentile(timings, 95),
                    'max_time_ms': np.max(timings),
                    'min_time_ms': np.min(timings),
                    'call_count': len(timings),
                    'target_violations': sum(1 for t in timings if t > self.target_latency_ms)
                }
        
        return analysis
    
    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """
        Analyze current memory usage.
        
        Returns:
        --------
        analysis : dict
            Memory usage analysis
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            vm_info = psutil.virtual_memory()
            
            analysis = {
                'current_memory_mb': memory_info.rss / 1024 / 1024,
                'peak_memory_mb': memory_info.peak_wset / 1024 / 1024 if hasattr(memory_info, 'peak_wset') else None,
                'available_memory_mb': vm_info.available / 1024 / 1024,
                'memory_percent': vm_info.percent,
                'precompute_cache_size': len(self.precomputed_features),
                'memory_pools': {name: len(pool) for name, pool in self.memory_pools.items()}
            }
            
        except Exception as e:
            logger.warning(f"Memory analysis failed: {e}")
            analysis = {'error': str(e)}
        
        return analysis
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """
        Generate optimization recommendations.
        
        Returns:
        --------
        recommendations : list of str
            Optimization recommendations
        """
        recommendations = []
        
        # Analyze performance data
        perf_summary = self.performance_monitor.get_summary()
        
        if perf_summary.get('avg_latency_ms', 0) > self.target_latency_ms:
            recommendations.append(f"Average latency ({perf_summary.get('avg_latency_ms', 0):.2f}ms) exceeds target ({self.target_latency_ms}ms)")
        
        if perf_summary.get('p95_latency_ms', 0) > self.target_latency_ms * 1.5:
            recommendations.append("High P95 latency suggests optimization needed for worst-case scenarios")
        
        # Memory recommendations
        memory_analysis = self._analyze_memory_usage()
        if memory_analysis.get('memory_percent', 0) > 80:
            recommendations.append("High memory usage detected, consider memory optimization")
        
        # Critical path recommendations
        critical_paths = self._analyze_critical_paths()
        for func_name, analysis in critical_paths.items():
            if analysis['avg_time_ms'] > self.target_latency_ms * 0.8:
                recommendations.append(f"Function '{func_name}' is taking {analysis['avg_time_ms']:.2f}ms on average")
        
        # General recommendations
        if not self.enable_async_processing:
            recommendations.append("Consider enabling async processing for better performance")
        
        if not self.enable_precomputation:
            recommendations.append("Consider enabling precomputation for frequently used features")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable targets")
        
        return recommendations
    
    def __del__(self):
        """Cleanup resources."""
        if self.async_executor:
            self.async_executor.shutdown(wait=True)


class PerformanceMonitor:
    """
    Real-time performance monitoring for fraud detection system.
    """
    
    def __init__(self, window_seconds: int = 60):
        """
        Initialize performance monitor.
        
        Parameters:
        -----------
        window_seconds : int, default=60
            Monitoring window in seconds
        """
        self.window_seconds = window_seconds
        
        # Performance metrics
        self.latencies = deque(maxlen=10000)
        self.operation_timings = defaultdict(lambda: deque(maxlen=1000))
        self.timestamps = deque(maxlen=10000)
        
        # System metrics
        self.cpu_usage = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=1000)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_system_metrics, daemon=True)
        self.monitor_thread.start()
    
    def record_latency(self, latency_ms: float) -> None:
        """
        Record prediction latency.
        
        Parameters:
        -----------
        latency_ms : float
            Latency in milliseconds
        """
        with self.lock:
            self.latencies.append(latency_ms)
            self.timestamps.append(datetime.now())
    
    def measure_operation(self, operation_name: str):
        """
        Context manager for measuring operation time.
        
        Parameters:
        -----------
        operation_name : str
            Name of operation to measure
            
        Returns:
        --------
        context_manager : OperationTimer
            Context manager for timing
        """
        return OperationTimer(self, operation_name)
    
    def record_operation_time(self, operation_name: str, time_ms: float) -> None:
        """
        Record operation timing.
        
        Parameters:
        -----------
        operation_name : str
            Operation name
        time_ms : float
            Time in milliseconds
        """
        with self.lock:
            self.operation_timings[operation_name].append(time_ms)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get performance summary.
        
        Returns:
        --------
        summary : dict
            Performance summary
        """
        with self.lock:
            summary = {}
            
            # Latency statistics
            if self.latencies:
                latencies_array = np.array(list(self.latencies))
                summary.update({
                    'avg_latency_ms': np.mean(latencies_array),
                    'median_latency_ms': np.median(latencies_array),
                    'p95_latency_ms': np.percentile(latencies_array, 95),
                    'p99_latency_ms': np.percentile(latencies_array, 99),
                    'max_latency_ms': np.max(latencies_array),
                    'min_latency_ms': np.min(latencies_array),
                    'total_predictions': len(latencies_array)
                })
            
            # Recent performance (last window)
            if self.timestamps:
                cutoff_time = datetime.now() - timedelta(seconds=self.window_seconds)
                recent_indices = [i for i, ts in enumerate(self.timestamps) if ts >= cutoff_time]
                
                if recent_indices:
                    recent_latencies = [self.latencies[i] for i in recent_indices]
                    summary.update({
                        'recent_avg_latency_ms': np.mean(recent_latencies),
                        'recent_predictions_count': len(recent_latencies),
                        'recent_predictions_per_second': len(recent_latencies) / self.window_seconds
                    })
            
            # Operation timings
            operation_summary = {}
            for op_name, timings in self.operation_timings.items():
                if timings:
                    timings_array = np.array(list(timings))
                    operation_summary[op_name] = {
                        'avg_time_ms': np.mean(timings_array),
                        'p95_time_ms': np.percentile(timings_array, 95),
                        'call_count': len(timings_array)
                    }
            
            if operation_summary:
                summary['operations'] = operation_summary
            
            # System metrics
            if self.cpu_usage:
                summary['avg_cpu_percent'] = np.mean(list(self.cpu_usage))
            
            if self.memory_usage:
                summary['avg_memory_mb'] = np.mean(list(self.memory_usage))
        
        return summary
    
    def get_real_time_metrics(self) -> Dict[str, float]:
        """
        Get real-time performance metrics.
        
        Returns:
        --------
        metrics : dict
            Real-time metrics
        """
        with self.lock:
            metrics = {}
            
            # Last 10 predictions
            if len(self.latencies) >= 10:
                recent_latencies = list(self.latencies)[-10:]
                metrics['last_10_avg_latency_ms'] = np.mean(recent_latencies)
            
            # Current system usage
            try:
                metrics['current_cpu_percent'] = psutil.cpu_percent()
                metrics['current_memory_mb'] = psutil.Process().memory_info().rss / 1024 / 1024
            except:
                pass
            
            # Throughput
            if self.timestamps:
                recent_count = sum(1 for ts in self.timestamps 
                                 if ts >= datetime.now() - timedelta(seconds=10))
                metrics['recent_throughput_per_second'] = recent_count / 10
        
        return metrics
    
    def _monitor_system_metrics(self) -> None:
        """Background system metrics monitoring."""
        while self.monitoring_active:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent()
                with self.lock:
                    self.cpu_usage.append(cpu_percent)
                
                # Memory usage
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                with self.lock:
                    self.memory_usage.append(memory_mb)
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                logger.warning(f"System metrics monitoring error: {e}")
                time.sleep(5)  # Wait longer on error
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)


class OperationTimer:
    """
    Context manager for timing operations.
    """
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        """
        Initialize operation timer.
        
        Parameters:
        -----------
        monitor : PerformanceMonitor
            Performance monitor
        operation_name : str
            Operation name
        """
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record."""
        if self.start_time is not None:
            elapsed_ms = (time.time() - self.start_time) * 1000
            self.monitor.record_operation_time(self.operation_name, elapsed_ms)


class AsyncLatencyOptimizer:
    """
    Asynchronous latency optimization for concurrent processing.
    """
    
    def __init__(self, max_concurrent: int = 10):
        """
        Initialize async latency optimizer.
        
        Parameters:
        -----------
        max_concurrent : int, default=10
            Maximum concurrent operations
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def process_batch_async(self, items: List[Any], 
                                 process_func: Callable) -> List[Any]:
        """
        Process batch of items asynchronously.
        
        Parameters:
        -----------
        items : list
            Items to process
        process_func : callable
            Processing function
            
        Returns:
        --------
        results : list
            Processing results
        """
        async def process_with_semaphore(item):
            async with self.semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, process_func, item)
        
        # Process all items concurrently
        tasks = [process_with_semaphore(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results