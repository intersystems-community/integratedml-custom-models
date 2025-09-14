"""
Batch Optimization for Real-time Fraud Detection

This module provides comprehensive batch processing optimization techniques
for handling multiple fraud detection requests efficiently.

Key Features:
- Intelligent batch formation and processing
- Dynamic batch size optimization
- Parallel batch processing
- Batch scheduling and prioritization
- Memory-efficient batch handling
- Throughput optimization

Author: IntegratedML Pluggable Models Team
"""

import numpy as np
import pandas as pd
import logging
import time
import threading
import asyncio
import queue
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta
from collections import deque, defaultdict
import concurrent.futures
from dataclasses import dataclass
from enum import Enum
import heapq
import math

logger = logging.getLogger(__name__)


class BatchPriority(Enum):
    """Batch processing priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BatchRequest:
    """Individual batch processing request."""

    request_id: str
    data: Any
    priority: BatchPriority = BatchPriority.NORMAL
    timestamp: datetime = None
    callback: Optional[Callable] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class BatchResult:
    """Batch processing result."""

    request_id: str
    result: Any
    processing_time_ms: float
    batch_id: str
    timestamp: datetime
    error: Optional[str] = None


class BatchOptimizer:
    """
    Comprehensive batch optimization for fraud detection system.

    This class provides intelligent batch formation, processing, and optimization
    to maximize throughput while maintaining low latency for individual requests.
    """

    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 100,
        max_wait_time_ms: float = 50.0,
        enable_dynamic_sizing: bool = True,
        enable_priority_queuing: bool = True,
        max_concurrent_batches: int = 4,
    ):
        """
        Initialize batch optimizer.

        Parameters:
        -----------
        min_batch_size : int, default=1
            Minimum batch size
        max_batch_size : int, default=100
            Maximum batch size
        max_wait_time_ms : float, default=50.0
            Maximum wait time for batch formation in milliseconds
        enable_dynamic_sizing : bool, default=True
            Whether to enable dynamic batch size optimization
        enable_priority_queuing : bool, default=True
            Whether to enable priority-based queuing
        max_concurrent_batches : int, default=4
            Maximum number of concurrent batches to process
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.enable_dynamic_sizing = enable_dynamic_sizing
        self.enable_priority_queuing = enable_priority_queuing
        self.max_concurrent_batches = max_concurrent_batches

        # Request queues by priority
        if self.enable_priority_queuing:
            self.request_queues = {
                BatchPriority.CRITICAL: queue.Queue(),
                BatchPriority.HIGH: queue.Queue(),
                BatchPriority.NORMAL: queue.Queue(),
                BatchPriority.LOW: queue.Queue(),
            }
        else:
            self.request_queue = queue.Queue()

        # Batch processing
        self.batch_processor = BatchProcessor(self.max_concurrent_batches)

        # Dynamic optimization
        if self.enable_dynamic_sizing:
            self.dynamic_optimizer = DynamicBatchOptimizer()

        # Performance monitoring
        self.performance_monitor = BatchPerformanceMonitor()

        # Scheduling
        self.batch_scheduler = BatchScheduler(
            min_batch_size=self.min_batch_size,
            max_batch_size=self.max_batch_size,
            max_wait_time_ms=self.max_wait_time_ms,
        )

        # Processing state
        self.processing_active = True
        self.processing_thread = threading.Thread(
            target=self._process_requests, daemon=True
        )
        self.processing_thread.start()

        logger.info("Initialized BatchOptimizer")

    def submit_request(self, request: BatchRequest) -> str:
        """
        Submit request for batch processing.

        Parameters:
        -----------
        request : BatchRequest
            Request to process

        Returns:
        --------
        request_id : str
            Request identifier
        """
        if self.enable_priority_queuing:
            self.request_queues[request.priority].put(request)
        else:
            self.request_queue.put(request)

        # Record submission
        self.performance_monitor.record_submission(request)

        return request.request_id

    def submit_batch_requests(self, requests: List[BatchRequest]) -> List[str]:
        """
        Submit multiple requests for batch processing.

        Parameters:
        -----------
        requests : list of BatchRequest
            Requests to process

        Returns:
        --------
        request_ids : list of str
            Request identifiers
        """
        request_ids = []
        for request in requests:
            request_id = self.submit_request(request)
            request_ids.append(request_id)

        return request_ids

    def process_batch_sync(
        self, requests: List[BatchRequest], processor_func: Callable
    ) -> List[BatchResult]:
        """
        Process batch synchronously.

        Parameters:
        -----------
        requests : list of BatchRequest
            Requests to process
        processor_func : callable
            Processing function

        Returns:
        --------
        results : list of BatchResult
            Processing results
        """
        batch_id = f"sync_batch_{int(time.time() * 1000)}"
        start_time = time.time()

        try:
            # Extract data for batch processing
            batch_data = [req.data for req in requests]

            # Process batch
            with self.performance_monitor.measure_batch_processing(len(requests)):
                batch_results = processor_func(batch_data)

            # Create results
            processing_time_ms = (time.time() - start_time) * 1000
            results = []

            for i, request in enumerate(requests):
                result = BatchResult(
                    request_id=request.request_id,
                    result=batch_results[i] if i < len(batch_results) else None,
                    processing_time_ms=processing_time_ms,
                    batch_id=batch_id,
                    timestamp=datetime.now(),
                )
                results.append(result)

            # Record performance
            self.performance_monitor.record_batch_completion(
                batch_id, len(requests), processing_time_ms
            )

            return results

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")

            # Create error results
            error_results = []
            for request in requests:
                result = BatchResult(
                    request_id=request.request_id,
                    result=None,
                    processing_time_ms=0,
                    batch_id=batch_id,
                    timestamp=datetime.now(),
                    error=str(e),
                )
                error_results.append(result)

            return error_results

    async def process_batch_async(
        self, requests: List[BatchRequest], processor_func: Callable
    ) -> List[BatchResult]:
        """
        Process batch asynchronously.

        Parameters:
        -----------
        requests : list of BatchRequest
            Requests to process
        processor_func : callable
            Processing function

        Returns:
        --------
        results : list of BatchResult
            Processing results
        """
        loop = asyncio.get_event_loop()

        # Run sync processing in executor
        results = await loop.run_in_executor(
            None, self.process_batch_sync, requests, processor_func
        )

        return results

    def optimize_batch_size(
        self, current_throughput: float, current_latency: float
    ) -> int:
        """
        Optimize batch size based on current performance.

        Parameters:
        -----------
        current_throughput : float
            Current throughput (requests/second)
        current_latency : float
            Current average latency (milliseconds)

        Returns:
        --------
        optimal_batch_size : int
            Optimized batch size
        """
        if not self.enable_dynamic_sizing:
            return self.max_batch_size

        return self.dynamic_optimizer.optimize_batch_size(
            current_throughput,
            current_latency,
            self.min_batch_size,
            self.max_batch_size,
        )

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status.

        Returns:
        --------
        status : dict
            Queue status information
        """
        if self.enable_priority_queuing:
            queue_sizes = {
                priority.name: q.qsize() for priority, q in self.request_queues.items()
            }
            total_queued = sum(queue_sizes.values())
        else:
            total_queued = self.request_queue.qsize()
            queue_sizes = {"total": total_queued}

        return {
            "queue_sizes": queue_sizes,
            "total_queued": total_queued,
            "processing_active": self.processing_active,
            "concurrent_batches": self.batch_processor.get_active_batch_count(),
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report.

        Returns:
        --------
        report : dict
            Performance report
        """
        return {
            "queue_status": self.get_queue_status(),
            "performance_metrics": self.performance_monitor.get_metrics(),
            "batch_processor_stats": self.batch_processor.get_stats(),
            "dynamic_optimization": (
                self.dynamic_optimizer.get_stats() if self.enable_dynamic_sizing else {}
            ),
        }

    def _process_requests(self) -> None:
        """Background request processing loop."""
        while self.processing_active:
            try:
                # Get next batch
                batch_requests = self._get_next_batch()

                if not batch_requests:
                    time.sleep(0.001)  # Short sleep if no requests
                    continue

                # Submit batch for processing
                self.batch_processor.submit_batch(batch_requests)

            except Exception as e:
                logger.error(f"Request processing error: {e}")
                time.sleep(0.01)

    def _get_next_batch(self) -> List[BatchRequest]:
        """
        Get next batch of requests to process.

        Returns:
        --------
        requests : list of BatchRequest
            Next batch of requests
        """
        return self.batch_scheduler.get_next_batch(
            self.request_queues if self.enable_priority_queuing else self.request_queue
        )

    def stop_processing(self) -> None:
        """Stop batch processing."""
        self.processing_active = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)

        self.batch_processor.shutdown()


class BatchProcessor:
    """
    Concurrent batch processor.
    """

    def __init__(self, max_concurrent_batches: int = 4):
        """
        Initialize batch processor.

        Parameters:
        -----------
        max_concurrent_batches : int, default=4
            Maximum concurrent batches
        """
        self.max_concurrent_batches = max_concurrent_batches
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent_batches
        )

        # Active batches tracking
        self.active_batches = {}
        self.batch_counter = 0
        self.lock = threading.RLock()

        # Statistics
        self.total_batches_processed = 0
        self.total_requests_processed = 0
        self.total_processing_time_ms = 0

    def submit_batch(self, requests: List[BatchRequest]) -> str:
        """
        Submit batch for processing.

        Parameters:
        -----------
        requests : list of BatchRequest
            Requests to process

        Returns:
        --------
        batch_id : str
            Batch identifier
        """
        with self.lock:
            self.batch_counter += 1
            batch_id = f"batch_{self.batch_counter}"

        # Submit to executor
        future = self.executor.submit(self._process_batch, batch_id, requests)

        with self.lock:
            self.active_batches[batch_id] = {
                "future": future,
                "requests": requests,
                "start_time": time.time(),
            }

        # Add completion callback
        future.add_done_callback(lambda f: self._batch_completed(batch_id, f))

        return batch_id

    def _process_batch(
        self, batch_id: str, requests: List[BatchRequest]
    ) -> List[BatchResult]:
        """
        Process a batch of requests.

        Parameters:
        -----------
        batch_id : str
            Batch identifier
        requests : list of BatchRequest
            Requests to process

        Returns:
        --------
        results : list of BatchResult
            Processing results
        """
        start_time = time.time()
        results = []

        try:
            # Process each request in the batch
            for request in requests:
                try:
                    # Simulate processing (in real implementation, call actual processor)
                    result_data = self._process_single_request(request)

                    result = BatchResult(
                        request_id=request.request_id,
                        result=result_data,
                        processing_time_ms=(time.time() - start_time) * 1000,
                        batch_id=batch_id,
                        timestamp=datetime.now(),
                    )

                    # Call callback if provided
                    if request.callback:
                        try:
                            request.callback(result)
                        except Exception as e:
                            logger.warning(
                                f"Callback failed for request {request.request_id}: {e}"
                            )

                except Exception as e:
                    logger.error(f"Failed to process request {request.request_id}: {e}")
                    result = BatchResult(
                        request_id=request.request_id,
                        result=None,
                        processing_time_ms=(time.time() - start_time) * 1000,
                        batch_id=batch_id,
                        timestamp=datetime.now(),
                        error=str(e),
                    )

                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Batch {batch_id} processing failed: {e}")

            # Create error results for all requests
            error_results = []
            for request in requests:
                result = BatchResult(
                    request_id=request.request_id,
                    result=None,
                    processing_time_ms=0,
                    batch_id=batch_id,
                    timestamp=datetime.now(),
                    error=str(e),
                )
                error_results.append(result)

            return error_results

    def _process_single_request(self, request: BatchRequest) -> Any:
        """
        Process a single request (placeholder implementation).

        Parameters:
        -----------
        request : BatchRequest
            Request to process

        Returns:
        --------
        result : any
            Processing result
        """
        # This is a placeholder - in real implementation,
        # this would call the actual fraud detection models
        return {"fraud_score": 0.1, "risk_level": "low"}

    def _batch_completed(
        self, batch_id: str, future: concurrent.futures.Future
    ) -> None:
        """
        Handle batch completion.

        Parameters:
        -----------
        batch_id : str
            Batch identifier
        future : Future
            Completed future
        """
        with self.lock:
            if batch_id in self.active_batches:
                batch_info = self.active_batches.pop(batch_id)

                # Update statistics
                self.total_batches_processed += 1
                self.total_requests_processed += len(batch_info["requests"])
                processing_time = time.time() - batch_info["start_time"]
                self.total_processing_time_ms += processing_time * 1000

                try:
                    results = future.result()
                    logger.debug(
                        f"Batch {batch_id} completed with {len(results)} results"
                    )
                except Exception as e:
                    logger.error(f"Batch {batch_id} failed: {e}")

    def get_active_batch_count(self) -> int:
        """
        Get number of active batches.

        Returns:
        --------
        count : int
            Number of active batches
        """
        with self.lock:
            return len(self.active_batches)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics.

        Returns:
        --------
        stats : dict
            Processor statistics
        """
        with self.lock:
            avg_processing_time = self.total_processing_time_ms / max(
                self.total_batches_processed, 1
            )

            return {
                "active_batches": len(self.active_batches),
                "total_batches_processed": self.total_batches_processed,
                "total_requests_processed": self.total_requests_processed,
                "avg_batch_processing_time_ms": avg_processing_time,
                "max_concurrent_batches": self.max_concurrent_batches,
            }

    def shutdown(self) -> None:
        """Shutdown batch processor."""
        self.executor.shutdown(wait=True)


class BatchScheduler:
    """
    Intelligent batch scheduling and formation.
    """

    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 100,
        max_wait_time_ms: float = 50.0,
    ):
        """
        Initialize batch scheduler.

        Parameters:
        -----------
        min_batch_size : int, default=1
            Minimum batch size
        max_batch_size : int, default=100
            Maximum batch size
        max_wait_time_ms : float, default=50.0
            Maximum wait time for batch formation
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms

        # Batch formation state
        self.pending_requests = []
        self.batch_start_time = None
        self.lock = threading.RLock()

    def get_next_batch(
        self, queues: Union[queue.Queue, Dict[BatchPriority, queue.Queue]]
    ) -> List[BatchRequest]:
        """
        Get next batch of requests to process.

        Parameters:
        -----------
        queues : Queue or dict of Queues
            Request queues

        Returns:
        --------
        requests : list of BatchRequest
            Next batch of requests
        """
        with self.lock:
            # Collect requests from queues
            if isinstance(queues, dict):
                # Priority-based queues
                requests = self._collect_priority_requests(queues)
            else:
                # Single queue
                requests = self._collect_single_queue_requests(queues)

            if not requests:
                return []

            # Add to pending requests
            self.pending_requests.extend(requests)

            # Set batch start time if first requests
            if self.batch_start_time is None and self.pending_requests:
                self.batch_start_time = time.time()

            # Check if batch should be formed
            should_form_batch = self._should_form_batch()

            if should_form_batch:
                batch_requests = self.pending_requests[: self.max_batch_size]
                self.pending_requests = self.pending_requests[self.max_batch_size :]

                # Reset batch timing if no more pending requests
                if not self.pending_requests:
                    self.batch_start_time = None

                return batch_requests

            return []

    def _collect_priority_requests(
        self, queues: Dict[BatchPriority, queue.Queue]
    ) -> List[BatchRequest]:
        """Collect requests from priority queues."""
        requests = []

        # Process in priority order
        for priority in [
            BatchPriority.CRITICAL,
            BatchPriority.HIGH,
            BatchPriority.NORMAL,
            BatchPriority.LOW,
        ]:
            queue_obj = queues[priority]

            # Get requests from this priority level
            while not queue_obj.empty() and len(requests) < self.max_batch_size:
                try:
                    request = queue_obj.get_nowait()
                    requests.append(request)
                except queue.Empty:
                    break

            # Stop if we have enough requests
            if len(requests) >= self.max_batch_size:
                break

        return requests

    def _collect_single_queue_requests(
        self, queue_obj: queue.Queue
    ) -> List[BatchRequest]:
        """Collect requests from single queue."""
        requests = []

        while not queue_obj.empty() and len(requests) < self.max_batch_size:
            try:
                request = queue_obj.get_nowait()
                requests.append(request)
            except queue.Empty:
                break

        return requests

    def _should_form_batch(self) -> bool:
        """
        Determine if batch should be formed.

        Returns:
        --------
        should_form : bool
            Whether to form batch now
        """
        if not self.pending_requests:
            return False

        # Form batch if we have enough requests
        if len(self.pending_requests) >= self.max_batch_size:
            return True

        # Form batch if minimum size reached and waited long enough
        if (
            len(self.pending_requests) >= self.min_batch_size
            and self.batch_start_time is not None
        ):

            wait_time_ms = (time.time() - self.batch_start_time) * 1000
            if wait_time_ms >= self.max_wait_time_ms:
                return True

        # Form batch immediately for critical priority requests
        for request in self.pending_requests:
            if request.priority == BatchPriority.CRITICAL:
                return True

        return False


class DynamicBatchOptimizer:
    """
    Dynamic batch size optimization based on performance feedback.
    """

    def __init__(self, adaptation_rate: float = 0.1):
        """
        Initialize dynamic optimizer.

        Parameters:
        -----------
        adaptation_rate : float, default=0.1
            Rate of adaptation for batch size changes
        """
        self.adaptation_rate = adaptation_rate

        # Performance history
        self.performance_history = deque(maxlen=100)

        # Current optimal batch size
        self.current_optimal_size = 50

        # Optimization statistics
        self.optimizations_performed = 0
        self.last_optimization_time = None

    def optimize_batch_size(
        self,
        current_throughput: float,
        current_latency: float,
        min_size: int,
        max_size: int,
    ) -> int:
        """
        Optimize batch size based on current performance.

        Parameters:
        -----------
        current_throughput : float
            Current throughput (requests/second)
        current_latency : float
            Current latency (milliseconds)
        min_size : int
            Minimum batch size
        max_size : int
            Maximum batch size

        Returns:
        --------
        optimal_size : int
            Optimized batch size
        """
        # Record current performance
        self.performance_history.append(
            {
                "throughput": current_throughput,
                "latency": current_latency,
                "batch_size": self.current_optimal_size,
                "timestamp": time.time(),
            }
        )

        # Need at least a few data points for optimization
        if len(self.performance_history) < 5:
            return self.current_optimal_size

        # Analyze performance trend
        recent_performance = list(self.performance_history)[-5:]

        # Calculate performance score (higher is better)
        def calculate_score(throughput, latency):
            # Normalize and combine throughput and latency
            # Higher throughput is better, lower latency is better
            normalized_throughput = min(throughput / 1000, 1.0)  # Normalize to 0-1
            normalized_latency = max(0, 1 - latency / 1000)  # Normalize to 0-1
            return normalized_throughput * 0.6 + normalized_latency * 0.4

        current_score = calculate_score(current_throughput, current_latency)

        # Get average score of recent performance
        recent_scores = [
            calculate_score(perf["throughput"], perf["latency"])
            for perf in recent_performance
        ]
        avg_recent_score = np.mean(recent_scores)

        # Determine optimization direction
        if current_score > avg_recent_score:
            # Performance is improving, continue in current direction
            if self.current_optimal_size < max_size:
                adjustment = max(
                    1, int(self.adaptation_rate * self.current_optimal_size)
                )
                new_size = min(self.current_optimal_size + adjustment, max_size)
            else:
                new_size = self.current_optimal_size
        else:
            # Performance is degrading, try opposite direction
            if self.current_optimal_size > min_size:
                adjustment = max(
                    1, int(self.adaptation_rate * self.current_optimal_size)
                )
                new_size = max(self.current_optimal_size - adjustment, min_size)
            else:
                new_size = self.current_optimal_size

        # Update optimal size
        if new_size != self.current_optimal_size:
            self.current_optimal_size = new_size
            self.optimizations_performed += 1
            self.last_optimization_time = datetime.now()

        return self.current_optimal_size

    def get_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics.

        Returns:
        --------
        stats : dict
            Optimization statistics
        """
        return {
            "current_optimal_size": self.current_optimal_size,
            "optimizations_performed": self.optimizations_performed,
            "last_optimization_time": (
                self.last_optimization_time.isoformat()
                if self.last_optimization_time
                else None
            ),
            "performance_data_points": len(self.performance_history),
        }


class BatchPerformanceMonitor:
    """
    Performance monitoring for batch processing.
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize performance monitor.

        Parameters:
        -----------
        window_size : int, default=1000
            Size of performance monitoring window
        """
        self.window_size = window_size

        # Performance metrics
        self.submission_times = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.batch_sizes = deque(maxlen=window_size)
        self.queue_sizes = deque(maxlen=window_size)

        # Counters
        self.total_requests_submitted = 0
        self.total_requests_processed = 0
        self.total_batches_processed = 0

        # Timing
        self.start_time = time.time()

        # Thread safety
        self.lock = threading.RLock()

    def record_submission(self, request: BatchRequest) -> None:
        """
        Record request submission.

        Parameters:
        -----------
        request : BatchRequest
            Submitted request
        """
        with self.lock:
            self.submission_times.append(time.time())
            self.total_requests_submitted += 1

    def record_batch_completion(
        self, batch_id: str, batch_size: int, processing_time_ms: float
    ) -> None:
        """
        Record batch completion.

        Parameters:
        -----------
        batch_id : str
            Batch identifier
        batch_size : int
            Size of completed batch
        processing_time_ms : float
            Processing time in milliseconds
        """
        with self.lock:
            self.processing_times.append(processing_time_ms)
            self.batch_sizes.append(batch_size)
            self.total_requests_processed += batch_size
            self.total_batches_processed += 1

    def measure_batch_processing(self, batch_size: int):
        """
        Context manager for measuring batch processing.

        Parameters:
        -----------
        batch_size : int
            Size of batch being processed

        Returns:
        --------
        context_manager : BatchProcessingTimer
            Context manager for timing
        """
        return BatchProcessingTimer(self, batch_size)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.

        Returns:
        --------
        metrics : dict
            Performance metrics
        """
        with self.lock:
            # Calculate throughput
            elapsed_time = time.time() - self.start_time
            throughput = self.total_requests_processed / max(elapsed_time, 1)

            # Calculate average metrics
            metrics = {
                "total_requests_submitted": self.total_requests_submitted,
                "total_requests_processed": self.total_requests_processed,
                "total_batches_processed": self.total_batches_processed,
                "throughput_requests_per_second": throughput,
                "elapsed_time_seconds": elapsed_time,
            }

            # Processing time statistics
            if self.processing_times:
                processing_array = np.array(list(self.processing_times))
                metrics.update(
                    {
                        "avg_processing_time_ms": np.mean(processing_array),
                        "median_processing_time_ms": np.median(processing_array),
                        "p95_processing_time_ms": np.percentile(processing_array, 95),
                        "max_processing_time_ms": np.max(processing_array),
                        "min_processing_time_ms": np.min(processing_array),
                    }
                )

            # Batch size statistics
            if self.batch_sizes:
                batch_array = np.array(list(self.batch_sizes))
                metrics.update(
                    {
                        "avg_batch_size": np.mean(batch_array),
                        "median_batch_size": np.median(batch_array),
                        "max_batch_size": np.max(batch_array),
                        "min_batch_size": np.min(batch_array),
                    }
                )

        return metrics


class BatchProcessingTimer:
    """
    Context manager for timing batch processing.
    """

    def __init__(self, monitor: BatchPerformanceMonitor, batch_size: int):
        """
        Initialize timer.

        Parameters:
        -----------
        monitor : BatchPerformanceMonitor
            Performance monitor
        batch_size : int
            Size of batch being processed
        """
        self.monitor = monitor
        self.batch_size = batch_size
        self.start_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record."""
        if self.start_time is not None:
            processing_time_ms = (time.time() - self.start_time) * 1000
            batch_id = f"timed_batch_{int(time.time() * 1000)}"
            self.monitor.record_batch_completion(
                batch_id, self.batch_size, processing_time_ms
            )


class StreamProcessor:
    """
    Stream processing for real-time fraud detection.

    Handles continuous stream of transactions with low-latency processing,
    optimized for real-time fraud detection scenarios.
    """

    def __init__(
        self,
        batch_timeout_ms: float = 10.0,
        max_stream_buffer: int = 1000,
        enable_backpressure: bool = True,
    ):
        """
        Initialize stream processor.

        Parameters:
        -----------
        batch_timeout_ms : float, default=10.0
            Maximum time to wait before processing incomplete batch
        max_stream_buffer : int, default=1000
            Maximum buffer size for stream
        enable_backpressure : bool, default=True
            Enable backpressure handling
        """
        self.batch_timeout_ms = batch_timeout_ms
        self.max_stream_buffer = max_stream_buffer
        self.enable_backpressure = enable_backpressure

        # Stream buffer
        self.stream_buffer = deque(maxlen=max_stream_buffer)
        self.buffer_lock = threading.RLock()

        # Processing state
        self.processing_active = False
        self.process_thread = None

        # Statistics
        self.total_processed = 0
        self.total_dropped = 0
        self.last_process_time = None

    def start(self):
        """Start stream processing."""
        if not self.processing_active:
            self.processing_active = True
            self.process_thread = threading.Thread(target=self._process_stream)
            self.process_thread.daemon = True
            self.process_thread.start()
            logger.info("Stream processor started")

    def stop(self):
        """Stop stream processing."""
        self.processing_active = False
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2)
        logger.info("Stream processor stopped")

    def submit(self, transaction_data: Dict[str, Any]) -> bool:
        """
        Submit transaction to stream.

        Parameters:
        -----------
        transaction_data : dict
            Transaction data to process

        Returns:
        --------
        accepted : bool
            Whether transaction was accepted into stream
        """
        with self.buffer_lock:
            if len(self.stream_buffer) >= self.max_stream_buffer:
                if self.enable_backpressure:
                    # Drop oldest if buffer full
                    self.stream_buffer.popleft()
                    self.total_dropped += 1
                else:
                    # Reject new transaction
                    return False

            self.stream_buffer.append(
                {"data": transaction_data, "timestamp": time.time()}
            )
            return True

    def _process_stream(self):
        """Process stream continuously."""
        last_batch_time = time.time()

        while self.processing_active:
            current_time = time.time()

            with self.buffer_lock:
                # Check if we should process
                should_process = (
                    len(self.stream_buffer) > 0
                    and (current_time - last_batch_time) * 1000 >= self.batch_timeout_ms
                )

                if should_process:
                    # Process available items
                    batch = []
                    while self.stream_buffer and len(batch) < 100:
                        batch.append(self.stream_buffer.popleft())

                    if batch:
                        self._process_batch(batch)
                        last_batch_time = current_time

            # Small sleep to prevent CPU spinning
            time.sleep(0.001)

    def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of stream items."""
        try:
            # Process transactions (placeholder for actual processing)
            for item in batch:
                # In real implementation, this would call fraud detection
                self.total_processed += 1
                self.last_process_time = time.time()

        except Exception as e:
            logger.error(f"Stream batch processing failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get stream processor statistics."""
        with self.buffer_lock:
            return {
                "buffer_size": len(self.stream_buffer),
                "total_processed": self.total_processed,
                "total_dropped": self.total_dropped,
                "processing_active": self.processing_active,
                "last_process_time": self.last_process_time,
            }
