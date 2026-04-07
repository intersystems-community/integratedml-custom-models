"""
Performance tests for fraud detection system.
Tests latency, throughput, and scalability requirements.
"""

import pytest
import time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from statistics import mean, median
import psutil
import os

from demos.fraud_detection.models.ensemble_fraud_detector import EnsembleFraudDetector
from demos.fraud_detection.features.realtime_features import RealTimeFeatureProcessor
from demos.fraud_detection.optimization.caching_strategies import (
    FraudDetectionCacheManager,
)


class PerformanceTimer:
    """Helper class for performance timing."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()

    def elapsed_ms(self):
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None


@pytest.mark.performance
class TestPerformanceRequirements:
    """Test performance requirements and latency targets."""

    def test_single_transaction_latency(
        self, mock_trained_ensemble, sample_single_transaction, test_config
    ):
        """Test single transaction prediction meets latency requirement."""
        # Setup mock predictions for fast execution
        mock_trained_ensemble.rule_based_detector.predict = lambda x: np.array([0.3])
        mock_trained_ensemble.anomaly_detector.predict = lambda x: np.array([0.4])
        mock_trained_ensemble.neural_detector.predict = lambda x: np.array([0.2])
        mock_trained_ensemble.behavioral_detector.predict = lambda x: np.array([0.35])

        # Convert transaction to DataFrame
        transaction_df = pd.DataFrame([sample_single_transaction])
        feature_columns = ["amount", "hour_of_day", "merchant_risk_score"]
        transaction_features = transaction_df[feature_columns].fillna(0)

        # Measure prediction latency multiple times
        latencies = []
        for _ in range(test_config["performance_iterations"]):
            timer = PerformanceTimer()
            timer.start()

            prediction = mock_trained_ensemble.predict(transaction_features)

            timer.stop()
            latencies.append(timer.elapsed_ms())

        # Calculate statistics
        avg_latency = mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = max(latencies)

        print(f"\n🎯 Single Transaction Latency Results:")
        print(f"   Average: {avg_latency:.2f}ms")
        print(f"   P95: {p95_latency:.2f}ms")
        print(f"   P99: {p99_latency:.2f}ms")
        print(f"   Max: {max_latency:.2f}ms")

        # Assert performance requirements
        assert (
            avg_latency <= test_config["max_latency_ms"]
        ), f"Average latency {avg_latency:.2f}ms exceeds target {test_config['max_latency_ms']}ms"
        assert (
            p95_latency <= test_config["max_latency_ms"] * 1.5
        ), f"P95 latency {p95_latency:.2f}ms exceeds acceptable threshold"
        assert (
            p99_latency <= test_config["max_latency_ms"] * 2.0
        ), f"P99 latency {p99_latency:.2f}ms exceeds acceptable threshold"

    def test_batch_processing_performance(
        self, mock_trained_ensemble, sample_features, test_config
    ):
        """Test batch processing performance and throughput."""
        batch_sizes = [1, 5, 10, 25, 50, 100]
        results = []

        # Setup mock predictions
        def mock_predict(features):
            batch_size = len(features)
            return np.random.uniform(0, 1, batch_size)

        mock_trained_ensemble.rule_based_detector.predict = mock_predict
        mock_trained_ensemble.anomaly_detector.predict = mock_predict
        mock_trained_ensemble.neural_detector.predict = mock_predict
        mock_trained_ensemble.behavioral_detector.predict = mock_predict

        for batch_size in batch_sizes:
            # Create batch of specified size
            batch_features = sample_features.head(batch_size)

            # Measure batch processing time
            timer = PerformanceTimer()
            timer.start()

            predictions = mock_trained_ensemble.predict(batch_features)

            timer.stop()

            total_time_ms = timer.elapsed_ms()
            avg_latency_per_transaction = total_time_ms / batch_size
            throughput_tps = (batch_size / total_time_ms) * 1000

            results.append(
                {
                    "batch_size": batch_size,
                    "total_time_ms": total_time_ms,
                    "avg_latency_ms": avg_latency_per_transaction,
                    "throughput_tps": throughput_tps,
                }
            )

            print(
                f"   Batch {batch_size}: {avg_latency_per_transaction:.2f}ms/txn, {throughput_tps:.1f} TPS"
            )

        # Verify that larger batches don't significantly degrade per-transaction latency
        single_latency = results[0]["avg_latency_ms"]
        batch_100_latency = results[-1]["avg_latency_ms"]

        # Allow some degradation but not excessive
        degradation_factor = batch_100_latency / single_latency
        assert (
            degradation_factor <= 3.0
        ), f"Batch processing degradation too high: {degradation_factor:.2f}x"

        # Verify minimum throughput
        max_throughput = max(result["throughput_tps"] for result in results)
        assert (
            max_throughput >= 100
        ), f"Maximum throughput {max_throughput:.1f} TPS below minimum requirement"

    def test_concurrent_processing_performance(
        self, mock_trained_ensemble, sample_features
    ):
        """Test concurrent processing capability."""

        # Setup mock predictions
        def mock_predict(features):
            time.sleep(0.01)  # Simulate some processing time
            batch_size = len(features)
            return np.random.uniform(0, 1, batch_size)

        mock_trained_ensemble.rule_based_detector.predict = mock_predict
        mock_trained_ensemble.anomaly_detector.predict = mock_predict
        mock_trained_ensemble.neural_detector.predict = mock_predict
        mock_trained_ensemble.behavioral_detector.predict = mock_predict

        # Test different concurrency levels
        concurrency_levels = [1, 2, 4, 8]
        single_transaction = sample_features.head(1)

        for num_threads in concurrency_levels:
            latencies = []

            def worker():
                timer = PerformanceTimer()
                timer.start()
                mock_trained_ensemble.predict(single_transaction)
                timer.stop()
                latencies.append(timer.elapsed_ms())

            # Execute concurrent predictions
            start_time = time.perf_counter()

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker) for _ in range(num_threads * 5)]
                for future in futures:
                    future.result()

            end_time = time.perf_counter()

            avg_latency = mean(latencies)
            total_time = (end_time - start_time) * 1000
            effective_throughput = len(latencies) / (total_time / 1000)

            print(
                f"   {num_threads} threads: {avg_latency:.2f}ms avg latency, {effective_throughput:.1f} TPS"
            )

            # Verify that concurrency doesn't cause excessive latency increase
            assert (
                avg_latency <= 200
            ), f"Concurrent processing latency {avg_latency:.2f}ms too high"

    def test_memory_usage_performance(self, mock_trained_ensemble, sample_features):
        """Test memory usage during processing."""
        process = psutil.Process(os.getpid())

        # Baseline memory usage
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Setup mock predictions
        def mock_predict(features):
            batch_size = len(features)
            return np.random.uniform(0, 1, batch_size)

        mock_trained_ensemble.rule_based_detector.predict = mock_predict
        mock_trained_ensemble.anomaly_detector.predict = mock_predict
        mock_trained_ensemble.neural_detector.predict = mock_predict
        mock_trained_ensemble.behavioral_detector.predict = mock_predict

        # Process multiple batches and monitor memory
        memory_measurements = []

        for i in range(10):
            # Process a batch
            batch_features = sample_features
            predictions = mock_trained_ensemble.predict(batch_features)

            # Measure memory
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_measurements.append(current_memory)

        # Calculate memory statistics
        max_memory = max(memory_measurements)
        avg_memory = mean(memory_measurements)
        memory_growth = max_memory - baseline_memory

        print(f"\n🧠 Memory Usage Results:")
        print(f"   Baseline: {baseline_memory:.1f} MB")
        print(f"   Average: {avg_memory:.1f} MB")
        print(f"   Peak: {max_memory:.1f} MB")
        print(f"   Growth: {memory_growth:.1f} MB")

        # Verify reasonable memory usage (adjust thresholds as needed)
        assert (
            memory_growth <= 100
        ), f"Memory growth {memory_growth:.1f} MB exceeds threshold"

        # Verify no significant memory leaks
        final_memory = memory_measurements[-1]
        initial_processing_memory = memory_measurements[0]
        leak_indicator = final_memory - initial_processing_memory

        assert (
            leak_indicator <= 10
        ), f"Potential memory leak detected: {leak_indicator:.1f} MB growth"

    def test_cache_performance_impact(
        self, feature_processor, sample_single_transaction
    ):
        """Test cache effectiveness on performance."""
        feature_processor.fit(pd.DataFrame([sample_single_transaction]))

        # Test without cache
        feature_processor._enable_caching = False

        no_cache_times = []
        for _ in range(20):
            timer = PerformanceTimer()
            timer.start()

            features = feature_processor.transform_single(
                sample_single_transaction
            )

            timer.stop()
            no_cache_times.append(timer.elapsed_ms())

        # Test with cache
        feature_processor._enable_caching = True

        # Prime the cache
        feature_processor.transform_single(
            sample_single_transaction
        )

        cached_times = []
        for _ in range(20):
            timer = PerformanceTimer()
            timer.start()

            features = feature_processor.transform_single(
                sample_single_transaction
            )

            timer.stop()
            cached_times.append(timer.elapsed_ms())

        # Compare performance
        avg_no_cache = mean(no_cache_times)
        avg_cached = mean(cached_times)
        speedup = avg_no_cache / avg_cached if avg_cached > 0 else 1

        print(f"\n⚡ Cache Performance Results:")
        print(f"   No cache: {avg_no_cache:.2f}ms")
        print(f"   With cache: {avg_cached:.2f}ms")
        print(f"   Speedup: {speedup:.2f}x")

        # Cache should provide some performance benefit
        assert speedup >= 1.1, f"Cache provides insufficient speedup: {speedup:.2f}x"

    def test_stress_testing(self, mock_trained_ensemble, transaction_generator):
        """Test system under stress conditions."""
        # Generate larger dataset for stress testing
        stress_data = transaction_generator.generate_transaction_data(
            num_transactions=1000
        )

        # Setup mock predictions for stress test
        def mock_predict(features):
            batch_size = len(features)
            time.sleep(0.001 * batch_size)  # Simulate processing time
            return np.random.uniform(0, 1, batch_size)

        mock_trained_ensemble.rule_based_detector.predict = mock_predict
        mock_trained_ensemble.anomaly_detector.predict = mock_predict
        mock_trained_ensemble.neural_detector.predict = mock_predict
        mock_trained_ensemble.behavioral_detector.predict = mock_predict

        # Process in batches and measure performance
        batch_size = 50
        latencies = []
        errors = 0

        for i in range(0, len(stress_data), batch_size):
            batch = stress_data.iloc[i : i + batch_size]

            # Create simple features for testing
            batch_features = pd.DataFrame(
                {
                    "amount": batch["amount"],
                    "hour_of_day": batch["hour_of_day"],
                    "merchant_risk_score": batch.get("merchant_risk_score", 0.5),
                }
            ).fillna(0)

            try:
                timer = PerformanceTimer()
                timer.start()

                predictions = mock_trained_ensemble.predict(batch_features)

                timer.stop()

                batch_latency = timer.elapsed_ms() / len(batch_features)
                latencies.append(batch_latency)

            except Exception as e:
                errors += 1
                print(f"Error processing batch {i//batch_size}: {e}")

        # Analyze stress test results
        if latencies:
            avg_latency = mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            error_rate = errors / (len(stress_data) // batch_size)

            print(f"\n🔥 Stress Test Results:")
            print(f"   Processed: {len(stress_data)} transactions")
            print(f"   Average latency: {avg_latency:.2f}ms")
            print(f"   P95 latency: {p95_latency:.2f}ms")
            print(f"   Error rate: {error_rate:.2%}")

            # Verify stress test performance
            assert (
                avg_latency <= 150
            ), f"Stress test average latency {avg_latency:.2f}ms too high"
            assert (
                error_rate <= 0.01
            ), f"Error rate {error_rate:.2%} too high during stress test"
        else:
            pytest.fail("No successful predictions during stress test")

    def test_warm_up_performance(self, mock_trained_ensemble, sample_features):
        """Test system performance after warm-up period."""

        # Setup mock predictions
        def mock_predict(features):
            batch_size = len(features)
            return np.random.uniform(0, 1, batch_size)

        mock_trained_ensemble.rule_based_detector.predict = mock_predict
        mock_trained_ensemble.anomaly_detector.predict = mock_predict
        mock_trained_ensemble.neural_detector.predict = mock_predict
        mock_trained_ensemble.behavioral_detector.predict = mock_predict

        single_transaction = sample_features.head(1)

        # Cold start measurements
        cold_start_times = []
        for _ in range(5):
            timer = PerformanceTimer()
            timer.start()
            mock_trained_ensemble.predict(single_transaction)
            timer.stop()
            cold_start_times.append(timer.elapsed_ms())

        # Warm-up period
        for _ in range(10):
            mock_trained_ensemble.predict(single_transaction)

        # Warmed up measurements
        warm_times = []
        for _ in range(20):
            timer = PerformanceTimer()
            timer.start()
            mock_trained_ensemble.predict(single_transaction)
            timer.stop()
            warm_times.append(timer.elapsed_ms())

        # Compare performance
        avg_cold = mean(cold_start_times)
        avg_warm = mean(warm_times)
        improvement = avg_cold / avg_warm if avg_warm > 0 else 1

        print(f"\n🌡️ Warm-up Performance Results:")
        print(f"   Cold start: {avg_cold:.2f}ms")
        print(f"   Warmed up: {avg_warm:.2f}ms")
        print(f"   Improvement: {improvement:.2f}x")

        # Verify warm-up provides benefit
        assert improvement >= 1.0, "Warm-up should not degrade performance"

        # Verify warmed-up performance meets requirements
        assert avg_warm <= 100, f"Warmed-up latency {avg_warm:.2f}ms exceeds target"


@pytest.mark.performance
class TestScalabilityRequirements:
    """Test scalability and resource efficiency."""

    def test_linear_scaling_performance(self, mock_trained_ensemble, sample_features):
        """Test that performance scales roughly linearly with batch size."""

        # Setup mock predictions
        def mock_predict(features):
            batch_size = len(features)
            time.sleep(0.001 * batch_size)  # Linear scaling simulation
            return np.random.uniform(0, 1, batch_size)

        mock_trained_ensemble.rule_based_detector.predict = mock_predict
        mock_trained_ensemble.anomaly_detector.predict = mock_predict
        mock_trained_ensemble.neural_detector.predict = mock_predict
        mock_trained_ensemble.behavioral_detector.predict = mock_predict

        batch_sizes = [1, 2, 4, 8, 16]
        per_transaction_times = []

        for batch_size in batch_sizes:
            batch_features = sample_features.head(batch_size)

            timer = PerformanceTimer()
            timer.start()
            predictions = mock_trained_ensemble.predict(batch_features)
            timer.stop()

            per_transaction_time = timer.elapsed_ms() / batch_size
            per_transaction_times.append(per_transaction_time)

        # Check that per-transaction time doesn't increase dramatically
        baseline = per_transaction_times[0]
        max_degradation = max(per_transaction_times) / baseline

        print(f"\n📈 Scaling Performance Results:")
        for i, (size, time_ms) in enumerate(zip(batch_sizes, per_transaction_times)):
            print(f"   Batch {size}: {time_ms:.2f}ms per transaction")

        assert (
            max_degradation <= 2.0
        ), f"Performance degradation {max_degradation:.2f}x too high"

    def test_resource_utilization_efficiency(
        self, mock_trained_ensemble, sample_features
    ):
        """Test CPU and memory efficiency under load."""
        process = psutil.Process(os.getpid())

        # Setup mock predictions
        def mock_predict(features):
            batch_size = len(features)
            # Simulate CPU-intensive work
            for _ in range(batch_size * 10):
                _ = sum(range(100))
            return np.random.uniform(0, 1, batch_size)

        mock_trained_ensemble.rule_based_detector.predict = mock_predict
        mock_trained_ensemble.anomaly_detector.predict = mock_predict
        mock_trained_ensemble.neural_detector.predict = mock_predict
        mock_trained_ensemble.behavioral_detector.predict = mock_predict

        # Measure resource usage during intensive processing
        cpu_percentages = []
        memory_usage = []

        initial_memory = process.memory_info().rss / 1024 / 1024

        for _ in range(5):
            # Process batch
            batch_features = sample_features
            predictions = mock_trained_ensemble.predict(batch_features)

            # Measure CPU and memory
            cpu_percent = process.cpu_percent()
            current_memory = process.memory_info().rss / 1024 / 1024

            cpu_percentages.append(cpu_percent)
            memory_usage.append(current_memory)

        avg_cpu = mean(cpu_percentages) if cpu_percentages else 0
        max_memory = max(memory_usage)
        memory_growth = max_memory - initial_memory

        print(f"\n🖥️ Resource Utilization Results:")
        print(f"   Average CPU: {avg_cpu:.1f}%")
        print(f"   Memory growth: {memory_growth:.1f} MB")
        print(f"   Peak memory: {max_memory:.1f} MB")

        # Verify reasonable resource usage
        # (CPU limits depend on system, so we're more lenient)
        assert memory_growth <= 50, f"Memory growth {memory_growth:.1f} MB excessive"
