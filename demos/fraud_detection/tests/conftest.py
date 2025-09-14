"""
Pytest configuration and shared fixtures for fraud detection tests.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from demos.fraud_detection.data.generate_transaction_data import (
    TransactionDataGenerator,
)
from demos.fraud_detection.models.ensemble_fraud_detector import EnsembleFraudDetector
from demos.fraud_detection.features.realtime_features import RealTimeFeatureProcessor
from demos.fraud_detection.optimization.caching_strategies import (
    FraudDetectionCacheManager,
)


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture."""
    return {
        "max_latency_ms": 100,
        "min_accuracy": 0.85,
        "test_data_size": 100,  # Smaller for fast tests
        "performance_iterations": 50,
        "tolerance": 0.05,
    }


@pytest.fixture(scope="session")
def transaction_generator():
    """Transaction data generator fixture."""
    return TransactionDataGenerator(
        num_customers=50,
        num_merchants=20,
        fraud_rate=0.1,
        enable_advanced_patterns=True,
    )


@pytest.fixture(scope="session")
def sample_transaction_data(transaction_generator):
    """Generate sample transaction data for testing."""
    return transaction_generator.generate_transaction_data(
        num_transactions=100,
        start_date=datetime.now() - timedelta(days=7),
        end_date=datetime.now(),
    )


@pytest.fixture(scope="session")
def fraud_transactions(sample_transaction_data):
    """Filter fraud transactions from sample data."""
    return sample_transaction_data[
        sample_transaction_data["is_fraud"] == True
    ].reset_index(drop=True)


@pytest.fixture(scope="session")
def legitimate_transactions(sample_transaction_data):
    """Filter legitimate transactions from sample data."""
    return sample_transaction_data[
        sample_transaction_data["is_fraud"] == False
    ].reset_index(drop=True)


@pytest.fixture
def feature_processor():
    """Real-time feature processor fixture."""
    return RealTimeFeatureProcessor(
        enable_caching=True,
        cache_ttl_seconds=300,
        enable_parallel_processing=False,  # Disable for testing
    )


@pytest.fixture
def cache_manager():
    """Cache manager fixture."""
    return FraudDetectionCacheManager()


@pytest.fixture
def ensemble_detector():
    """Ensemble fraud detector fixture."""
    return EnsembleFraudDetector(
        combination_strategy="weighted_voting",
        weights={
            "rule_based": 0.25,
            "anomaly": 0.25,
            "neural": 0.25,
            "behavioral": 0.25,
        },
        enable_confidence_scoring=True,
        enable_explanation=True,
    )


@pytest.fixture
def mock_trained_ensemble(ensemble_detector):
    """Mock a trained ensemble detector for testing."""
    # Mock the training state
    ensemble_detector._is_trained = True
    ensemble_detector._training_metrics = {
        "accuracy": 0.92,
        "precision": 0.89,
        "recall": 0.94,
        "f1_score": 0.91,
        "auc_roc": 0.96,
    }
    return ensemble_detector


@pytest.fixture
def sample_features():
    """Sample feature data for testing."""
    return pd.DataFrame(
        {
            "amount": [100.0, 1500.0, 50.0, 2000.0, 25.0],
            "hour_of_day": [14, 2, 10, 23, 8],
            "day_of_week": [1, 6, 3, 0, 4],
            "merchant_risk_score": [0.3, 0.8, 0.2, 0.9, 0.1],
            "customer_age_days": [365, 100, 730, 50, 1000],
            "velocity_1h": [1, 5, 1, 3, 1],
            "velocity_24h": [3, 12, 2, 8, 2],
            "amount_percentile": [0.4, 0.9, 0.1, 0.95, 0.05],
        }
    )


@pytest.fixture
def sample_single_transaction():
    """Single transaction for testing."""
    return {
        "transaction_id": "TXN_TEST_001",
        "customer_id": "CUST_001",
        "merchant_id": "MERCH_001",
        "amount": 125.50,
        "transaction_timestamp": datetime.now(),
        "merchant_category": "grocery",
        "payment_method": "credit_card",
        "is_fraud": False,
        "fraud_type": None,
        "hour_of_day": 14,
        "day_of_week": 2,
        "merchant_risk_score": 0.3,
        "customer_age_days": 365,
    }


@pytest.fixture
def high_risk_transaction():
    """High-risk transaction for testing."""
    return {
        "transaction_id": "TXN_FRAUD_001",
        "customer_id": "CUST_FRAUD_001",
        "merchant_id": "MERCH_RISK_001",
        "amount": 2500.00,
        "transaction_timestamp": datetime.now().replace(hour=2),  # Unusual time
        "merchant_category": "cash_advance",
        "payment_method": "credit_card",
        "is_fraud": True,
        "fraud_type": "stolen_card",
        "hour_of_day": 2,
        "day_of_week": 6,
        "merchant_risk_score": 0.9,
        "customer_age_days": 30,
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Set random seed for reproducible tests
    np.random.seed(42)

    # Clear any caches
    if hasattr(RealTimeFeatureProcessor, "_cache"):
        RealTimeFeatureProcessor._cache.clear()

    yield

    # Cleanup after test
    pass


@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()

        def elapsed_ms(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time) * 1000
            return None

    return Timer()


# Pytest markers
pytest_plugins = []


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interaction"
    )
    config.addinivalue_line("markers", "performance: Performance and latency tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line(
        "markers", "requires_data: Tests that require large datasets"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add unit marker to unit test files
        if "test_unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)

        # Add integration marker to integration test files
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Add performance marker to performance test files
        if "test_performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
