"""
Credit Risk Assessment Demo Test Suite.

This package contains comprehensive tests for the credit risk assessment demo,
including unit tests, integration tests, and performance benchmarks.

Test Modules:
- test_credit_risk_classifier: Tests for the core classifier model
- test_data_preprocessing: Tests for data preprocessing pipeline
- test_integration: End-to-end integration tests
- test_performance: Performance and benchmark tests
"""

__version__ = "1.0.0"
__author__ = "IntegratedML Pluggable Models Team"

# Test configuration
TEST_CONFIG = {
    "random_seed": 42,
    "default_test_samples": 100,
    "performance_test_samples": 1000,
    "test_timeout": 300,  # 5 minutes
    "min_accuracy_threshold": 0.6,
    "min_auc_threshold": 0.6,
}

# Import test classes for easier access
from .test_credit_risk_classifier import (
    TestCustomCreditRiskClassifier,
    TestUtilityFunctions,
)
from .test_data_preprocessing import (
    TestCreditDataPreprocessor,
    TestPreprocessingUtilities,
)
from .test_integration import (
    TestEndToEndWorkflow,
    TestDataPipelineIntegration,
    TestErrorHandlingIntegration,
)

__all__ = [
    "TestCustomCreditRiskClassifier",
    "TestUtilityFunctions",
    "TestCreditDataPreprocessor",
    "TestPreprocessingUtilities",
    "TestEndToEndWorkflow",
    "TestDataPipelineIntegration",
    "TestErrorHandlingIntegration",
    "TEST_CONFIG",
]
