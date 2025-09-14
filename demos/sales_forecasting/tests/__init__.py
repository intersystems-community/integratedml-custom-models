"""
Test Suite for Sales Forecasting System.

This package provides comprehensive testing for the sales forecasting system,
including unit tests, integration tests, and performance benchmarks.

Test Modules:
- test_hybrid_forecasting_model: Core model testing with time series validation
- test_components: Individual component testing (data generation, feature engineering, analytics)
- test_integration: End-to-end pipeline and performance testing
- run_tests: Main test runner with multiple test suite options

Usage:
    # Run all tests
    python run_tests.py --suite all

    # Run quick smoke tests
    python run_tests.py --suite smoke

    # Run specific test suite
    python run_tests.py --suite unit
    python run_tests.py --suite integration

    # Generate test report
    python run_tests.py --suite all --report test_results.txt
"""

__version__ = "1.0.0"

# Test configuration
TEST_CONFIG = {
    "random_seed": 42,
    "test_data_size": "small",  # small, medium, large
    "performance_benchmarks": True,
    "integration_tests": True,
    "verbose_output": True,
    "save_test_artifacts": False,
}

# Test data specifications
TEST_DATA_SPECS = {
    "small": {
        "date_range": 90,  # days
        "stores": 1,
        "categories": 1,
        "training_samples": 300,
    },
    "medium": {
        "date_range": 365,  # days
        "stores": 2,
        "categories": 2,
        "training_samples": 1000,
    },
    "large": {
        "date_range": 730,  # days
        "stores": 5,
        "categories": 3,
        "training_samples": 2000,
    },
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "training_time_max": 300,  # seconds
    "prediction_speed_min": 100,  # predictions per second
    "memory_usage_max": 1000,  # MB
    "mape_threshold": 50,  # percentage
    "r2_threshold": 0.3,  # correlation
}

# Test suite information
TEST_SUITES = {
    "unit": {
        "description": "Unit tests for individual components",
        "modules": ["test_components"],
        "estimated_time": "2-3 minutes",
    },
    "model": {
        "description": "Model-specific tests with time series validation",
        "modules": ["test_hybrid_forecasting_model"],
        "estimated_time": "5-10 minutes",
    },
    "integration": {
        "description": "End-to-end pipeline and performance tests",
        "modules": ["test_integration"],
        "estimated_time": "10-15 minutes",
    },
    "performance": {
        "description": "Performance benchmarks and optimization tests",
        "modules": ["test_integration.TestPerformanceBenchmarks"],
        "estimated_time": "5-10 minutes",
    },
    "smoke": {
        "description": "Quick validation of core functionality",
        "modules": ["run_tests.smoke_tests"],
        "estimated_time": "1-2 minutes",
    },
    "quick": {
        "description": "Essential tests for development",
        "modules": ["run_tests.quick_tests"],
        "estimated_time": "3-5 minutes",
    },
    "all": {
        "description": "Complete test suite with all validations",
        "modules": ["all"],
        "estimated_time": "15-25 minutes",
    },
}


def get_test_config():
    """Get current test configuration."""
    return TEST_CONFIG.copy()


def get_test_data_spec(size="small"):
    """Get test data specification for given size."""
    return TEST_DATA_SPECS.get(size, TEST_DATA_SPECS["small"])


def get_performance_thresholds():
    """Get performance testing thresholds."""
    return PERFORMANCE_THRESHOLDS.copy()


def list_test_suites():
    """List available test suites with descriptions."""
    return TEST_SUITES.copy()
