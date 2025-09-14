"""
Fraud Detection Ensemble Test Suite
==================================

Comprehensive testing framework for the fraud detection system including:
- Unit tests for individual components
- Integration tests for ensemble orchestration
- Performance tests for latency requirements
- End-to-end workflow testing
"""

__version__ = "1.0.0"
__author__ = "IntegratedML Team"

# Test configuration
TEST_CONFIG = {
    "max_latency_ms": 100,
    "min_accuracy": 0.85,
    "test_data_size": 1000,
    "performance_iterations": 100,
    "tolerance": 0.05,
}
