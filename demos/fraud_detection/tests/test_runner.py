"""
Test runner for fraud detection system.
Provides convenient test execution and reporting.
"""

import pytest
import sys
import os
import time
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def run_unit_tests():
    """Run unit tests only."""
    print("🧪 Running Unit Tests...")
    return pytest.main([
        "-v",
        "-m", "unit",
        str(Path(__file__).parent),
        "--tb=short"
    ])


def run_integration_tests():
    """Run integration tests only."""
    print("🔗 Running Integration Tests...")
    return pytest.main([
        "-v",
        "-m", "integration",
        str(Path(__file__).parent),
        "--tb=short"
    ])


def run_performance_tests():
    """Run performance tests only."""
    print("⚡ Running Performance Tests...")
    return pytest.main([
        "-v",
        "-m", "performance",
        str(Path(__file__).parent),
        "--tb=short"
    ])


def run_all_tests():
    """Run all tests."""
    print("🚀 Running All Tests...")
    return pytest.main([
        "-v",
        str(Path(__file__).parent),
        "--tb=short"
    ])


def run_quick_tests():
    """Run quick tests (excluding slow performance tests)."""
    print("⚡ Running Quick Tests...")
    return pytest.main([
        "-v",
        "-m", "not slow",
        str(Path(__file__).parent),
        "--tb=short"
    ])


def run_latency_verification():
    """Run specific latency verification tests."""
    print("🎯 Running Latency Verification Tests...")
    return pytest.main([
        "-v",
        "-k", "latency",
        str(Path(__file__).parent),
        "--tb=short"
    ])


def run_with_coverage():
    """Run tests with coverage reporting."""
    print("📊 Running Tests with Coverage...")
    return pytest.main([
        "-v",
        "--cov=demos.fraud_detection",
        "--cov-report=html",
        "--cov-report=term-missing",
        str(Path(__file__).parent),
        "--tb=short"
    ])


def main():
    """Main test runner with command line interface."""
    parser = argparse.ArgumentParser(description="Fraud Detection Test Runner")
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "performance", "all", "quick", "latency", "coverage"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print("🔍 Fraud Detection Test Suite")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run specified test type
    if args.test_type == "unit":
        exit_code = run_unit_tests()
    elif args.test_type == "integration":
        exit_code = run_integration_tests()
    elif args.test_type == "performance":
        exit_code = run_performance_tests()
    elif args.test_type == "all":
        exit_code = run_all_tests()
    elif args.test_type == "quick":
        exit_code = run_quick_tests()
    elif args.test_type == "latency":
        exit_code = run_latency_verification()
    elif args.test_type == "coverage":
        exit_code = run_with_coverage()
    else:
        print(f"❌ Unknown test type: {args.test_type}")
        return 1
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("=" * 50)
    print(f"⏱️ Test execution completed in {duration:.2f} seconds")
    
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return exit_code


if __name__ == "__main__":
    exit(main())