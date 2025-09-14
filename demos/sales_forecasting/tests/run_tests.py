"""
Main Test Runner for Sales Forecasting System.

This script provides a unified interface to run all test suites for the
sales forecasting system, including unit tests, integration tests, and
performance benchmarks.
"""

import sys
import os
import unittest
import time
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(os.path.abspath("../../.."))

# Import test modules
from test_hybrid_forecasting_model import (
    TestHybridForecastingModel,
    TestTimeSeriesValidation,
    TestModelComponents,
    TestBusinessLogic,
    TestDataQuality,
    run_all_tests as run_model_tests,
)

from test_integration import (
    TestEndToEndPipeline,
    TestPerformanceBenchmarks,
    run_integration_tests,
)

from test_components import (
    TestSalesDataGenerator,
    TestFeatureEngineer,
    TestForecastEvaluator,
    TestBusinessIntelligence,
    run_component_tests,
)

warnings.filterwarnings("ignore")


class TestResult:
    """Container for test results."""

    def __init__(
        self,
        name: str,
        tests_run: int,
        failures: int,
        errors: int,
        skipped: int = 0,
        duration: float = 0.0,
    ):
        self.name = name
        self.tests_run = tests_run
        self.failures = failures
        self.errors = errors
        self.skipped = skipped
        self.duration = duration
        self.success_rate = (
            (tests_run - failures - errors) / tests_run * 100 if tests_run > 0 else 0
        )

    def __str__(self):
        return (
            f"{self.name}: {self.tests_run} tests, "
            f"{self.failures} failures, {self.errors} errors, "
            f"{self.skipped} skipped, {self.success_rate:.1f}% success"
        )


class SalesTestRunner:
    """Comprehensive test runner for sales forecasting system."""

    def __init__(self, verbosity: int = 2):
        self.verbosity = verbosity
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None

    def run_test_suite(
        self, suite_name: str, test_classes: List, standalone_runner: Optional = None
    ) -> TestResult:
        """Run a specific test suite."""
        print(f"\n{'='*60}")
        print(f"RUNNING {suite_name.upper()}")
        print(f"{'='*60}")

        start_time = time.time()

        if standalone_runner:
            # Use standalone runner function
            result = standalone_runner()
            duration = time.time() - start_time

            test_result = TestResult(
                name=suite_name,
                tests_run=result.testsRun,
                failures=len(result.failures),
                errors=len(result.errors),
                skipped=len(getattr(result, "skipped", [])),
                duration=duration,
            )
        else:
            # Create test suite manually
            test_loader = unittest.TestLoader()
            test_suite = unittest.TestSuite()

            for test_class in test_classes:
                tests = test_loader.loadTestsFromTestCase(test_class)
                test_suite.addTests(tests)

            # Run tests
            runner = unittest.TextTestRunner(verbosity=self.verbosity)
            result = runner.run(test_suite)
            duration = time.time() - start_time

            test_result = TestResult(
                name=suite_name,
                tests_run=result.testsRun,
                failures=len(result.failures),
                errors=len(result.errors),
                skipped=len(getattr(result, "skipped", [])),
                duration=duration,
            )

        self.results.append(test_result)
        print(f"\n{suite_name} completed in {duration:.2f} seconds")
        return test_result

    def run_unit_tests(self) -> TestResult:
        """Run unit tests for individual components."""
        test_classes = [
            TestSalesDataGenerator,
            TestFeatureEngineer,
            TestForecastEvaluator,
            TestBusinessIntelligence,
        ]

        return self.run_test_suite("Unit Tests", test_classes)

    def run_model_tests(self) -> TestResult:
        """Run model-specific tests."""
        test_classes = [
            TestHybridForecastingModel,
            TestTimeSeriesValidation,
            TestModelComponents,
            TestBusinessLogic,
            TestDataQuality,
        ]

        return self.run_test_suite("Model Tests", test_classes)

    def run_integration_tests(self) -> TestResult:
        """Run integration tests."""
        test_classes = [TestEndToEndPipeline, TestPerformanceBenchmarks]

        return self.run_test_suite("Integration Tests", test_classes)

    def run_performance_tests(self) -> TestResult:
        """Run performance benchmark tests."""
        test_classes = [TestPerformanceBenchmarks]

        return self.run_test_suite("Performance Tests", test_classes)

    def run_all_tests(self) -> List[TestResult]:
        """Run all test suites."""
        print("=" * 80)
        print("SALES FORECASTING SYSTEM - COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.start_time = time.time()

        try:
            # Run test suites in order
            self.run_unit_tests()
            self.run_model_tests()
            self.run_integration_tests()

        except KeyboardInterrupt:
            print("\n\nTest run interrupted by user")
            return self.results

        except Exception as e:
            print(f"\n\nUnexpected error during test run: {e}")
            return self.results

        finally:
            self.end_time = time.time()
            self.print_final_summary()

        return self.results

    def run_quick_tests(self) -> List[TestResult]:
        """Run a quick subset of tests for development."""
        print("=" * 80)
        print("SALES FORECASTING SYSTEM - QUICK TEST SUITE")
        print("=" * 80)

        self.start_time = time.time()

        # Run essential tests only
        essential_classes = [
            TestHybridForecastingModel,
            TestSalesDataGenerator,
            TestFeatureEngineer,
            TestForecastEvaluator,
        ]

        self.run_test_suite("Quick Tests", essential_classes)

        self.end_time = time.time()
        self.print_final_summary()

        return self.results

    def run_smoke_tests(self) -> List[TestResult]:
        """Run basic smoke tests to verify system functionality."""
        print("=" * 80)
        print("SALES FORECASTING SYSTEM - SMOKE TESTS")
        print("=" * 80)

        self.start_time = time.time()

        # Create minimal smoke test suite
        smoke_loader = unittest.TestLoader()
        smoke_suite = unittest.TestSuite()

        # Add one test method from each critical component
        smoke_tests = [
            TestHybridForecastingModel("test_model_initialization"),
            TestHybridForecastingModel("test_model_fitting"),
            TestHybridForecastingModel("test_model_prediction"),
            TestSalesDataGenerator("test_full_dataset_generation"),
            TestFeatureEngineer("test_full_feature_engineering"),
            TestForecastEvaluator("test_evaluate_forecast"),
        ]

        for test in smoke_tests:
            smoke_suite.addTest(test)

        start_time = time.time()
        runner = unittest.TextTestRunner(verbosity=self.verbosity)
        result = runner.run(smoke_suite)
        duration = time.time() - start_time

        smoke_result = TestResult(
            name="Smoke Tests",
            tests_run=result.testsRun,
            failures=len(result.failures),
            errors=len(result.errors),
            skipped=len(getattr(result, "skipped", [])),
            duration=duration,
        )

        self.results.append(smoke_result)
        self.end_time = time.time()
        self.print_final_summary()

        return self.results

    def print_final_summary(self):
        """Print comprehensive test summary."""
        if not self.results:
            print("No test results to summarize")
            return

        total_duration = (
            self.end_time - self.start_time if self.end_time and self.start_time else 0
        )

        print("\n" + "=" * 80)
        print("FINAL TEST SUMMARY")
        print("=" * 80)

        # Overall statistics
        total_tests = sum(r.tests_run for r in self.results)
        total_failures = sum(r.failures for r in self.results)
        total_errors = sum(r.errors for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        total_passed = total_tests - total_failures - total_errors
        overall_success_rate = (
            total_passed / total_tests * 100 if total_tests > 0 else 0
        )

        print(f"Total Duration: {total_duration:.2f} seconds")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failures}")
        print(f"Errors: {total_errors}")
        print(f"Skipped: {total_skipped}")
        print(f"Overall Success Rate: {overall_success_rate:.1f}%")

        # Suite breakdown
        print(f"\nSuite Breakdown:")
        print("-" * 60)
        for result in self.results:
            status = (
                "✓ PASS" if result.failures == 0 and result.errors == 0 else "✗ FAIL"
            )
            print(f"{status} {result}")

        # Performance summary
        if any("Performance" in r.name for r in self.results):
            print(f"\nPerformance Summary:")
            print("-" * 60)
            for result in self.results:
                if "Performance" in result.name:
                    print(f"Performance tests completed in {result.duration:.2f}s")

        # Recommendations
        print(f"\nRecommendations:")
        print("-" * 60)

        if overall_success_rate >= 95:
            print("✓ Excellent test coverage and system health")
        elif overall_success_rate >= 85:
            print("⚠ Good test coverage, minor issues to address")
        elif overall_success_rate >= 70:
            print("⚠ Moderate test coverage, several issues need attention")
        else:
            print(
                "✗ Poor test coverage, significant issues require immediate attention"
            )

        if total_failures > 0:
            print(f"• Address {total_failures} test failures")

        if total_errors > 0:
            print(f"• Fix {total_errors} test errors")

        if total_skipped > total_tests * 0.1:  # More than 10% skipped
            print(f"• Review {total_skipped} skipped tests")

        # Final status
        print("\n" + "=" * 80)
        if total_failures == 0 and total_errors == 0:
            print("🎉 ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT")
        else:
            print("⚠️  SOME TESTS FAILED - REVIEW REQUIRED BEFORE DEPLOYMENT")
        print("=" * 80)

    def generate_test_report(self, output_file: str = "test_report.txt"):
        """Generate detailed test report file."""
        if not self.results:
            print("No test results to report")
            return

        with open(output_file, "w") as f:
            f.write("SALES FORECASTING SYSTEM - TEST REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary
            total_tests = sum(r.tests_run for r in self.results)
            total_failures = sum(r.failures for r in self.results)
            total_errors = sum(r.errors for r in self.results)
            total_passed = total_tests - total_failures - total_errors
            overall_success_rate = (
                total_passed / total_tests * 100 if total_tests > 0 else 0
            )

            f.write(f"SUMMARY\n")
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"Passed: {total_passed}\n")
            f.write(f"Failed: {total_failures}\n")
            f.write(f"Errors: {total_errors}\n")
            f.write(f"Success Rate: {overall_success_rate:.1f}%\n\n")

            # Detailed results
            f.write("DETAILED RESULTS\n")
            f.write("-" * 30 + "\n")
            for result in self.results:
                f.write(f"{result}\n")

            f.write(f"\nReport saved to: {output_file}\n")

        print(f"Test report saved to: {output_file}")


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Sales Forecasting System Test Runner")
    parser.add_argument(
        "--suite",
        choices=[
            "all",
            "unit",
            "model",
            "integration",
            "performance",
            "quick",
            "smoke",
        ],
        default="all",
        help="Test suite to run",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Test output verbosity level",
    )
    parser.add_argument("--report", type=str, help="Generate test report file")
    parser.add_argument(
        "--no-warnings", action="store_true", help="Suppress warning messages"
    )

    args = parser.parse_args()

    if args.no_warnings:
        warnings.filterwarnings("ignore")

    # Create test runner
    runner = SalesTestRunner(verbosity=args.verbosity)

    # Run selected test suite
    if args.suite == "all":
        results = runner.run_all_tests()
    elif args.suite == "unit":
        results = [runner.run_unit_tests()]
    elif args.suite == "model":
        results = [runner.run_model_tests()]
    elif args.suite == "integration":
        results = [runner.run_integration_tests()]
    elif args.suite == "performance":
        results = [runner.run_performance_tests()]
    elif args.suite == "quick":
        results = runner.run_quick_tests()
    elif args.suite == "smoke":
        results = runner.run_smoke_tests()
    else:
        print(f"Unknown test suite: {args.suite}")
        sys.exit(1)

    # Generate report if requested
    if args.report:
        runner.generate_test_report(args.report)

    # Exit with appropriate code
    total_failures = sum(r.failures for r in results)
    total_errors = sum(r.errors for r in results)

    if total_failures > 0 or total_errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
