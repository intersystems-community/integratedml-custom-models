"""
Comprehensive test runner for the Credit Risk Assessment demo.

This script runs all tests in the credit risk demo test suite and provides
detailed reporting on test results, coverage, and performance metrics.

Usage:
    python run_tests.py [options]

Options:
    --verbose, -v: Verbose output
    --coverage, -c: Run with coverage analysis
    --performance, -p: Include performance benchmarks
    --integration-only, -i: Run only integration tests
    --unit-only, -u: Run only unit tests
    --fast, -f: Run only fast tests (skip performance tests)
    --report: Generate detailed HTML report
"""

import unittest
import sys
import os
import argparse
import time
import json
from pathlib import Path
from io import StringIO

# Add project root to path
project_root = Path(__file__).parents[3]
sys.path.append(str(project_root))

# Import test modules
from demos.credit_risk.tests.test_credit_risk_classifier import (
    TestCustomCreditRiskClassifier,
    TestUtilityFunctions,
)
from demos.credit_risk.tests.test_data_preprocessing import (
    TestCreditDataPreprocessor,
    TestPreprocessingUtilities,
)
from demos.credit_risk.tests.test_integration import (
    TestEndToEndWorkflow,
    TestDataPipelineIntegration,
    TestErrorHandlingIntegration,
)


class TestResults:
    """Container for test execution results."""

    def __init__(self):
        self.total_tests = 0
        self.passed = 0
        self.failed = 0
        self.errors = 0
        self.skipped = 0
        self.execution_time = 0.0
        self.failures = []
        self.errors_list = []
        self.test_details = {}


class ColoredTextTestResult(unittest.TextTestResult):
    """Custom test result class with colored output and detailed tracking."""

    def __init__(self, stream, verbosity, results_container):
        super().__init__(stream, None, verbosity)
        self.results = results_container
        self.start_time = None

    def startTest(self, test):
        super().startTest(test)
        self.start_time = time.time()
        if self.verbosity > 1:
            self.stream.write(f"Running {test._testMethodName}... ")
            self.stream.flush()

    def addSuccess(self, test):
        super().addSuccess(test)
        self.results.passed += 1
        test_time = time.time() - self.start_time
        self.results.test_details[str(test)] = {"status": "PASS", "time": test_time}
        if self.verbosity > 1:
            self.stream.write(f"✓ PASS ({test_time:.3f}s)\n")

    def addError(self, test, err):
        super().addError(test, err)
        self.results.errors += 1
        self.results.errors_list.append((test, err))
        test_time = time.time() - self.start_time
        self.results.test_details[str(test)] = {
            "status": "ERROR",
            "time": test_time,
            "error": str(err[1]),
        }
        if self.verbosity > 1:
            self.stream.write(f"✗ ERROR ({test_time:.3f}s)\n")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.results.failed += 1
        self.results.failures.append((test, err))
        test_time = time.time() - self.start_time
        self.results.test_details[str(test)] = {
            "status": "FAIL",
            "time": test_time,
            "error": str(err[1]),
        }
        if self.verbosity > 1:
            self.stream.write(f"✗ FAIL ({test_time:.3f}s)\n")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.results.skipped += 1
        test_time = time.time() - self.start_time
        self.results.test_details[str(test)] = {
            "status": "SKIP",
            "time": test_time,
            "reason": reason,
        }
        if self.verbosity > 1:
            self.stream.write(f"- SKIP ({test_time:.3f}s): {reason}\n")


def create_test_suite(test_type="all"):
    """Create test suite based on specified test type."""
    suite = unittest.TestSuite()

    if test_type in ["all", "unit"]:
        # Unit tests
        suite.addTest(
            unittest.TestLoader().loadTestsFromTestCase(TestCustomCreditRiskClassifier)
        )
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestUtilityFunctions))
        suite.addTest(
            unittest.TestLoader().loadTestsFromTestCase(TestCreditDataPreprocessor)
        )
        suite.addTest(
            unittest.TestLoader().loadTestsFromTestCase(TestPreprocessingUtilities)
        )

    if test_type in ["all", "integration"]:
        # Integration tests
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestEndToEndWorkflow))
        suite.addTest(
            unittest.TestLoader().loadTestsFromTestCase(TestDataPipelineIntegration)
        )
        suite.addTest(
            unittest.TestLoader().loadTestsFromTestCase(TestErrorHandlingIntegration)
        )

    return suite


def create_result_class(results_container):
    """Create a custom result class factory."""

    class CustomResult(ColoredTextTestResult):
        def __init__(self, stream, descriptions, verbosity):
            # Call the parent's __init__ with the correct arguments
            super().__init__(stream, verbosity, results_container)

    return CustomResult


def run_with_coverage(suite, verbosity=1):
    """Run tests with coverage analysis if coverage package is available."""
    try:
        import coverage

        cov = coverage.Coverage(source=["demos.credit_risk"])
        cov.start()

        # Run tests
        results = TestResults()
        runner = unittest.TextTestRunner(
            verbosity=verbosity, resultclass=create_result_class(results)
        )

        start_time = time.time()
        test_result = runner.run(suite)
        results.execution_time = time.time() - start_time

        cov.stop()
        cov.save()

        # Generate coverage report
        print("\n" + "=" * 60)
        print("COVERAGE REPORT")
        print("=" * 60)
        cov.report()

        return results, test_result

    except ImportError:
        print(
            "Warning: coverage package not installed. Running without coverage analysis."
        )
        return run_without_coverage(suite, verbosity)


def run_without_coverage(suite, verbosity=1):
    """Run tests without coverage analysis."""
    results = TestResults()
    runner = unittest.TextTestRunner(
        verbosity=verbosity, resultclass=create_result_class(results)
    )

    start_time = time.time()
    test_result = runner.run(suite)
    results.execution_time = time.time() - start_time

    return results, test_result


def print_summary(results, test_result):
    """Print detailed test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    total = test_result.testsRun
    passed = total - len(test_result.failures) - len(test_result.errors)

    print(f"Total Tests:     {total}")
    print(f"Passed:          {passed} ✓")
    print(f"Failed:          {len(test_result.failures)} ✗")
    print(f"Errors:          {len(test_result.errors)} ✗")
    print(f"Skipped:         {len(test_result.skipped)} -")
    print(f"Execution Time:  {results.execution_time:.2f} seconds")

    if total > 0:
        success_rate = (passed / total) * 100
        print(f"Success Rate:    {success_rate:.1f}%")

        if success_rate >= 95:
            print("Status:          EXCELLENT ✓✓✓")
        elif success_rate >= 90:
            print("Status:          GOOD ✓✓")
        elif success_rate >= 80:
            print("Status:          ACCEPTABLE ✓")
        else:
            print("Status:          NEEDS IMPROVEMENT ✗")

    # Print failures and errors if any
    if test_result.failures:
        print(f"\n{'='*20} FAILURES {'='*20}")
        for test, error in test_result.failures:
            print(f"\nFAIL: {test}")
            print("-" * 40)
            print(error)

    if test_result.errors:
        print(f"\n{'='*20} ERRORS {'='*20}")
        for test, error in test_result.errors:
            print(f"\nERROR: {test}")
            print("-" * 40)
            print(error)


def generate_json_report(results, test_result, output_file="test_results.json"):
    """Generate JSON report of test results."""
    report = {
        "summary": {
            "total_tests": test_result.testsRun,
            "passed": test_result.testsRun
            - len(test_result.failures)
            - len(test_result.errors),
            "failed": len(test_result.failures),
            "errors": len(test_result.errors),
            "skipped": len(test_result.skipped),
            "execution_time": results.execution_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "test_details": results.test_details,
        "failures": [
            {"test": str(test), "error": str(error)}
            for test, error in test_result.failures
        ],
        "errors": [
            {"test": str(test), "error": str(error)}
            for test, error in test_result.errors
        ],
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed JSON report saved to: {output_file}")


def run_performance_benchmarks():
    """Run performance benchmarks and report results."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 60)

    try:
        from demos.credit_risk.data.generate_sample_data import CreditDataGenerator
        from demos.credit_risk.models.credit_risk_classifier import (
            CustomCreditRiskClassifier,
        )
        from demos.credit_risk.scripts.data_preprocessing import CreditDataPreprocessor
        from sklearn.model_selection import train_test_split
        import time

        # Benchmark data generation
        print("1. Data Generation Benchmark")
        generator = CreditDataGenerator(random_seed=42)

        start_time = time.time()
        X, y = generator.generate_dataset(n_samples=1000, default_rate=0.3)
        generation_time = time.time() - start_time
        print(f"   Generated 1000 samples in {generation_time:.3f} seconds")

        # Benchmark preprocessing
        print("2. Data Preprocessing Benchmark")
        preprocessor = CreditDataPreprocessor(
            handle_missing=True, remove_outliers=True, normalize_features=True
        )

        start_time = time.time()
        X_processed, y_processed = preprocessor.fit_transform(X, y)
        preprocessing_time = time.time() - start_time
        print(f"   Preprocessed 1000 samples in {preprocessing_time:.3f} seconds")

        # Benchmark model training
        print("3. Model Training Benchmark")
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.3, random_state=42
        )

        configurations = [
            (
                "Baseline",
                {
                    "enable_debt_ratio": False,
                    "enable_interaction_terms": False,
                    "enable_risk_scoring": False,
                },
            ),
            (
                "Full Features",
                {
                    "enable_debt_ratio": True,
                    "enable_interaction_terms": True,
                    "enable_risk_scoring": True,
                },
            ),
        ]

        for config_name, params in configurations:
            model = CustomCreditRiskClassifier(**params)

            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            start_time = time.time()
            predictions = model.predict(X_test)
            prediction_time = time.time() - start_time

            print(f"   {config_name}:")
            print(f"     Training time: {training_time:.3f} seconds")
            print(
                f"     Prediction time: {prediction_time:.3f} seconds ({len(X_test)} samples)"
            )
            print(f"     Predictions/second: {len(X_test)/prediction_time:.0f}")

        total_time = (
            generation_time + preprocessing_time + training_time + prediction_time
        )
        print(f"\nTotal pipeline time: {total_time:.3f} seconds")

    except Exception as e:
        print(f"Performance benchmark failed: {e}")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run Credit Risk Assessment tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "-c", "--coverage", action="store_true", help="Run with coverage analysis"
    )
    parser.add_argument(
        "-p",
        "--performance",
        action="store_true",
        help="Include performance benchmarks",
    )
    parser.add_argument(
        "-i",
        "--integration-only",
        action="store_true",
        help="Run only integration tests",
    )
    parser.add_argument(
        "-u", "--unit-only", action="store_true", help="Run only unit tests"
    )
    parser.add_argument("-f", "--fast", action="store_true", help="Run only fast tests")
    parser.add_argument("--report", type=str, help="Generate JSON report to file")

    args = parser.parse_args()

    # Determine test type
    if args.integration_only:
        test_type = "integration"
    elif args.unit_only:
        test_type = "unit"
    else:
        test_type = "all"

    # Determine verbosity
    verbosity = 2 if args.verbose else 1

    print("Credit Risk Assessment Demo - Test Suite")
    print("=" * 60)
    print(f"Test Type: {test_type.upper()}")
    print(f"Verbosity: {'High' if args.verbose else 'Standard'}")
    if args.coverage:
        print("Coverage: Enabled")
    print()

    # Create and run test suite
    suite = create_test_suite(test_type)

    if args.coverage:
        results, test_result = run_with_coverage(suite, verbosity)
    else:
        results, test_result = run_without_coverage(suite, verbosity)

    # Print summary
    print_summary(results, test_result)

    # Generate report if requested
    if args.report:
        generate_json_report(results, test_result, args.report)

    # Run performance benchmarks if requested
    if args.performance and not args.fast:
        run_performance_benchmarks()

    # Return appropriate exit code
    if test_result.failures or test_result.errors:
        return 1
    else:
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
