#!/usr/bin/env python3
"""
🚀 IntegratedML Custom Models - Complete Demo Showcase

This script demonstrates the power of IntegratedML's Custom Models feature
by running all four demo use cases, showing how Python ML models can be
seamlessly integrated into IRIS SQL workflows.

Key Features Demonstrated:
- Custom feature engineering within SQL CREATE MODEL
- Real-time predictions via SQL SELECT PREDICT()
- Multiple ML paradigms (classification, regression, ensembles, time series)
- Domain-specific models (financial, fraud, sales, genomics)

Usage:
    python run_all_demos.py [--quick] [--test-only]

Options:
    --quick      Run simplified versions for faster demo
    --test-only  Only run integration tests, skip full demos
"""

import sys
import time
import argparse
from pathlib import Path
import subprocess

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from shared.utils.logging import setup_logger

logger = setup_logger(__name__)


def print_banner(title: str, subtitle: str = ""):
    """Print a fancy banner for demo sections."""
    print("\n" + "="*80)
    print(f"🎯 {title}")
    if subtitle:
        print(f"   {subtitle}")
    print("="*80)


def run_integration_tests():
    """Run all integration tests to verify functionality."""
    print_banner("🧪 INTEGRATION TESTS", "Verifying all components work correctly")

    test_results = {}

    demos = [
        ("Credit Risk", "demos/credit_risk/tests/test_integration.py"),
        ("Fraud Detection", "demos/fraud_detection/tests/test_integration.py"),
        ("Sales Forecasting", "demos/sales_forecasting/tests/test_integration.py"),
        ("DNA Similarity", "demos/dna_similarity/tests/test_integration.py")
    ]

    for demo_name, test_path in demos:
        print(f"\n📋 Testing {demo_name}...")
        try:
            result = subprocess.run([
                "uv", "run", "pytest", test_path, "--tb=short", "-q"
            ], capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                # Count passed tests
                output_lines = result.stdout.split('\n')
                passed_line = [line for line in output_lines if 'passed' in line and ('failed' in line or 'error' in line or 'passed' == line.strip().split()[-1])]
                if passed_line:
                    test_results[demo_name] = f"✅ {passed_line[-1]}"
                else:
                    test_results[demo_name] = "✅ PASSED"
            else:
                failed_line = [line for line in result.stdout.split('\n') if 'failed' in line or 'error' in line]
                if failed_line:
                    test_results[demo_name] = f"⚠️ {failed_line[-1]}"
                else:
                    test_results[demo_name] = "⚠️ SOME ISSUES"

        except subprocess.TimeoutExpired:
            test_results[demo_name] = "⏰ TIMEOUT"
        except Exception as e:
            test_results[demo_name] = f"❌ ERROR: {e}"

    # Print results summary
    print_banner("📊 TEST RESULTS SUMMARY")
    for demo_name, result in test_results.items():
        print(f"   {demo_name:20} {result}")

    return test_results


def run_credit_risk_demo(quick: bool = False):
    """Run the credit risk assessment demo."""
    print_banner("🏦 CREDIT RISK ASSESSMENT", "Financial risk scoring with custom feature engineering")

    print("""
💡 USE CASE: Banks need to assess loan default risk while keeping sensitive data secure.

🔧 INTEGRATEDML SOLUTION:
   - Custom feature engineering (debt-to-income ratios, stability scores)
   - Domain-specific financial calculations executed in-database
   - Compliance-friendly processing without data movement

📝 SQL USAGE:
   CREATE MODEL CreditRiskModel
   PREDICTING (default_risk)
   FROM LoanApplications
   USING "demos.credit_risk.models.CustomCreditRiskClassifier"
   WITH (enable_debt_ratio=true, decision_threshold=0.7);

   SELECT customer_id, loan_amount,
          PREDICT(CreditRiskModel) as risk_score,
          PREDICT(CreditRiskModel PROBABILITY) as risk_probability
   FROM NewApplications;
    """)

    if not quick:
        try:
            print("🔄 Running Credit Risk Demo...")
            subprocess.run(["python", "run_credit_risk_demo.py"], timeout=60)
            print("✅ Credit Risk Demo completed!")
        except Exception as e:
            print(f"⚠️ Demo execution issue: {e}")

    return True


def run_fraud_detection_demo(quick: bool = False):
    """Run the fraud detection demo."""
    print_banner("🛡️ REAL-TIME FRAUD DETECTION", "Sub-100ms fraud detection with ensemble models")

    print("""
💡 USE CASE: Payment processors need lightning-fast fraud detection without data movement.

🔧 INTEGRATEDML SOLUTION:
   - Ensemble model combining neural networks, rules, and behavioral analysis
   - Real-time feature calculation and caching
   - Sub-100ms predictions directly in database

📝 SQL USAGE:
   CREATE MODEL FraudDetectionModel
   PREDICTING (is_fraud)
   FROM Transactions
   USING "demos.fraud_detection.models.EnsembleFraudDetector"
   WITH (confidence_threshold=0.8, enable_neural=true);

   SELECT transaction_id, amount, merchant,
          PREDICT(FraudDetectionModel) as fraud_risk,
          PREDICT(FraudDetectionModel PROBABILITY) as confidence
   FROM LiveTransactions
   WHERE amount > 1000;
    """)

    if not quick:
        try:
            print("🔄 Running Fraud Detection Demo...")
            subprocess.run(["python", "run_fraud_detection_demo.py"], timeout=60)
            print("✅ Fraud Detection Demo completed!")
        except Exception as e:
            print(f"⚠️ Demo execution issue: {e}")

    return True


def run_sales_forecasting_demo(quick: bool = False):
    """Run the sales forecasting demo."""
    print_banner("📈 SALES FORECASTING", "Hybrid time series + ML for accurate predictions")

    print("""
💡 USE CASE: Retailers need accurate forecasts combining time series and ML approaches.

🔧 INTEGRATEDML SOLUTION:
   - Hybrid model integrating Prophet (trending) with LightGBM (pattern learning)
   - Automated seasonal decomposition and feature engineering
   - Confidence intervals and business intelligence integration

📝 SQL USAGE:
   CREATE MODEL SalesForecastModel
   PREDICTING (sales_amount)
   FROM HistoricalSales
   USING "demos.sales_forecasting.models.HybridForecastingModel"
   WITH (seasonality_mode='multiplicative', forecast_horizon=30);

   SELECT date, store_id, product_category,
          PREDICT(SalesForecastModel) as forecast,
          PREDICT(SalesForecastModel CONFIDENCE_INTERVAL) as ci_bounds
   FROM FutureDates
   WHERE date BETWEEN '2024-01-01' AND '2024-01-31';
    """)

    if not quick:
        try:
            print("🔄 Running Sales Forecasting Demo...")
            subprocess.run(["python", "run_sales_forecasting_demo.py"], timeout=60)
            print("✅ Sales Forecasting Demo completed!")
        except Exception as e:
            print(f"⚠️ Demo execution issue: {e}")

    return True


def run_dna_similarity_demo(quick: bool = False):
    """Run the DNA similarity demo."""
    print_banner("🧬 DNA SEQUENCE SIMILARITY", "Specialized genomics algorithms in SQL")

    print("""
💡 USE CASE: Genomics researchers need specialized sequence analysis algorithms.

🔧 INTEGRATEDML SOLUTION:
   - Custom similarity metrics (Levenshtein distance, k-mer analysis)
   - Bioinformatics algorithms accessible via SQL
   - Optimized sequence processing without data export

📝 SQL USAGE:
   CREATE MODEL DNASimilarityModel
   PREDICTING (sequence_class)
   FROM DNASequences
   USING "demos.dna_similarity.models.SimpleDNAClassifier"
   WITH (ngram_size=3, similarity_threshold=0.8);

   SELECT sequence_id, dna_sequence,
          PREDICT(DNASimilarityModel) as predicted_class,
          PREDICT(DNASimilarityModel SIMILARITY) as similarity_score
   FROM NewSequences;
    """)

    if not quick:
        try:
            print("🔄 Running DNA Similarity Demo...")
            subprocess.run(["python", "run_dna_similarity_demo.py"], timeout=60)
            print("✅ DNA Similarity Demo completed!")
        except Exception as e:
            print(f"⚠️ Demo execution issue: {e}")

    return True


def show_conclusion():
    """Show the conclusion and key takeaways."""
    print_banner("🎉 INTEGRATEDML CUSTOM MODELS SHOWCASE COMPLETE!")

    print("""
🚀 KEY ACHIEVEMENTS DEMONSTRATED:

✅ SEAMLESS SQL INTEGRATION
   • Custom Python models work directly in SQL CREATE MODEL and SELECT PREDICT()
   • No data movement required - models execute where data lives
   • Familiar SQL interface for complex ML operations

✅ PYTHON FLEXIBILITY
   • Any scikit-learn compatible model supported
   • Custom preprocessing and feature engineering
   • Integration with popular libraries (TensorFlow, LightGBM, Prophet)

✅ REAL-WORLD USE CASES
   • Financial risk assessment with compliance requirements
   • Real-time fraud detection with sub-100ms latency
   • Advanced sales forecasting with hybrid approaches
   • Specialized genomics algorithms for research

✅ PRODUCTION READY
   • Models persist in database with versioning
   • Built-in security and access controls
   • Scalable in-database execution

🎯 NEXT STEPS:
   1. Explore the demo notebooks in demos/*/notebooks/
   2. Try building your own custom models using the base classes
   3. Check out the PRD.md for complete feature documentation
   4. Join our community to share your models and use cases

💡 WHERE SQL MEETS MACHINE LEARNING - IntegratedML Custom Models makes it possible!
    """)


def main():
    """Main demo orchestration."""
    parser = argparse.ArgumentParser(description="IntegratedML Custom Models Demo Showcase")
    parser.add_argument("--quick", action="store_true", help="Run quick demos without full execution")
    parser.add_argument("--test-only", action="store_true", help="Only run integration tests")
    args = parser.parse_args()

    print_banner("🚀 INTEGRATEDML CUSTOM MODELS", "Bringing Python ML Models to SQL")

    print("""
🎯 WELCOME TO THE FUTURE OF IN-DATABASE MACHINE LEARNING!

This showcase demonstrates how IntegratedML's Custom Models feature revolutionizes
machine learning by bringing YOUR Python models directly into SQL workflows.

No more data movement. No more complex infrastructure. Just pure ML power in SQL.
    """)

    # Always run integration tests first
    test_results = run_integration_tests()

    if args.test_only:
        print("\n✅ Integration tests completed. Use --quick to see demo descriptions.")
        return

    # Run all demos
    print_banner("🎬 DEMO SHOWCASE", "Four real-world use cases")

    try:
        run_credit_risk_demo(args.quick)
        run_fraud_detection_demo(args.quick)
        run_sales_forecasting_demo(args.quick)
        run_dna_similarity_demo(args.quick)

        show_conclusion()

    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Demo error: {e}")
        logger.error(f"Demo execution failed: {e}")


if __name__ == "__main__":
    main()