#!/usr/bin/env python
"""
Complete E2E Test for ALL IntegratedML Custom Models Demos
Tests Credit Risk, Fraud Detection, Sales Forecasting, and DNA Similarity
"""
import os
import sys
import time
import numpy as np
from datetime import datetime, timedelta

sys.path.append(".")
from shared.database.connection import get_connection

os.environ["IRIS_PORT"] = "1974"


def test_credit_risk(conn):
    """Test Credit Risk Assessment"""
    print("\n" + "=" * 70)
    print("TESTING CREDIT RISK ASSESSMENT")
    print("=" * 70)

    try:
        # Clean up
        try:
            conn.execute_sql("DROP MODEL IF EXISTS CreditRiskTest")
            conn.execute_sql("DROP TABLE IF EXISTS CreditRiskTest")
        except:
            pass

        # Create table
        conn.execute_sql(
            """
            CREATE TABLE CreditRiskTest (
                customer_id INT PRIMARY KEY,
                age INT,
                income DECIMAL(10,2),
                debt_ratio DECIMAL(5,3),
                credit_score INT,
                employment_length INT,
                default_risk INT
            )
        """
        )

        # Generate data
        print("Generating 10,000 credit records...")
        np.random.seed(42)

        for i in range(10000):
            age = int(np.clip(np.random.normal(40, 12), 18, 70))
            employment = int(np.random.uniform(0, min(age - 18, 40)))
            income = int(
                np.clip(
                    np.random.normal(50000 + employment * 2000, 15000), 15000, 200000
                )
            )
            credit_score = int(
                np.clip(np.random.normal(650 + employment * 2, 80), 300, 850)
            )
            debt_ratio = np.clip(np.random.beta(2, 5), 0, 1)

            risk = 0
            if credit_score < 600:
                risk += 0.3
            if debt_ratio > 0.5:
                risk += 0.2
            if income < 30000:
                risk += 0.2
            default_risk = 1 if risk > 0.4 else 0

            conn.execute_sql(
                f"""
                INSERT INTO CreditRiskTest VALUES (
                    {i+1}, {age}, {income}, {debt_ratio:.3f},
                    {credit_score}, {employment}, {default_risk}
                )
            """
            )

        # Create and train model
        print("Creating and training model...")
        conn.execute_sql(
            """
            CREATE MODEL CreditRiskTest
            PREDICTING (default_risk)
            FROM CreditRiskTest
            USING {
                "path_to_classifiers": "/opt/iris/mgr/python/custom_models/classifiers",
                "model_name": "CustomCreditRiskClassifier",
                "isc_models_disabled": 1,
                "user_params": {
                    "enable_debt_ratio": 1,
                    "enable_interaction_terms": 0,
                    "enable_risk_scoring": 1
                }
            }
        """
        )

        start = time.time()
        conn.execute_sql("TRAIN MODEL CreditRiskTest")
        train_time = time.time() - start

        # Test predictions
        results = conn.execute_sql(
            """
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN default_risk = ROUND(PREDICT(CreditRiskTest), 0) THEN 1 ELSE 0 END) as correct
            FROM CreditRiskTest
            WHERE customer_id <= 500
        """
        )[0]

        accuracy = results["correct"] / results["total"] * 100

        # Cleanup
        conn.execute_sql("DROP MODEL CreditRiskTest")
        conn.execute_sql("DROP TABLE CreditRiskTest")

        print(f"✅ PASSED - Train: {train_time:.1f}s, Accuracy: {accuracy:.1f}%")
        return True, train_time, accuracy

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, 0, 0


def test_fraud_detection(conn):
    """Test Fraud Detection"""
    print("\n" + "=" * 70)
    print("TESTING FRAUD DETECTION")
    print("=" * 70)

    try:
        # Clean up
        try:
            conn.execute_sql("DROP MODEL IF EXISTS FraudTest")
            conn.execute_sql("DROP TABLE IF EXISTS FraudTest")
        except:
            pass

        # Create table
        conn.execute_sql(
            """
            CREATE TABLE FraudTest (
                transaction_id VARCHAR(20) PRIMARY KEY,
                amount DECIMAL(15,2),
                merchant_country CHAR(2),
                transaction_hour INTEGER,
                anomaly_score DECIMAL(8,6),
                is_fraud INTEGER
            )
        """
        )

        # Generate data
        print("Generating 25,000 transactions...")
        np.random.seed(42)
        fraud_count = 0

        for i in range(25000):
            amount = np.clip(np.random.lognormal(3.5, 1.5), 1, 10000)
            merchant_country = "US" if np.random.random() > 0.15 else "FR"
            hour = np.random.randint(0, 24)
            anomaly_score = np.random.beta(2, 5)

            fraud_prob = 0.02
            if amount > 5000:
                fraud_prob += 0.05
            if merchant_country != "US":
                fraud_prob += 0.03
            if hour < 6 or hour > 22:
                fraud_prob += 0.02
            if anomaly_score > 0.7:
                fraud_prob += 0.05

            is_fraud = 1 if np.random.random() < fraud_prob else 0
            if is_fraud:
                fraud_count += 1

            conn.execute_sql(
                f"""
                INSERT INTO FraudTest VALUES (
                    'TXN{i+1:08d}', {amount:.2f}, '{merchant_country}',
                    {hour}, {anomaly_score:.6f}, {is_fraud}
                )
            """
            )

        print(f"Generated {fraud_count} fraudulent transactions")

        # Create and train model
        print("Creating and training model...")
        conn.execute_sql(
            """
            CREATE MODEL FraudTest
            PREDICTING (is_fraud)
            FROM FraudTest
            USING {
                "path_to_classifiers": "/opt/iris/mgr/python/custom_models/classifiers",
                "model_name": "EnsembleFraudDetector",
                "isc_models_disabled": 1,
                "user_params": {
                    "enable_neural": 1,
                    "enable_rules": 1,
                    "enable_anomaly": 1,
                    "enable_behavioral": 0
                }
            }
        """
        )

        start = time.time()
        conn.execute_sql("TRAIN MODEL FraudTest")
        train_time = time.time() - start

        # Test predictions
        results = conn.execute_sql(
            """
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN PREDICT(FraudTest) > 0.5 THEN 1 ELSE 0 END) as flagged
            FROM FraudTest
            WHERE amount > 1000
        """
        )[0]

        # Cleanup
        conn.execute_sql("DROP MODEL FraudTest")
        conn.execute_sql("DROP TABLE FraudTest")

        print(
            f"✅ PASSED - Train: {train_time:.1f}s, Flagged: {results['flagged']} high-risk"
        )
        return True, train_time, results["flagged"]

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, 0, 0


def test_sales_forecasting(conn):
    """Test Sales Forecasting"""
    print("\n" + "=" * 70)
    print("TESTING SALES FORECASTING")
    print("=" * 70)

    try:
        # Clean up
        try:
            conn.execute_sql("DROP MODEL IF EXISTS SalesTest")
            conn.execute_sql("DROP TABLE IF EXISTS SalesTest")
        except:
            pass

        # Create table
        conn.execute_sql(
            """
            CREATE TABLE SalesTest (
                sales_date DATE,
                store_id VARCHAR(20),
                sales_amount DECIMAL(15,2),
                PRIMARY KEY (sales_date, store_id)
            )
        """
        )

        # Generate data
        print("Generating 365 days of sales data...")
        base_date = datetime.now() - timedelta(days=365)

        for day in range(365):
            current_date = base_date + timedelta(days=day)
            for store in ["STORE001", "STORE002", "STORE003", "STORE004", "STORE005"]:
                seasonal = 1 + 0.3 * np.sin((day - 45) * 2 * np.pi / 365)
                base_sales = (
                    5000
                    if store == "STORE001"
                    else 4000 if store == "STORE002"
                    else 3000 if store == "STORE003"
                    else 3500 if store == "STORE004"
                    else 2500
                )
                sales = base_sales * seasonal * np.random.uniform(0.8, 1.2)

                conn.execute_sql(
                    f"""
                    INSERT INTO SalesTest VALUES (
                        '{current_date.date()}', '{store}', {sales:.2f}
                    )
                """
                )

        # Create and train model
        print("Creating and training model...")
        conn.execute_sql(
            """
            CREATE MODEL SalesTest
            PREDICTING (sales_amount)
            FROM SalesTest
            USING {
                "path_to_regressors": "/opt/iris/mgr/python/custom_models/regressors",
                "model_name": "HybridForecastingModel",
                "isc_models_disabled": 1,
                "user_params": {
                    "prophet_config": {
                        "seasonality_mode": "multiplicative",
                        "weekly_seasonality": true,
                        "yearly_seasonality": false
                    }
                }
            }
        """
        )

        start = time.time()
        conn.execute_sql("TRAIN MODEL SalesTest")
        train_time = time.time() - start

        # Test predictions
        results = conn.execute_sql(
            """
            SELECT AVG(sales_amount) as avg_actual,
                   AVG(PREDICT(SalesTest)) as avg_forecast
            FROM SalesTest
            WHERE sales_date >= CURRENT_DATE - 7
        """
        )[0]

        mape = (
            abs(results["avg_forecast"] - results["avg_actual"])
            / results["avg_actual"]
            * 100
        )

        # Cleanup
        conn.execute_sql("DROP MODEL SalesTest")
        conn.execute_sql("DROP TABLE SalesTest")

        print(f"✅ PASSED - Train: {train_time:.1f}s, MAPE: {mape:.1f}%")
        return True, train_time, mape

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, 0, 0


def test_dna_similarity(conn):
    """Test DNA Similarity Analysis"""
    print("\n" + "=" * 70)
    print("TESTING DNA SIMILARITY ANALYSIS")
    print("=" * 70)

    try:
        # Clean up
        try:
            conn.execute_sql("DROP MODEL IF EXISTS DNATest")
            conn.execute_sql("DROP TABLE IF EXISTS DNATest")
        except:
            pass

        # Create table
        conn.execute_sql(
            """
            CREATE TABLE DNATest (
                sequence_id VARCHAR(50) PRIMARY KEY,
                sequence_data VARCHAR(1000),
                gc_content DECIMAL(5,2),
                sequence_length INT,
                has_mutations INT,
                is_pathogenic INT
            )
        """
        )

        # Generate DNA sequences
        print("Generating 5,000 DNA sequences...")
        np.random.seed(42)

        bases = ["A", "T", "G", "C"]
        mutation_patterns = ["ATCG", "GGCC", "TATA", "CAGT", "CCAA", "TTGG"]

        for i in range(5000):
            # Generate random sequence
            seq_length = np.random.randint(100, 500)
            sequence = "".join(np.random.choice(bases, seq_length))

            # Calculate GC content
            gc_count = sequence.count("G") + sequence.count("C")
            gc_content = gc_count / seq_length * 100

            # Check for mutations
            has_mutations = 0
            for pattern in mutation_patterns:
                if pattern in sequence:
                    has_mutations = 1
                    break

            # Determine if pathogenic based on features
            pathogenic_prob = 0.1
            if gc_content > 60:
                pathogenic_prob += 0.2
            if gc_content < 30:
                pathogenic_prob += 0.15
            if has_mutations:
                pathogenic_prob += 0.25
            if seq_length > 400:
                pathogenic_prob += 0.1

            is_pathogenic = 1 if np.random.random() < pathogenic_prob else 0

            conn.execute_sql(
                f"""
                INSERT INTO DNATest VALUES (
                    'SEQ{i+1:04d}', '{sequence}', {gc_content:.2f},
                    {seq_length}, {has_mutations}, {is_pathogenic}
                )
            """
            )

        # Create and train model
        print("Creating and training model...")
        conn.execute_sql(
            """
            CREATE MODEL DNATest
            PREDICTING (is_pathogenic)
            FROM DNATest
            USING {
                "path_to_classifiers": "/opt/iris/mgr/python/custom_models/classifiers",
                "model_name": "DNASimilarityAnalyzer",
                "isc_models_disabled": 1,
                "user_params": {
                    "k": 5,
                    "use_gc_content": 1,
                    "use_motif_search": 1,
                    "similarity_threshold": 0.8
                }
            }
        """
        )

        start = time.time()
        conn.execute_sql("TRAIN MODEL DNATest")
        train_time = time.time() - start

        # Test predictions
        results = conn.execute_sql(
            """
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN is_pathogenic = ROUND(PREDICT(DNATest), 0) THEN 1 ELSE 0 END) as correct
            FROM DNATest
            WHERE sequence_id LIKE 'SEQ00%'
        """
        )[0]

        accuracy = results["correct"] / results["total"] * 100

        # Test a specific sequence
        test_result = conn.execute_sql(
            """
            SELECT sequence_id, gc_content, has_mutations,
                   is_pathogenic as actual,
                   PREDICT(DNATest) as predicted
            FROM DNATest
            WHERE sequence_id = 'SEQ0001'
        """
        )[0]

        print(
            f"Sample prediction: SEQ0001 - GC: {test_result['gc_content']:.1f}%, "
            f"Mutations: {test_result['has_mutations']}, "
            f"Actual: {test_result['actual']}, "
            f"Predicted: {test_result['predicted']:.3f}"
        )

        # Cleanup
        conn.execute_sql("DROP MODEL DNATest")
        conn.execute_sql("DROP TABLE DNATest")

        print(f"✅ PASSED - Train: {train_time:.1f}s, Accuracy: {accuracy:.1f}%")
        return True, train_time, accuracy

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, 0, 0


def main():
    print("=" * 70)
    print("INTEGRATEDML CUSTOM MODELS - COMPLETE E2E TEST")
    print("=" * 70)
    print(f"Start Time: {datetime.now()}")

    try:
        # Connect
        print("\nConnecting to IRIS...")
        conn = get_connection()
        result = conn.execute_sql("SELECT 1 as test")
        if not result or result[0]["test"] != 1:
            raise Exception("Connection test failed")
        print("✅ Connected to IRIS")

        # Run all tests
        results = {
            "Credit Risk": test_credit_risk(conn),
            "Fraud Detection": test_fraud_detection(conn),
            "Sales Forecasting": test_sales_forecasting(conn),
            "DNA Similarity": test_dna_similarity(conn),
        }

        # Summary
        print("\n" + "=" * 70)
        print("E2E TEST SUMMARY")
        print("=" * 70)
        print("Demo                Status    Train Time   Metric")
        print("-" * 70)

        all_passed = True
        for demo, (passed, train_time, metric) in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"

            if demo == "Credit Risk" or demo == "DNA Similarity":
                metric_str = f"Accuracy: {metric:.1f}%"
            elif demo == "Fraud Detection":
                metric_str = f"Flagged: {int(metric)}"
            else:  # Sales Forecasting
                metric_str = f"MAPE: {metric:.1f}%"

            print(f"{demo:18s}  {status}      {train_time:6.1f}s     {metric_str}")

            if not passed:
                all_passed = False

        print("\n" + "=" * 70)
        if all_passed:
            print("✅ ALL FOUR DEMOS PASSED!")
            print("\nValidated:")
            print("- Credit Risk Assessment with custom classifier")
            print("- Fraud Detection with ensemble model")
            print("- Sales Forecasting with hybrid Prophet/LightGBM")
            print("- DNA Similarity Analysis with sequence algorithms")
            print("- All using IRIS 2025.2 JSON USING syntax")
        else:
            print("❌ SOME TESTS FAILED")
        print("=" * 70)

        return all_passed

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
