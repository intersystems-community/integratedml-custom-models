#!/usr/bin/env python3
"""
IntegratedML Flexible Model Integration Demo - Quick Start Example

This example demonstrates how to use the IntegratedML demo models
in a standalone Python environment before deploying to IntegratedML.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import demo models
from demos.credit_risk.models.credit_risk_classifier import CustomCreditRiskClassifier
from demos.fraud_detection.models.ensemble_fraud_detector import EnsembleFraudDetector
from shared.models.classification import ClassificationModel


def create_sample_credit_data():
    """Create sample credit risk data for demonstration."""
    print("Creating sample credit risk data...")

    # Generate synthetic data similar to German Credit dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_classes=2,
        class_sep=0.8,
        random_state=42,
    )

    # Create DataFrame with meaningful column names
    feature_names = [
        "checking_balance",
        "duration_months",
        "credit_history",
        "purpose",
        "amount",
        "savings_balance",
        "employment_duration",
        "installment_rate",
        "personal_status",
        "other_debtors",
        "residence_history",
        "property",
        "age",
        "installment_plans",
        "housing",
    ]

    df = pd.DataFrame(X, columns=feature_names)
    df["default_risk"] = y

    return df


def demo_credit_risk_model():
    """Demonstrate the Custom Credit Risk Classifier."""
    print("\n" + "=" * 60)
    print("DEMO 1: Credit Risk Assessment with Custom Feature Engineering")
    print("=" * 60)

    # Create sample data
    data = create_sample_credit_data()
    X = data.drop("default_risk", axis=1)
    y = data["default_risk"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Initialize and train model
    print("\nInitializing Custom Credit Risk Classifier...")
    model = CustomCreditRiskClassifier(
        enable_debt_ratio=True,
        enable_interaction_terms=True,
        enable_risk_scoring=True,
        decision_threshold=0.6,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy:.3f}")

    # Show some example predictions
    print("\nSample Predictions:")
    print("-" * 40)
    for i in range(5):
        print(
            f"Customer {i+1}: Risk Probability = {probabilities[i, 1]:.3f}, "
            f"Decision = {'HIGH RISK' if predictions[i] == 1 else 'LOW RISK'}"
        )

    # Show model info
    print(f"\nModel Information:")
    model_info = model.get_model_info()
    print(f"Model Class: {model_info['model_class']}")
    print(f"Is Fitted: {model_info['is_fitted']}")
    print(f"Features: {model_info['n_features_in']}")

    return model


def demo_fraud_detection_ensemble():
    """Demonstrate the Ensemble Fraud Detector."""
    print("\n" + "=" * 60)
    print("DEMO 2: Real-time Fraud Detection with Ensemble Models")
    print("=" * 60)

    # Create sample transaction data
    print("Creating sample transaction data...")
    X, y = make_classification(
        n_samples=5000,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        class_sep=0.9,
        weights=[0.95, 0.05],  # Imbalanced: 5% fraud
        random_state=42,
    )

    feature_names = [
        "amount",
        "merchant_category",
        "hour_of_day",
        "day_of_week",
        "customer_age",
        "account_balance",
        "transaction_velocity",
        "location_risk_score",
        "merchant_risk_score",
        "payment_method",
        "previous_fraud_count",
        "account_age_days",
    ]

    X_df = pd.DataFrame(X, columns=feature_names)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Fraud rate in training: {np.mean(y_train):.3%}")

    # Initialize ensemble detector
    print("\nInitializing Ensemble Fraud Detector...")
    detector = EnsembleFraudDetector(
        voting="weighted",
        confidence_threshold=0.8,
        enable_rule_engine=True,
        enable_anomaly_detection=True,
        enable_ml_classifier=True,
    )

    print("Training ensemble model...")
    try:
        detector.fit(X_train, y_train)

        # Make predictions
        print("Making fraud predictions...")
        predictions = detector.predict(X_test)
        probabilities = detector.predict_proba(X_test)

        # Calculate metrics
        accuracy = np.mean(predictions == y_test)
        fraud_detected = np.sum((predictions == 1) & (y_test == 1))
        total_fraud = np.sum(y_test == 1)

        print(f"Accuracy: {accuracy:.3f}")
        print(
            f"Fraud Detection Rate: {fraud_detected}/{total_fraud} "
            f"({fraud_detected/total_fraud:.1%})"
        )

        # Show some example predictions
        print("\nSample Fraud Predictions:")
        print("-" * 50)
        fraud_indices = np.where(y_test == 1)[0][:3]
        for idx in fraud_indices:
            print(
                f"Transaction {idx}: Fraud Probability = {probabilities[idx, 1]:.3f}, "
                f"Detected = {'YES' if predictions[idx] == 1 else 'NO'}"
            )

    except Exception as e:
        print(f"Note: Ensemble training skipped due to missing dependencies: {e}")
        print("Install required packages: pip install xgboost scikit-learn")

    return detector


def demo_model_serialization():
    """Demonstrate model saving and loading."""
    print("\n" + "=" * 60)
    print("DEMO: Model Serialization and Loading")
    print("=" * 60)

    # Create and train a simple model
    data = create_sample_credit_data()
    X = data.drop("default_risk", axis=1)
    y = data["default_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = CustomCreditRiskClassifier(decision_threshold=0.5)
    model.fit(X_train, y_train)

    # Save model
    model_path = "/tmp/credit_risk_model.pkl"
    print(f"Saving model to {model_path}...")
    model.save_model(model_path)

    # Load model
    print("Loading model from file...")
    loaded_model = CustomCreditRiskClassifier.load_model(model_path)

    # Test loaded model
    original_pred = model.predict(X_test[:5])
    loaded_pred = loaded_model.predict(X_test[:5])

    print("Comparing predictions from original and loaded model:")
    print(f"Original predictions: {original_pred}")
    print(f"Loaded predictions:   {loaded_pred}")
    print(f"Predictions match: {np.array_equal(original_pred, loaded_pred)}")

    # Clean up
    os.remove(model_path)
    print("Temporary model file cleaned up.")


def main():
    """Run all demo examples."""
    print("IntegratedML Flexible Model Integration Demo - Quick Start Examples")
    print("========================================================")

    try:
        # Demo 1: Credit Risk
        credit_model = demo_credit_risk_model()

        # Demo 2: Fraud Detection (may skip if dependencies missing)
        fraud_detector = demo_fraud_detection_ensemble()

        # Demo: Model Serialization
        demo_model_serialization()

        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)
        print("\nNext Steps:")
        print("1. Explore individual demo directories for detailed implementations")
        print("2. Run Jupyter notebooks for interactive exploration")
        print("3. Deploy models to IntegratedML using provided SQL scripts")
        print("4. Check out the documentation in docs/ for advanced usage")

    except Exception as e:
        print(f"\nError running demos: {e}")
        print("\nThis is normal if some dependencies are missing.")
        print("Install requirements: pip install -r requirements.txt")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
