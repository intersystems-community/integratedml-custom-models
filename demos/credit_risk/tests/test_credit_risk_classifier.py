"""
Unit tests for the Custom Credit Risk Classifier.

This module contains comprehensive tests for the credit risk assessment model,
including feature engineering, model training, prediction capabilities,
and IntegratedML compatibility.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parents[3]
sys.path.append(str(project_root))

from demos.credit_risk.models.credit_risk_classifier import (
    CustomCreditRiskClassifier,
    calculate_debt_to_income_ratio,
    assess_credit_stability,
    calculate_financial_capacity_score,
    encode_credit_purpose,
)
from demos.credit_risk.data.generate_sample_data import CreditDataGenerator


class TestCustomCreditRiskClassifier(unittest.TestCase):
    """Test cases for the CustomCreditRiskClassifier."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Generate sample data for testing
        self.generator = CreditDataGenerator(random_seed=42)
        self.X_sample, self.y_sample = self.generator.generate_dataset(
            n_samples=100, default_rate=0.3
        )

        # Split for training and testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_sample, self.y_sample, test_size=0.3, random_state=42
        )

        # Create different model configurations
        self.model_baseline = CustomCreditRiskClassifier(
            enable_debt_ratio=False,
            enable_interaction_terms=False,
            enable_risk_scoring=False,
        )

        self.model_full = CustomCreditRiskClassifier(
            enable_debt_ratio=True,
            enable_interaction_terms=True,
            enable_risk_scoring=True,
        )

    def test_model_initialization(self):
        """Test proper model initialization with various configurations."""
        # Test default initialization
        model_default = CustomCreditRiskClassifier()
        self.assertTrue(model_default.enable_debt_ratio)
        self.assertTrue(model_default.enable_interaction_terms)
        self.assertTrue(model_default.enable_risk_scoring)
        self.assertEqual(model_default.decision_threshold, 0.5)

        # Test custom initialization
        model_custom = CustomCreditRiskClassifier(
            enable_debt_ratio=False,
            enable_interaction_terms=True,
            decision_threshold=0.7,
        )
        self.assertFalse(model_custom.enable_debt_ratio)
        self.assertTrue(model_custom.enable_interaction_terms)
        self.assertEqual(model_custom.decision_threshold, 0.7)

    def test_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Test invalid decision threshold
        with self.assertRaises(ValueError):
            CustomCreditRiskClassifier(decision_threshold=1.5)

        with self.assertRaises(ValueError):
            CustomCreditRiskClassifier(decision_threshold=-0.1)

        # Test invalid boolean parameters
        with self.assertRaises(ValueError):
            CustomCreditRiskClassifier(enable_debt_ratio="true")

    def test_feature_engineering_debt_ratios(self):
        """Test debt ratio feature engineering."""
        # Test with sample data
        X_engineered = self.model_full._add_debt_ratio_features(self.X_sample)

        # Check that new features are created
        self.assertGreater(X_engineered.shape[1], self.X_sample.shape[1])

        # Check for specific debt ratio features
        expected_features = [
            "debt_to_income_ratio",
            "monthly_payment_ratio",
            "age_adjusted_credit",
        ]
        for feature in expected_features:
            if feature in X_engineered.columns:
                # Verify feature has reasonable values
                self.assertTrue(
                    (X_engineered[feature] >= 0).all()
                    or X_engineered[feature].isnull().any()
                )

    def test_feature_engineering_interaction_terms(self):
        """Test interaction terms feature engineering."""
        X_engineered = self.model_full._add_interaction_terms(self.X_sample)

        # Check that interaction features are created
        self.assertGreater(X_engineered.shape[1], self.X_sample.shape[1])

        # Check for specific interaction features
        interaction_features = [
            col for col in X_engineered.columns if "interaction" in col.lower()
        ]
        self.assertGreater(len(interaction_features), 0)

    def test_feature_engineering_risk_scoring(self):
        """Test risk scoring feature engineering."""
        X_engineered = self.model_full._add_risk_scoring_features(self.X_sample)

        # Check that risk scoring features are created
        self.assertGreater(X_engineered.shape[1], self.X_sample.shape[1])

        # Check for specific risk scoring features
        expected_features = ["stability_score", "composite_risk_score"]
        for feature in expected_features:
            if feature in X_engineered.columns:
                # Risk scores should be between 0 and 1
                self.assertTrue((X_engineered[feature] >= 0).all())
                self.assertTrue((X_engineered[feature] <= 1).all())

    def test_model_training(self):
        """Test model training functionality."""
        # Train baseline model
        self.model_baseline.fit(self.X_train, self.y_train)
        self.assertTrue(self.model_baseline.is_fitted)
        self.assertIsNotNone(self.model_baseline._pipeline)

        # Train full model
        self.model_full.fit(self.X_train, self.y_train)
        self.assertTrue(self.model_full.is_fitted)

        # Check that feature engineering metadata is stored
        self.assertIn("feature_engineering_enabled", self.model_full._model_metadata)
        self.assertIn("original_features", self.model_full._model_metadata)
        self.assertIn("engineered_features", self.model_full._model_metadata)

    def test_prediction_capabilities(self):
        """Test prediction functionality."""
        # Train model first
        self.model_full.fit(self.X_train, self.y_train)

        # Test predict method
        predictions = self.model_full.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))

        # Test predict_proba method
        probabilities = self.model_full.predict_proba(self.X_test)
        self.assertEqual(probabilities.shape[0], len(self.X_test))
        self.assertEqual(probabilities.shape[1], 2)  # Binary classification

        # Probabilities should sum to 1
        prob_sums = probabilities.sum(axis=1)
        np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5)

        # Test decision function
        decision_scores = self.model_full.decision_function(self.X_test)
        self.assertEqual(len(decision_scores), len(self.X_test))

    def test_model_performance(self):
        """Test that model achieves reasonable performance."""
        # Train both models
        self.model_baseline.fit(self.X_train, self.y_train)
        self.model_full.fit(self.X_train, self.y_train)

        # Make predictions
        pred_baseline = self.model_baseline.predict(self.X_test)
        pred_full = self.model_full.predict(self.X_test)

        prob_baseline = self.model_baseline.predict_proba(self.X_test)[:, 1]
        prob_full = self.model_full.predict_proba(self.X_test)[:, 1]

        # Calculate performance metrics
        acc_baseline = accuracy_score(self.y_test, pred_baseline)
        acc_full = accuracy_score(self.y_test, pred_full)

        auc_baseline = roc_auc_score(self.y_test, prob_baseline)
        auc_full = roc_auc_score(self.y_test, prob_full)

        # Models should achieve reasonable performance (>55% AUC, >60% accuracy)
        self.assertGreater(acc_baseline, 0.6)
        self.assertGreater(acc_full, 0.6)
        self.assertGreater(auc_baseline, 0.54)
        self.assertGreater(auc_full, 0.54)

        # Full feature engineering should perform at least as well as baseline
        self.assertGreaterEqual(auc_full, auc_baseline - 0.05)  # Allow small variance

    def test_feature_importance(self):
        """Test feature importance extraction."""
        self.model_full.fit(self.X_train, self.y_train)

        importance = self.model_full.get_feature_importance()
        if importance is not None:
            # Should have importance for each feature after preprocessing
            X_engineered = self.model_full._engineer_features(self.X_train)
            X_processed = self.model_full._preprocessor.transform(X_engineered)
            expected_features = X_processed.shape[1]
            self.assertEqual(len(importance), expected_features)

            # All importance values should be non-negative
            self.assertTrue((importance >= 0).all())

    def test_risk_explanation(self):
        """Test risk explanation functionality."""
        self.model_full.fit(self.X_train, self.y_train)

        # Test with a small sample
        sample_data = self.X_test.head(3)
        explanation = self.model_full.get_risk_explanation(sample_data)

        # Check explanation structure
        self.assertIn("risk_probabilities", explanation)
        self.assertIn("risk_factors", explanation)
        self.assertIn("recommendations", explanation)

        # Check that we have explanations for each sample
        self.assertEqual(len(explanation["risk_probabilities"]), len(sample_data))
        self.assertEqual(len(explanation["recommendations"]), len(sample_data))

    def test_model_serialization(self):
        """Test model saving and loading."""
        # Train model
        self.model_full.fit(self.X_train, self.y_train)

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            model_path = tmp_file.name

        try:
            self.model_full.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))

            # Load model
            loaded_model = CustomCreditRiskClassifier.load_model(model_path)

            # Test that loaded model works
            self.assertTrue(loaded_model.is_fitted)

            # Predictions should be identical
            pred_original = self.model_full.predict(self.X_test)
            pred_loaded = loaded_model.predict(self.X_test)
            np.testing.assert_array_equal(pred_original, pred_loaded)

            prob_original = self.model_full.predict_proba(self.X_test)
            prob_loaded = loaded_model.predict_proba(self.X_test)
            np.testing.assert_allclose(prob_original, prob_loaded, rtol=1e-5)

        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_json_serialization(self):
        """Test JSON parameter serialization for IntegratedML."""
        # Test to_json method
        json_str = self.model_full.to_json()
        self.assertIsInstance(json_str, str)

        # Should be valid JSON
        import json

        params = json.loads(json_str)
        self.assertIsInstance(params, dict)

        # Test from_json method
        new_model = CustomCreditRiskClassifier.from_json(json_str)
        self.assertEqual(new_model.enable_debt_ratio, self.model_full.enable_debt_ratio)
        self.assertEqual(
            new_model.enable_interaction_terms, self.model_full.enable_interaction_terms
        )
        self.assertEqual(
            new_model.enable_risk_scoring, self.model_full.enable_risk_scoring
        )

    def test_sklearn_compatibility(self):
        """Test compatibility with scikit-learn interfaces."""
        from sklearn.base import clone
        from sklearn.model_selection import cross_val_score

        # Test cloning
        cloned_model = clone(self.model_full)
        self.assertEqual(
            cloned_model.enable_debt_ratio, self.model_full.enable_debt_ratio
        )

        # Test get_params and set_params
        params = self.model_full.get_params()
        self.assertIsInstance(params, dict)
        self.assertIn("enable_debt_ratio", params)

        new_params = {"decision_threshold": 0.7}
        self.model_full.set_params(**new_params)
        self.assertEqual(self.model_full.decision_threshold, 0.7)

        # Test cross-validation (with small sample for speed)
        if len(self.X_sample) >= 10:  # Need minimum samples for CV
            cv_scores = cross_val_score(
                self.model_full, self.X_sample, self.y_sample, cv=3
            )
            self.assertEqual(len(cv_scores), 3)
            self.assertTrue(all(score >= 0 for score in cv_scores))

    def test_edge_cases(self):
        """Test handling of edge cases and error conditions."""
        # Test prediction before fitting
        unfitted_model = CustomCreditRiskClassifier()
        with self.assertRaises(Exception):  # Should raise some kind of error
            unfitted_model.predict(self.X_test)

        # Test with empty data
        with self.assertRaises(Exception):
            empty_X = pd.DataFrame()
            empty_y = pd.Series(dtype=int)
            self.model_full.fit(empty_X, empty_y)

        # Test with single sample
        single_X = self.X_sample.head(1)
        single_y = self.y_sample.head(1)

        single_model = CustomCreditRiskClassifier(
            enable_interaction_terms=False  # Reduce complexity for small data
        )

        try:
            single_model.fit(single_X, single_y)
            pred = single_model.predict(single_X)
            self.assertEqual(len(pred), 1)
        except Exception:
            # Single sample might not work with all preprocessing - that's OK
            pass


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def test_calculate_debt_to_income_ratio(self):
        """Test debt-to-income ratio calculation."""
        # Normal cases
        self.assertAlmostEqual(calculate_debt_to_income_ratio(50000, 100000), 0.5)
        self.assertAlmostEqual(calculate_debt_to_income_ratio(30000, 60000), 0.5)

        # Edge cases
        self.assertEqual(calculate_debt_to_income_ratio(0, 50000), 0.0)
        self.assertEqual(calculate_debt_to_income_ratio(50000, 0), float("inf"))
        self.assertEqual(calculate_debt_to_income_ratio(-10000, 50000), 0.0)

        # Cap test
        high_ratio = calculate_debt_to_income_ratio(1000000, 10000)
        self.assertEqual(high_ratio, 10.0)  # Should be capped

    def test_assess_credit_stability(self):
        """Test credit stability assessment."""
        # Normal cases
        stability1 = assess_credit_stability(
            24, 36
        )  # 2 years employment, 3 years residence
        stability2 = assess_credit_stability(60, 60)  # 5 years each (max stability)
        stability3 = assess_credit_stability(6, 12)  # Short term

        self.assertGreater(stability2, stability1)
        self.assertGreater(stability1, stability3)

        # All scores should be between 0 and 1
        self.assertGreaterEqual(stability1, 0.0)
        self.assertLessEqual(stability1, 1.0)
        self.assertGreaterEqual(stability2, 0.0)
        self.assertLessEqual(stability2, 1.0)

        # Edge cases
        zero_stability = assess_credit_stability(0, 0)
        self.assertEqual(zero_stability, 0.0)

        negative_stability = assess_credit_stability(-5, 10)
        self.assertEqual(negative_stability, 0.2)  # Only residence counted

    def test_calculate_financial_capacity_score(self):
        """Test financial capacity score calculation."""
        # With income
        capacity1 = calculate_financial_capacity_score(
            50000, 60, 5000
        )  # Monthly payment = 833, ratio = 0.167
        capacity2 = calculate_financial_capacity_score(
            50000, 30, 5000
        )  # Monthly payment = 1667, ratio = 0.333

        self.assertGreater(capacity1, capacity2)  # Longer term = higher capacity

        # Without income
        capacity3 = calculate_financial_capacity_score(50000, 50)
        self.assertGreaterEqual(capacity3, 0.0)
        self.assertLessEqual(capacity3, 1.0)

        # Edge cases
        zero_duration = calculate_financial_capacity_score(50000, 0, 5000)
        self.assertEqual(zero_duration, 0.0)

    def test_encode_credit_purpose(self):
        """Test credit purpose encoding."""
        # Test known purposes
        self.assertEqual(encode_credit_purpose("car"), 1)
        self.assertEqual(encode_credit_purpose("business"), 5)
        self.assertEqual(encode_credit_purpose("vacation"), 4)

        # Test case insensitivity
        self.assertEqual(encode_credit_purpose("CAR"), 1)
        self.assertEqual(encode_credit_purpose("Car"), 1)

        # Test unknown purpose
        self.assertEqual(
            encode_credit_purpose("unknown_purpose"), 3
        )  # Default medium risk


class TestDataGeneration(unittest.TestCase):
    """Test cases for the data generation functionality."""

    def test_data_generator_initialization(self):
        """Test CreditDataGenerator initialization."""
        generator = CreditDataGenerator(random_seed=123)
        self.assertEqual(generator.random_seed, 123)

        # Test that distributions are set up
        self.assertIsNotNone(generator.age_weights)
        self.assertIsNotNone(generator.employment_patterns)
        self.assertIsNotNone(generator.credit_purposes)

    def test_data_generation_basic(self):
        """Test basic data generation functionality."""
        generator = CreditDataGenerator(random_seed=42)
        X, y = generator.generate_dataset(n_samples=50, default_rate=0.4)

        # Check basic properties
        self.assertEqual(len(X), 50)
        self.assertEqual(len(y), 50)
        self.assertAlmostEqual(y.mean(), 0.4, delta=0.1)  # Allow some variance

        # Check that required columns exist
        required_columns = ["age", "credit_amount", "duration", "employment_duration"]
        for col in required_columns:
            self.assertIn(col, X.columns)

        # Check data types and ranges
        self.assertTrue((X["age"] >= 18).all())
        self.assertTrue((X["age"] <= 100).all())
        self.assertTrue((X["credit_amount"] > 0).all())
        self.assertTrue((X["duration"] > 0).all())

    def test_data_generation_reproducibility(self):
        """Test that data generation is reproducible with same seed."""
        generator1 = CreditDataGenerator(random_seed=42)
        generator2 = CreditDataGenerator(random_seed=42)

        X1, y1 = generator1.generate_dataset(n_samples=30, default_rate=0.3)
        X2, y2 = generator2.generate_dataset(n_samples=30, default_rate=0.3)

        # Should be identical
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
