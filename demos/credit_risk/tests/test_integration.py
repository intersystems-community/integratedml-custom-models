"""
Integration tests for the Credit Risk Assessment demo.

This module contains comprehensive integration tests that verify the complete
end-to-end workflow from data generation through model training to prediction
and evaluation.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
import json
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parents[3]
sys.path.append(str(project_root))

from demos.credit_risk.models.credit_risk_classifier import CustomCreditRiskClassifier
from demos.credit_risk.data.generate_sample_data import CreditDataGenerator
from demos.credit_risk.scripts.data_preprocessing import CreditDataPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


class TestEndToEndWorkflow(unittest.TestCase):
    """Integration tests for the complete credit risk assessment workflow."""

    def setUp(self):
        """Set up test fixtures for integration testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = CreditDataGenerator(random_seed=42)
        self.preprocessor = CreditDataPreprocessor(
            handle_missing=True, remove_outliers=True, normalize_features=True
        )

    def tearDown(self):
        """Clean up temporary files after each test."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_complete_workflow_baseline(self):
        """Test complete workflow with baseline model configuration."""
        # Step 1: Generate data
        X_raw, y_raw = self.generator.generate_dataset(n_samples=200, default_rate=0.3)
        self.assertEqual(len(X_raw), 200)

        # Step 2: Preprocess data
        X_processed, y_processed = self.preprocessor.fit_transform(X_raw, y_raw)
        self.assertEqual(len(X_processed), len(y_processed))
        self.assertLessEqual(len(X_processed), len(X_raw))  # May remove outliers

        # Step 3: Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.3, random_state=42
        )

        # Step 4: Train baseline model
        model = CustomCreditRiskClassifier(
            enable_debt_ratio=False,
            enable_interaction_terms=False,
            enable_risk_scoring=False,
        )
        model.fit(X_train, y_train)
        self.assertTrue(model.is_fitted)

        # Step 5: Make predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        # Step 6: Evaluate performance
        accuracy = accuracy_score(y_test, predictions)
        auc_score = roc_auc_score(y_test, probabilities[:, 1])

        # Assertions
        self.assertGreater(accuracy, 0.50)  # Should achieve reasonable accuracy
        self.assertGreater(auc_score, 0.50)  # Should achieve reasonable AUC
        self.assertEqual(len(predictions), len(X_test))
        self.assertEqual(probabilities.shape, (len(X_test), 2))

    def test_complete_workflow_full_features(self):
        """Test complete workflow with full feature engineering."""
        # Step 1: Generate larger dataset for better feature engineering performance
        X_raw, y_raw = self.generator.generate_dataset(n_samples=300, default_rate=0.25)

        # Step 2: Preprocess data
        X_processed, y_processed = self.preprocessor.fit_transform(X_raw, y_raw)

        # Step 3: Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.3, random_state=42
        )

        # Step 4: Train full model with all feature engineering
        model = CustomCreditRiskClassifier(
            enable_debt_ratio=True,
            enable_interaction_terms=True,
            enable_risk_scoring=True,
            decision_threshold=0.6,
        )
        model.fit(X_train, y_train)

        # Step 5: Make predictions and get explanations
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        explanations = model.get_risk_explanation(X_test.head(5))

        # Step 6: Evaluate performance
        accuracy = accuracy_score(y_test, predictions)
        auc_score = roc_auc_score(y_test, probabilities[:, 1])

        # Assertions
        self.assertGreater(accuracy, 0.50)
        self.assertGreater(auc_score, 0.50)
        self.assertIn("risk_probabilities", explanations)
        self.assertIn("risk_factors", explanations)
        self.assertEqual(len(explanations["risk_probabilities"]), 5)

    def test_workflow_with_dirty_data(self):
        """Test workflow robustness with problematic data."""
        # Step 1: Generate data and introduce problems
        X_raw, y_raw = self.generator.generate_dataset(n_samples=150, default_rate=0.4)

        # Introduce various data quality issues
        X_dirty = X_raw.copy()
        X_dirty.loc[0:10, "age"] = np.nan  # Missing values
        X_dirty.loc[15, "credit_amount"] = -1000  # Negative amount
        X_dirty.loc[20, "age"] = 200  # Unrealistic age

        # Step 2: Preprocess should handle these issues
        X_processed, y_processed = self.preprocessor.fit_transform(X_dirty, y_raw)

        # Should have no missing values and reasonable ranges
        self.assertEqual(X_processed.isnull().sum().sum(), 0)

        # Step 3: Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.3, random_state=42
        )

        model = CustomCreditRiskClassifier()
        model.fit(X_train, y_train)

        # Step 4: Should still make reasonable predictions
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        self.assertGreater(
            accuracy, 0.45
        )  # Should still work reasonably well with dirty data

    def test_model_serialization_workflow(self):
        """Test complete workflow including model saving and loading."""
        # Step 1: Train model
        X_raw, y_raw = self.generator.generate_dataset(n_samples=100, default_rate=0.3)
        X_processed, y_processed = self.preprocessor.fit_transform(X_raw, y_raw)

        model = CustomCreditRiskClassifier()
        model.fit(X_processed, y_processed)

        # Step 2: Save model
        model_path = os.path.join(self.temp_dir, "test_model.pkl")
        model.save_model(model_path)
        self.assertTrue(os.path.exists(model_path))

        # Step 3: Save preprocessor
        preprocessor_path = os.path.join(self.temp_dir, "test_preprocessor.pkl")
        self.preprocessor.save_preprocessor(preprocessor_path)
        self.assertTrue(os.path.exists(preprocessor_path))

        # Step 4: Load model and preprocessor
        loaded_model = CustomCreditRiskClassifier.load_model(model_path)
        loaded_preprocessor = CreditDataPreprocessor.load_preprocessor(
            preprocessor_path
        )

        # Step 5: Generate new test data and predict
        X_new, y_new = self.generator.generate_dataset(n_samples=50, default_rate=0.3)
        X_new_processed = loaded_preprocessor.transform(X_new)

        predictions_original = model.predict(X_new_processed)
        predictions_loaded = loaded_model.predict(X_new_processed)

        # Should get identical predictions
        np.testing.assert_array_equal(predictions_original, predictions_loaded)

    def test_json_serialization_workflow(self):
        """Test JSON serialization for IntegratedML compatibility."""
        # Step 1: Create and configure model
        model = CustomCreditRiskClassifier(
            enable_debt_ratio=True,
            enable_interaction_terms=False,
            decision_threshold=0.7,
        )

        # Step 2: Serialize to JSON
        json_params = model.to_json()
        self.assertIsInstance(json_params, str)

        # Step 3: Verify JSON structure
        params_dict = json.loads(json_params)
        self.assertIn("enable_debt_ratio", params_dict)
        self.assertEqual(params_dict["enable_debt_ratio"], True)
        self.assertEqual(params_dict["decision_threshold"], 0.7)

        # Step 4: Recreate model from JSON
        new_model = CustomCreditRiskClassifier.from_json(json_params)
        self.assertEqual(new_model.enable_debt_ratio, model.enable_debt_ratio)
        self.assertEqual(new_model.decision_threshold, model.decision_threshold)

    def test_performance_comparison_workflow(self):
        """Test performance comparison between different model configurations."""
        # Generate test data
        X_raw, y_raw = self.generator.generate_dataset(n_samples=250, default_rate=0.3)
        X_processed, y_processed = self.preprocessor.fit_transform(X_raw, y_raw)
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.3, random_state=42
        )

        # Test different model configurations
        configurations = [
            {
                "name": "baseline",
                "params": {
                    "enable_debt_ratio": False,
                    "enable_interaction_terms": False,
                    "enable_risk_scoring": False,
                },
            },
            {
                "name": "debt_ratios_only",
                "params": {
                    "enable_debt_ratio": True,
                    "enable_interaction_terms": False,
                    "enable_risk_scoring": False,
                },
            },
            {
                "name": "full_features",
                "params": {
                    "enable_debt_ratio": True,
                    "enable_interaction_terms": True,
                    "enable_risk_scoring": True,
                },
            },
        ]

        results = {}

        for config in configurations:
            model = CustomCreditRiskClassifier(**config["params"])
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, predictions)
            auc_score = roc_auc_score(y_test, probabilities)

            results[config["name"]] = {"accuracy": accuracy, "auc": auc_score}

        # All configurations should achieve reasonable performance
        for config_name, metrics in results.items():
            self.assertGreater(
                metrics["accuracy"],
                0.45,
                f"{config_name} accuracy too low: {metrics['accuracy']}",
            )
            self.assertGreater(
                metrics["auc"], 0.45, f"{config_name} AUC too low: {metrics['auc']}"
            )

        # Feature engineering should generally improve performance
        # (Allow some tolerance for random variation)
        baseline_auc = results["baseline"]["auc"]
        full_auc = results["full_features"]["auc"]

        # Full features should be at least competitive with baseline
        self.assertGreaterEqual(full_auc, baseline_auc - 0.1)

    def test_batch_prediction_workflow(self):
        """Test batch prediction capabilities."""
        # Generate and prepare data
        X_raw, y_raw = self.generator.generate_dataset(n_samples=200, default_rate=0.3)
        X_processed, y_processed = self.preprocessor.fit_transform(X_raw, y_raw)

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.4, random_state=42
        )

        model = CustomCreditRiskClassifier()
        model.fit(X_train, y_train)

        # Test different batch sizes
        batch_sizes = [1, 5, 10, len(X_test)]

        for batch_size in batch_sizes:
            if batch_size <= len(X_test):
                batch_data = X_test.head(batch_size)

                # Predictions should work for any batch size
                predictions = model.predict(batch_data)
                probabilities = model.predict_proba(batch_data)

                self.assertEqual(len(predictions), batch_size)
                self.assertEqual(probabilities.shape[0], batch_size)

                # All predictions should be valid
                self.assertTrue(all(pred in [0, 1] for pred in predictions))
                self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))

    def test_feature_importance_workflow(self):
        """Test feature importance extraction and interpretation."""
        # Generate data with clear signal
        X_raw, y_raw = self.generator.generate_dataset(n_samples=300, default_rate=0.3)
        X_processed, y_processed = self.preprocessor.fit_transform(X_raw, y_raw)

        # Train model with feature engineering
        model = CustomCreditRiskClassifier(
            enable_debt_ratio=True,
            enable_interaction_terms=True,
            enable_risk_scoring=True,
        )
        model.fit(X_processed, y_processed)

        # Get feature importance
        importance = model.get_feature_importance()

        if importance is not None:
            # Should have importance for each feature that was actually used in training
            X_engineered = model._engineer_features(X_processed)
            X_final = model._preprocessor.transform(X_engineered)
            self.assertEqual(len(importance), X_final.shape[1])

            # Should be normalized (sum to 1)
            self.assertAlmostEqual(importance.sum(), 1.0, places=5)

            # All values should be non-negative
            self.assertTrue((importance >= 0).all())

            # Should identify some features as more important than others
            self.assertGreater(importance.max(), importance.min())


class TestDataPipelineIntegration(unittest.TestCase):
    """Integration tests for data generation and preprocessing pipeline."""

    def test_data_generation_preprocessing_compatibility(self):
        """Test that generated data works well with preprocessing pipeline."""
        generator = CreditDataGenerator(random_seed=42)
        preprocessor = CreditDataPreprocessor()

        # Test with different dataset sizes
        for n_samples in [50, 100, 200]:
            X, y = generator.generate_dataset(n_samples=n_samples, default_rate=0.3)

            # Should preprocess without errors
            X_processed, y_processed = preprocessor.fit_transform(X, y)

            # Basic integrity checks
            self.assertGreater(len(X_processed), 0)
            self.assertEqual(len(X_processed), len(y_processed))
            self.assertLessEqual(len(X_processed), n_samples)

    def test_data_quality_across_pipeline(self):
        """Test that data quality is maintained across the pipeline."""
        generator = CreditDataGenerator(random_seed=42)
        preprocessor = CreditDataPreprocessor(
            handle_missing=True, remove_outliers=True, normalize_features=True
        )

        X_raw, y_raw = generator.generate_dataset(n_samples=200, default_rate=0.3)

        # Check raw data quality
        raw_quality = self._calculate_basic_quality_score(X_raw)

        # Process data
        X_processed, y_processed = preprocessor.fit_transform(X_raw, y_raw)

        # Check processed data quality
        processed_quality = self._calculate_basic_quality_score(X_processed)

        # Processed data should have higher quality (no missing values, etc.)
        self.assertGreaterEqual(processed_quality, raw_quality)

        # Should have no missing values after processing
        self.assertEqual(X_processed.isnull().sum().sum(), 0)

    def _calculate_basic_quality_score(self, data):
        """Calculate a basic data quality score."""
        if len(data) == 0:
            return 0.0

        # Factors that affect quality
        missing_penalty = data.isnull().sum().sum() / (len(data) * len(data.columns))

        # Basic quality score (1 = perfect, 0 = terrible)
        quality_score = 1.0 - missing_penalty

        return max(0.0, quality_score)


class TestErrorHandlingIntegration(unittest.TestCase):
    """Integration tests for error handling across the pipeline."""

    def test_graceful_failure_handling(self):
        """Test that the pipeline handles failures gracefully."""
        generator = CreditDataGenerator(random_seed=42)

        # Test with minimal data
        try:
            X_minimal, y_minimal = generator.generate_dataset(
                n_samples=5, default_rate=0.4
            )

            preprocessor = CreditDataPreprocessor()
            X_processed, y_processed = preprocessor.fit_transform(X_minimal, y_minimal)

            # Even with minimal data, should not crash
            self.assertGreater(len(X_processed), 0)

        except Exception as e:
            # If it fails, it should fail with a clear error message
            self.assertIsInstance(e, (ValueError, RuntimeError))

    def test_input_validation_integration(self):
        """Test input validation across integrated components."""
        generator = CreditDataGenerator(random_seed=42)
        X, y = generator.generate_dataset(n_samples=100, default_rate=0.3)

        preprocessor = CreditDataPreprocessor()
        X_processed, y_processed = preprocessor.fit_transform(X, y)

        model = CustomCreditRiskClassifier()
        model.fit(X_processed, y_processed)

        # Test invalid inputs
        with self.assertRaises(Exception):
            # Empty data
            model.predict(pd.DataFrame())

        with self.assertRaises(Exception):
            # Wrong number of features
            wrong_features = X_processed.iloc[:, :3]  # Take only first 3 features
            model.predict(wrong_features)


if __name__ == "__main__":
    # Run integration tests with verbose output
    unittest.main(verbosity=2)
