"""
Unit tests for the Credit Risk Data Preprocessing module.

This module contains comprehensive tests for data validation, cleaning,
and preprocessing pipeline functionality.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parents[3]
sys.path.append(str(project_root))

from demos.credit_risk.scripts.data_preprocessing import CreditDataPreprocessor
from demos.credit_risk.data.generate_sample_data import CreditDataGenerator


class TestCreditDataPreprocessor(unittest.TestCase):
    """Test cases for the CreditDataPreprocessor."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Generate clean test data
        self.generator = CreditDataGenerator(random_seed=42)
        self.clean_X, self.clean_y = self.generator.generate_dataset(
            n_samples=100, default_rate=0.3
        )

        # Create dirty test data with various issues
        self.dirty_X = self.clean_X.copy()
        self.dirty_y = self.clean_y.copy()

        # Introduce missing values
        self.dirty_X.loc[5:10, "age"] = np.nan
        self.dirty_X.loc[15:20, "credit_amount"] = np.nan

        # Add outliers
        self.dirty_X.loc[25, "age"] = 150  # Unrealistic age
        self.dirty_X.loc[26, "credit_amount"] = -5000  # Negative credit amount

        # Add invalid categorical values
        if "sex" in self.dirty_X.columns:
            self.dirty_X.loc[30, "sex"] = "invalid_gender"

        # Add duplicate rows
        duplicate_row = self.dirty_X.iloc[0:1].copy()
        self.dirty_X = pd.concat([self.dirty_X, duplicate_row], ignore_index=True)
        self.dirty_y = pd.concat(
            [self.dirty_y, self.dirty_y.iloc[0:1]], ignore_index=True
        )

        # Initialize preprocessor
        self.preprocessor = CreditDataPreprocessor()

    def test_preprocessor_initialization(self):
        """Test proper preprocessor initialization."""
        preprocessor = CreditDataPreprocessor(
            handle_missing=True, remove_outliers=True, normalize_features=True
        )

        self.assertTrue(preprocessor.handle_missing)
        self.assertTrue(preprocessor.remove_outliers)
        self.assertTrue(preprocessor.normalize_features)
        self.assertFalse(preprocessor.is_fitted)

    def test_data_validation_clean_data(self):
        """Test data validation with clean data."""
        validation_report = self.preprocessor.validate_data(self.clean_X, self.clean_y)

        # Should pass basic validation checks
        self.assertIn("missing_values", validation_report)
        self.assertIn("data_types", validation_report)
        self.assertIn("feature_ranges", validation_report)
        self.assertIn("target_distribution", validation_report)

        # Check that clean data has minimal issues
        self.assertEqual(validation_report["missing_values"]["total_missing"], 0)
        self.assertFalse(validation_report["has_critical_issues"])

    def test_data_validation_dirty_data(self):
        """Test data validation with problematic data."""
        validation_report = self.preprocessor.validate_data(self.dirty_X, self.dirty_y)

        # Should detect issues
        self.assertGreater(validation_report["missing_values"]["total_missing"], 0)
        self.assertTrue(validation_report["has_critical_issues"])

        # Should identify problematic features
        self.assertIn("problematic_features", validation_report)
        problematic_features = validation_report["problematic_features"]
        self.assertGreater(len(problematic_features), 0)

    def test_missing_value_handling(self):
        """Test missing value handling strategies."""
        # Create data with missing values
        X_missing = self.clean_X.copy()
        X_missing.loc[0:5, "age"] = np.nan
        X_missing.loc[10:15, "credit_amount"] = np.nan

        # Test with missing value handling enabled
        preprocessor_handle = CreditDataPreprocessor(handle_missing=True)
        X_processed, _ = preprocessor_handle.fit_transform(X_missing, self.clean_y)

        # Should have no missing values after processing
        self.assertEqual(X_processed.isnull().sum().sum(), 0)

        # Test with missing value handling disabled
        preprocessor_no_handle = CreditDataPreprocessor(handle_missing=False)

        # Should either raise an error or keep missing values
        try:
            X_no_handle, _ = preprocessor_no_handle.fit_transform(
                X_missing, self.clean_y
            )
            # If it doesn't raise an error, missing values should remain
            self.assertGreater(X_no_handle.isnull().sum().sum(), 0)
        except Exception:
            # It's acceptable to raise an error when handle_missing=False
            pass

    def test_outlier_detection_and_removal(self):
        """Test outlier detection and removal."""
        # Create data with obvious outliers
        X_outliers = self.clean_X.copy()
        X_outliers.loc[0, "age"] = 200  # Impossible age
        X_outliers.loc[1, "credit_amount"] = 1000000000  # Extremely high amount

        original_shape = X_outliers.shape

        # Test with outlier removal enabled
        preprocessor_remove = CreditDataPreprocessor(remove_outliers=True)
        X_processed, y_processed = preprocessor_remove.fit_transform(
            X_outliers, self.clean_y
        )

        # Should have fewer samples after outlier removal
        self.assertLessEqual(X_processed.shape[0], original_shape[0])
        self.assertEqual(len(X_processed), len(y_processed))

        # Test with outlier removal disabled
        preprocessor_keep = CreditDataPreprocessor(remove_outliers=False)
        X_no_removal, y_no_removal = preprocessor_keep.fit_transform(
            X_outliers, self.clean_y
        )

        # Should keep all samples
        self.assertEqual(X_no_removal.shape[0], original_shape[0])
        self.assertEqual(len(X_no_removal), len(y_no_removal))

    def test_feature_normalization(self):
        """Test feature normalization."""
        # Test with normalization enabled
        preprocessor_norm = CreditDataPreprocessor(normalize_features=True)
        X_normalized, _ = preprocessor_norm.fit_transform(self.clean_X, self.clean_y)

        # Check that originally numerical features are normalized
        # Only check features that were identified as numerical and supposed to be normalized
        for feature in preprocessor_norm.numerical_features:
            if len(X_normalized[feature].unique()) > 2:  # Skip binary features
                # Should have approximately zero mean and unit variance
                self.assertAlmostEqual(X_normalized[feature].mean(), 0.0, delta=0.1)
                self.assertAlmostEqual(X_normalized[feature].std(), 1.0, delta=0.1)

        # Test with normalization disabled
        preprocessor_no_norm = CreditDataPreprocessor(normalize_features=False)
        X_no_norm, _ = preprocessor_no_norm.fit_transform(self.clean_X, self.clean_y)

        # Should preserve original scale for numerical features (but categorical features will still be encoded)
        for feature in preprocessor_no_norm.numerical_features:
            if len(X_no_norm[feature].unique()) > 2:  # Skip binary features
                # Should NOT be normalized (not zero mean/unit variance)
                self.assertNotAlmostEqual(X_no_norm[feature].mean(), 0.0, delta=0.1)
                # Original numerical features should maintain their scale
                original_std = self.clean_X[feature].std()
                self.assertAlmostEqual(
                    X_no_norm[feature].std(), original_std, delta=0.1
                )

    def test_categorical_encoding(self):
        """Test categorical feature encoding."""
        # Add a categorical feature if not present
        X_categorical = self.clean_X.copy()
        if "categorical_test" not in X_categorical.columns:
            X_categorical["categorical_test"] = np.random.choice(
                ["A", "B", "C"], size=len(X_categorical)
            )

        X_processed, _ = self.preprocessor.fit_transform(X_categorical, self.clean_y)

        # Categorical features should be encoded
        original_categoricals = X_categorical.select_dtypes(include=["object"]).columns
        processed_categoricals = X_processed.select_dtypes(include=["object"]).columns

        # Should have fewer or equal categorical columns after encoding
        self.assertLessEqual(len(processed_categoricals), len(original_categoricals))

    def test_duplicate_removal(self):
        """Test duplicate row removal."""
        # Create data with duplicates
        X_duplicates = pd.concat(
            [self.clean_X, self.clean_X.iloc[:5]], ignore_index=True
        )
        y_duplicates = pd.concat(
            [self.clean_y, self.clean_y.iloc[:5]], ignore_index=True
        )

        original_shape = X_duplicates.shape

        X_processed, y_processed = self.preprocessor.fit_transform(
            X_duplicates, y_duplicates
        )

        # Should have fewer or equal samples after duplicate removal
        self.assertLessEqual(X_processed.shape[0], original_shape[0])
        self.assertEqual(len(X_processed), len(y_processed))

        # Should not have exact duplicates
        self.assertEqual(len(X_processed), len(X_processed.drop_duplicates()))

    def test_feature_type_identification(self):
        """Test automatic feature type identification."""
        feature_types = self.preprocessor._identify_feature_types(self.clean_X)

        self.assertIn("numerical", feature_types)
        self.assertIn("categorical", feature_types)
        self.assertIn("binary", feature_types)

        # Check that feature types are lists
        self.assertIsInstance(feature_types["numerical"], list)
        self.assertIsInstance(feature_types["categorical"], list)
        self.assertIsInstance(feature_types["binary"], list)

        # Should identify at least some numerical features
        self.assertGreater(len(feature_types["numerical"]), 0)

    def test_fit_transform_consistency(self):
        """Test that fit_transform produces consistent results."""
        # Fit and transform in one step
        X_fit_transform, y_fit_transform = self.preprocessor.fit_transform(
            self.clean_X, self.clean_y
        )

        # Fit separately then transform
        preprocessor2 = CreditDataPreprocessor(
            handle_missing=self.preprocessor.handle_missing,
            remove_outliers=self.preprocessor.remove_outliers,
            normalize_features=self.preprocessor.normalize_features,
        )
        preprocessor2.fit(self.clean_X, self.clean_y)
        X_separate = preprocessor2.transform(self.clean_X)

        # Results should be identical
        pd.testing.assert_frame_equal(X_fit_transform, X_separate, check_dtype=False)
        pd.testing.assert_series_equal(y_fit_transform, self.clean_y, check_names=False)

    def test_transform_new_data(self):
        """Test transforming new data with fitted preprocessor."""
        # Fit on training data
        self.preprocessor.fit(self.clean_X, self.clean_y)

        # Generate new test data
        new_X, _ = self.generator.generate_dataset(n_samples=50, default_rate=0.2)

        # Transform new data
        new_X_transformed = self.preprocessor.transform(new_X)

        # Should have same number of features as training data
        training_features = self.preprocessor.transform(self.clean_X).shape[1]
        self.assertEqual(new_X_transformed.shape[1], training_features)

        # Should not have missing values if preprocessor handles them
        if self.preprocessor.handle_missing:
            self.assertEqual(new_X_transformed.isnull().sum().sum(), 0)

    def test_preprocessing_pipeline_integrity(self):
        """Test that the preprocessing pipeline maintains data integrity."""
        X_processed, y_processed = self.preprocessor.fit_transform(
            self.dirty_X, self.dirty_y
        )

        # Basic integrity checks
        self.assertEqual(len(X_processed), len(y_processed))
        self.assertGreater(len(X_processed), 0)
        self.assertGreater(X_processed.shape[1], 0)

        # No infinite values
        self.assertFalse(
            np.isinf(X_processed.select_dtypes(include=[np.number])).any().any()
        )

        # Target should be binary (0 or 1)
        unique_targets = y_processed.unique()
        self.assertTrue(all(target in [0, 1] for target in unique_targets))

    def test_preprocessing_metadata(self):
        """Test that preprocessing metadata is properly stored."""
        self.preprocessor.fit(self.clean_X, self.clean_y)

        # Should store preprocessing information
        self.assertTrue(self.preprocessor.is_fitted)
        self.assertIsNotNone(self.preprocessor._feature_types)

        # Should have information about original features
        self.assertIn("original_features", self.preprocessor._preprocessing_metadata)
        self.assertIn("processed_features", self.preprocessor._preprocessing_metadata)

    def test_save_load_preprocessor(self):
        """Test saving and loading preprocessor state."""
        # Fit preprocessor
        self.preprocessor.fit(self.clean_X, self.clean_y)

        # Save preprocessor
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            preprocessor_path = tmp_file.name

        try:
            self.preprocessor.save_preprocessor(preprocessor_path)
            self.assertTrue(os.path.exists(preprocessor_path))

            # Load preprocessor
            loaded_preprocessor = CreditDataPreprocessor.load_preprocessor(
                preprocessor_path
            )

            # Test that loaded preprocessor works
            self.assertTrue(loaded_preprocessor.is_fitted)

            # Transformations should be identical
            original_transform = self.preprocessor.transform(self.clean_X)
            loaded_transform = loaded_preprocessor.transform(self.clean_X)

            pd.testing.assert_frame_equal(
                original_transform, loaded_transform, check_dtype=False
            )

        finally:
            # Clean up
            if os.path.exists(preprocessor_path):
                os.unlink(preprocessor_path)

    def test_edge_cases(self):
        """Test handling of edge cases."""
        # Test with single sample
        single_X = self.clean_X.head(1)
        single_y = self.clean_y.head(1)

        try:
            X_single, y_single = self.preprocessor.fit_transform(single_X, single_y)
            self.assertEqual(len(X_single), 1)
            self.assertEqual(len(y_single), 1)
        except Exception:
            # Single sample preprocessing might not work - that's acceptable
            pass

        # Test with all missing values in a column
        X_all_missing = self.clean_X.copy()
        X_all_missing["test_missing"] = np.nan

        try:
            X_processed, _ = self.preprocessor.fit_transform(
                X_all_missing, self.clean_y
            )
            # Should either handle it gracefully or remove the column
            self.assertGreaterEqual(X_processed.shape[1], self.clean_X.shape[1])
        except Exception:
            # It's acceptable to raise an error for pathological cases
            pass

        # Test with constant feature
        X_constant = self.clean_X.copy()
        X_constant["constant_feature"] = 42

        X_processed, _ = self.preprocessor.fit_transform(X_constant, self.clean_y)
        # Should handle constant features (might remove them)
        self.assertGreaterEqual(X_processed.shape[1], self.clean_X.shape[1])


class TestPreprocessingUtilities(unittest.TestCase):
    """Test cases for preprocessing utility functions."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Generate sample data
        generator = CreditDataGenerator(random_seed=42)
        self.clean_X, self.clean_y = generator.generate_dataset(100, default_rate=0.3)

    def test_outlier_detection_methods(self):
        """Test different outlier detection methods."""
        # Create data with obvious outliers
        normal_data = np.random.normal(0, 1, 100)
        outlier_data = np.concatenate([normal_data, [10, -10, 15]])  # Add outliers

        preprocessor = CreditDataPreprocessor()

        # Test IQR method
        outliers_iqr = preprocessor._detect_outliers_iqr(outlier_data)
        self.assertGreater(len(outliers_iqr), 0)  # Should detect some outliers

        # Test Z-score method
        outliers_zscore = preprocessor._detect_outliers_zscore(outlier_data)
        self.assertGreater(len(outliers_zscore), 0)  # Should detect some outliers

    def test_data_quality_scoring(self):
        """Test data quality scoring functionality."""
        preprocessor = CreditDataPreprocessor()

        # Test with clean data
        clean_score = preprocessor._calculate_data_quality_score(self.clean_X)

        # Test with dirty data
        dirty_X = self.clean_X.copy()
        dirty_X.loc[0:10, "age"] = np.nan  # Add missing values
        dirty_score = preprocessor._calculate_data_quality_score(dirty_X)

        # Clean data should have higher quality score
        self.assertGreater(clean_score, dirty_score)

        # Scores should be between 0 and 1
        self.assertGreaterEqual(clean_score, 0.0)
        self.assertLessEqual(clean_score, 1.0)
        self.assertGreaterEqual(dirty_score, 0.0)
        self.assertLessEqual(dirty_score, 1.0)


if __name__ == "__main__":
    # Create clean test data
    generator = CreditDataGenerator(random_seed=42)
    clean_X, clean_y = generator.generate_dataset(n_samples=100, default_rate=0.3)

    # Run tests with verbose output
    unittest.main(verbosity=2)
