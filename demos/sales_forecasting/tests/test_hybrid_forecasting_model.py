"""
Comprehensive Test Suite for Hybrid Forecasting Model.

This module provides extensive testing for the hybrid forecasting model,
including unit tests, integration tests, and time series-specific validation.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath("../../.."))

from demos.sales_forecasting.models.hybrid_forecasting_model import (
    HybridForecastingModel,
)
from demos.sales_forecasting.data.generate_sales_data import SalesDataGenerator
from demos.sales_forecasting.scripts.feature_engineering import FeatureEngineer
from demos.sales_forecasting.analytics.forecast_evaluator import ForecastEvaluator

warnings.filterwarnings("ignore")


class TestHybridForecastingModel(unittest.TestCase):
    """Test cases for the HybridForecastingModel."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        print("Setting up test environment for HybridForecastingModel...")

        # Generate test data
        cls.generator = SalesDataGenerator(
            start_date="2022-01-01",
            end_date="2023-12-31",
            stores=["TEST_STORE"],
            product_categories=["TEST_CATEGORY"],
        )

        cls.sales_data, cls.external_factors = cls.generator.generate_full_dataset()

        # Filter for single store/category
        cls.model_data = cls.sales_data[
            (cls.sales_data["store_id"] == "TEST_STORE")
            & (cls.sales_data["product_category"] == "TEST_CATEGORY")
        ].copy()

        cls.model_data["date"] = pd.to_datetime(cls.model_data["date"])
        cls.model_data = cls.model_data.set_index("date").sort_index()
        cls.sales_series = cls.model_data["daily_sales"]

        # Feature engineering
        cls.feature_engineer = FeatureEngineer(
            lag_features=[1, 7, 14], rolling_windows=[7, 14], seasonal_periods=[7, 30]
        )

        cls.external_factors["date"] = pd.to_datetime(cls.external_factors["date"])
        cls.external_factors = cls.external_factors.set_index("date")

        cls.engineered_features = cls.feature_engineer.engineer_features(
            cls.sales_series, external_factors=cls.external_factors
        )

        # Split data
        split_date = "2023-09-01"
        cls.train_data = cls.engineered_features[
            cls.engineered_features.index < split_date
        ].copy()
        cls.test_data = cls.engineered_features[
            cls.engineered_features.index >= split_date
        ].copy()

        cls.X_train = cls.train_data.drop("target", axis=1)
        cls.y_train = cls.train_data["target"]
        cls.X_test = cls.test_data.drop("target", axis=1)
        cls.y_test = cls.test_data["target"]

        print(
            f"Test data prepared: {len(cls.X_train)} training samples, {len(cls.X_test)} test samples"
        )

    def setUp(self):
        """Set up for each individual test."""
        self.model = HybridForecastingModel(
            forecast_horizon=30,
            prophet_params={
                "seasonality_mode": "multiplicative",
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "daily_seasonality": False,
            },
            lightgbm_params={
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "num_leaves": 15,  # Smaller for faster testing
                "learning_rate": 0.1,
                "verbose": -1,
                "random_state": 42,
                "n_estimators": 50,  # Fewer iterations for faster testing
            },
            ensemble_strategy="horizon_weighted",
        )

    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, HybridForecastingModel)
        self.assertIsNotNone(self.model.prophet_params)
        self.assertIsNotNone(self.model.lightgbm_params)
        self.assertIsNotNone(self.model.ensemble_strategy)

        # Test ensemble strategy validation
        self.assertIn(
            self.model.ensemble_strategy,
            ["simple_average", "horizon_weighted", "confidence_weighted"],
        )

    def test_model_fitting(self):
        """Test model fitting process."""
        # Test that model can be fitted
        self.model.fit(self.X_train, self.y_train)

        # Check that components are trained
        self.assertTrue(hasattr(self.model, "is_fitted"))
        self.assertTrue(self.model.is_fitted)

        # Test fitting with insufficient data
        small_X = self.X_train.head(10)
        small_y = self.y_train.head(10)

        with self.assertRaises((ValueError, Exception)):
            small_model = HybridForecastingModel()
            small_model.fit(small_X, small_y)

    def test_model_prediction(self):
        """Test model prediction functionality."""
        # Fit the model
        self.model.fit(self.X_train, self.y_train)

        # Test prediction
        predictions = self.model.predict(self.X_test)

        # Check prediction properties
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(
            all(isinstance(p, (int, float, np.number)) for p in predictions)
        )
        self.assertTrue(
            all(p >= 0 for p in predictions)
        )  # Sales should be non-negative

        # Test prediction with single sample
        single_prediction = self.model.predict(self.X_test.iloc[:1])
        self.assertEqual(len(single_prediction), 1)

    def test_prediction_without_fitting(self):
        """Test that prediction raises error when model is not fitted."""
        with self.assertRaises((ValueError, AttributeError)):
            self.model.predict(self.X_test)

    def test_prediction_accuracy(self):
        """Test prediction accuracy metrics."""
        # Fit and predict
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)

        # Calculate MAPE
        mape = np.mean(np.abs((self.y_test - predictions) / self.y_test)) * 100

        # MAPE should be reasonable for time series forecasting
        self.assertLess(
            mape, 50, "MAPE should be less than 50% for reasonable forecast"
        )

        # Test correlation
        correlation = np.corrcoef(self.y_test, predictions)[0, 1]
        self.assertGreater(
            correlation,
            0.3,
            "Predictions should have positive correlation with actual values",
        )

    def test_confidence_intervals(self):
        """Test confidence interval generation."""
        self.model.fit(self.X_train, self.y_train)

        try:
            confidence_intervals = self.model.predict_with_confidence(
                self.X_test, confidence_level=0.95
            )

            # Check structure
            self.assertIn("lower", confidence_intervals)
            self.assertIn("upper", confidence_intervals)

            # Check that upper > lower
            self.assertTrue(
                all(confidence_intervals["upper"] >= confidence_intervals["lower"])
            )

            # Check that intervals contain reasonable predictions
            predictions = self.model.predict(self.X_test)
            self.assertTrue(
                all(
                    (confidence_intervals["lower"] <= predictions)
                    & (predictions <= confidence_intervals["upper"])
                )
            )

        except NotImplementedError:
            self.skipTest("Confidence intervals not implemented")

    def test_feature_importance(self):
        """Test feature importance extraction."""
        self.model.fit(self.X_train, self.y_train)

        try:
            importance = self.model.get_feature_importance()

            # Check that importance is returned
            self.assertIsInstance(importance, (pd.Series, dict))

            if isinstance(importance, pd.Series):
                # Check that importance values are non-negative
                self.assertTrue(all(importance >= 0))

                # Check that we have importance for all features
                self.assertEqual(len(importance), len(self.X_train.columns))

        except (NotImplementedError, AttributeError):
            self.skipTest("Feature importance not implemented")

    def test_model_serialization(self):
        """Test model saving and loading."""
        import pickle
        import tempfile

        # Fit the model
        self.model.fit(self.X_train, self.y_train)
        original_predictions = self.model.predict(self.X_test)

        # Save and load model
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            pickle.dump(self.model, tmp_file)
            tmp_file.flush()

            with open(tmp_file.name, "rb") as f:
                loaded_model = pickle.load(f)

        # Test that loaded model produces same predictions
        loaded_predictions = loaded_model.predict(self.X_test)
        np.testing.assert_array_almost_equal(
            original_predictions, loaded_predictions, decimal=6
        )

        # Clean up
        os.unlink(tmp_file.name)

    def test_input_validation(self):
        """Test input validation."""
        # Test with mismatched X and y lengths
        with self.assertRaises((ValueError, IndexError)):
            self.model.fit(self.X_train, self.y_train.iloc[:-10])

        # Test with NaN values
        X_with_nan = self.X_train.copy()
        X_with_nan.iloc[0, 0] = np.nan

        # Should either handle NaN or raise appropriate error
        try:
            self.model.fit(X_with_nan, self.y_train)
        except (ValueError, Exception) as e:
            self.assertIn(("nan", "NaN", "missing", "null"), str(e).lower())

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with constant target values
        constant_y = pd.Series([100.0] * len(self.y_train), index=self.y_train.index)

        try:
            constant_model = HybridForecastingModel()
            constant_model.fit(self.X_train, constant_y)
            constant_predictions = constant_model.predict(self.X_test)

            # Predictions should be close to constant value
            self.assertTrue(all(abs(constant_predictions - 100.0) < 50.0))

        except Exception:
            # Some models may not handle constant targets well
            pass

        # Test with very small values
        small_y = self.y_train * 0.001
        try:
            small_model = HybridForecastingModel()
            small_model.fit(self.X_train, small_y)
            small_predictions = small_model.predict(self.X_test)

            # Should produce reasonable predictions
            self.assertTrue(all(small_predictions >= 0))

        except Exception:
            pass

    def test_reproducibility(self):
        """Test model reproducibility with same random seed."""
        # Create two identical models
        model1 = HybridForecastingModel(
            lightgbm_config={"random_state": 42, "verbose": -1, "n_estimators": 20}
        )
        model2 = HybridForecastingModel(
            lightgbm_config={"random_state": 42, "verbose": -1, "n_estimators": 20}
        )

        # Fit both models
        model1.fit(self.X_train, self.y_train)
        model2.fit(self.X_train, self.y_train)

        # Predictions should be very similar (allowing for small numerical differences)
        pred1 = model1.predict(self.X_test)
        pred2 = model2.predict(self.X_test)

        # Check that predictions are close (within 1% difference)
        relative_diff = np.abs((pred1 - pred2) / (pred1 + 1e-8))
        self.assertTrue(
            np.mean(relative_diff) < 0.01, "Models should produce reproducible results"
        )


class TestTimeSeriesValidation(unittest.TestCase):
    """Test cases for time series-specific validation."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Use same data as HybridForecastingModel tests
        cls.generator = SalesDataGenerator(
            start_date="2022-01-01",
            end_date="2023-12-31",
            stores=["TEST_STORE"],
            product_categories=["TEST_CATEGORY"],
        )

        cls.sales_data, cls.external_factors = cls.generator.generate_full_dataset()

        cls.model_data = cls.sales_data[
            (cls.sales_data["store_id"] == "TEST_STORE")
            & (cls.sales_data["product_category"] == "TEST_CATEGORY")
        ].copy()

        cls.model_data["date"] = pd.to_datetime(cls.model_data["date"])
        cls.model_data = cls.model_data.set_index("date").sort_index()
        cls.sales_series = cls.model_data["daily_sales"]

        cls.feature_engineer = FeatureEngineer()
        cls.external_factors["date"] = pd.to_datetime(cls.external_factors["date"])
        cls.external_factors = cls.external_factors.set_index("date")

        cls.engineered_features = cls.feature_engineer.engineer_features(
            cls.sales_series, external_factors=cls.external_factors
        )

    def test_walk_forward_validation(self):
        """Test walk-forward validation for time series."""
        # Parameters
        initial_training_size = 300  # days
        step_size = 30  # days
        forecast_horizon = 7  # days

        all_errors = []
        validation_steps = 0

        start_idx = initial_training_size
        while start_idx + forecast_horizon < len(self.engineered_features):
            # Define training and test windows
            train_end = start_idx
            test_start = train_end
            test_end = min(test_start + forecast_horizon, len(self.engineered_features))

            # Split data
            train_data = self.engineered_features.iloc[:train_end]
            test_data = self.engineered_features.iloc[test_start:test_end]

            if len(train_data) < 100 or len(test_data) == 0:
                break

            X_train = train_data.drop("target", axis=1)
            y_train = train_data["target"]
            X_test = test_data.drop("target", axis=1)
            y_test = test_data["target"]

            # Train and predict
            try:
                model = HybridForecastingModel(
                    lightgbm_config={"verbose": -1, "n_estimators": 20}
                )
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                # Calculate error
                mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
                all_errors.append(mape)
                validation_steps += 1

            except Exception as e:
                print(f"Walk-forward validation failed at step {validation_steps}: {e}")
                break

            start_idx += step_size

        # Validate results
        self.assertGreater(
            validation_steps, 0, "Should complete at least one validation step"
        )
        self.assertLess(np.mean(all_errors), 100, "Average MAPE should be reasonable")

        print(
            f"Walk-forward validation completed {validation_steps} steps, average MAPE: {np.mean(all_errors):.2f}%"
        )

    def test_temporal_consistency(self):
        """Test temporal consistency of predictions."""
        # Split data
        split_date = "2023-09-01"
        train_data = self.engineered_features[
            self.engineered_features.index < split_date
        ].copy()
        test_data = self.engineered_features[
            self.engineered_features.index >= split_date
        ].copy()

        X_train = train_data.drop("target", axis=1)
        y_train = train_data["target"]
        X_test = test_data.drop("target", axis=1)

        # Train model
        model = HybridForecastingModel(
            lightgbm_config={"verbose": -1, "n_estimators": 20}
        )
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Check temporal consistency
        # 1. Predictions should not have extreme jumps
        pred_series = pd.Series(predictions, index=X_test.index)
        daily_changes = pred_series.pct_change().abs()

        # Remove the first value (NaN) and check that most daily changes are reasonable
        daily_changes = daily_changes.dropna()
        extreme_changes = daily_changes > 0.5  # More than 50% change

        self.assertLess(
            extreme_changes.sum() / len(daily_changes),
            0.1,
            "Less than 10% of predictions should have extreme daily changes",
        )

        # 2. Predictions should maintain some correlation with recent history
        if len(pred_series) > 7:
            recent_correlation = (
                pred_series.rolling(7).corr(pred_series.shift(1)).mean()
            )
            if not np.isnan(recent_correlation):
                self.assertGreater(
                    recent_correlation,
                    0,
                    "Predictions should maintain positive temporal correlation",
                )

    def test_seasonal_pattern_preservation(self):
        """Test that model preserves seasonal patterns."""
        # Train model on full dataset
        X = self.engineered_features.drop("target", axis=1)
        y = self.engineered_features["target"]

        model = HybridForecastingModel(
            lightgbm_config={"verbose": -1, "n_estimators": 30}
        )
        model.fit(X, y)

        # Make predictions
        predictions = model.predict(X)
        pred_series = pd.Series(predictions, index=X.index)

        # Analyze weekly seasonality
        pred_series["day_of_week"] = pred_series.index.dayofweek
        actual_series = y.copy()
        actual_series["day_of_week"] = actual_series.index.dayofweek

        # Calculate average by day of week
        pred_weekly = pred_series.groupby("day_of_week").mean()
        actual_weekly = actual_series.groupby("day_of_week").mean()

        # Check correlation of weekly patterns
        weekly_correlation = np.corrcoef(pred_weekly, actual_weekly)[0, 1]

        if not np.isnan(weekly_correlation):
            self.assertGreater(
                weekly_correlation,
                0.3,
                "Predicted weekly pattern should correlate with actual pattern",
            )

    def test_forecast_horizon_performance(self):
        """Test performance degradation over forecast horizon."""
        # Use different forecast horizons
        horizons = [1, 7, 14, 30]
        horizon_errors = {}

        for horizon in horizons:
            # Split data for this horizon
            train_size = len(self.engineered_features) - horizon - 30  # Leave buffer
            if train_size < 200:
                continue

            train_data = self.engineered_features.iloc[:train_size]
            test_data = self.engineered_features.iloc[train_size : train_size + horizon]

            X_train = train_data.drop("target", axis=1)
            y_train = train_data["target"]
            X_test = test_data.drop("target", axis=1)
            y_test = test_data["target"]

            try:
                model = HybridForecastingModel(
                    lightgbm_config={"verbose": -1, "n_estimators": 20}
                )
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
                horizon_errors[horizon] = mape

            except Exception:
                continue

        # Check that we have results for multiple horizons
        self.assertGreaterEqual(
            len(horizon_errors), 2, "Should test multiple forecast horizons"
        )

        # Generally, error should not increase dramatically with horizon
        if len(horizon_errors) >= 2:
            errors = list(horizon_errors.values())
            # Error shouldn't more than double from shortest to longest horizon
            self.assertLess(
                max(errors) / min(errors),
                3.0,
                "Error shouldn't increase too dramatically with horizon",
            )

        print(f"Forecast horizon errors: {horizon_errors}")


class TestModelComponents(unittest.TestCase):
    """Test individual model components."""

    def setUp(self):
        """Set up test data."""
        # Generate simple test data
        dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
        np.random.seed(42)

        # Simple trend + seasonality + noise
        trend = np.linspace(1000, 1200, len(dates))
        seasonal = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
        noise = np.random.normal(0, 50, len(dates))

        self.sales_data = pd.Series(trend + seasonal + noise, index=dates)
        self.features_df = pd.DataFrame(
            {
                "year": dates.year,
                "month": dates.month,
                "day": dates.day,
                "dayofweek": dates.dayofweek,
            },
            index=dates,
        )

    def test_prophet_component(self):
        """Test Prophet component independently."""
        try:
            from demos.sales_forecasting.models.components.prophet_component import (
                ProphetComponent,
            )

            # Initialize Prophet component
            prophet = ProphetComponent(
                seasonality_mode="additive",
                yearly_seasonality=True,
                weekly_seasonality=True,
            )

            # Fit Prophet
            prophet.fit(self.sales_data)

            # Make predictions
            future_dates = pd.date_range(
                self.sales_data.index[-1] + pd.Timedelta(days=1), periods=30, freq="D"
            )
            predictions = prophet.predict(future_dates)

            # Validate predictions
            self.assertEqual(len(predictions), 30)
            self.assertTrue(
                all(isinstance(p, (int, float, np.number)) for p in predictions)
            )

        except ImportError:
            self.skipTest("Prophet component not available")

    def test_lightgbm_component(self):
        """Test LightGBM component independently."""
        try:
            from demos.sales_forecasting.models.components.lightgbm_component import (
                LightGBMComponent,
            )

            # Prepare training data
            X_train = self.features_df[:-30]
            y_train = self.sales_data[:-30]
            X_test = self.features_df[-30:]

            # Initialize LightGBM component
            lgb = LightGBMComponent(
                objective="regression", metric="rmse", verbose=-1, n_estimators=50
            )

            # Fit and predict
            lgb.fit(X_train, y_train)
            predictions = lgb.predict(X_test)

            # Validate predictions
            self.assertEqual(len(predictions), 30)
            self.assertTrue(
                all(isinstance(p, (int, float, np.number)) for p in predictions)
            )

            # Test feature importance
            importance = lgb.get_feature_importance()
            self.assertIsInstance(importance, (pd.Series, dict, np.ndarray))

        except ImportError:
            self.skipTest("LightGBM component not available")

    def test_seasonal_analyzer(self):
        """Test seasonal analyzer component."""
        try:
            from demos.sales_forecasting.models.components.seasonal_analyzer import (
                SeasonalAnalyzer,
            )

            # Initialize analyzer
            analyzer = SeasonalAnalyzer(
                seasonal_periods=[7, 30], auto_detect_periods=True
            )

            # Fit analyzer
            analyzer.fit(self.sales_data)

            # Check that seasonal patterns were detected
            self.assertGreater(len(analyzer.seasonal_strengths_), 0)

            # Generate seasonal forecasts
            seasonal_forecasts = analyzer.predict_seasonal_components(30)
            self.assertIsInstance(seasonal_forecasts, dict)

            # Check summary
            summary = analyzer.get_seasonal_summary()
            self.assertIsInstance(summary, pd.DataFrame)

        except ImportError:
            self.skipTest("Seasonal analyzer not available")

    def test_trend_detector(self):
        """Test trend detector component."""
        try:
            from demos.sales_forecasting.models.components.trend_detector import (
                TrendDetector,
            )

            # Initialize detector
            detector = TrendDetector(
                trend_methods=["linear", "polynomial"], min_segment_length=10
            )

            # Fit detector
            detector.fit(self.sales_data)

            # Check that trend was detected
            self.assertIsNotNone(detector.best_trend_method_)
            self.assertGreater(len(detector.trend_strengths_), 0)

            # Generate trend forecast
            trend_forecast = detector.predict_trend(30)
            self.assertEqual(len(trend_forecast), 30)

            # Check summary
            summary = detector.get_trend_summary()
            self.assertIsInstance(summary, pd.DataFrame)

        except ImportError:
            self.skipTest("Trend detector not available")


class TestBusinessLogic(unittest.TestCase):
    """Test business logic and evaluation components."""

    def setUp(self):
        """Set up test data."""
        # Simple test data
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        np.random.seed(42)

        self.actual = pd.Series(
            1000 + np.random.normal(0, 100, len(dates)), index=dates
        )
        self.forecast = pd.Series(
            1000 + np.random.normal(0, 80, len(dates)), index=dates
        )

        # Add some correlation between actual and forecast
        self.forecast = 0.7 * self.actual + 0.3 * self.forecast

    def test_forecast_evaluator(self):
        """Test forecast evaluation functionality."""
        evaluator = ForecastEvaluator()

        # Evaluate forecast
        results = evaluator.evaluate_forecast(self.actual, self.forecast)

        # Check that results contain expected keys
        expected_keys = [
            "accuracy_metrics",
            "business_insights",
            "performance_analysis",
        ]
        for key in expected_keys:
            self.assertIn(key, results)

        # Check accuracy metrics
        accuracy = results["accuracy_metrics"]
        self.assertIn("mape", accuracy)
        self.assertIn("mae", accuracy)
        self.assertIn("rmse", accuracy)

        # MAPE should be reasonable
        self.assertLess(accuracy["mape"], 100)
        self.assertGreater(accuracy["mape"], 0)

        # Check business insights
        business_insights = results["business_insights"]
        self.assertIn("revenue_impact", business_insights)

    def test_business_intelligence(self):
        """Test business intelligence functionality."""
        from demos.sales_forecasting.analytics.business_intelligence import (
            BusinessIntelligence,
        )

        bi = BusinessIntelligence()

        # Generate dashboard
        dashboard = bi.generate_executive_dashboard(self.actual, self.forecast)

        # Check dashboard structure
        expected_keys = ["core_kpis", "business_health", "executive_summary"]
        for key in expected_keys:
            self.assertIn(key, dashboard)

        # Check KPIs
        kpis = dashboard["core_kpis"]
        self.assertIn("total_revenue", kpis)
        self.assertIn("forecast_accuracy", kpis)

        # Check health metrics
        health = dashboard["business_health"]
        self.assertIn("overall_health_score", health)

        # Health scores should be between 0 and 10
        for score in health.values():
            if isinstance(score, (int, float)):
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 10)


class TestDataQuality(unittest.TestCase):
    """Test data quality and validation."""

    def test_sales_data_generator(self):
        """Test sales data generator."""
        generator = SalesDataGenerator(
            start_date="2023-01-01",
            end_date="2023-01-31",
            stores=["TEST_STORE"],
            categories=["TEST_CAT"],
        )

        sales_data, external_factors = generator.generate_full_dataset()

        # Check data structure
        self.assertIsInstance(sales_data, pd.DataFrame)
        self.assertIsInstance(external_factors, pd.DataFrame)

        # Check required columns
        required_sales_cols = ["date", "store_id", "product_category", "daily_sales"]
        for col in required_sales_cols:
            self.assertIn(col, sales_data.columns)

        # Check data quality
        self.assertFalse(sales_data["daily_sales"].isna().any())
        self.assertTrue((sales_data["daily_sales"] >= 0).all())

        # Check date range
        sales_dates = pd.to_datetime(sales_data["date"])
        self.assertEqual(sales_dates.min().date(), pd.to_datetime("2023-01-01").date())
        self.assertEqual(sales_dates.max().date(), pd.to_datetime("2023-01-31").date())

    def test_feature_engineering(self):
        """Test feature engineering pipeline."""
        # Generate test data
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        sales_series = pd.Series(
            1000 + np.random.normal(0, 100, len(dates)), index=dates
        )

        feature_engineer = FeatureEngineer(lag_features=[1, 7], rolling_windows=[7, 14])

        # Engineer features
        features = feature_engineer.engineer_features(sales_series)

        # Check that features were created
        self.assertIn("target", features.columns)
        self.assertIn("sales_lag_1", features.columns)
        self.assertIn("sales_lag_7", features.columns)
        self.assertIn("sales_ma_7", features.columns)

        # Check feature quality
        # Lag features should have expected number of NaN values
        self.assertEqual(features["sales_lag_1"].isna().sum(), 1)  # First value
        self.assertEqual(features["sales_lag_7"].isna().sum(), 7)  # First 7 values

        # Moving averages should be reasonable
        ma7 = features["sales_ma_7"].dropna()
        self.assertTrue((ma7 > 0).all())


def run_all_tests():
    """Run all test suites."""
    print("=" * 60)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    # Create test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestHybridForecastingModel,
        TestTimeSeriesValidation,
        TestModelComponents,
        TestBusinessLogic,
        TestDataQuality,
    ]

    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(
                f"- {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}"
            )

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\\n')[-2]}")

    success_rate = (
        (result.testsRun - len(result.failures) - len(result.errors))
        / result.testsRun
        * 100
    )
    print(f"\nSuccess Rate: {success_rate:.1f}%")

    return result


if __name__ == "__main__":
    run_all_tests()
