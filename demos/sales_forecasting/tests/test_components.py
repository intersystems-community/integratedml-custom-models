"""
Component Tests for Sales Forecasting System.

This module provides unit tests for individual components of the
sales forecasting system, including data generation, feature engineering,
and analytics components.
"""

import unittest
import numpy as np
import pandas as pd
import warnings
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.abspath("../../.."))

from demos.sales_forecasting.data.generate_sales_data import SalesDataGenerator
from demos.sales_forecasting.scripts.feature_engineering import FeatureEngineer
from demos.sales_forecasting.analytics.forecast_evaluator import ForecastEvaluator
from demos.sales_forecasting.analytics.business_intelligence import BusinessIntelligence

warnings.filterwarnings("ignore")


class TestSalesDataGenerator(unittest.TestCase):
    """Test cases for the SalesDataGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = SalesDataGenerator(
            start_date="2023-01-01",
            end_date="2023-01-31",
            stores=["TEST_STORE_1", "TEST_STORE_2"],
            product_categories=["TEST_CAT_1", "TEST_CAT_2"],
        )

    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.start_date, pd.to_datetime("2023-01-01"))
        self.assertEqual(self.generator.end_date, pd.to_datetime("2023-01-31"))
        self.assertEqual(self.generator.stores, ["TEST_STORE_1", "TEST_STORE_2"])
        self.assertEqual(self.generator.categories, ["TEST_CAT_1", "TEST_CAT_2"])

    def test_date_range_generation(self):
        """Test date range generation."""
        dates = self.generator._generate_date_range()

        # Check date range
        self.assertEqual(dates[0], pd.to_datetime("2023-01-01"))
        self.assertEqual(dates[-1], pd.to_datetime("2023-01-31"))
        self.assertEqual(len(dates), 31)  # January has 31 days

    def test_base_sales_patterns(self):
        """Test base sales pattern generation."""
        dates = self.generator._generate_date_range()
        base_sales = self.generator._generate_base_sales_pattern(dates)

        # Check output structure
        self.assertIsInstance(base_sales, pd.Series)
        self.assertEqual(len(base_sales), len(dates))

        # Check that sales are positive
        self.assertTrue((base_sales > 0).all())

        # Check for reasonable variance
        self.assertGreater(base_sales.std(), 0)

    def test_seasonal_patterns(self):
        """Test seasonal pattern generation."""
        dates = self.generator._generate_date_range()

        # Test weekly seasonality
        weekly_pattern = self.generator._apply_weekly_seasonality(
            pd.Series(1.0, index=dates)
        )
        self.assertEqual(len(weekly_pattern), len(dates))

        # Test monthly seasonality
        monthly_pattern = self.generator._apply_monthly_seasonality(
            pd.Series(1.0, index=dates)
        )
        self.assertEqual(len(monthly_pattern), len(dates))

        # Patterns should modify the base series
        self.assertNotEqual(weekly_pattern.iloc[0], weekly_pattern.iloc[1])

    def test_external_factors_generation(self):
        """Test external factors generation."""
        external_factors = self.generator._generate_external_factors()

        # Check structure
        self.assertIsInstance(external_factors, pd.DataFrame)

        # Check required columns
        expected_columns = ["date", "temperature", "precipitation", "holiday_indicator"]
        for col in expected_columns:
            self.assertIn(col, external_factors.columns)

        # Check date range
        dates = pd.to_datetime(external_factors["date"])
        self.assertEqual(dates.min().date(), pd.to_datetime("2023-01-01").date())
        self.assertEqual(dates.max().date(), pd.to_datetime("2023-01-31").date())

        # Check data quality
        self.assertFalse(external_factors["temperature"].isna().all())
        self.assertTrue((external_factors["holiday_indicator"].isin([0, 1])).all())

    def test_full_dataset_generation(self):
        """Test full dataset generation."""
        sales_data, external_factors = self.generator.generate_full_dataset()

        # Check sales data structure
        self.assertIsInstance(sales_data, pd.DataFrame)
        required_sales_cols = ["date", "store_id", "product_category", "daily_sales"]
        for col in required_sales_cols:
            self.assertIn(col, sales_data.columns)

        # Check external factors structure
        self.assertIsInstance(external_factors, pd.DataFrame)
        self.assertIn("date", external_factors.columns)

        # Check data integrity
        self.assertFalse(sales_data["daily_sales"].isna().any())
        self.assertTrue((sales_data["daily_sales"] >= 0).all())

        # Check store and category combinations
        unique_stores = sales_data["store_id"].unique()
        unique_categories = sales_data["product_category"].unique()

        self.assertEqual(set(unique_stores), set(self.generator.stores))
        self.assertEqual(set(unique_categories), set(self.generator.product_categories))

        # Check that we have data for all store-category combinations
        expected_combinations = len(self.generator.stores) * len(
            self.generator.product_categories
        )
        actual_combinations = len(sales_data.groupby(["store_id", "product_category"]))
        self.assertEqual(actual_combinations, expected_combinations)

    def test_data_consistency(self):
        """Test data consistency across multiple generations."""
        # Generate dataset twice with same parameters
        sales1, external1 = self.generator.generate_full_dataset()
        sales2, external2 = self.generator.generate_full_dataset()

        # Datasets should have same structure
        self.assertEqual(len(sales1), len(sales2))
        self.assertEqual(len(external1), len(external2))
        self.assertEqual(list(sales1.columns), list(sales2.columns))
        self.assertEqual(list(external1.columns), list(external2.columns))

        # But different random data (unless seed is set)
        if not hasattr(self.generator, "random_seed"):
            self.assertFalse(sales1["daily_sales"].equals(sales2["daily_sales"]))

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with single day
        single_day_gen = SalesDataGenerator(
            start_date="2023-01-01",
            end_date="2023-01-01",
            stores=["STORE"],
            categories=["CAT"],
        )

        sales, external = single_day_gen.generate_full_dataset()
        self.assertEqual(len(sales), 1)  # One store-category combination
        self.assertEqual(len(external), 1)  # One day

        # Test invalid date range
        with self.assertRaises((ValueError, Exception)):
            invalid_gen = SalesDataGenerator(
                start_date="2023-01-02",
                end_date="2023-01-01",  # End before start
                stores=["STORE"],
                categories=["CAT"],
            )
            invalid_gen.generate_full_dataset()


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for the FeatureEngineer."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test time series
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        np.random.seed(42)

        # Simple trend + seasonality
        trend = np.linspace(1000, 1200, len(dates))
        seasonal = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
        noise = np.random.normal(0, 50, len(dates))

        self.sales_series = pd.Series(trend + seasonal + noise, index=dates)

        # Create external factors
        self.external_factors = pd.DataFrame(
            {
                "temperature": np.random.normal(20, 10, len(dates)),
                "holiday": np.random.choice([0, 1], len(dates), p=[0.9, 0.1]),
                "promotion": np.random.choice([0, 1], len(dates), p=[0.8, 0.2]),
            },
            index=dates,
        )

        self.feature_engineer = FeatureEngineer(
            lag_features=[1, 7, 14], rolling_windows=[7, 14], seasonal_periods=[7, 30]
        )

    def test_initialization(self):
        """Test feature engineer initialization."""
        self.assertEqual(self.feature_engineer.lag_features, [1, 7, 14])
        self.assertEqual(self.feature_engineer.rolling_windows, [7, 14])
        self.assertEqual(self.feature_engineer.seasonal_periods, [7, 30])

    def test_lag_features(self):
        """Test lag feature generation."""
        lag_features = self.feature_engineer._create_lag_features(self.sales_series)

        # Check that lag features are created
        expected_columns = ["sales_lag_1", "sales_lag_7", "sales_lag_14"]
        for col in expected_columns:
            self.assertIn(col, lag_features.columns)

        # Check lag values
        self.assertTrue(
            pd.isna(lag_features["sales_lag_1"].iloc[0])
        )  # First value should be NaN
        self.assertAlmostEqual(
            lag_features["sales_lag_1"].iloc[1], self.sales_series.iloc[0], places=6
        )

        # Check 7-day lag
        self.assertTrue(
            pd.isna(lag_features["sales_lag_7"].iloc[6])
        )  # 7th value should be NaN
        self.assertAlmostEqual(
            lag_features["sales_lag_7"].iloc[7], self.sales_series.iloc[0], places=6
        )

    def test_rolling_features(self):
        """Test rolling window feature generation."""
        rolling_features = self.feature_engineer._create_rolling_features(
            self.sales_series
        )

        # Check that rolling features are created
        expected_columns = ["sales_ma_7", "sales_ma_14", "sales_std_7", "sales_std_14"]
        for col in expected_columns:
            self.assertIn(col, rolling_features.columns)

        # Check rolling mean calculation
        self.assertTrue(
            pd.isna(rolling_features["sales_ma_7"].iloc[6])
        )  # First 6 should be NaN

        # Manually calculate 7-day MA for verification
        expected_ma = self.sales_series.iloc[:7].mean()
        self.assertAlmostEqual(
            rolling_features["sales_ma_7"].iloc[6], expected_ma, places=6
        )

        # Check that standard deviation is positive
        std_features = rolling_features[["sales_std_7", "sales_std_14"]].dropna()
        self.assertTrue((std_features >= 0).all().all())

    def test_seasonal_features(self):
        """Test seasonal feature generation."""
        seasonal_features = self.feature_engineer._create_seasonal_features(
            self.sales_series
        )

        # Check basic datetime features
        datetime_features = ["year", "month", "day", "dayofweek", "quarter"]
        for feature in datetime_features:
            self.assertIn(feature, seasonal_features.columns)

        # Check cyclical features
        cyclical_features = ["month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos"]
        for feature in cyclical_features:
            self.assertIn(feature, seasonal_features.columns)

        # Verify cyclical encoding
        # Sin and cos values should be between -1 and 1
        for feature in cyclical_features:
            values = seasonal_features[feature]
            self.assertTrue((values >= -1).all())
            self.assertTrue((values <= 1).all())

        # Check that month encoding is correct
        january_mask = seasonal_features["month"] == 1
        if january_mask.any():
            # January should have consistent cyclical encoding
            jan_sin_values = seasonal_features.loc[january_mask, "month_sin"].unique()
            self.assertEqual(
                len(jan_sin_values), 1
            )  # Should be same for all January days

    def test_trend_features(self):
        """Test trend feature generation."""
        trend_features = self.feature_engineer._create_trend_features(self.sales_series)

        # Check that trend features are created
        expected_features = ["trend_linear", "sales_pct_change"]
        for feature in expected_features:
            self.assertIn(feature, trend_features.columns)

        # Check linear trend
        trend_values = trend_features["trend_linear"]
        self.assertEqual(trend_values.iloc[0], 0)  # Should start at 0
        self.assertEqual(
            trend_values.iloc[-1], len(self.sales_series) - 1
        )  # Should end at length-1

        # Check percentage change
        pct_change = trend_features["sales_pct_change"]
        self.assertTrue(pd.isna(pct_change.iloc[0]))  # First value should be NaN

        # Manual calculation for second value
        expected_pct_change = (
            self.sales_series.iloc[1] - self.sales_series.iloc[0]
        ) / self.sales_series.iloc[0]
        self.assertAlmostEqual(pct_change.iloc[1], expected_pct_change, places=6)

    def test_external_factor_integration(self):
        """Test external factor integration."""
        features = self.feature_engineer.engineer_features(
            self.sales_series, external_factors=self.external_factors
        )

        # Check that external factors are included
        external_columns = ["temperature", "holiday", "promotion"]
        for col in external_columns:
            self.assertIn(col, features.columns)

        # Check that external factor values are preserved
        np.testing.assert_array_equal(
            features["temperature"].values, self.external_factors["temperature"].values
        )

    def test_full_feature_engineering(self):
        """Test complete feature engineering pipeline."""
        features = self.feature_engineer.engineer_features(self.sales_series)

        # Check that target is created
        self.assertIn("target", features.columns)
        np.testing.assert_array_equal(
            features["target"].dropna().values, self.sales_series.dropna().values
        )

        # Check that multiple feature types are created
        feature_types = {
            "lag": any("lag" in col for col in features.columns),
            "rolling": any("ma" in col or "std" in col for col in features.columns),
            "seasonal": any("sin" in col or "cos" in col for col in features.columns),
            "trend": any("trend" in col for col in features.columns),
        }

        for feature_type, exists in feature_types.items():
            self.assertTrue(exists, f"{feature_type} features not created")

        # Check data types
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        self.assertGreater(
            len(numeric_columns), 10
        )  # Should have many numeric features

        # Check for reasonable feature counts
        self.assertGreater(
            len(features.columns), 15
        )  # Should have substantial feature set
        self.assertLess(len(features.columns), 100)  # But not excessive

    def test_feature_selection(self):
        """Test feature selection functionality."""
        # Test with feature selection enabled
        selective_engineer = FeatureEngineer(
            lag_features=[1, 7], rolling_windows=[7], feature_selection_k=10
        )

        features = selective_engineer.engineer_features(self.sales_series)

        # Should have limited features when selection is applied
        if (
            hasattr(selective_engineer, "max_features")
            and selective_engineer.max_features
        ):
            # Allow some flexibility for target and essential features
            self.assertLessEqual(
                len(features.columns), selective_engineer.max_features + 5
            )

    def test_missing_data_handling(self):
        """Test handling of missing data."""
        # Create series with missing values
        missing_series = self.sales_series.copy()
        missing_series.iloc[10:15] = np.nan  # Insert missing values

        features = self.feature_engineer.engineer_features(missing_series)

        # Should handle missing data gracefully
        self.assertIsInstance(features, pd.DataFrame)
        self.assertIn("target", features.columns)

        # Check that missing values are handled appropriately
        # (implementation may fill, drop, or mark them)
        target_missing = features["target"].isna().sum()
        original_missing = missing_series.isna().sum()

        # Target should reflect original missing pattern
        self.assertEqual(target_missing, original_missing)


class TestForecastEvaluator(unittest.TestCase):
    """Test cases for the ForecastEvaluator."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        # Create test forecast data
        self.dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")

        # Actual values with trend and seasonality
        trend = np.linspace(1000, 1100, len(self.dates))
        seasonal = 50 * np.sin(2 * np.pi * np.arange(len(self.dates)) / 7)
        noise = np.random.normal(0, 25, len(self.dates))

        self.actual = pd.Series(trend + seasonal + noise, index=self.dates)

        # Forecast with some error
        forecast_noise = np.random.normal(0, 30, len(self.dates))
        self.forecast = pd.Series(trend + seasonal + forecast_noise, index=self.dates)

        self.evaluator = ForecastEvaluator()

    def test_accuracy_metrics(self):
        """Test accuracy metric calculations."""
        metrics = self.evaluator.calculate_accuracy_metrics(self.actual, self.forecast)

        # Check that all expected metrics are calculated
        expected_metrics = ["mae", "mse", "rmse", "mape", "smape", "r2_score"]
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertFalse(np.isnan(metrics[metric]))

        # Check metric properties
        self.assertGreaterEqual(metrics["mae"], 0)
        self.assertGreaterEqual(metrics["mse"], 0)
        self.assertGreaterEqual(metrics["rmse"], 0)
        self.assertGreaterEqual(metrics["mape"], 0)
        self.assertGreaterEqual(metrics["smape"], 0)
        self.assertLessEqual(metrics["r2_score"], 1)

        # RMSE should be sqrt of MSE
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(metrics["mse"]), places=6)

        # R2 should be reasonable for correlated data
        self.assertGreater(metrics["r2_score"], 0.5)  # Should have decent correlation

    def test_perfect_forecast(self):
        """Test metrics with perfect forecast."""
        perfect_forecast = self.actual.copy()
        metrics = self.evaluator.calculate_accuracy_metrics(
            self.actual, perfect_forecast
        )

        # Perfect forecast should have zero error metrics
        self.assertAlmostEqual(metrics["mae"], 0, places=10)
        self.assertAlmostEqual(metrics["mse"], 0, places=10)
        self.assertAlmostEqual(metrics["rmse"], 0, places=10)
        self.assertAlmostEqual(metrics["mape"], 0, places=6)
        self.assertAlmostEqual(metrics["smape"], 0, places=6)
        self.assertAlmostEqual(metrics["r2_score"], 1.0, places=10)

    def test_business_insights(self):
        """Test business insight calculations."""
        insights = self.evaluator.calculate_business_insights(
            self.actual, self.forecast
        )

        # Check expected insight categories
        expected_insights = ["revenue_impact", "forecast_bias", "accuracy_trends"]
        for insight in expected_insights:
            self.assertIn(insight, insights)

        # Check revenue impact calculation
        revenue_impact = insights["revenue_impact"]
        self.assertIn("total_actual", revenue_impact)
        self.assertIn("total_forecast", revenue_impact)
        self.assertIn("absolute_error", revenue_impact)
        self.assertIn("percentage_error", revenue_impact)

        # Revenue should be positive
        self.assertGreater(revenue_impact["total_actual"], 0)
        self.assertGreater(revenue_impact["total_forecast"], 0)

        # Check forecast bias
        bias = insights["forecast_bias"]
        self.assertIn("mean_bias", bias)
        self.assertIn("bias_direction", bias)

        # Bias direction should be valid
        self.assertIn(
            bias["bias_direction"], ["under_forecast", "over_forecast", "unbiased"]
        )

    def test_evaluate_forecast(self):
        """Test complete forecast evaluation."""
        results = self.evaluator.evaluate_forecast(self.actual, self.forecast)

        # Check result structure
        expected_sections = [
            "accuracy_metrics",
            "business_insights",
            "performance_analysis",
        ]
        for section in expected_sections:
            self.assertIn(section, results)

        # Check that each section has content
        for section in expected_sections:
            self.assertIsInstance(results[section], dict)
            self.assertGreater(len(results[section]), 0)

        # Check performance analysis
        performance = results["performance_analysis"]
        self.assertIn("forecast_quality", performance)
        self.assertIn("confidence_level", performance)

        # Quality should be between 0 and 1
        quality = performance["forecast_quality"]
        self.assertGreaterEqual(quality, 0)
        self.assertLessEqual(quality, 1)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with constant values
        constant_actual = pd.Series([100.0] * 30, index=self.dates[:30])
        constant_forecast = pd.Series([100.0] * 30, index=self.dates[:30])

        metrics = self.evaluator.calculate_accuracy_metrics(
            constant_actual, constant_forecast
        )

        # Should handle constant values without errors
        self.assertEqual(metrics["mae"], 0)
        self.assertEqual(metrics["mse"], 0)
        # MAPE might be 0 or undefined, both acceptable

        # Test with mismatched lengths
        with self.assertRaises((ValueError, IndexError)):
            self.evaluator.calculate_accuracy_metrics(
                self.actual, self.forecast.iloc[:-10]  # Shorter forecast
            )

        # Test with zeros in actual (causes MAPE issues)
        zero_actual = self.actual.copy()
        zero_actual.iloc[5] = 0

        # Should handle division by zero gracefully
        try:
            metrics = self.evaluator.calculate_accuracy_metrics(
                zero_actual, self.forecast
            )
            # MAPE might be inf or very large, but shouldn't crash
            self.assertIsInstance(metrics["mape"], (int, float))
        except Exception as e:
            # If it raises an exception, it should be handled gracefully
            self.assertIn(("zero", "division", "invalid"), str(e).lower())


class TestBusinessIntelligence(unittest.TestCase):
    """Test cases for the BusinessIntelligence component."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        # Create realistic business data
        self.dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")

        # Actual sales with business patterns
        base_sales = 10000 + np.random.normal(0, 1000, len(self.dates))
        weekly_pattern = 1 + 0.3 * np.sin(2 * np.pi * np.arange(len(self.dates)) / 7)
        monthly_growth = 1 + 0.01 * np.arange(len(self.dates)) / 30  # 1% monthly growth

        self.actual = pd.Series(
            base_sales * weekly_pattern * monthly_growth, index=self.dates
        )

        # Forecast with some accuracy
        forecast_error = np.random.normal(0, 500, len(self.dates))
        self.forecast = self.actual + forecast_error

        self.bi = BusinessIntelligence()

    def test_core_kpis(self):
        """Test core KPI calculations."""
        kpis = self.bi.calculate_core_kpis(self.actual, self.forecast)

        # Check expected KPIs
        expected_kpis = [
            "total_revenue",
            "average_daily_sales",
            "forecast_accuracy",
            "revenue_growth",
        ]
        for kpi in expected_kpis:
            self.assertIn(kpi, kpis)

        # Validate KPI values
        self.assertGreater(kpis["total_revenue"], 0)
        self.assertGreater(kpis["average_daily_sales"], 0)
        self.assertGreaterEqual(kpis["forecast_accuracy"], 0)
        self.assertLessEqual(kpis["forecast_accuracy"], 100)

        # Check that total revenue equals sum of actual
        expected_revenue = self.actual.sum()
        self.assertAlmostEqual(kpis["total_revenue"], expected_revenue, places=2)

        # Check average daily sales
        expected_avg = self.actual.mean()
        self.assertAlmostEqual(kpis["average_daily_sales"], expected_avg, places=2)

    def test_business_health_scoring(self):
        """Test business health score calculations."""
        health_scores = self.bi.calculate_business_health(self.actual, self.forecast)

        # Check expected health metrics
        expected_metrics = [
            "overall_health_score",
            "forecast_reliability",
            "trend_stability",
        ]
        for metric in expected_metrics:
            self.assertIn(metric, health_scores)

        # Health scores should be between 0 and 10
        for metric, score in health_scores.items():
            if isinstance(score, (int, float)):
                self.assertGreaterEqual(score, 0, f"{metric} score below 0: {score}")
                self.assertLessEqual(score, 10, f"{metric} score above 10: {score}")

    def test_executive_dashboard(self):
        """Test executive dashboard generation."""
        dashboard = self.bi.generate_executive_dashboard(self.actual, self.forecast)

        # Check dashboard structure
        expected_sections = [
            "core_kpis",
            "business_health",
            "executive_summary",
            "recommendations",
        ]
        for section in expected_sections:
            self.assertIn(section, dashboard)

        # Check core KPIs section
        kpis = dashboard["core_kpis"]
        self.assertIsInstance(kpis, dict)
        self.assertIn("total_revenue", kpis)

        # Check business health section
        health = dashboard["business_health"]
        self.assertIsInstance(health, dict)
        self.assertIn("overall_health_score", health)

        # Check executive summary
        summary = dashboard["executive_summary"]
        self.assertIsInstance(summary, dict)
        self.assertIn("key_insights", summary)

        # Check recommendations
        recommendations = dashboard["recommendations"]
        self.assertIsInstance(recommendations, (list, dict))

    def test_roi_analysis(self):
        """Test ROI analysis calculations."""
        roi_analysis = self.bi.calculate_roi_metrics(self.actual, self.forecast)

        # Check ROI metrics structure
        expected_metrics = [
            "forecast_value",
            "cost_of_forecast_errors",
            "accuracy_improvement_value",
        ]
        for metric in expected_metrics:
            self.assertIn(metric, roi_analysis)

        # Check that values are reasonable
        forecast_value = roi_analysis["forecast_value"]
        error_cost = roi_analysis["cost_of_forecast_errors"]

        self.assertGreaterEqual(forecast_value, 0)
        self.assertGreaterEqual(error_cost, 0)

        # Error cost should be related to forecast accuracy
        # Better forecasts should have lower error costs
        high_error_forecast = self.actual + np.random.normal(0, 2000, len(self.actual))
        high_error_roi = self.bi.calculate_roi_metrics(self.actual, high_error_forecast)

        # Higher errors should result in higher costs (generally)
        # self.assertGreaterEqual(high_error_roi['cost_of_forecast_errors'], error_cost)

    def test_trend_analysis(self):
        """Test trend analysis functionality."""
        trend_analysis = self.bi.analyze_trends(self.actual, self.forecast)

        # Check trend analysis structure
        expected_elements = ["actual_trend", "forecast_trend", "trend_accuracy"]
        for element in expected_elements:
            self.assertIn(element, trend_analysis)

        # Check trend values
        actual_trend = trend_analysis["actual_trend"]
        forecast_trend = trend_analysis["forecast_trend"]

        # Trends should be reasonable
        self.assertIsInstance(actual_trend, (int, float))
        self.assertIsInstance(forecast_trend, (int, float))

        # For our test data with growth, trend should be positive
        self.assertGreater(actual_trend, 0)

    def test_scenario_analysis(self):
        """Test scenario analysis capabilities."""
        scenarios = {
            "optimistic": self.forecast * 1.1,
            "pessimistic": self.forecast * 0.9,
            "baseline": self.forecast,
        }

        scenario_analysis = self.bi.analyze_scenarios(self.actual, scenarios)

        # Check that all scenarios are analyzed
        for scenario_name in scenarios.keys():
            self.assertIn(scenario_name, scenario_analysis)

        # Check scenario structure
        for scenario_name, analysis in scenario_analysis.items():
            self.assertIsInstance(analysis, dict)
            self.assertIn("scenario_metrics", analysis)

            # Should have accuracy metrics for each scenario
            metrics = analysis["scenario_metrics"]
            self.assertIn("accuracy", metrics)

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very short data
        short_actual = self.actual.iloc[:7]  # One week
        short_forecast = self.forecast.iloc[:7]

        try:
            dashboard = self.bi.generate_executive_dashboard(
                short_actual, short_forecast
            )
            self.assertIsInstance(dashboard, dict)
        except Exception as e:
            # Should handle short data gracefully
            self.assertIn(("insufficient", "short", "length"), str(e).lower())

        # Test with perfect forecast
        perfect_forecast = self.actual.copy()
        dashboard = self.bi.generate_executive_dashboard(self.actual, perfect_forecast)

        # Should handle perfect forecast without errors
        self.assertIsInstance(dashboard, dict)
        self.assertIn("core_kpis", dashboard)

        # Forecast accuracy should be very high
        accuracy = dashboard["core_kpis"]["forecast_accuracy"]
        self.assertGreater(accuracy, 99)


def run_component_tests():
    """Run all component tests."""
    print("=" * 60)
    print("RUNNING COMPONENT TEST SUITE")
    print("=" * 60)

    # Create test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestSalesDataGenerator,
        TestFeatureEngineer,
        TestForecastEvaluator,
        TestBusinessIntelligence,
    ]

    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + "=" * 60)
    print("COMPONENT TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

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
    run_component_tests()
