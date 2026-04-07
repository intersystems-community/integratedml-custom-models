"""
Advanced Forecast Evaluation and Business Analytics.

This module provides comprehensive forecast evaluation metrics and business-oriented
analytics for the hybrid sales forecasting system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings
from scipy import stats
import json


class ForecastEvaluator:
    """
    Advanced forecast evaluation with business-oriented metrics and insights.

    This component provides:
    - Comprehensive forecast accuracy metrics
    - Business impact analysis and ROI calculations
    - Time-based performance analytics
    - Scenario analysis and what-if modeling
    - Model interpretability and feature importance
    - Business intelligence and actionable insights
    """

    def __init__(
        self,
        business_metrics: List[str] = None,
        confidence_levels: List[float] = None,
        seasonality_periods: List[int] = None,
        cost_parameters: Dict[str, float] = None,
    ):
        """
        Initialize the forecast evaluator.

        Parameters:
        -----------
        business_metrics : list of str, optional
            Business metrics to calculate
        confidence_levels : list of float, optional
            Confidence levels for intervals
        seasonality_periods : list of int, optional
            Seasonal periods for analysis
        cost_parameters : dict, optional
            Business cost parameters for ROI analysis
        """
        self.business_metrics = business_metrics or [
            "revenue_impact",
            "inventory_optimization",
            "stockout_cost",
            "holding_cost",
            "forecast_value_added",
        ]
        self.confidence_levels = confidence_levels or [0.68, 0.95, 0.99]
        self.seasonality_periods = seasonality_periods or [7, 30, 365]

        # Default cost parameters for business analysis
        self.cost_parameters = cost_parameters or {
            "stockout_cost_per_unit": 5.0,
            "holding_cost_per_unit": 0.5,
            "ordering_cost": 100.0,
            "revenue_per_unit": 20.0,
            "forecast_cost_per_period": 10.0,
        }

        # Evaluation results
        self.accuracy_metrics_ = {}
        self.business_insights_ = {}
        self.performance_analysis_ = {}
        self.scenario_results_ = {}
        self.is_evaluated_ = False

    def evaluate_forecast(
        self,
        actual: pd.Series,
        forecast: pd.Series,
        forecast_intervals: Optional[Dict[str, pd.Series]] = None,
        external_factors: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive forecast evaluation.

        Parameters:
        -----------
        actual : Series
            Actual values
        forecast : Series
            Forecast values
        forecast_intervals : dict, optional
            Forecast confidence intervals
        external_factors : DataFrame, optional
            External factors for analysis

        Returns:
        --------
        evaluation_results : dict
            Comprehensive evaluation results
        """
        print("Starting comprehensive forecast evaluation...")

        # Align data
        common_index = actual.index.intersection(forecast.index)
        actual_aligned = actual.loc[common_index]
        forecast_aligned = forecast.loc[common_index]

        if len(common_index) == 0:
            raise ValueError("No overlapping time periods between actual and forecast")

        # Calculate accuracy metrics
        self.accuracy_metrics_ = self._calculate_accuracy_metrics(
            actual_aligned, forecast_aligned, forecast_intervals
        )

        # Calculate business insights
        self.business_insights_ = self._calculate_business_insights(
            actual_aligned, forecast_aligned, external_factors
        )

        # Performance analysis
        self.performance_analysis_ = self._analyze_performance_patterns(
            actual_aligned, forecast_aligned
        )

        # Generate actionable insights
        actionable_insights = self._generate_actionable_insights()

        self.is_evaluated_ = True

        results = {
            "accuracy_metrics": self.accuracy_metrics_,
            "business_insights": self.business_insights_,
            "performance_analysis": self.performance_analysis_,
            "actionable_insights": actionable_insights,
            "evaluation_summary": self._create_evaluation_summary(),
        }

        print(
            f"Evaluation complete. Overall MAPE: {self.accuracy_metrics_['mape']:.2f}%"
        )

        return results

    def analyze_business_scenarios(
        self, forecast: pd.Series, scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze different business scenarios and their impact.

        Parameters:
        -----------
        forecast : Series
            Base forecast
        scenarios : dict
            Business scenarios to analyze

        Returns:
        --------
        scenario_results : dict
            Results for each scenario
        """
        results = {}

        for scenario_name, scenario_params in scenarios.items():
            scenario_result = self._evaluate_scenario(forecast, scenario_params)
            results[scenario_name] = scenario_result

        self.scenario_results_ = results
        return results

    def calculate_forecast_value_added(
        self, forecast: pd.Series, naive_forecast: pd.Series, actual: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate Forecast Value Added (FVA) metrics.

        Parameters:
        -----------
        forecast : Series
            Model forecast
        naive_forecast : Series
            Naive forecast (baseline)
        actual : Series
            Actual values

        Returns:
        --------
        fva_metrics : dict
            Forecast Value Added metrics
        """
        # Align data
        common_index = forecast.index.intersection(naive_forecast.index).intersection(
            actual.index
        )
        forecast_aligned = forecast.loc[common_index]
        naive_aligned = naive_forecast.loc[common_index]
        actual_aligned = actual.loc[common_index]

        # Calculate errors
        model_errors = np.abs(actual_aligned - forecast_aligned)
        naive_errors = np.abs(actual_aligned - naive_aligned)

        # FVA calculation
        fva = (naive_errors.mean() - model_errors.mean()) / naive_errors.mean() * 100

        # Additional FVA metrics
        fva_median = (
            (naive_errors.median() - model_errors.median())
            / naive_errors.median()
            * 100
        )

        # Percentage of periods where model beats naive
        periods_better = (model_errors < naive_errors).mean() * 100

        return {
            "forecast_value_added_mean": fva,
            "forecast_value_added_median": fva_median,
            "periods_better_than_naive": periods_better,
            "relative_improvement": fva / 100,
        }

    def generate_business_report(self) -> str:
        """
        Generate a comprehensive business-oriented report.

        Returns:
        --------
        report : str
            Business report text
        """
        if not self.is_evaluated_:
            return "No evaluation results available. Run evaluate_forecast() first."

        report_sections = []

        # Executive Summary
        report_sections.append("# SALES FORECAST EVALUATION REPORT")
        report_sections.append("=" * 50)
        report_sections.append("")

        summary = self._create_evaluation_summary()
        report_sections.append("## Executive Summary")
        report_sections.append(f"Overall Forecast Accuracy: {summary['overall_grade']}")
        report_sections.append(
            f"Mean Absolute Percentage Error: {self.accuracy_metrics_['mape']:.2f}%"
        )
        report_sections.append(
            f"Business Impact Score: {summary['business_impact_score']:.2f}/10"
        )
        report_sections.append("")

        # Key Findings
        report_sections.append("## Key Findings")
        for insight in summary["key_insights"]:
            report_sections.append(f"• {insight}")
        report_sections.append("")

        # Accuracy Analysis
        report_sections.append("## Forecast Accuracy Analysis")
        report_sections.append(
            f"Mean Absolute Error (MAE): {self.accuracy_metrics_['mae']:.2f}"
        )
        report_sections.append(
            f"Root Mean Square Error (RMSE): {self.accuracy_metrics_['rmse']:.2f}"
        )
        report_sections.append(
            f"Mean Absolute Percentage Error (MAPE): {self.accuracy_metrics_['mape']:.2f}%"
        )
        report_sections.append(f"Forecast Bias: {self.accuracy_metrics_['bias']:.2f}")
        report_sections.append(
            f"Tracking Signal: {self.accuracy_metrics_['tracking_signal']:.2f}"
        )
        report_sections.append("")

        # Business Impact
        report_sections.append("## Business Impact Analysis")
        if "revenue_impact" in self.business_insights_:
            revenue_impact = self.business_insights_["revenue_impact"]
            report_sections.append(
                f"Estimated Revenue Impact: ${revenue_impact['total_impact']:,.2f}"
            )
            report_sections.append(
                f"Revenue at Risk: ${revenue_impact['revenue_at_risk']:,.2f}"
            )

        if "inventory_optimization" in self.business_insights_:
            inventory = self.business_insights_["inventory_optimization"]
            report_sections.append(
                f"Inventory Optimization Savings: ${inventory['potential_savings']:,.2f}"
            )
            report_sections.append(
                f"Stockout Risk Reduction: {inventory['stockout_reduction']:.1f}%"
            )

        report_sections.append("")

        # Recommendations
        report_sections.append("## Recommendations")
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            report_sections.append(f"{i}. {rec}")

        return "\n".join(report_sections)

    def plot_evaluation_dashboard(
        self,
        actual: pd.Series,
        forecast: pd.Series,
        figsize: Tuple[int, int] = (20, 12),
    ) -> None:
        """
        Create comprehensive evaluation dashboard.

        Parameters:
        -----------
        actual : Series
            Actual values
        forecast : Series
            Forecast values
        figsize : tuple
            Figure size
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("Matplotlib and seaborn required for plotting")
            return

        if not self.is_evaluated_:
            print("Run evaluate_forecast() first")
            return

        # Create dashboard
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle(
            "Sales Forecast Evaluation Dashboard", fontsize=16, fontweight="bold"
        )

        # 1. Actual vs Forecast
        axes[0, 0].plot(actual, label="Actual", linewidth=2)
        axes[0, 0].plot(forecast, label="Forecast", linewidth=2, alpha=0.8)
        axes[0, 0].set_title("Actual vs Forecast")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residuals
        residuals = actual - forecast
        axes[0, 1].plot(residuals, color="red", alpha=0.7)
        axes[0, 1].axhline(y=0, color="black", linestyle="--")
        axes[0, 1].set_title("Forecast Residuals")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Residual Distribution
        axes[0, 2].hist(residuals.dropna(), bins=30, alpha=0.7, color="skyblue")
        axes[0, 2].set_title("Residual Distribution")
        axes[0, 2].axvline(x=0, color="red", linestyle="--")

        # 4. MAPE by Period
        if "mape_by_period" in self.performance_analysis_:
            mape_series = self.performance_analysis_["mape_by_period"]
            axes[1, 0].plot(mape_series, color="orange")
            axes[1, 0].set_title("MAPE by Period")
            axes[1, 0].set_ylabel("MAPE (%)")
            axes[1, 0].grid(True, alpha=0.3)

        # 5. Accuracy Metrics Bar Chart
        metrics = ["MAE", "RMSE", "MAPE"]
        values = [
            self.accuracy_metrics_["mae"],
            self.accuracy_metrics_["rmse"],
            self.accuracy_metrics_["mape"],
        ]
        axes[1, 1].bar(metrics, values, color=["skyblue", "lightgreen", "coral"])
        axes[1, 1].set_title("Accuracy Metrics")

        # 6. Forecast vs Actual Scatter
        axes[1, 2].scatter(actual, forecast, alpha=0.6)
        min_val = min(actual.min(), forecast.min())
        max_val = max(actual.max(), forecast.max())
        axes[1, 2].plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8)
        axes[1, 2].set_xlabel("Actual")
        axes[1, 2].set_ylabel("Forecast")
        axes[1, 2].set_title("Forecast vs Actual")

        # 7. Cumulative Error
        cumulative_error = residuals.cumsum()
        axes[2, 0].plot(cumulative_error, color="purple")
        axes[2, 0].set_title("Cumulative Forecast Error")
        axes[2, 0].axhline(y=0, color="black", linestyle="--")
        axes[2, 0].grid(True, alpha=0.3)

        # 8. Business Impact
        if "revenue_impact" in self.business_insights_:
            categories = ["Revenue Impact", "Cost Savings", "Risk Reduction"]
            values = [
                self.business_insights_["revenue_impact"]["total_impact"],
                self.business_insights_.get("inventory_optimization", {}).get(
                    "potential_savings", 0
                ),
                abs(self.business_insights_["revenue_impact"]["revenue_at_risk"]),
            ]
            axes[2, 1].bar(categories, values, color=["green", "blue", "orange"])
            axes[2, 1].set_title("Business Impact ($)")
            axes[2, 1].tick_params(axis="x", rotation=45)

        # 9. Performance Summary
        axes[2, 2].axis("off")
        summary_text = f"""
        PERFORMANCE SUMMARY
        
        Overall Grade: {self._create_evaluation_summary()['overall_grade']}
        
        Accuracy:
        • MAPE: {self.accuracy_metrics_['mape']:.2f}%
        • MAE: {self.accuracy_metrics_['mae']:.2f}
        • RMSE: {self.accuracy_metrics_['rmse']:.2f}
        
        Business Impact:
        • Revenue Impact: ${self.business_insights_.get('revenue_impact', {}).get('total_impact', 0):,.0f}
        • Forecast Bias: {self.accuracy_metrics_['bias']:.2f}
        """
        axes[2, 2].text(
            0.1,
            0.9,
            summary_text,
            transform=axes[2, 2].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

        plt.tight_layout()
        plt.show()

    def _calculate_accuracy_metrics(
        self,
        actual: pd.Series,
        forecast: pd.Series,
        forecast_intervals: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Calculate comprehensive accuracy metrics."""
        # Remove any NaN values
        mask = ~(actual.isna() | forecast.isna())
        actual_clean = actual[mask]
        forecast_clean = forecast[mask]

        if len(actual_clean) == 0:
            return {}

        # Basic error metrics
        errors = actual_clean - forecast_clean
        abs_errors = np.abs(errors)
        squared_errors = errors**2

        # Standard metrics
        mae = abs_errors.mean()
        rmse = np.sqrt(squared_errors.mean())
        mse = squared_errors.mean()

        # Percentage errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            percentage_errors = (errors / actual_clean) * 100
            abs_percentage_errors = np.abs(percentage_errors)

        mape = abs_percentage_errors.mean()
        median_ape = abs_percentage_errors.median()

        # Bias and tracking signal
        bias = errors.mean()
        tracking_signal = bias / mae if mae > 0 else 0

        # Theil's U statistic
        theil_u = self._calculate_theil_u(actual_clean, forecast_clean)

        # Directional accuracy
        actual_direction = np.diff(actual_clean) > 0
        forecast_direction = np.diff(forecast_clean) > 0
        directional_accuracy = (actual_direction == forecast_direction).mean() * 100

        # R-squared
        r_squared = 1 - (
            np.sum(squared_errors) / np.sum((actual_clean - actual_clean.mean()) ** 2)
        )

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "mse": mse,
            "mape": mape,
            "median_ape": median_ape,
            "bias": bias,
            "tracking_signal": tracking_signal,
            "theil_u": theil_u,
            "directional_accuracy": directional_accuracy,
            "r_squared": max(0, r_squared),  # Ensure non-negative
            "forecast_periods": len(actual_clean),
        }

        # Interval-based metrics if available
        if forecast_intervals:
            interval_metrics = self._calculate_interval_metrics(
                actual_clean, forecast_intervals
            )
            metrics.update(interval_metrics)

        return metrics

    def _calculate_business_insights(
        self,
        actual: pd.Series,
        forecast: pd.Series,
        external_factors: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Calculate business-oriented insights."""
        insights = {}

        # Revenue impact analysis
        revenue_impact = self._calculate_revenue_impact(actual, forecast)
        insights["revenue_impact"] = revenue_impact

        # Inventory optimization
        inventory_analysis = self._calculate_inventory_optimization(actual, forecast)
        insights["inventory_optimization"] = inventory_analysis

        # Forecast reliability
        reliability = self._assess_forecast_reliability(actual, forecast)
        insights["forecast_reliability"] = reliability

        # Seasonal performance
        seasonal_performance = self._analyze_seasonal_performance(actual, forecast)
        insights["seasonal_performance"] = seasonal_performance

        # Risk assessment
        risk_assessment = self._calculate_business_risk(actual, forecast)
        insights["risk_assessment"] = risk_assessment

        return insights

    def _analyze_performance_patterns(
        self, actual: pd.Series, forecast: pd.Series
    ) -> Dict[str, Any]:
        """Analyze performance patterns over time."""
        analysis = {}

        # Performance by time period
        errors = np.abs(actual - forecast)
        percentage_errors = np.abs((actual - forecast) / actual) * 100

        # MAPE by period
        analysis["mape_by_period"] = percentage_errors

        # Performance by day of week (if daily data)
        if hasattr(actual.index, "dayofweek"):
            dow_performance = percentage_errors.groupby(actual.index.dayofweek).mean()
            analysis["performance_by_day_of_week"] = dow_performance

        # Performance by month
        if hasattr(actual.index, "month"):
            monthly_performance = percentage_errors.groupby(actual.index.month).mean()
            analysis["performance_by_month"] = monthly_performance

        # Trend in performance
        window_size = min(30, len(errors) // 4)
        if window_size > 1:
            rolling_mape = percentage_errors.rolling(window=window_size).mean()
            analysis["performance_trend"] = rolling_mape

        # Performance distribution
        analysis["performance_quartiles"] = {
            "q25": percentage_errors.quantile(0.25),
            "q50": percentage_errors.quantile(0.50),
            "q75": percentage_errors.quantile(0.75),
            "q90": percentage_errors.quantile(0.90),
            "q95": percentage_errors.quantile(0.95),
        }

        return analysis

    def _calculate_revenue_impact(
        self, actual: pd.Series, forecast: pd.Series
    ) -> Dict[str, float]:
        """Calculate revenue impact of forecast accuracy."""
        errors = (
            forecast - actual
        )  # Positive = over-forecast, Negative = under-forecast

        # Revenue per unit
        revenue_per_unit = self.cost_parameters["revenue_per_unit"]

        # Over-forecasting impact (holding costs)
        over_forecast = errors[errors > 0].sum()
        holding_cost_total = (
            over_forecast * self.cost_parameters["holding_cost_per_unit"]
        )

        # Under-forecasting impact (stockout costs)
        under_forecast = abs(errors[errors < 0].sum())
        stockout_cost_total = (
            under_forecast * self.cost_parameters["stockout_cost_per_unit"]
        )

        # Total revenue impact
        total_impact = -(holding_cost_total + stockout_cost_total)

        # Revenue at risk (potential lost sales)
        revenue_at_risk = under_forecast * revenue_per_unit

        # Opportunity cost
        opportunity_cost = (
            over_forecast * revenue_per_unit * 0.1
        )  # 10% opportunity cost

        return {
            "total_impact": total_impact,
            "holding_cost": holding_cost_total,
            "stockout_cost": stockout_cost_total,
            "revenue_at_risk": revenue_at_risk,
            "opportunity_cost": opportunity_cost,
            "over_forecast_units": over_forecast,
            "under_forecast_units": under_forecast,
        }

    def _calculate_inventory_optimization(
        self, actual: pd.Series, forecast: pd.Series
    ) -> Dict[str, float]:
        """Calculate inventory optimization metrics."""
        # Safety stock reduction potential
        forecast_std = forecast.std()
        actual_std = actual.std()

        # Improved forecast accuracy reduces safety stock needs
        if actual_std > 0:
            accuracy_improvement = 1 - (forecast_std / actual_std)
            safety_stock_reduction = max(
                0, accuracy_improvement * 0.5
            )  # Conservative estimate
        else:
            safety_stock_reduction = 0

        # Average inventory level
        avg_forecast = forecast.mean()

        # Potential savings from reduced safety stock
        safety_stock_savings = (
            avg_forecast
            * safety_stock_reduction
            * self.cost_parameters["holding_cost_per_unit"]
        )

        # Stockout reduction
        errors = actual - forecast
        stockout_periods = (errors > 0).sum()
        total_periods = len(errors)
        stockout_rate = stockout_periods / total_periods if total_periods > 0 else 0

        # Baseline stockout rate (assume 10% without forecasting)
        baseline_stockout_rate = 0.10
        stockout_reduction = max(
            0, (baseline_stockout_rate - stockout_rate) / baseline_stockout_rate * 100
        )

        return {
            "potential_savings": safety_stock_savings,
            "safety_stock_reduction": safety_stock_reduction * 100,
            "stockout_reduction": stockout_reduction,
            "current_stockout_rate": stockout_rate * 100,
            "inventory_turnover_improvement": (
                accuracy_improvement * 100 if "accuracy_improvement" in locals() else 0
            ),
        }

    def _assess_forecast_reliability(
        self, actual: pd.Series, forecast: pd.Series
    ) -> Dict[str, float]:
        """Assess forecast reliability metrics."""
        errors = actual - forecast
        abs_errors = np.abs(errors)

        # Consistency of errors
        error_std = errors.std()
        error_cv = (
            error_std / abs_errors.mean() if abs_errors.mean() > 0 else float("inf")
        )

        # Reliability score (lower CV = higher reliability)
        reliability_score = max(0, min(10, 10 * (1 - min(error_cv / 2, 1))))

        # Forecast stability (how much forecast changes between periods)
        forecast_changes = np.abs(np.diff(forecast))
        forecast_stability = (
            1 - (forecast_changes.std() / forecast.mean()) if forecast.mean() > 0 else 0
        )
        forecast_stability = max(0, min(1, forecast_stability))

        # Predictive power (correlation between forecast and actual)
        correlation = np.corrcoef(actual, forecast)[0, 1] if len(actual) > 1 else 0
        if np.isnan(correlation):
            correlation = 0

        return {
            "reliability_score": reliability_score,
            "error_consistency": 1 / (1 + error_cv),  # Higher is better
            "forecast_stability": forecast_stability,
            "predictive_power": max(0, correlation),
            "error_coefficient_of_variation": error_cv,
        }

    def _analyze_seasonal_performance(
        self, actual: pd.Series, forecast: pd.Series
    ) -> Dict[str, Any]:
        """Analyze performance across different seasonal periods."""
        seasonal_analysis = {}

        for period in self.seasonality_periods:
            if len(actual) >= 2 * period:
                # Group by seasonal period
                seasonal_groups = {}
                for i in range(period):
                    mask = np.arange(len(actual)) % period == i
                    if mask.sum() > 0:
                        actual_seasonal = actual.iloc[mask]
                        forecast_seasonal = forecast.iloc[mask]

                        # Calculate MAPE for this seasonal period
                        if len(actual_seasonal) > 0:
                            errors = (
                                np.abs(
                                    (actual_seasonal - forecast_seasonal)
                                    / actual_seasonal
                                )
                                * 100
                            )
                            seasonal_groups[i] = {
                                "mape": errors.mean(),
                                "periods": len(actual_seasonal),
                            }

                seasonal_analysis[f"period_{period}"] = seasonal_groups

        return seasonal_analysis

    def _calculate_business_risk(
        self, actual: pd.Series, forecast: pd.Series
    ) -> Dict[str, float]:
        """Calculate business risk metrics."""
        errors = actual - forecast

        # Value at Risk (VaR) - potential losses
        var_95 = np.percentile(errors, 5)  # 5th percentile (95% VaR)
        var_99 = np.percentile(errors, 1)  # 1st percentile (99% VaR)

        # Maximum loss potential
        max_loss = errors.min() * self.cost_parameters["revenue_per_unit"]

        # Volatility risk
        forecast_volatility = (
            forecast.std() / forecast.mean() if forecast.mean() > 0 else 0
        )
        actual_volatility = actual.std() / actual.mean() if actual.mean() > 0 else 0

        volatility_risk = abs(forecast_volatility - actual_volatility)

        # Tail risk (extreme events)
        extreme_errors = errors[np.abs(errors) > 2 * errors.std()]
        tail_risk = len(extreme_errors) / len(errors) * 100

        return {
            "value_at_risk_95": var_95,
            "value_at_risk_99": var_99,
            "maximum_loss_potential": max_loss,
            "volatility_risk": volatility_risk,
            "tail_risk_percentage": tail_risk,
            "risk_score": min(
                10, tail_risk + volatility_risk * 10
            ),  # Combined risk score
        }

    def _calculate_theil_u(self, actual: np.ndarray, forecast: np.ndarray) -> float:
        """Calculate Theil's U statistic."""
        try:
            numerator = np.sqrt(np.mean((forecast - actual) ** 2))
            denominator = np.sqrt(np.mean(actual**2)) + np.sqrt(np.mean(forecast**2))

            if denominator == 0:
                return float("inf")

            return numerator / denominator
        except:
            return float("inf")

    def _calculate_interval_metrics(
        self, actual: pd.Series, intervals: Dict
    ) -> Dict[str, float]:
        """Calculate prediction interval metrics."""
        interval_metrics = {}

        for conf_level, interval_data in intervals.items():
            if "lower" in interval_data and "upper" in interval_data:
                lower = interval_data["lower"]
                upper = interval_data["upper"]

                # Coverage probability
                coverage = ((actual >= lower) & (actual <= upper)).mean() * 100

                # Average interval width
                avg_width = (upper - lower).mean()

                interval_metrics[f"coverage_{conf_level}"] = coverage
                interval_metrics[f"avg_width_{conf_level}"] = avg_width

        return interval_metrics

    def _evaluate_scenario(
        self, forecast: pd.Series, scenario_params: Dict[str, float]
    ) -> Dict[str, float]:
        """Evaluate a specific business scenario."""
        # Apply scenario modifications to forecast
        modified_forecast = forecast.copy()

        if "demand_multiplier" in scenario_params:
            modified_forecast *= scenario_params["demand_multiplier"]

        if "seasonal_boost" in scenario_params:
            # Apply seasonal boost (simplified)
            seasonal_effect = scenario_params["seasonal_boost"]
            modified_forecast *= 1 + seasonal_effect

        # Calculate business metrics for this scenario
        total_demand = modified_forecast.sum()
        revenue_estimate = total_demand * self.cost_parameters["revenue_per_unit"]

        # Inventory requirements
        safety_stock = modified_forecast.std() * 2  # 2 standard deviations
        holding_cost = safety_stock * self.cost_parameters["holding_cost_per_unit"]

        return {
            "total_demand": total_demand,
            "revenue_estimate": revenue_estimate,
            "safety_stock_needed": safety_stock,
            "holding_cost": holding_cost,
            "profit_estimate": revenue_estimate - holding_cost,
        }

    def _generate_actionable_insights(self) -> List[str]:
        """Generate actionable business insights."""
        insights = []

        if not self.is_evaluated_:
            return insights

        # Accuracy insights
        mape = self.accuracy_metrics_["mape"]
        if mape < 5:
            insights.append(
                "Excellent forecast accuracy achieved. Consider using this model for critical business decisions."
            )
        elif mape < 10:
            insights.append(
                "Good forecast accuracy. Model is suitable for operational planning."
            )
        elif mape < 20:
            insights.append(
                "Moderate forecast accuracy. Consider improving feature engineering or model tuning."
            )
        else:
            insights.append(
                "Poor forecast accuracy. Recommend model revision or additional data sources."
            )

        # Bias insights
        bias = self.accuracy_metrics_["bias"]
        if abs(bias) > self.accuracy_metrics_["mae"] * 0.5:
            if bias > 0:
                insights.append(
                    "Forecast shows significant positive bias (over-forecasting). Review demand assumptions."
                )
            else:
                insights.append(
                    "Forecast shows significant negative bias (under-forecasting). May lead to stockouts."
                )

        # Business impact insights
        if "revenue_impact" in self.business_insights_:
            revenue_impact = self.business_insights_["revenue_impact"]["total_impact"]
            if revenue_impact < -1000:
                insights.append(
                    f"Forecast errors creating significant business impact (${abs(revenue_impact):,.0f}). Prioritize accuracy improvements."
                )

        # Seasonal performance insights
        if "seasonal_performance" in self.business_insights_:
            insights.append(
                "Monitor seasonal performance patterns to identify improvement opportunities."
            )

        # Risk insights
        if "risk_assessment" in self.business_insights_:
            risk_score = self.business_insights_["risk_assessment"]["risk_score"]
            if risk_score > 7:
                insights.append(
                    "High forecast risk detected. Implement robust contingency planning."
                )

        return insights

    def _create_evaluation_summary(self) -> Dict[str, Any]:
        """Create evaluation summary."""
        if not self.is_evaluated_:
            return {}

        # Overall grade based on MAPE
        mape = self.accuracy_metrics_["mape"]
        if mape < 5:
            grade = "A"
        elif mape < 10:
            grade = "B"
        elif mape < 20:
            grade = "C"
        elif mape < 30:
            grade = "D"
        else:
            grade = "F"

        # Business impact score
        revenue_impact = self.business_insights_.get("revenue_impact", {}).get(
            "total_impact", 0
        )
        reliability_score = self.business_insights_.get("forecast_reliability", {}).get(
            "reliability_score", 5
        )

        business_impact_score = min(
            10, max(0, reliability_score - abs(revenue_impact) / 10000)
        )

        # Key insights
        key_insights = self._generate_actionable_insights()[:3]  # Top 3 insights

        return {
            "overall_grade": grade,
            "business_impact_score": business_impact_score,
            "forecast_periods": self.accuracy_metrics_["forecast_periods"],
            "key_insights": key_insights,
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate specific recommendations for improvement."""
        recommendations = []

        if not self.is_evaluated_:
            return recommendations

        mape = self.accuracy_metrics_["mape"]
        bias = self.accuracy_metrics_["bias"]

        # Accuracy recommendations
        if mape > 15:
            recommendations.append(
                "Improve model accuracy by adding more relevant features or trying different algorithms"
            )

        # Bias recommendations
        if abs(bias) > self.accuracy_metrics_["mae"] * 0.3:
            recommendations.append(
                "Address forecast bias through calibration or model adjustment"
            )

        # Business recommendations
        if "revenue_impact" in self.business_insights_:
            revenue_impact = self.business_insights_["revenue_impact"]["total_impact"]
            if revenue_impact < -5000:
                recommendations.append(
                    "Implement forecast accuracy improvement initiative due to significant business impact"
                )

        # Risk recommendations
        if "risk_assessment" in self.business_insights_:
            tail_risk = self.business_insights_["risk_assessment"][
                "tail_risk_percentage"
            ]
            if tail_risk > 10:
                recommendations.append(
                    "Develop contingency plans for extreme demand scenarios"
                )

        recommendations.append("Establish regular forecast review and updating process")
        recommendations.append("Implement forecast performance monitoring dashboard")

        return recommendations

    def calculate_accuracy_metrics(self, actual: pd.Series, forecast: pd.Series) -> dict:
        if len(actual) != len(forecast):
            raise ValueError(f"Length mismatch: actual={len(actual)}, forecast={len(forecast)}")
        import warnings
        mask = ~(actual.isna() | forecast.isna())
        a, f = actual[mask].values, forecast[mask].values
        if len(a) == 0:
            raise ValueError("No data after removing NaNs")
        errors = a - f
        abs_errors = np.abs(errors)
        mae = float(abs_errors.mean())
        mse = float((errors ** 2).mean())
        rmse = float(np.sqrt(mse))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pct_errors = np.abs(errors / np.where(a == 0, np.nan, a)) * 100
        mape = float(np.nanmean(pct_errors))
        smape = float((2 * abs_errors / (np.abs(a) + np.abs(f) + 1e-10) * 100).mean())
        ss_tot = float(((a - a.mean()) ** 2).sum())
        r2 = float(np.corrcoef(a, f)[0, 1] ** 2) if ss_tot > 0 else 1.0
        return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "smape": smape, "r2_score": r2}

    def calculate_business_insights(self, actual: pd.Series, forecast: pd.Series) -> dict:
        total_actual = float(actual.sum())
        total_forecast = float(forecast.sum())
        abs_error = abs(total_actual - total_forecast)
        pct_error = abs_error / total_actual * 100 if total_actual != 0 else 0.0
        bias = float((forecast - actual).mean())
        direction = "over_forecast" if bias > 0.01 else ("under_forecast" if bias < -0.01 else "unbiased")
        window = min(30, max(1, len(actual) // 4))
        return {
            "revenue_impact": {
                "total_actual": total_actual,
                "total_forecast": total_forecast,
                "absolute_error": abs_error,
                "percentage_error": pct_error,
            },
            "forecast_bias": {"mean_bias": bias, "bias_direction": direction},
            "accuracy_trends": {"rolling_mean_error": (forecast - actual).rolling(window).mean()},
        }

    def evaluate_forecast(self, actual: pd.Series, forecast: pd.Series, **kwargs) -> dict:
        metrics = self.calculate_accuracy_metrics(actual, forecast)
        insights = self.calculate_business_insights(actual, forecast)
        forecast_quality = float(np.clip(1.0 - metrics.get("mape", 100.0) / 100.0, 0.0, 1.0))
        confidence_level = float(np.clip(forecast_quality * 0.9 + 0.05, 0.0, 1.0))
        return {
            "accuracy_metrics": metrics,
            "business_insights": insights,
            "performance_analysis": {
                "forecast_quality": forecast_quality,
                "confidence_level": confidence_level,
            },
        }


def main():
    """Example usage of ForecastEvaluator."""
    print("=== Forecast Evaluator Demo ===")

    # Generate synthetic data for demonstration
    np.random.seed(42)

    # Create 100 days of actual and forecast data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

    # Actual sales with trend and seasonality
    trend = np.linspace(1000, 1200, 100)
    seasonal = 100 * np.sin(2 * np.pi * np.arange(100) / 7)  # Weekly seasonality
    noise = np.random.normal(0, 50, 100)
    actual_sales = trend + seasonal + noise

    # Forecast with some error
    forecast_error = np.random.normal(0, 30, 100)
    forecast_sales = actual_sales + forecast_error

    # Create Series
    actual = pd.Series(actual_sales, index=dates)
    forecast = pd.Series(forecast_sales, index=dates)

    print(f"Generated {len(actual)} days of sales data")
    print(f"Actual sales range: ${actual.min():.0f} - ${actual.max():.0f}")
    print(f"Forecast sales range: ${forecast.min():.0f} - ${forecast.max():.0f}")

    # Initialize evaluator
    evaluator = ForecastEvaluator(
        cost_parameters={
            "stockout_cost_per_unit": 5.0,
            "holding_cost_per_unit": 0.5,
            "revenue_per_unit": 25.0,
            "ordering_cost": 100.0,
        }
    )

    # Evaluate forecast
    print("\nEvaluating forecast performance...")
    results = evaluator.evaluate_forecast(actual, forecast)

    # Display results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    print(f"\nAccuracy Metrics:")
    accuracy = results["accuracy_metrics"]
    print(f"  MAPE: {accuracy['mape']:.2f}%")
    print(f"  MAE: {accuracy['mae']:.2f}")
    print(f"  RMSE: {accuracy['rmse']:.2f}")
    print(f"  Bias: {accuracy['bias']:.2f}")
    print(f"  Directional Accuracy: {accuracy['directional_accuracy']:.1f}%")

    print(f"\nBusiness Impact:")
    revenue_impact = results["business_insights"]["revenue_impact"]
    print(f"  Total Revenue Impact: ${revenue_impact['total_impact']:,.2f}")
    print(f"  Stockout Cost: ${revenue_impact['stockout_cost']:,.2f}")
    print(f"  Holding Cost: ${revenue_impact['holding_cost']:,.2f}")

    inventory = results["business_insights"]["inventory_optimization"]
    print(f"  Potential Savings: ${inventory['potential_savings']:,.2f}")
    print(f"  Stockout Reduction: {inventory['stockout_reduction']:.1f}%")

    print(f"\nOverall Assessment:")
    summary = results["evaluation_summary"]
    print(f"  Grade: {summary['overall_grade']}")
    print(f"  Business Impact Score: {summary['business_impact_score']:.1f}/10")

    print(f"\nKey Insights:")
    for insight in results["actionable_insights"][:3]:
        print(f"  • {insight}")

    # Generate business report
    print("\n" + "=" * 50)
    print("BUSINESS REPORT")
    print("=" * 50)
    report = evaluator.generate_business_report()
    print(report)

    print("\nForecast evaluation demonstration complete!")


if __name__ == "__main__":
    main()
