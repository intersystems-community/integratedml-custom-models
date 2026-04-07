"""
Business Intelligence and Analytics for Sales Forecasting.

This module provides advanced business intelligence capabilities, KPI tracking,
and executive-level analytics for the hybrid sales forecasting system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta, date
import warnings
import json
from collections import defaultdict


class BusinessIntelligence:
    """
    Business Intelligence and KPI analytics for sales forecasting.

    This component provides:
    - Executive dashboard metrics and KPIs
    - Business performance analytics
    - Strategic insights and recommendations
    - Comparative analysis across time periods
    - ROI and financial impact assessment
    - Market trend analysis
    - Operational efficiency metrics
    """

    def __init__(
        self,
        kpi_targets: Dict[str, float] = None,
        business_calendar: Dict[str, List[str]] = None,
        market_segments: List[str] = None,
        reporting_periods: List[str] = None,
    ):
        """
        Initialize Business Intelligence analytics.

        Parameters:
        -----------
        kpi_targets : dict, optional
            Target values for KPIs
        business_calendar : dict, optional
            Business calendar with important dates
        market_segments : list, optional
            Market segments for analysis
        reporting_periods : list, optional
            Standard reporting periods
        """
        self.kpi_targets = kpi_targets or {
            "revenue_growth_rate": 0.15,  # 15% growth target
            "forecast_accuracy": 0.95,  # 95% accuracy target
            "inventory_turnover": 12,  # 12 times per year
            "customer_satisfaction": 0.90,  # 90% satisfaction
            "cost_efficiency": 0.85,  # 85% efficiency target
        }

        self.business_calendar = business_calendar or {
            "holidays": ["2023-12-25", "2024-01-01", "2024-07-04"],
            "peak_seasons": ["2023-11-01:2023-12-31", "2024-06-01:2024-08-31"],
            "promotions": ["2023-11-24:2023-11-27", "2024-03-15:2024-03-22"],
        }

        self.market_segments = market_segments or [
            "Premium",
            "Standard",
            "Budget",
            "Enterprise",
            "Consumer",
        ]

        self.reporting_periods = reporting_periods or [
            "daily",
            "weekly",
            "monthly",
            "quarterly",
            "yearly",
        ]

        # Analytics results
        self.kpi_dashboard_ = {}
        self.performance_analytics_ = {}
        self.strategic_insights_ = {}
        self.competitive_analysis_ = {}
        self.is_analyzed_ = False

    def generate_executive_dashboard(
        self,
        sales_data: pd.Series,
        forecast_data: pd.Series,
        additional_metrics: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive executive dashboard.

        Parameters:
        -----------
        sales_data : Series
            Historical sales data
        forecast_data : Series
            Forecast data
        additional_metrics : dict, optional
            Additional business metrics

        Returns:
        --------
        dashboard : dict
            Executive dashboard metrics and insights
        """
        print("Generating executive dashboard...")

        # Core KPIs
        core_kpis = self._calculate_core_kpis(sales_data, forecast_data)

        # Performance trends
        performance_trends = self._analyze_performance_trends(sales_data, forecast_data)

        # Business health metrics
        health_metrics = self._calculate_business_health(sales_data, forecast_data)

        # Strategic recommendations
        strategic_recommendations = self._generate_strategic_recommendations(
            sales_data, forecast_data, core_kpis
        )

        # Risk indicators
        risk_indicators = self._assess_business_risks(sales_data, forecast_data)

        # Opportunity analysis
        opportunities = self._identify_opportunities(sales_data, forecast_data)

        dashboard = {
            "summary": {
                "report_date": datetime.now().strftime("%Y-%m-%d"),
                "data_period": f"{sales_data.index[0].strftime('%Y-%m-%d')} to {sales_data.index[-1].strftime('%Y-%m-%d')}",
                "total_periods": len(sales_data),
                "forecast_periods": len(forecast_data),
            },
            "core_kpis": core_kpis,
            "performance_trends": performance_trends,
            "business_health": health_metrics,
            "strategic_recommendations": strategic_recommendations,
            "risk_indicators": risk_indicators,
            "opportunities": opportunities,
            "executive_summary": self._create_executive_summary(
                core_kpis, health_metrics
            ),
        }

        self.kpi_dashboard_ = dashboard
        self.is_analyzed_ = True

        print("Executive dashboard generated successfully")

        return dashboard

    def analyze_market_performance(
        self,
        sales_data: pd.Series,
        market_data: Optional[pd.DataFrame] = None,
        competitor_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Analyze market performance and competitive position.

        Parameters:
        -----------
        sales_data : Series
            Sales performance data
        market_data : DataFrame, optional
            Market trend data
        competitor_data : DataFrame, optional
            Competitor performance data

        Returns:
        --------
        market_analysis : dict
            Market performance analysis
        """
        analysis = {}

        # Market share analysis
        if market_data is not None:
            market_share = self._calculate_market_share(sales_data, market_data)
            analysis["market_share"] = market_share

        # Competitive positioning
        if competitor_data is not None:
            competitive_position = self._analyze_competitive_position(
                sales_data, competitor_data
            )
            analysis["competitive_position"] = competitive_position

        # Market trend analysis
        market_trends = self._analyze_market_trends(sales_data)
        analysis["market_trends"] = market_trends

        # Growth opportunities
        growth_opportunities = self._identify_growth_opportunities(sales_data)
        analysis["growth_opportunities"] = growth_opportunities

        self.competitive_analysis_ = analysis

        return analysis

    def calculate_roi_metrics(
        self,
        sales_data: pd.Series,
        forecast_data: pd.Series,
        investment_costs: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate ROI and financial impact metrics.

        Parameters:
        -----------
        sales_data : Series
            Historical sales data
        forecast_data : Series
            Forecast data
        investment_costs : dict
            Investment costs for forecasting system

        Returns:
        --------
        roi_metrics : dict
            ROI and financial metrics
        """
        # Revenue calculations
        historical_revenue = sales_data.sum()
        forecast_revenue = forecast_data.sum()

        # Revenue improvement from better forecasting
        # Assume 2% revenue improvement from better inventory management
        revenue_improvement = historical_revenue * 0.02

        # Cost savings from improved forecasting
        inventory_savings = historical_revenue * 0.01  # 1% inventory cost savings
        stockout_reduction = historical_revenue * 0.005  # 0.5% stockout cost reduction

        total_benefits = revenue_improvement + inventory_savings + stockout_reduction
        total_costs = sum(investment_costs.values())

        # ROI calculations
        roi = (
            (total_benefits - total_costs) / total_costs * 100 if total_costs > 0 else 0
        )
        payback_period = (
            total_costs / (total_benefits / 12) if total_benefits > 0 else float("inf")
        )  # months

        # NPV calculation (simplified, 5-year horizon, 10% discount rate)
        annual_benefits = total_benefits
        discount_rate = 0.10
        npv = (
            sum([annual_benefits / (1 + discount_rate) ** year for year in range(1, 6)])
            - total_costs
        )

        return {
            "total_benefits": total_benefits,
            "total_costs": total_costs,
            "net_benefit": total_benefits - total_costs,
            "roi_percentage": roi,
            "payback_period_months": min(payback_period, 999),  # Cap at 999 months
            "npv": npv,
            "revenue_improvement": revenue_improvement,
            "cost_savings": inventory_savings + stockout_reduction,
            "benefit_cost_ratio": (
                total_benefits / total_costs if total_costs > 0 else 0
            ),
        }

    def generate_periodic_reports(
        self, sales_data: pd.Series, forecast_data: pd.Series, period: str = "monthly"
    ) -> Dict[str, Any]:
        """
        Generate periodic business reports.

        Parameters:
        -----------
        sales_data : Series
            Historical sales data
        forecast_data : Series
            Forecast data
        period : str
            Reporting period ('daily', 'weekly', 'monthly', 'quarterly')

        Returns:
        --------
        periodic_report : dict
            Periodic business report
        """
        if period not in self.reporting_periods:
            raise ValueError(
                f"Period {period} not supported. Use one of {self.reporting_periods}"
            )

        # Group data by period
        if period == "weekly":
            grouped_sales = sales_data.resample("W").sum()
            grouped_forecast = forecast_data.resample("W").sum()
        elif period == "monthly":
            grouped_sales = sales_data.resample("M").sum()
            grouped_forecast = forecast_data.resample("M").sum()
        elif period == "quarterly":
            grouped_sales = sales_data.resample("Q").sum()
            grouped_forecast = forecast_data.resample("Q").sum()
        else:  # daily
            grouped_sales = sales_data
            grouped_forecast = forecast_data

        # Calculate period-over-period metrics
        sales_growth = grouped_sales.pct_change().mean() * 100

        # Performance consistency
        cv_sales = (
            grouped_sales.std() / grouped_sales.mean()
            if grouped_sales.mean() > 0
            else 0
        )

        # Seasonal patterns
        seasonal_analysis = self._analyze_seasonal_patterns(grouped_sales, period)

        # Trend analysis
        trend_analysis = self._analyze_trend_patterns(grouped_sales)

        report = {
            "period": period,
            "summary_stats": {
                "total_sales": grouped_sales.sum(),
                "average_per_period": grouped_sales.mean(),
                "growth_rate": sales_growth,
                "volatility": cv_sales,
                "periods_analyzed": len(grouped_sales),
            },
            "seasonal_analysis": seasonal_analysis,
            "trend_analysis": trend_analysis,
            "performance_ranking": self._rank_periods(grouped_sales),
            "recommendations": self._generate_period_recommendations(
                grouped_sales, period
            ),
        }

        return report

    def create_kpi_scorecard(
        self,
        current_metrics: Dict[str, float],
        target_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Create KPI scorecard with performance ratings.

        Parameters:
        -----------
        current_metrics : dict
            Current KPI values
        target_metrics : dict, optional
            Target KPI values

        Returns:
        --------
        scorecard : dict
            KPI scorecard with ratings
        """
        targets = target_metrics or self.kpi_targets
        scorecard = {}

        total_score = 0
        total_weight = 0

        for kpi, current_value in current_metrics.items():
            if kpi in targets:
                target_value = targets[kpi]

                # Calculate performance ratio
                if target_value != 0:
                    performance_ratio = current_value / target_value
                else:
                    performance_ratio = 1.0 if current_value == 0 else 0.0

                # Determine rating
                if performance_ratio >= 1.0:
                    rating = "Excellent"
                    score = 5
                elif performance_ratio >= 0.9:
                    rating = "Good"
                    score = 4
                elif performance_ratio >= 0.8:
                    rating = "Satisfactory"
                    score = 3
                elif performance_ratio >= 0.7:
                    rating = "Needs Improvement"
                    score = 2
                else:
                    rating = "Poor"
                    score = 1

                scorecard[kpi] = {
                    "current_value": current_value,
                    "target_value": target_value,
                    "performance_ratio": performance_ratio,
                    "rating": rating,
                    "score": score,
                    "gap": target_value - current_value,
                }

                total_score += score
                total_weight += 5  # Max score per KPI

        # Overall scorecard rating
        overall_score = (total_score / total_weight * 100) if total_weight > 0 else 0

        if overall_score >= 90:
            overall_rating = "Excellent"
        elif overall_score >= 80:
            overall_rating = "Good"
        elif overall_score >= 70:
            overall_rating = "Satisfactory"
        elif overall_score >= 60:
            overall_rating = "Needs Improvement"
        else:
            overall_rating = "Poor"

        scorecard["overall"] = {
            "score": overall_score,
            "rating": overall_rating,
            "total_kpis": len(current_metrics),
        }

        return scorecard

    def export_dashboard_data(self, format_type: str = "json") -> str:
        """
        Export dashboard data in specified format.

        Parameters:
        -----------
        format_type : str
            Export format ('json', 'csv', 'excel')

        Returns:
        --------
        exported_data : str
            Exported data as string
        """
        if not self.is_analyzed_:
            raise ValueError(
                "No dashboard data available. Run generate_executive_dashboard() first."
            )

        if format_type == "json":
            return json.dumps(self.kpi_dashboard_, indent=2, default=str)
        elif format_type == "csv":
            # Convert to flat structure for CSV
            flat_data = self._flatten_dashboard_data()
            df = pd.DataFrame([flat_data])
            return df.to_csv(index=False)
        else:
            return json.dumps(self.kpi_dashboard_, indent=2, default=str)

    def _calculate_core_kpis(
        self, sales_data: pd.Series, forecast_data: pd.Series
    ) -> Dict[str, float]:
        """Calculate core business KPIs."""
        # Revenue metrics
        total_revenue = sales_data.sum()
        avg_daily_revenue = sales_data.mean()

        # Growth metrics
        if len(sales_data) > 30:
            recent_sales = sales_data.tail(30).mean()
            older_sales = sales_data.head(30).mean()
            growth_rate = (
                ((recent_sales - older_sales) / older_sales * 100)
                if older_sales > 0
                else 0
            )
        else:
            growth_rate = 0

        # Forecast accuracy
        common_index = sales_data.index.intersection(forecast_data.index)
        if len(common_index) > 0:
            actual_subset = sales_data.loc[common_index]
            forecast_subset = forecast_data.loc[common_index]
            forecast_accuracy = (
                100
                - np.mean(np.abs((actual_subset - forecast_subset) / actual_subset))
                * 100
            )
        else:
            forecast_accuracy = 0

        # Variability
        cv = sales_data.std() / sales_data.mean() if sales_data.mean() > 0 else 0

        # Trend strength
        x = np.arange(len(sales_data))
        correlation = (
            np.corrcoef(x, sales_data.values)[0, 1] if len(sales_data) > 1 else 0
        )

        return {
            "total_revenue": total_revenue,
            "avg_daily_revenue": avg_daily_revenue,
            "revenue_growth_rate": growth_rate,
            "forecast_accuracy": max(0, forecast_accuracy),
            "sales_volatility": cv,
            "trend_strength": abs(correlation) if not np.isnan(correlation) else 0,
            "peak_sales_day": sales_data.max(),
            "min_sales_day": sales_data.min(),
        }

    def _analyze_performance_trends(
        self, sales_data: pd.Series, forecast_data: pd.Series
    ) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        trends = {}

        # Sales trend analysis
        if len(sales_data) > 7:
            weekly_sales = sales_data.resample("W").sum()
            weekly_growth = weekly_sales.pct_change().mean() * 100
            trends["weekly_growth_rate"] = weekly_growth

        if len(sales_data) > 30:
            monthly_sales = sales_data.resample("M").sum()
            monthly_growth = monthly_sales.pct_change().mean() * 100
            trends["monthly_growth_rate"] = monthly_growth

        # Momentum indicators
        if len(sales_data) > 10:
            recent_trend = sales_data.tail(10).mean()
            older_trend = sales_data.head(10).mean()
            momentum = (
                ((recent_trend - older_trend) / older_trend * 100)
                if older_trend > 0
                else 0
            )
            trends["momentum"] = momentum

        # Seasonality strength
        if len(sales_data) > 14:
            weekly_pattern = [
                sales_data[sales_data.index.dayofweek == i].mean() for i in range(7)
            ]
            seasonality_strength = (
                np.std(weekly_pattern) / np.mean(weekly_pattern)
                if np.mean(weekly_pattern) > 0
                else 0
            )
            trends["weekly_seasonality"] = seasonality_strength

        return trends

    def _calculate_business_health(
        self, sales_data: pd.Series, forecast_data: pd.Series
    ) -> Dict[str, float]:
        """Calculate business health indicators."""
        health = {}

        # Stability score (lower volatility = higher stability)
        cv = sales_data.std() / sales_data.mean() if sales_data.mean() > 0 else 0
        stability_score = max(0, min(10, 10 * (1 - cv)))
        health["stability_score"] = stability_score

        # Growth sustainability
        if len(sales_data) > 30:
            growth_consistency = 1 - sales_data.pct_change().std()
            growth_sustainability = max(0, min(10, growth_consistency * 10))
        else:
            growth_sustainability = 5
        health["growth_sustainability"] = growth_sustainability

        # Performance consistency
        rolling_mean = sales_data.rolling(window=7).mean()
        consistency = (
            1 - (rolling_mean.std() / rolling_mean.mean())
            if rolling_mean.mean() > 0
            else 0
        )
        performance_consistency = max(0, min(10, consistency * 10))
        health["performance_consistency"] = performance_consistency

        # Overall health score
        overall_health = (
            stability_score + growth_sustainability + performance_consistency
        ) / 3
        health["overall_health_score"] = overall_health

        return health

    def _generate_strategic_recommendations(
        self, sales_data: pd.Series, forecast_data: pd.Series, kpis: Dict[str, float]
    ) -> List[str]:
        """Generate strategic business recommendations."""
        recommendations = []

        # Growth recommendations
        growth_rate = kpis.get("revenue_growth_rate", 0)
        if growth_rate < 5:
            recommendations.append(
                "Consider growth initiatives: market expansion, new products, or customer acquisition"
            )
        elif growth_rate > 20:
            recommendations.append(
                "High growth detected: ensure operational capacity can support continued expansion"
            )

        # Volatility recommendations
        volatility = kpis.get("sales_volatility", 0)
        if volatility > 0.3:
            recommendations.append(
                "High sales volatility: implement demand smoothing strategies or diversification"
            )

        # Forecast accuracy recommendations
        accuracy = kpis.get("forecast_accuracy", 0)
        if accuracy < 80:
            recommendations.append(
                "Improve forecasting accuracy through better data collection and model enhancement"
            )

        # Trend recommendations
        trend_strength = kpis.get("trend_strength", 0)
        if trend_strength < 0.3:
            recommendations.append(
                "Weak trend signal: focus on identifying and strengthening growth drivers"
            )

        # Seasonal recommendations
        if hasattr(sales_data.index, "dayofweek"):
            dow_variation = sales_data.groupby(sales_data.index.dayofweek).mean().std()
            dow_mean = sales_data.groupby(sales_data.index.dayofweek).mean().mean()
            if dow_variation / dow_mean > 0.2:
                recommendations.append(
                    "Strong day-of-week patterns: optimize staffing and inventory for peak days"
                )

        return recommendations

    def _assess_business_risks(
        self, sales_data: pd.Series, forecast_data: pd.Series
    ) -> Dict[str, Any]:
        """Assess business risks and warning indicators."""
        risks = {}

        # Volatility risk
        cv = sales_data.std() / sales_data.mean() if sales_data.mean() > 0 else 0
        volatility_risk = "High" if cv > 0.3 else "Medium" if cv > 0.15 else "Low"
        risks["volatility_risk"] = volatility_risk

        # Trend reversal risk
        if len(sales_data) > 20:
            recent_trend = np.polyfit(range(10), sales_data.tail(10).values, 1)[0]
            overall_trend = np.polyfit(range(len(sales_data)), sales_data.values, 1)[0]
            trend_divergence = (
                abs(recent_trend - overall_trend) / abs(overall_trend)
                if overall_trend != 0
                else 0
            )

            trend_risk = (
                "High"
                if trend_divergence > 0.5
                else "Medium" if trend_divergence > 0.2 else "Low"
            )
            risks["trend_reversal_risk"] = trend_risk

        # Forecast reliability risk
        common_index = sales_data.index.intersection(forecast_data.index)
        if len(common_index) > 0:
            forecast_errors = np.abs(
                sales_data.loc[common_index] - forecast_data.loc[common_index]
            )
            error_cv = (
                forecast_errors.std() / forecast_errors.mean()
                if forecast_errors.mean() > 0
                else 0
            )

            forecast_risk = (
                "High" if error_cv > 0.5 else "Medium" if error_cv > 0.25 else "Low"
            )
            risks["forecast_reliability_risk"] = forecast_risk

        # Seasonal dependency risk
        if len(sales_data) > 30:
            monthly_sales = sales_data.resample("M").sum()
            if len(monthly_sales) > 1:
                seasonal_cv = monthly_sales.std() / monthly_sales.mean()
                seasonal_risk = (
                    "High"
                    if seasonal_cv > 0.4
                    else "Medium" if seasonal_cv > 0.2 else "Low"
                )
                risks["seasonal_dependency_risk"] = seasonal_risk

        return risks

    def _identify_opportunities(
        self, sales_data: pd.Series, forecast_data: pd.Series
    ) -> List[Dict[str, Any]]:
        """Identify business opportunities."""
        opportunities = []

        # Growth acceleration opportunity
        if len(sales_data) > 30:
            recent_growth = sales_data.tail(15).mean()
            older_growth = sales_data.head(15).mean()

            if recent_growth > older_growth * 1.1:
                opportunities.append(
                    {
                        "type": "Growth Acceleration",
                        "description": "Recent sales showing strong upward momentum",
                        "potential_impact": "High",
                        "timeframe": "Short-term",
                    }
                )

        # Seasonality optimization
        if hasattr(sales_data.index, "dayofweek"):
            dow_sales = sales_data.groupby(sales_data.index.dayofweek).mean()
            peak_day = dow_sales.idxmax()
            low_day = dow_sales.idxmin()

            if dow_sales.max() > dow_sales.min() * 1.5:
                opportunities.append(
                    {
                        "type": "Seasonal Optimization",
                        "description": f"Significant variation between peak (day {peak_day}) and low (day {low_day}) sales",
                        "potential_impact": "Medium",
                        "timeframe": "Medium-term",
                    }
                )

        # Forecast improvement opportunity
        common_index = sales_data.index.intersection(forecast_data.index)
        if len(common_index) > 0:
            accuracy = (
                100
                - np.mean(
                    np.abs(
                        (sales_data.loc[common_index] - forecast_data.loc[common_index])
                        / sales_data.loc[common_index]
                    )
                )
                * 100
            )

            if accuracy < 85:
                opportunities.append(
                    {
                        "type": "Forecast Enhancement",
                        "description": "Forecast accuracy below target - potential for operational improvements",
                        "potential_impact": "Medium",
                        "timeframe": "Medium-term",
                    }
                )

        return opportunities

    def _create_executive_summary(
        self, kpis: Dict[str, float], health: Dict[str, float]
    ) -> str:
        """Create executive summary text."""
        summary_parts = []

        # Revenue summary
        revenue = kpis.get("total_revenue", 0)
        growth = kpis.get("revenue_growth_rate", 0)
        summary_parts.append(
            f"Total revenue: ${revenue:,.0f} with {growth:+.1f}% growth rate"
        )

        # Performance summary
        accuracy = kpis.get("forecast_accuracy", 0)
        summary_parts.append(f"Forecast accuracy: {accuracy:.1f}%")

        # Health summary
        health_score = health.get("overall_health_score", 0)
        if health_score >= 8:
            health_status = "excellent"
        elif health_score >= 6:
            health_status = "good"
        elif health_score >= 4:
            health_status = "fair"
        else:
            health_status = "poor"

        summary_parts.append(
            f"Business health: {health_status} ({health_score:.1f}/10)"
        )

        return ". ".join(summary_parts) + "."

    def _analyze_seasonal_patterns(
        self, data: pd.Series, period: str
    ) -> Dict[str, Any]:
        """Analyze seasonal patterns for given period."""
        analysis = {}

        if period == "monthly" and hasattr(data.index, "month"):
            monthly_avg = data.groupby(data.index.month).mean()
            peak_month = monthly_avg.idxmax()
            low_month = monthly_avg.idxmin()

            analysis["peak_period"] = peak_month
            analysis["low_period"] = low_month
            analysis["seasonal_variation"] = (
                monthly_avg.max() - monthly_avg.min()
            ) / monthly_avg.mean()

        elif period == "weekly" and hasattr(data.index, "dayofweek"):
            weekly_avg = data.groupby(data.index.dayofweek).mean()
            peak_day = weekly_avg.idxmax()
            low_day = weekly_avg.idxmin()

            analysis["peak_period"] = peak_day
            analysis["low_period"] = low_day
            analysis["seasonal_variation"] = (
                weekly_avg.max() - weekly_avg.min()
            ) / weekly_avg.mean()

        return analysis

    def _analyze_trend_patterns(self, data: pd.Series) -> Dict[str, float]:
        """Analyze trend patterns in the data."""
        if len(data) < 3:
            return {}

        # Linear trend
        x = np.arange(len(data))
        slope, intercept = np.polyfit(x, data.values, 1)

        # Trend strength (R-squared)
        y_pred = slope * x + intercept
        ss_res = np.sum((data.values - y_pred) ** 2)
        ss_tot = np.sum((data.values - data.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            "trend_slope": slope,
            "trend_strength": max(0, r_squared),
            "trend_direction": (
                "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Stable"
            ),
        }

    def _rank_periods(self, data: pd.Series) -> Dict[str, Any]:
        """Rank periods by performance."""
        if len(data) == 0:
            return {}

        # Sort periods by value
        sorted_data = data.sort_values(ascending=False)

        return {
            "best_period": {
                "date": sorted_data.index[0].strftime("%Y-%m-%d"),
                "value": sorted_data.iloc[0],
            },
            "worst_period": {
                "date": sorted_data.index[-1].strftime("%Y-%m-%d"),
                "value": sorted_data.iloc[-1],
            },
            "median_performance": sorted_data.median(),
            "top_quartile_threshold": sorted_data.quantile(0.75),
        }

    def _generate_period_recommendations(
        self, data: pd.Series, period: str
    ) -> List[str]:
        """Generate recommendations based on period analysis."""
        recommendations = []

        if len(data) == 0:
            return recommendations

        # Performance consistency
        cv = data.std() / data.mean() if data.mean() > 0 else 0
        if cv > 0.3:
            recommendations.append(
                f"High {period} variability detected - investigate causes and implement stabilization measures"
            )

        # Growth trend
        if len(data) > 2:
            growth_rate = (
                (data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100
                if data.iloc[0] > 0
                else 0
            )
            if growth_rate < -5:
                recommendations.append(
                    f"Declining {period} trend - urgent action needed to reverse negative trajectory"
                )
            elif growth_rate > 20:
                recommendations.append(
                    f"Strong {period} growth - ensure capacity planning to support continued expansion"
                )

        return recommendations

    def _flatten_dashboard_data(self) -> Dict[str, Any]:
        """Flatten dashboard data for CSV export."""
        flat_data = {}

        def flatten_dict(d, prefix=""):
            for key, value in d.items():
                if isinstance(value, dict):
                    flatten_dict(value, f"{prefix}{key}_")
                elif isinstance(value, list):
                    flat_data[f"{prefix}{key}"] = str(value)
                else:
                    flat_data[f"{prefix}{key}"] = value

        flatten_dict(self.kpi_dashboard_)
        return flat_data

    def _calculate_market_share(
        self, sales_data: pd.Series, market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate market share metrics."""
        # This is a simplified example - would need actual market data
        total_market = (
            market_data.sum().sum() if not market_data.empty else sales_data.sum() * 10
        )
        our_sales = sales_data.sum()

        market_share = (our_sales / total_market * 100) if total_market > 0 else 0

        return {
            "current_market_share": market_share,
            "market_size": total_market,
            "our_sales": our_sales,
        }

    def _analyze_competitive_position(
        self, sales_data: pd.Series, competitor_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze competitive position."""
        # Simplified competitive analysis
        our_growth = sales_data.pct_change().mean() * 100

        # This would analyze competitor data in a real implementation
        competitive_position = {
            "our_growth_rate": our_growth,
            "market_position": (
                "Strong" if our_growth > 10 else "Average" if our_growth > 0 else "Weak"
            ),
            "competitive_advantage": (
                "Growth rate"
                if our_growth > 10
                else "Stability" if our_growth > 0 else "None identified"
            ),
        }

        return competitive_position

    def _analyze_market_trends(self, sales_data: pd.Series) -> Dict[str, Any]:
        """Analyze market trends."""
        # Simplified market trend analysis based on our sales data
        if len(sales_data) > 30:
            recent_trend = np.polyfit(range(30), sales_data.tail(30).values, 1)[0]
            overall_trend = np.polyfit(range(len(sales_data)), sales_data.values, 1)[0]

            trend_analysis = {
                "recent_trend": "Positive" if recent_trend > 0 else "Negative",
                "overall_trend": "Positive" if overall_trend > 0 else "Negative",
                "trend_acceleration": recent_trend - overall_trend,
                "market_momentum": (
                    "Strong"
                    if recent_trend > overall_trend * 1.2
                    else "Moderate" if recent_trend > overall_trend else "Weak"
                ),
            }
        else:
            trend_analysis = {"insufficient_data": True}

        return trend_analysis

    def _identify_growth_opportunities(
        self, sales_data: pd.Series
    ) -> List[Dict[str, str]]:
        """Identify growth opportunities."""
        opportunities = []

        # Trend-based opportunities
        if len(sales_data) > 10:
            recent_avg = sales_data.tail(5).mean()
            older_avg = sales_data.head(5).mean()

            if recent_avg > older_avg * 1.1:
                opportunities.append(
                    {
                        "opportunity": "Momentum Capitalization",
                        "description": "Recent positive momentum presents expansion opportunity",
                        "priority": "High",
                    }
                )

        # Volatility-based opportunities
        cv = sales_data.std() / sales_data.mean() if sales_data.mean() > 0 else 0
        if cv > 0.2:
            opportunities.append(
                {
                    "opportunity": "Market Stabilization",
                    "description": "High volatility suggests untapped market segments",
                    "priority": "Medium",
                }
            )

        return opportunities

    def calculate_core_kpis(self, actual: pd.Series, forecast: pd.Series) -> dict:
        raw_kpis = self._calculate_core_kpis(actual, forecast)
        return {
            "total_revenue": raw_kpis["total_revenue"],
            "average_daily_sales": raw_kpis["avg_daily_revenue"],
            "forecast_accuracy": raw_kpis["forecast_accuracy"],
            "revenue_growth": raw_kpis["revenue_growth_rate"],
        }

    def calculate_business_health(self, actual: pd.Series, forecast: pd.Series) -> dict:
        raw_health = self._calculate_business_health(actual, forecast)
        raw_kpis = self._calculate_core_kpis(actual, forecast)
        return {
            "overall_health_score": raw_health["overall_health_score"],
            "forecast_reliability": float(
                np.clip(raw_kpis["forecast_accuracy"] / 10.0, 0.0, 10.0)
            ),
            "trend_stability": float(
                np.clip(raw_kpis["trend_strength"] * 10.0, 0.0, 10.0)
            ),
        }

    def analyze_trends(self, actual: pd.Series, forecast: pd.Series) -> dict:
        actual_trend, forecast_trend = [
            float(np.polyfit(np.arange(len(series)), series.values, 1)[0])
            if len(series) > 1
            else 0.0
            for series in (actual, forecast)
        ]
        trend_accuracy = 1.0 - abs(actual_trend - forecast_trend) / (abs(actual_trend) + 1e-10)
        return {
            "actual_trend": actual_trend,
            "forecast_trend": forecast_trend,
            "trend_accuracy": float(max(0.0, min(1.0, trend_accuracy))),
        }

    def analyze_scenarios(self, actual: pd.Series, scenarios: dict) -> dict:
        actual_values = actual.values
        denom = np.where(actual_values == 0, np.nan, actual_values)
        results = {}
        for name, scenario_forecast in scenarios.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                pct_errs = np.abs((actual_values - scenario_forecast.values) / denom)
            mape = float(np.nanmean(pct_errs) * 100)
            results[name] = {
                "scenario_metrics": {
                    "accuracy": max(0.0, 100.0 - mape),
                    "mape": mape,
                }
            }
        return results

    def calculate_roi_metrics(self, actual: pd.Series, forecast: pd.Series) -> dict:
        errors = np.abs(actual.values - forecast.values)
        cost_of_errors = float(errors.sum() * 0.05)
        forecast_value = float(actual.sum() * 0.02)
        return {
            "forecast_value": forecast_value,
            "cost_of_forecast_errors": cost_of_errors,
            "accuracy_improvement_value": max(0.0, forecast_value - cost_of_errors * 0.1),
        }

    def generate_executive_dashboard(
        self,
        sales_data: pd.Series,
        forecast_data: pd.Series,
        additional_metrics=None,
    ) -> dict:
        raw_kpis = self._calculate_core_kpis(sales_data, forecast_data)
        core_kpis = {
            "total_revenue": raw_kpis["total_revenue"],
            "average_daily_sales": raw_kpis["avg_daily_revenue"],
            "forecast_accuracy": raw_kpis["forecast_accuracy"],
            "revenue_growth": raw_kpis["revenue_growth_rate"],
        }
        health = self.calculate_business_health(sales_data, forecast_data)
        strategic = self._generate_strategic_recommendations(
            sales_data, forecast_data, raw_kpis
        )
        summary = {
            "key_insights": [
                f"Total revenue: {core_kpis['total_revenue']:.0f}",
                f"Forecast accuracy: {core_kpis['forecast_accuracy']:.1f}%",
            ]
        }
        return {
            "core_kpis": core_kpis,
            "business_health": health,
            "executive_summary": summary,
            "recommendations": strategic,
        }


def main():
    """Example usage of BusinessIntelligence."""
    print("=== Business Intelligence Demo ===")

    # Generate synthetic sales data
    np.random.seed(42)

    # Create 6 months of daily sales data
    dates = pd.date_range(start="2023-01-01", periods=180, freq="D")

    # Sales with trend, seasonality, and noise
    trend = np.linspace(1000, 1500, 180)
    weekly_season = 200 * np.sin(2 * np.pi * np.arange(180) / 7)
    monthly_season = 150 * np.sin(2 * np.pi * np.arange(180) / 30)
    noise = np.random.normal(0, 100, 180)

    sales_values = trend + weekly_season + monthly_season + noise
    sales_data = pd.Series(sales_values, index=dates)

    # Create forecast data (with some error)
    forecast_error = np.random.normal(0, 50, 180)
    forecast_data = pd.Series(sales_values + forecast_error, index=dates)

    print(f"Generated {len(sales_data)} days of sales data")
    print(f"Sales range: ${sales_data.min():.0f} - ${sales_data.max():.0f}")

    # Initialize Business Intelligence
    bi = BusinessIntelligence(
        kpi_targets={
            "revenue_growth_rate": 0.15,
            "forecast_accuracy": 0.90,
            "inventory_turnover": 12,
        }
    )

    # Generate executive dashboard
    print("\nGenerating executive dashboard...")
    dashboard = bi.generate_executive_dashboard(sales_data, forecast_data)

    # Display key results
    print("\n" + "=" * 50)
    print("EXECUTIVE DASHBOARD")
    print("=" * 50)

    print(f"\nExecutive Summary:")
    print(dashboard["executive_summary"])

    print(f"\nCore KPIs:")
    kpis = dashboard["core_kpis"]
    print(f"  Total Revenue: ${kpis['total_revenue']:,.0f}")
    print(f"  Growth Rate: {kpis['revenue_growth_rate']:+.1f}%")
    print(f"  Forecast Accuracy: {kpis['forecast_accuracy']:.1f}%")
    print(f"  Trend Strength: {kpis['trend_strength']:.2f}")

    print(f"\nBusiness Health:")
    health = dashboard["business_health"]
    print(f"  Overall Health Score: {health['overall_health_score']:.1f}/10")
    print(f"  Stability Score: {health['stability_score']:.1f}/10")
    print(f"  Growth Sustainability: {health['growth_sustainability']:.1f}/10")

    print(f"\nStrategic Recommendations:")
    for i, rec in enumerate(dashboard["strategic_recommendations"], 1):
        print(f"  {i}. {rec}")

    print(f"\nRisk Indicators:")
    risks = dashboard["risk_indicators"]
    for risk_type, level in risks.items():
        print(f"  {risk_type.replace('_', ' ').title()}: {level}")

    # Calculate ROI metrics
    print("\n" + "=" * 50)
    print("ROI ANALYSIS")
    print("=" * 50)

    investment_costs = {
        "software_development": 50000,
        "data_infrastructure": 25000,
        "training": 10000,
        "maintenance_annual": 15000,
    }

    roi_metrics = bi.calculate_roi_metrics(sales_data, forecast_data, investment_costs)

    print(f"\nROI Metrics:")
    print(f"  Total Benefits: ${roi_metrics['total_benefits']:,.0f}")
    print(f"  Total Costs: ${roi_metrics['total_costs']:,.0f}")
    print(f"  ROI: {roi_metrics['roi_percentage']:.1f}%")
    print(f"  Payback Period: {roi_metrics['payback_period_months']:.1f} months")
    print(f"  NPV: ${roi_metrics['npv']:,.0f}")

    # Generate monthly report
    print("\n" + "=" * 50)
    print("MONTHLY PERFORMANCE REPORT")
    print("=" * 50)

    monthly_report = bi.generate_periodic_reports(sales_data, forecast_data, "monthly")

    print(f"\nMonthly Summary:")
    summary = monthly_report["summary_stats"]
    print(f"  Total Sales: ${summary['total_sales']:,.0f}")
    print(f"  Average per Month: ${summary['average_per_period']:,.0f}")
    print(f"  Growth Rate: {summary['growth_rate']:+.1f}%")
    print(f"  Volatility: {summary['volatility']:.2f}")

    # Create KPI scorecard
    current_kpis = {
        "revenue_growth_rate": kpis["revenue_growth_rate"] / 100,
        "forecast_accuracy": kpis["forecast_accuracy"] / 100,
        "inventory_turnover": 10,  # Example value
    }

    scorecard = bi.create_kpi_scorecard(current_kpis)

    print(f"\nKPI Scorecard:")
    print(
        f"  Overall Rating: {scorecard['overall']['rating']} ({scorecard['overall']['score']:.0f}%)"
    )

    for kpi, details in scorecard.items():
        if kpi != "overall":
            print(
                f"  {kpi.replace('_', ' ').title()}: {details['rating']} "
                f"({details['current_value']:.2f} vs target {details['target_value']:.2f})"
            )

    print("\nBusiness Intelligence demonstration complete!")


if __name__ == "__main__":
    main()
