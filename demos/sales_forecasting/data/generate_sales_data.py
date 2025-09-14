"""
Sales Data Generator for IntegratedML Sales Forecasting Demo.

This module generates realistic multi-store retail sales data with complex
seasonal patterns, external factors, and business discontinuities for
demonstrating advanced forecasting capabilities.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
from pathlib import Path


class SalesDataGenerator:
    """
    Generate realistic sales data with multiple seasonal patterns and external factors.

    Features:
    - Multiple seasonal patterns (yearly, monthly, weekly, daily)
    - Store and product category variations
    - External factors (holidays, promotions, weather, economic indicators)
    - Realistic noise and business discontinuities
    - Configurable parameters for different business scenarios
    """

    def __init__(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
        stores: List[str] = None,
        categories: List[str] = None,
        product_categories: List[str] = None,
        base_sales_range: Tuple[float, float] = (1000, 5000),
        seasonality_strength: float = 0.3,
        noise_level: float = 0.1,
        random_seed: int = 42,
    ):
        """
        Initialize the sales data generator.

        Parameters:
        -----------
        start_date : str
            Start date for data generation
        end_date : str
            End date for data generation
        stores : list of str, optional
            Store identifiers
        categories : list of str, optional
            Product category names (alias for product_categories)
        product_categories : list of str, optional
            Product category names
        base_sales_range : tuple
            Range for base daily sales (min, max)
        seasonality_strength : float
            Strength of seasonal patterns (0-1)
        noise_level : float
            Level of random noise (0-1)
        random_seed : int
            Random seed for reproducibility
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.stores = stores or [f"Store_{i:03d}" for i in range(1, 11)]  # 10 stores

        # Support both 'categories' and 'product_categories' for backward compatibility
        if categories is not None:
            self.product_categories = categories
        elif product_categories is not None:
            self.product_categories = product_categories
        else:
            self.product_categories = [
                "Electronics",
                "Clothing",
                "Home_Garden",
                "Sports_Outdoors",
                "Health_Beauty",
                "Books_Media",
                "Food_Beverages",
                "Toys_Games",
            ]

        # Also set categories attribute for test compatibility
        self.categories = self.product_categories
        self.base_sales_range = base_sales_range
        self.seasonality_strength = seasonality_strength
        self.noise_level = noise_level

        # Set random seed for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Generate date range
        self.date_range = pd.date_range(
            start=self.start_date, end=self.end_date, freq="D"
        )
        self.n_days = len(self.date_range)

        # Initialize store and category characteristics
        self._initialize_store_characteristics()
        self._initialize_category_characteristics()

        # Generate external factors
        self.holidays = self._generate_holidays()
        self.economic_indicators = self._generate_economic_indicators()
        self.weather_data = self._generate_weather_data()
        self.promotions = self._generate_promotions()

    def _initialize_store_characteristics(self) -> None:
        """Initialize store-specific characteristics."""
        self.store_characteristics = {}

        for store in self.stores:
            # Store size factor (affects base sales)
            size_factor = np.random.uniform(0.7, 1.5)

            # Location type (urban, suburban, rural)
            location_type = np.random.choice(
                ["urban", "suburban", "rural"], p=[0.3, 0.5, 0.2]
            )

            # Regional economic factor
            economic_factor = np.random.uniform(0.8, 1.3)

            # Store opening hours effect on sales
            operating_hours = np.random.choice([8, 10, 12, 14], p=[0.1, 0.3, 0.4, 0.2])

            self.store_characteristics[store] = {
                "size_factor": size_factor,
                "location_type": location_type,
                "economic_factor": economic_factor,
                "operating_hours": operating_hours,
                "base_sales": np.random.uniform(*self.base_sales_range) * size_factor,
            }

    def _initialize_category_characteristics(self) -> None:
        """Initialize product category characteristics."""
        self.category_characteristics = {}

        # Define category-specific patterns
        category_patterns = {
            "Electronics": {
                "seasonal_peaks": [11, 12],
                "growth_rate": 0.05,
                "volatility": 0.15,
            },
            "Clothing": {
                "seasonal_peaks": [3, 6, 9, 11],
                "growth_rate": 0.02,
                "volatility": 0.20,
            },
            "Home_Garden": {
                "seasonal_peaks": [4, 5, 6],
                "growth_rate": 0.03,
                "volatility": 0.12,
            },
            "Sports_Outdoors": {
                "seasonal_peaks": [5, 6, 7, 8],
                "growth_rate": 0.04,
                "volatility": 0.18,
            },
            "Health_Beauty": {
                "seasonal_peaks": [1, 6],
                "growth_rate": 0.06,
                "volatility": 0.10,
            },
            "Books_Media": {
                "seasonal_peaks": [9, 12],
                "growth_rate": -0.02,
                "volatility": 0.25,
            },
            "Food_Beverages": {
                "seasonal_peaks": [7, 11, 12],
                "growth_rate": 0.01,
                "volatility": 0.08,
            },
            "Toys_Games": {
                "seasonal_peaks": [11, 12],
                "growth_rate": 0.03,
                "volatility": 0.30,
            },
        }

        for category in self.product_categories:
            if category in category_patterns:
                self.category_characteristics[category] = category_patterns[category]
            else:
                # Default characteristics for unknown categories
                self.category_characteristics[category] = {
                    "seasonal_peaks": [np.random.randint(1, 13)],
                    "growth_rate": np.random.uniform(-0.01, 0.05),
                    "volatility": np.random.uniform(0.10, 0.25),
                }

    def _generate_holidays(self) -> pd.DataFrame:
        """Generate holiday calendar with impact factors."""
        holidays_data = []

        # Major holidays with sales impact
        holiday_definitions = {
            "New Year": {"month": 1, "day": 1, "impact": 0.8, "duration": 1},
            "Valentines Day": {"month": 2, "day": 14, "impact": 1.2, "duration": 1},
            "Easter": {
                "month": 4,
                "day": 15,
                "impact": 1.1,
                "duration": 2,
            },  # Simplified
            "Memorial Day": {
                "month": 5,
                "day": 25,
                "impact": 1.3,
                "duration": 3,
            },  # Last Monday
            "Independence Day": {"month": 7, "day": 4, "impact": 1.2, "duration": 1},
            "Labor Day": {
                "month": 9,
                "day": 6,
                "impact": 1.1,
                "duration": 3,
            },  # First Monday
            "Halloween": {"month": 10, "day": 31, "impact": 1.4, "duration": 1},
            "Thanksgiving": {
                "month": 11,
                "day": 25,
                "impact": 0.3,
                "duration": 1,
            },  # Stores closed
            "Black Friday": {"month": 11, "day": 26, "impact": 2.5, "duration": 1},
            "Cyber Monday": {"month": 11, "day": 29, "impact": 2.0, "duration": 1},
            "Christmas": {
                "month": 12,
                "day": 25,
                "impact": 0.2,
                "duration": 1,
            },  # Stores closed
            "Boxing Day": {"month": 12, "day": 26, "impact": 1.8, "duration": 1},
        }

        for year in range(self.start_date.year, self.end_date.year + 1):
            for holiday_name, details in holiday_definitions.items():
                try:
                    holiday_date = datetime(year, details["month"], details["day"])
                    if self.start_date <= pd.to_datetime(holiday_date) <= self.end_date:
                        holidays_data.append(
                            {
                                "date": holiday_date,
                                "holiday": holiday_name,
                                "impact_factor": details["impact"],
                                "duration_days": details["duration"],
                            }
                        )
                except ValueError:
                    # Handle invalid dates (e.g., Feb 29 in non-leap years)
                    continue

        return pd.DataFrame(holidays_data)

    def _generate_economic_indicators(self) -> pd.DataFrame:
        """Generate economic indicators affecting sales."""
        dates = self.date_range
        n_days = len(dates)

        # Generate base economic trends
        base_gdp_growth = 0.02  # 2% annual growth
        base_unemployment = 5.0  # 5% base unemployment
        base_inflation = 2.0  # 2% base inflation

        # Add economic cycles and shocks
        # GDP growth with business cycles
        gdp_cycle = (
            np.sin(2 * np.pi * np.arange(n_days) / (4 * 365)) * 0.01
        )  # 4-year cycle
        gdp_shock_2020 = np.where(
            (dates >= "2020-03-01") & (dates <= "2020-06-01"), -0.05, 0
        )
        gdp_growth = base_gdp_growth + gdp_cycle + gdp_shock_2020

        # Unemployment with seasonal patterns
        unemployment_seasonal = np.sin(2 * np.pi * dates.dayofyear / 365) * 0.5
        unemployment_shock_2020 = np.where(
            (dates >= "2020-03-01") & (dates <= "2020-08-01"), 3.0, 0
        )
        unemployment_rate = (
            base_unemployment + unemployment_seasonal + unemployment_shock_2020
        )

        # Inflation with trend and volatility
        inflation_trend = np.cumsum(np.random.normal(0, 0.001, n_days))
        inflation_rate = base_inflation + inflation_trend

        # Consumer confidence index (0-100)
        confidence_base = 75
        confidence_cycle = np.sin(2 * np.pi * np.arange(n_days) / (2 * 365)) * 10
        confidence_shock_2020 = np.where(
            (dates >= "2020-03-01") & (dates <= "2020-12-01"), -20, 0
        )
        consumer_confidence = confidence_base + confidence_cycle + confidence_shock_2020

        return pd.DataFrame(
            {
                "date": dates,
                "gdp_growth_rate": gdp_growth,
                "unemployment_rate": unemployment_rate,
                "inflation_rate": inflation_rate,
                "consumer_confidence": consumer_confidence,
            }
        )

    def _generate_weather_data(self) -> pd.DataFrame:
        """Generate weather data affecting sales."""
        dates = self.date_range
        n_days = len(dates)

        # Temperature with seasonal patterns
        base_temp = 15  # Celsius
        seasonal_temp = 10 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365)
        daily_variation = np.random.normal(0, 3, n_days)
        temperature = base_temp + seasonal_temp + daily_variation

        # Precipitation (affects foot traffic)
        precipitation_base = np.random.exponential(2, n_days)  # mm per day
        seasonal_precip = 2 * np.sin(2 * np.pi * (dates.dayofyear - 60) / 365)
        precipitation = np.maximum(0, precipitation_base + seasonal_precip)

        # Weather categories
        weather_conditions = []
        for i in range(n_days):
            if precipitation[i] > 5:
                if temperature[i] < 0:
                    condition = "snow"
                else:
                    condition = "rain"
            elif temperature[i] > 25:
                condition = "hot"
            elif temperature[i] < 5:
                condition = "cold"
            else:
                condition = "mild"
            weather_conditions.append(condition)

        return pd.DataFrame(
            {
                "date": dates,
                "temperature_celsius": temperature,
                "precipitation_mm": precipitation,
                "weather_condition": weather_conditions,
            }
        )

    def _generate_promotions(self) -> pd.DataFrame:
        """Generate promotional campaigns."""
        promotions_data = []

        # Regular promotional patterns
        promotion_types = {
            "Weekly Sale": {"frequency": 7, "duration": 2, "discount": 0.15},
            "Monthly Special": {"frequency": 30, "duration": 3, "discount": 0.25},
            "Seasonal Clearance": {"frequency": 90, "duration": 7, "discount": 0.40},
            "Flash Sale": {"frequency": 45, "duration": 1, "discount": 0.30},
        }

        for promo_type, details in promotion_types.items():
            current_date = self.start_date
            while current_date <= self.end_date:
                # Add some randomness to promotion timing
                next_promo = current_date + timedelta(
                    days=details["frequency"] + np.random.randint(-5, 6)
                )

                if next_promo <= self.end_date:
                    # Random selection of stores and categories
                    participating_stores = np.random.choice(
                        self.stores,
                        size=np.random.randint(1, len(self.stores) + 1),
                        replace=False,
                    )
                    participating_categories = np.random.choice(
                        self.product_categories,
                        size=np.random.randint(1, len(self.product_categories) + 1),
                        replace=False,
                    )

                    for day in range(details["duration"]):
                        promo_date = next_promo + timedelta(days=day)
                        if promo_date <= self.end_date:
                            promotions_data.append(
                                {
                                    "date": promo_date,
                                    "promotion_type": promo_type,
                                    "discount_rate": details["discount"],
                                    "participating_stores": ",".join(
                                        participating_stores
                                    ),
                                    "participating_categories": ",".join(
                                        participating_categories
                                    ),
                                }
                            )

                current_date = next_promo

        return pd.DataFrame(promotions_data)

    def generate_sales_data(self) -> pd.DataFrame:
        """
        Generate complete sales dataset with all factors.

        Returns:
        --------
        sales_data : DataFrame
            Complete sales dataset with all stores, categories, and external factors
        """
        sales_records = []

        print(
            f"Generating sales data for {len(self.stores)} stores and {len(self.product_categories)} categories..."
        )
        print(f"Date range: {self.start_date.date()} to {self.end_date.date()}")

        # Process each date
        for date_idx, date in enumerate(self.date_range):
            if date_idx % 365 == 0:
                print(f"Processing year {date.year}...")

            # Get external factors for this date
            day_of_year = date.dayofyear
            day_of_week = date.dayofweek
            month = date.month
            year = date.year

            # Holiday effects
            holiday_factor = self._get_holiday_factor(date)

            # Economic effects
            economic_factor = self._get_economic_factor(date)

            # Weather effects
            weather_factor = self._get_weather_factor(date)

            # Base trends (overall market growth/decline)
            years_since_start = (date - self.start_date).days / 365.25

            # Generate sales for each store and category combination
            for store in self.stores:
                store_chars = self.store_characteristics[store]

                for category in self.product_categories:
                    category_chars = self.category_characteristics[category]

                    # Base sales for this store-category combination
                    base_sales = store_chars["base_sales"] * np.random.uniform(
                        0.1, 0.4
                    )  # Category share

                    # Apply growth trend
                    growth_factor = (
                        1 + category_chars["growth_rate"]
                    ) ** years_since_start

                    # Seasonal patterns
                    seasonal_factor = self._calculate_seasonal_factor(
                        day_of_year,
                        month,
                        day_of_week,
                        category_chars["seasonal_peaks"],
                    )

                    # Store-specific factors
                    store_factor = (
                        store_chars["size_factor"] * store_chars["economic_factor"]
                    )

                    # Promotion effects
                    promotion_factor = self._get_promotion_factor(date, store, category)

                    # Calculate final sales
                    daily_sales = (
                        base_sales
                        * growth_factor
                        * seasonal_factor
                        * store_factor
                        * holiday_factor
                        * economic_factor
                        * weather_factor
                        * promotion_factor
                    )

                    # Add noise and ensure non-negative
                    noise = np.random.normal(
                        1, category_chars["volatility"] * self.noise_level
                    )
                    daily_sales = max(0, daily_sales * noise)

                    # Create sales record
                    sales_records.append(
                        {
                            "date": date,
                            "store_id": store,
                            "product_category": category,
                            "sales_amount": round(daily_sales, 2),
                            "day_of_week": day_of_week,
                            "month": month,
                            "quarter": (month - 1) // 3 + 1,
                            "year": year,
                            "is_weekend": int(day_of_week >= 5),
                            "is_holiday": int(holiday_factor != 1.0),
                            "holiday_factor": holiday_factor,
                            "economic_factor": economic_factor,
                            "weather_factor": weather_factor,
                            "promotion_factor": promotion_factor,
                            "seasonal_factor": seasonal_factor,
                        }
                    )

        print(f"Generated {len(sales_records)} sales records")
        return pd.DataFrame(sales_records)

    def generate_full_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate complete dataset including sales data and external factors.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Sales data and external factors dataframes
        """
        sales_data = self.generate_sales_data()
        external_factors = self._create_external_factors_dataset()

        return sales_data, external_factors

    def _get_holiday_factor(self, date: pd.Timestamp) -> float:
        """Get holiday impact factor for a specific date."""
        if self.holidays.empty:
            return 1.0

        # Check if date matches any holiday
        matching_holidays = self.holidays[self.holidays["date"].dt.date == date.date()]

        if not matching_holidays.empty:
            return matching_holidays.iloc[0]["impact_factor"]

        # Check for extended holiday effects
        for _, holiday in self.holidays.iterrows():
            holiday_date = pd.to_datetime(holiday["date"])
            days_diff = (date - holiday_date).days

            if 0 < days_diff <= holiday["duration_days"]:
                # Diminishing effect over duration
                factor = holiday["impact_factor"]
                decay = (holiday["duration_days"] - days_diff) / holiday[
                    "duration_days"
                ]
                return 1.0 + (factor - 1.0) * decay

        return 1.0

    def _get_economic_factor(self, date: pd.Timestamp) -> float:
        """Get economic impact factor for a specific date."""
        econ_data = self.economic_indicators[
            self.economic_indicators["date"].dt.date == date.date()
        ]

        if econ_data.empty:
            return 1.0

        econ_row = econ_data.iloc[0]

        # Combine economic indicators
        confidence_effect = (
            econ_row["consumer_confidence"] - 75
        ) / 100  # Normalize around 75
        unemployment_effect = (
            -(econ_row["unemployment_rate"] - 5) / 20
        )  # Normalize around 5%

        economic_factor = 1.0 + (confidence_effect + unemployment_effect) * 0.1

        return max(0.5, min(1.5, economic_factor))  # Bound between 0.5 and 1.5

    def _get_weather_factor(self, date: pd.Timestamp) -> float:
        """Get weather impact factor for a specific date."""
        weather_data = self.weather_data[
            self.weather_data["date"].dt.date == date.date()
        ]

        if weather_data.empty:
            return 1.0

        weather_row = weather_data.iloc[0]
        condition = weather_row["weather_condition"]

        # Weather effects on foot traffic and sales
        weather_effects = {
            "mild": 1.0,
            "hot": 0.95,  # Slightly lower foot traffic
            "cold": 0.90,  # Lower foot traffic
            "rain": 0.85,  # Reduced foot traffic
            "snow": 0.75,  # Significant reduction
        }

        return weather_effects.get(condition, 1.0)

    def _get_promotion_factor(
        self, date: pd.Timestamp, store: str, category: str
    ) -> float:
        """Get promotion impact factor for a specific date, store, and category."""
        if self.promotions.empty:
            return 1.0

        # Check for promotions on this date
        date_promotions = self.promotions[
            self.promotions["date"].dt.date == date.date()
        ]

        for _, promo in date_promotions.iterrows():
            participating_stores = promo["participating_stores"].split(",")
            participating_categories = promo["participating_categories"].split(",")

            if store in participating_stores and category in participating_categories:
                # Convert discount to sales boost
                discount_rate = promo["discount_rate"]
                # Assume sales increase due to promotion (elastic demand)
                sales_boost = 1 + (
                    discount_rate * 2
                )  # 15% discount -> 30% sales increase
                return sales_boost

        return 1.0

    def _calculate_seasonal_factor(
        self, day_of_year: int, month: int, day_of_week: int, seasonal_peaks: List[int]
    ) -> float:
        """Calculate seasonal impact factor."""
        seasonal_factor = 1.0

        # Yearly seasonality based on category peaks
        for peak_month in seasonal_peaks:
            # Gaussian-like peak around each seasonal month
            month_distance = min(abs(month - peak_month), 12 - abs(month - peak_month))
            month_effect = np.exp(-(month_distance**2) / (2 * 2**2))  # Sigma = 2 months
            seasonal_factor += month_effect * self.seasonality_strength

        # Weekly seasonality (weekend vs weekday)
        if day_of_week in [5, 6]:  # Saturday, Sunday
            seasonal_factor *= 1.2  # Weekend boost
        elif day_of_week == 0:  # Monday
            seasonal_factor *= 0.9  # Monday dip

        # Monthly patterns (end of month salary effect)
        day_of_month = day_of_year % 30  # Approximate
        if day_of_month > 25:  # End of month
            seasonal_factor *= 1.1  # Payday effect

        return seasonal_factor

    def save_datasets(
        self, output_dir: str = "demos/sales_forecasting/data/"
    ) -> Dict[str, str]:
        """
        Generate and save all datasets.

        Parameters:
        -----------
        output_dir : str
            Directory to save the generated datasets

        Returns:
        --------
        file_paths : dict
            Dictionary of dataset names and their file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Generating comprehensive sales datasets...")

        # Generate main sales data
        sales_data = self.generate_sales_data()

        # Save main dataset
        sales_file = output_path / "retail_sales_data.csv"
        sales_data.to_csv(sales_file, index=False)
        print(f"Saved main sales data: {sales_file}")

        # Save external factors
        external_factors = self._create_external_factors_dataset()
        external_file = output_path / "external_factors.csv"
        external_factors.to_csv(external_file, index=False)
        print(f"Saved external factors: {external_file}")

        # Save aggregated data for different forecasting scenarios
        aggregated_data = self._create_aggregated_datasets(sales_data)

        file_paths = {
            "main_sales": str(sales_file),
            "external_factors": str(external_file),
        }

        for name, data in aggregated_data.items():
            agg_file = output_path / f"{name}_sales.csv"
            data.to_csv(agg_file, index=False)
            file_paths[name] = str(agg_file)
            print(f"Saved {name} data: {agg_file}")

        # Save metadata
        metadata = self._create_metadata()
        metadata_file = output_path / "dataset_metadata.json"

        import json

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        file_paths["metadata"] = str(metadata_file)
        print(f"Saved metadata: {metadata_file}")

        print(f"\nDataset generation complete! Generated {len(file_paths)} files.")
        print(f"Total sales records: {len(sales_data):,}")
        print(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Stores: {len(self.stores)}")
        print(f"Product categories: {len(self.product_categories)}")

        return file_paths

    def _create_external_factors_dataset(self) -> pd.DataFrame:
        """Create consolidated external factors dataset."""
        # Merge all external factor datasets
        external_data = self.economic_indicators.copy()

        # Add holiday indicators
        external_data["is_holiday"] = (
            external_data["date"].isin(self.holidays["date"]).astype(int)
        )

        # Add weather data
        weather_subset = self.weather_data[
            ["date", "temperature_celsius", "precipitation_mm", "weather_condition"]
        ]
        external_data = external_data.merge(weather_subset, on="date", how="left")

        # Add promotion indicators
        promo_summary = (
            self.promotions.groupby("date")
            .agg({"discount_rate": "mean", "promotion_type": "count"})
            .rename(columns={"promotion_type": "active_promotions"})
            .reset_index()
        )

        external_data = external_data.merge(promo_summary, on="date", how="left")
        external_data["discount_rate"] = external_data["discount_rate"].fillna(0)
        external_data["active_promotions"] = external_data["active_promotions"].fillna(
            0
        )

        return external_data

    def _create_aggregated_datasets(
        self, sales_data: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Create aggregated datasets for different forecasting scenarios."""
        aggregated_datasets = {}

        # Monthly aggregation (most common for business forecasting)
        monthly_data = (
            sales_data.groupby(["year", "month", "store_id", "product_category"])
            .agg(
                {
                    "daily_sales": "sum",
                    "is_weekend": "mean",
                    "is_holiday": "mean",
                    "holiday_factor": "mean",
                    "economic_factor": "mean",
                    "weather_factor": "mean",
                    "promotion_factor": "mean",
                    "seasonal_factor": "mean",
                }
            )
            .reset_index()
        )

        monthly_data["date"] = pd.to_datetime(
            monthly_data[["year", "month"]].assign(day=1)
        )
        monthly_data = monthly_data.rename(columns={"daily_sales": "monthly_sales"})
        aggregated_datasets["monthly"] = monthly_data

        # Weekly aggregation
        sales_data["week"] = sales_data["date"].dt.isocalendar().week
        weekly_data = (
            sales_data.groupby(["year", "week", "store_id", "product_category"])
            .agg(
                {
                    "daily_sales": "sum",
                    "is_weekend": "mean",
                    "is_holiday": "mean",
                    "holiday_factor": "mean",
                    "economic_factor": "mean",
                    "weather_factor": "mean",
                    "promotion_factor": "mean",
                    "seasonal_factor": "mean",
                }
            )
            .reset_index()
        )

        weekly_data["date"] = (
            sales_data.groupby(["year", "week"])["date"].first().reset_index()["date"]
        )
        weekly_data = weekly_data.rename(columns={"daily_sales": "weekly_sales"})
        aggregated_datasets["weekly"] = weekly_data

        # Store-level aggregation (combining all categories)
        store_daily = (
            sales_data.groupby(["date", "store_id"])
            .agg(
                {
                    "daily_sales": "sum",
                    "is_weekend": "first",
                    "is_holiday": "first",
                    "holiday_factor": "mean",
                    "economic_factor": "mean",
                    "weather_factor": "mean",
                    "promotion_factor": "mean",
                    "seasonal_factor": "mean",
                }
            )
            .reset_index()
        )

        store_daily = store_daily.rename(columns={"daily_sales": "total_store_sales"})
        aggregated_datasets["store_daily"] = store_daily

        # Category-level aggregation (combining all stores)
        category_daily = (
            sales_data.groupby(["date", "product_category"])
            .agg(
                {
                    "daily_sales": "sum",
                    "is_weekend": "first",
                    "is_holiday": "first",
                    "holiday_factor": "mean",
                    "economic_factor": "mean",
                    "weather_factor": "mean",
                    "promotion_factor": "mean",
                    "seasonal_factor": "mean",
                }
            )
            .reset_index()
        )

        category_daily = category_daily.rename(
            columns={"daily_sales": "total_category_sales"}
        )
        aggregated_datasets["category_daily"] = category_daily

        return aggregated_datasets

    def _create_metadata(self) -> Dict:
        """Create metadata describing the generated datasets."""
        return {
            "generation_info": {
                "generated_at": datetime.now().isoformat(),
                "date_range": {
                    "start": self.start_date.isoformat(),
                    "end": self.end_date.isoformat(),
                    "total_days": self.n_days,
                },
                "parameters": {
                    "seasonality_strength": self.seasonality_strength,
                    "noise_level": self.noise_level,
                    "base_sales_range": self.base_sales_range,
                },
            },
            "data_structure": {
                "stores": {"count": len(self.stores), "list": self.stores},
                "product_categories": {
                    "count": len(self.product_categories),
                    "list": self.product_categories,
                },
                "external_factors": [
                    "holidays",
                    "economic_indicators",
                    "weather",
                    "promotions",
                ],
            },
            "business_scenarios": {
                "seasonal_patterns": {
                    "yearly": "Major holidays and annual cycles",
                    "monthly": "Category-specific monthly patterns",
                    "weekly": "Weekend vs weekday differences",
                    "daily": "Intra-week variations",
                },
                "external_influences": {
                    "economic": "GDP growth, unemployment, inflation, consumer confidence",
                    "weather": "Temperature, precipitation, weather conditions",
                    "promotions": "Discounts, special events, flash sales",
                    "holidays": "Major holidays and their sales impact",
                },
                "business_discontinuities": {
                    "covid_impact": "Economic shock in 2020",
                    "seasonal_clearances": "Regular clearance events",
                    "promotional_campaigns": "Various promotional strategies",
                },
            },
            "usage_recommendations": {
                "forecasting_horizons": {
                    "short_term": "1-7 days (daily data)",
                    "medium_term": "1-3 months (weekly/monthly data)",
                    "long_term": "3-12 months (monthly data)",
                },
                "model_training": {
                    "min_history": "2 years for seasonal pattern detection",
                    "validation_split": "Last 6-12 months for testing",
                    "cross_validation": "Time series split with expanding window",
                },
            },
        }


def main():
    """Generate and save sales forecasting datasets."""
    print("=== Sales Forecasting Data Generator ===")
    print("Generating realistic retail sales data with complex patterns...")

    # Initialize generator with realistic parameters
    generator = SalesDataGenerator(
        start_date="2020-01-01",
        end_date="2023-12-31",
        seasonality_strength=0.3,
        noise_level=0.1,
        random_seed=42,
    )

    # Generate and save all datasets
    file_paths = generator.save_datasets()

    print("\n=== Generation Summary ===")
    for dataset_name, file_path in file_paths.items():
        print(f"{dataset_name}: {file_path}")

    print("\n=== Next Steps ===")
    print("1. Explore the generated data in Jupyter notebooks")
    print("2. Train the hybrid forecasting model")
    print("3. Evaluate forecasting performance")
    print("4. Deploy to IntegratedML for SQL integration")


if __name__ == "__main__":
    main()
