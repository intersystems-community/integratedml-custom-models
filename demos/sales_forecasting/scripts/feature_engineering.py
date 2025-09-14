"""
Advanced Feature Engineering for Sales Forecasting.

This module provides sophisticated feature engineering capabilities for time series
forecasting, including lag features, rolling statistics, seasonal decomposition,
and external factor transformations.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression


class TimeSeriesFeatureEngineer:
    """
    Advanced feature engineering for time series forecasting.

    This class provides comprehensive feature engineering capabilities including:
    - Lag features with adaptive selection
    - Rolling statistics with multiple windows
    - Seasonal decomposition and encoding
    - Trend detection and change point features
    - External regressor transformations
    - Feature selection and importance analysis
    """

    def __init__(
        self,
        target_col: str = "y",
        date_col: str = "ds",
        freq: str = "D",
        lag_features: List[int] = None,
        rolling_windows: List[int] = None,
        seasonal_periods: List[int] = None,
        include_holidays: bool = True,
        feature_selection_k: int = None,
        scaling: bool = True,
    ):
        """
        Initialize the feature engineer.

        Parameters:
        -----------
        target_col : str
            Name of the target column
        date_col : str
            Name of the date column
        freq : str
            Frequency of the time series ('D', 'W', 'M', etc.)
        lag_features : list of int, optional
            Lag periods to create features for
        rolling_windows : list of int, optional
            Rolling window sizes for statistics
        seasonal_periods : list of int, optional
            Seasonal periods for decomposition
        include_holidays : bool
            Whether to include holiday features
        feature_selection_k : int, optional
            Number of top features to select
        scaling : bool
            Whether to scale features
        """
        self.target_col = target_col
        self.date_col = date_col
        self.freq = freq
        self.lag_features = lag_features or [1, 2, 3, 7, 14, 30]
        self.rolling_windows = rolling_windows or [7, 14, 30, 90]
        self.seasonal_periods = seasonal_periods or [7, 30, 365]
        self.include_holidays = include_holidays
        self.feature_selection_k = feature_selection_k
        self.scaling = scaling

        # State variables
        self.feature_names_ = []
        self.scaler_ = StandardScaler() if scaling else None
        self.feature_selector_ = None
        self.label_encoders_ = {}
        self.is_fitted_ = False

        # Feature importance tracking
        self.feature_importance_ = {}

    def fit(
        self, data: pd.DataFrame, external_data: Optional[pd.DataFrame] = None
    ) -> "TimeSeriesFeatureEngineer":
        """
        Fit the feature engineer on training data.

        Parameters:
        -----------
        data : DataFrame
            Training time series data
        external_data : DataFrame, optional
            External regressors data

        Returns:
        --------
        self : TimeSeriesFeatureEngineer
            Fitted feature engineer
        """
        print("Fitting feature engineer...")

        # Prepare data
        data_clean = self._prepare_data(data, external_data)

        # Generate all features
        features = self._generate_all_features(data_clean)

        # Remove features with too many missing values
        features = self._handle_missing_values(features)

        # Fit label encoders for categorical features
        self._fit_categorical_encoders(features)

        # Encode categorical features
        features_encoded = self._encode_categorical_features(features)

        # Fit scaler if needed
        if self.scaler_ is not None:
            numeric_cols = features_encoded.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                self.scaler_.fit(features_encoded[numeric_cols])

        # Feature selection if specified
        if self.feature_selection_k is not None:
            self._fit_feature_selector(features_encoded, data_clean[self.target_col])

        # Store feature names
        self.feature_names_ = features_encoded.columns.tolist()

        self.is_fitted_ = True
        print(
            f"Feature engineering fitted. Generated {len(self.feature_names_)} features."
        )

        return self

    def transform(
        self, data: pd.DataFrame, external_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Transform data using fitted feature engineer.

        Parameters:
        -----------
        data : DataFrame
            Time series data to transform
        external_data : DataFrame, optional
            External regressors data

        Returns:
        --------
        features : DataFrame
            Engineered features
        """
        if not self.is_fitted_:
            raise ValueError("Feature engineer must be fitted before transform")

        # Prepare data
        data_clean = self._prepare_data(data, external_data)

        # Generate all features
        features = self._generate_all_features(data_clean)

        # Handle missing values
        features = self._handle_missing_values(features, is_training=False)

        # Encode categorical features
        features_encoded = self._encode_categorical_features(features)

        # Ensure all training features are present
        for col in self.feature_names_:
            if col not in features_encoded.columns:
                features_encoded[col] = 0  # Default value for missing features

        # Reorder columns to match training
        features_encoded = features_encoded[self.feature_names_]

        # Scale features
        if self.scaler_ is not None:
            numeric_cols = features_encoded.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                features_encoded[numeric_cols] = self.scaler_.transform(
                    features_encoded[numeric_cols]
                )

        # Apply feature selection
        if self.feature_selector_ is not None:
            selected_features = self.feature_selector_.transform(features_encoded)
            selected_feature_names = [
                self.feature_names_[i]
                for i in self.feature_selector_.get_support(indices=True)
            ]
            features_encoded = pd.DataFrame(
                selected_features,
                columns=selected_feature_names,
                index=features_encoded.index,
            )

        return features_encoded

    def fit_transform(
        self, data: pd.DataFrame, external_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Fit the feature engineer and transform data in one step.

        Parameters:
        -----------
        data : DataFrame
            Training time series data
        external_data : DataFrame, optional
            External regressors data

        Returns:
        --------
        features : DataFrame
            Engineered features
        """
        return self.fit(data, external_data).transform(data, external_data)

    def _prepare_data(
        self, data: pd.DataFrame, external_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Prepare and validate input data.

        Parameters:
        -----------
        data : DataFrame
            Time series data
        external_data : DataFrame, optional
            External regressors

        Returns:
        --------
        data_clean : DataFrame
            Cleaned and prepared data
        """
        data_clean = data.copy()

        # Ensure date column is datetime
        if self.date_col in data_clean.columns:
            data_clean[self.date_col] = pd.to_datetime(data_clean[self.date_col])
        else:
            raise ValueError(f"Date column '{self.date_col}' not found in data")

        # Ensure target column exists
        if self.target_col not in data_clean.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")

        # Sort by date
        data_clean = data_clean.sort_values(self.date_col).reset_index(drop=True)

        # Merge external data if provided
        if external_data is not None:
            if self.date_col in external_data.columns:
                external_clean = external_data.copy()
                external_clean[self.date_col] = pd.to_datetime(
                    external_clean[self.date_col]
                )
                data_clean = pd.merge(
                    data_clean, external_clean, on=self.date_col, how="left"
                )
            else:
                print("Warning: External data doesn't have date column, skipping merge")

        return data_clean

    def _generate_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all types of features.

        Parameters:
        -----------
        data : DataFrame
            Prepared time series data

        Returns:
        --------
        features : DataFrame
            All generated features
        """
        features = pd.DataFrame(index=data.index)

        # Date-based features
        date_features = self._create_date_features(data[self.date_col])
        features = pd.concat([features, date_features], axis=1)

        # Lag features
        lag_features = self._create_lag_features(data[self.target_col])
        features = pd.concat([features, lag_features], axis=1)

        # Rolling statistics
        rolling_features = self._create_rolling_features(data[self.target_col])
        features = pd.concat([features, rolling_features], axis=1)

        # Trend features
        trend_features = self._create_trend_features(
            data[self.target_col], data[self.date_col]
        )
        features = pd.concat([features, trend_features], axis=1)

        # Seasonal features
        seasonal_features = self._create_seasonal_features(data[self.date_col])
        features = pd.concat([features, seasonal_features], axis=1)

        # Holiday features
        if self.include_holidays:
            holiday_features = self._create_holiday_features(data[self.date_col])
            features = pd.concat([features, holiday_features], axis=1)

        # Change point features
        change_features = self._create_change_point_features(data[self.target_col])
        features = pd.concat([features, change_features], axis=1)

        # External regressor features
        external_features = self._create_external_features(data)
        features = pd.concat([features, external_features], axis=1)

        # Interaction features
        interaction_features = self._create_interaction_features(features)
        features = pd.concat([features, interaction_features], axis=1)

    def engineer_features(
        self, target_series: pd.Series, external_factors: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Engineer features for the target time series.

        Parameters:
        -----------
        target_series : pd.Series
            Time series data with datetime index
        external_factors : pd.DataFrame, optional
            External factors with datetime index

        Returns:
        --------
        pd.DataFrame
            Engineered features with target column
        """
        # Prepare data
        data = pd.DataFrame({self.target_col: target_series})
        data[self.date_col] = target_series.index

        # Add external factors if provided
        if external_factors is not None:
            data = data.merge(
                external_factors, left_on=self.date_col, right_index=True, how="left"
            )

        # Generate all features
        features = self._generate_all_features(data)

        # Handle case where feature generation returns None
        if features is None:
            print("Warning: Feature generation returned None, using input data")
            features = data.copy()

        # Rename target column to 'target' for consistency
        if "target" not in features.columns and self.target_col in features.columns:
            features = features.rename(columns={self.target_col: "target"})

        return features

    def _create_date_features(self, dates: pd.Series) -> pd.DataFrame:
        """Create date-based features."""
        features = pd.DataFrame(index=dates.index)

        # Basic date components
        features["year"] = dates.dt.year
        features["month"] = dates.dt.month
        features["quarter"] = dates.dt.quarter
        features["day_of_month"] = dates.dt.day
        features["day_of_year"] = dates.dt.dayofyear
        features["week_of_year"] = dates.dt.isocalendar().week
        features["day_of_week"] = dates.dt.dayofweek

        # Binary indicators
        features["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
        features["is_month_start"] = dates.dt.is_month_start.astype(int)
        features["is_month_end"] = dates.dt.is_month_end.astype(int)
        features["is_quarter_start"] = dates.dt.is_quarter_start.astype(int)
        features["is_quarter_end"] = dates.dt.is_quarter_end.astype(int)
        features["is_year_start"] = dates.dt.is_year_start.astype(int)
        features["is_year_end"] = dates.dt.is_year_end.astype(int)

        # Days since epoch (for trend)
        epoch = pd.Timestamp("2020-01-01")
        features["days_since_epoch"] = (dates - epoch).dt.days

        return features

    def _create_lag_features(self, target: pd.Series) -> pd.DataFrame:
        """Create lag features."""
        features = pd.DataFrame(index=target.index)

        for lag in self.lag_features:
            features[f"lag_{lag}"] = target.shift(lag)

            # Lag differences
            if lag > 1:
                features[f"lag_diff_{lag}"] = target.shift(lag) - target.shift(lag + 1)

        return features

    def _create_rolling_features(self, target: pd.Series) -> pd.DataFrame:
        """Create rolling statistics features."""
        features = pd.DataFrame(index=target.index)

        for window in self.rolling_windows:
            # Basic rolling statistics
            features[f"rolling_mean_{window}"] = target.rolling(
                window=window, min_periods=1
            ).mean()
            features[f"rolling_std_{window}"] = target.rolling(
                window=window, min_periods=1
            ).std()
            features[f"rolling_min_{window}"] = target.rolling(
                window=window, min_periods=1
            ).min()
            features[f"rolling_max_{window}"] = target.rolling(
                window=window, min_periods=1
            ).max()
            features[f"rolling_median_{window}"] = target.rolling(
                window=window, min_periods=1
            ).median()

            # Rolling differences
            features[f"rolling_range_{window}"] = (
                features[f"rolling_max_{window}"] - features[f"rolling_min_{window}"]
            )

            # Rolling percentiles
            features[f"rolling_q25_{window}"] = target.rolling(
                window=window, min_periods=1
            ).quantile(0.25)
            features[f"rolling_q75_{window}"] = target.rolling(
                window=window, min_periods=1
            ).quantile(0.75)

            # Position relative to rolling statistics
            features[f"target_vs_rolling_mean_{window}"] = (
                target / features[f"rolling_mean_{window}"]
            )
            features[f"target_rolling_zscore_{window}"] = (
                target - features[f"rolling_mean_{window}"]
            ) / features[f"rolling_std_{window}"]

        return features

    def _create_trend_features(
        self, target: pd.Series, dates: pd.Series
    ) -> pd.DataFrame:
        """Create trend-based features."""
        features = pd.DataFrame(index=target.index)

        # Linear trend
        features["trend"] = np.arange(len(target))
        features["trend_squared"] = features["trend"] ** 2

        # Local trends (slopes over different windows)
        for window in [7, 30, 90]:
            if len(target) > window:
                slopes = []
                for i in range(len(target)):
                    start_idx = max(0, i - window + 1)
                    end_idx = i + 1

                    if end_idx - start_idx >= 2:
                        y_vals = target.iloc[start_idx:end_idx].values
                        x_vals = np.arange(len(y_vals))

                        # Calculate slope using linear regression
                        if len(y_vals) > 1 and not np.all(np.isnan(y_vals)):
                            slope = np.polyfit(x_vals, y_vals, 1)[0]
                        else:
                            slope = 0
                    else:
                        slope = 0

                    slopes.append(slope)

                features[f"trend_slope_{window}"] = slopes

        # Detrended values
        if len(target) > 2:
            x_vals = np.arange(len(target))
            y_vals = target.values

            # Fit linear trend
            valid_mask = ~np.isnan(y_vals)
            if np.sum(valid_mask) > 1:
                trend_coef = np.polyfit(x_vals[valid_mask], y_vals[valid_mask], 1)
                trend_line = np.polyval(trend_coef, x_vals)
                features["detrended"] = y_vals - trend_line
            else:
                features["detrended"] = 0

        return features

    def _create_seasonal_features(self, dates: pd.Series) -> pd.DataFrame:
        """Create seasonal decomposition features."""
        features = pd.DataFrame(index=dates.index)

        # Cyclical encoding for different periods
        periods = {
            "year": 365.25,
            "month": 30.44,  # Average days per month
            "week": 7,
            "day": 1,
        }

        for period_name, period_days in periods.items():
            if period_name == "year":
                values = dates.dt.dayofyear / period_days
            elif period_name == "month":
                values = dates.dt.day / period_days
            elif period_name == "week":
                values = dates.dt.dayofweek / period_days
            else:
                values = dates.dt.hour / 24  # For daily, use hour if available

            # Sine and cosine encoding for cyclical nature
            features[f"sin_{period_name}"] = np.sin(2 * np.pi * values)
            features[f"cos_{period_name}"] = np.cos(2 * np.pi * values)

        # Month indicators (for seasonal patterns)
        for month in range(1, 13):
            features[f"month_{month}"] = (dates.dt.month == month).astype(int)

        # Quarter indicators
        for quarter in range(1, 5):
            features[f"quarter_{quarter}"] = (dates.dt.quarter == quarter).astype(int)

        # Day of week indicators
        for dow in range(7):
            features[f"dow_{dow}"] = (dates.dt.dayofweek == dow).astype(int)

        return features

    def _create_holiday_features(self, dates: pd.Series) -> pd.DataFrame:
        """Create holiday-based features."""
        features = pd.DataFrame(index=dates.index)

        # Major holidays (simplified)
        holidays = {
            "new_year": (1, 1),
            "valentines": (2, 14),
            "independence": (7, 4),
            "halloween": (10, 31),
            "christmas": (12, 25),
        }

        for holiday_name, (month, day) in holidays.items():
            # Exact holiday
            is_holiday = (dates.dt.month == month) & (dates.dt.day == day)
            features[f"is_{holiday_name}"] = is_holiday.astype(int)

            # Days before/after holiday
            for offset in [-3, -2, -1, 1, 2, 3]:
                holiday_dates = pd.to_datetime(
                    [f"{year}-{month:02d}-{day:02d}" for year in dates.dt.year.unique()]
                )
                offset_dates = holiday_dates + pd.Timedelta(days=offset)

                is_offset = dates.isin(offset_dates)
                features[f"is_{holiday_name}_plus_{offset}"] = is_offset.astype(int)

        # Holiday season indicators
        features["is_holiday_season"] = (
            (dates.dt.month == 11) | (dates.dt.month == 12)
        ).astype(int)

        features["is_back_to_school"] = (
            (dates.dt.month == 8) | (dates.dt.month == 9)
        ).astype(int)

        return features

    def _create_change_point_features(self, target: pd.Series) -> pd.DataFrame:
        """Create change point detection features."""
        features = pd.DataFrame(index=target.index)

        # Simple change point detection using rolling statistics
        for window in [14, 30, 60]:
            if len(target) > window * 2:
                # Rolling mean change
                rolling_mean = target.rolling(window=window, min_periods=1).mean()
                mean_change = rolling_mean.diff(window)
                features[f"mean_change_{window}"] = mean_change

                # Rolling variance change
                rolling_var = target.rolling(window=window, min_periods=1).var()
                var_change = rolling_var.diff(window)
                features[f"var_change_{window}"] = var_change

                # Z-score of recent changes
                recent_values = target.rolling(window=window // 2, min_periods=1).mean()
                historical_mean = (
                    target.rolling(window=window, min_periods=1)
                    .mean()
                    .shift(window // 2)
                )
                historical_std = (
                    target.rolling(window=window, min_periods=1)
                    .std()
                    .shift(window // 2)
                )

                change_zscore = (recent_values - historical_mean) / (
                    historical_std + 1e-8
                )
                features[f"change_zscore_{window}"] = change_zscore

        return features

    def _create_external_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features from external regressors."""
        features = pd.DataFrame(index=data.index)

        # Find external regressor columns (exclude target and date)
        exclude_cols = {self.target_col, self.date_col}
        external_cols = [col for col in data.columns if col not in exclude_cols]

        for col in external_cols:
            if col in data.columns:
                # Original values
                features[f"ext_{col}"] = data[col]

                # Lagged external values
                for lag in [1, 7, 30]:
                    features[f"ext_{col}_lag_{lag}"] = data[col].shift(lag)

                # Only apply numeric operations to numeric columns
                if data[col].dtype in [np.float64, np.int64]:
                    # Changes in external values
                    features[f"ext_{col}_diff"] = data[col].diff()
                    features[f"ext_{col}_pct_change"] = data[col].pct_change()

                    # Rolling statistics of external values
                    for window in [7, 30]:
                        features[f"ext_{col}_rolling_mean_{window}"] = (
                            data[col].rolling(window=window, min_periods=1).mean()
                        )

        return features

    def _create_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables."""
        interaction_features = pd.DataFrame(index=features.index)

        # Select key features for interactions
        key_patterns = ["lag_", "rolling_mean_", "trend", "is_weekend", "month_"]
        key_features = []

        for pattern in key_patterns:
            matching_cols = [col for col in features.columns if pattern in col]
            key_features.extend(matching_cols[:2])  # Limit to prevent explosion

        # Create interactions between key features
        for i, feat1 in enumerate(key_features):
            for feat2 in key_features[i + 1 :]:
                if feat1 in features.columns and feat2 in features.columns:
                    # Multiplicative interaction
                    if features[feat1].dtype in [np.float64, np.int64] and features[
                        feat2
                    ].dtype in [np.float64, np.int64]:

                        interaction_name = f"interact_{feat1}_{feat2}"
                        if len(interaction_name) < 50:  # Avoid very long names
                            interaction_features[interaction_name] = (
                                features[feat1] * features[feat2]
                            )

        return interaction_features

    def _handle_missing_values(
        self, features: pd.DataFrame, is_training: bool = True
    ) -> pd.DataFrame:
        """Handle missing values in features."""
        features_clean = features.copy()

        if is_training:
            # Remove features with too many missing values (>50%)
            missing_pct = features_clean.isnull().sum() / len(features_clean)
            features_to_keep = missing_pct[missing_pct <= 0.5].index
            features_clean = features_clean[features_to_keep]

        # Fill remaining missing values
        for col in features_clean.columns:
            if features_clean[col].dtype in [np.float64, np.int64]:
                # Numeric: fill with median
                features_clean[col] = features_clean[col].fillna(
                    features_clean[col].median()
                )
            else:
                # Categorical: fill with mode or 'unknown'
                mode_val = features_clean[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else "unknown"
                features_clean[col] = features_clean[col].fillna(fill_val)

        return features_clean

    def _fit_categorical_encoders(self, features: pd.DataFrame) -> None:
        """Fit label encoders for categorical features."""
        for col in features.columns:
            if (
                features[col].dtype == "object"
                or features[col].dtype.name == "category"
            ):
                encoder = LabelEncoder()
                encoder.fit(features[col].fillna("unknown").astype(str))
                self.label_encoders_[col] = encoder

    def _encode_categorical_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using fitted encoders."""
        features_encoded = features.copy()

        for col, encoder in self.label_encoders_.items():
            if col in features_encoded.columns:
                # Handle unseen categories
                col_values = features_encoded[col].fillna("unknown").astype(str)
                encoded_values = []

                for val in col_values:
                    try:
                        encoded_val = encoder.transform([val])[0]
                    except ValueError:
                        # Unseen category, use 0
                        encoded_val = 0
                    encoded_values.append(encoded_val)

                features_encoded[col] = encoded_values

        return features_encoded

    def _fit_feature_selector(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Fit feature selector to select top k features."""
        # Remove target-related features to prevent leakage
        valid_features = features.copy()

        # Align indices
        common_idx = features.index.intersection(target.index)
        valid_features = valid_features.loc[common_idx]
        target_aligned = target.loc[common_idx]

        # Remove rows with missing target
        valid_mask = ~target_aligned.isnull()
        valid_features = valid_features[valid_mask]
        target_clean = target_aligned[valid_mask]

        if len(valid_features) > 0 and len(target_clean) > 0:
            # Use mutual information for feature selection
            self.feature_selector_ = SelectKBest(
                score_func=mutual_info_regression,
                k=min(self.feature_selection_k, len(valid_features.columns)),
            )

            self.feature_selector_.fit(valid_features, target_clean)

            # Store feature importance scores
            feature_scores = self.feature_selector_.scores_
            selected_mask = self.feature_selector_.get_support()

            self.feature_importance_ = dict(
                zip(
                    valid_features.columns[selected_mask], feature_scores[selected_mask]
                )
            )

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance_

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names_

    def save_state(self, filepath: str) -> None:
        """Save the fitted feature engineer state."""
        import pickle

        state = {
            "feature_names_": self.feature_names_,
            "scaler_": self.scaler_,
            "feature_selector_": self.feature_selector_,
            "label_encoders_": self.label_encoders_,
            "feature_importance_": self.feature_importance_,
            "is_fitted_": self.is_fitted_,
            "config": {
                "target_col": self.target_col,
                "date_col": self.date_col,
                "freq": self.freq,
                "lag_features": self.lag_features,
                "rolling_windows": self.rolling_windows,
                "seasonal_periods": self.seasonal_periods,
                "include_holidays": self.include_holidays,
                "feature_selection_k": self.feature_selection_k,
                "scaling": self.scaling,
            },
        }

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        print(f"Feature engineer state saved to {filepath}")

    def load_state(self, filepath: str) -> None:
        """Load a fitted feature engineer state."""
        import pickle

        with open(filepath, "rb") as f:
            state = pickle.load(f)

        # Restore state
        self.feature_names_ = state["feature_names_"]
        self.scaler_ = state["scaler_"]
        self.feature_selector_ = state["feature_selector_"]
        self.label_encoders_ = state["label_encoders_"]
        self.feature_importance_ = state["feature_importance_"]
        self.is_fitted_ = state["is_fitted_"]

        # Restore config
        config = state["config"]
        self.target_col = config["target_col"]
        self.date_col = config["date_col"]
        self.freq = config["freq"]
        self.lag_features = config["lag_features"]
        self.rolling_windows = config["rolling_windows"]
        self.seasonal_periods = config["seasonal_periods"]
        self.include_holidays = config["include_holidays"]
        self.feature_selection_k = config["feature_selection_k"]
        self.scaling = config["scaling"]

        print(f"Feature engineer state loaded from {filepath}")


def create_forecasting_features(
    data: pd.DataFrame,
    target_col: str = "daily_sales",
    date_col: str = "date",
    external_data: Optional[pd.DataFrame] = None,
    config: Dict[str, Any] = None,
) -> Tuple[pd.DataFrame, TimeSeriesFeatureEngineer]:
    """
    Convenience function to create forecasting features with recommended settings.

    Parameters:
    -----------
    data : DataFrame
        Time series data
    target_col : str
        Name of target column
    date_col : str
        Name of date column
    external_data : DataFrame, optional
        External regressors
    config : dict, optional
        Custom configuration

    Returns:
    --------
    features : DataFrame
        Engineered features
    engineer : TimeSeriesFeatureEngineer
        Fitted feature engineer
    """
    default_config = {
        "lag_features": [1, 2, 3, 7, 14, 30],
        "rolling_windows": [7, 14, 30, 90],
        "seasonal_periods": [7, 30, 365],
        "include_holidays": True,
        "feature_selection_k": 50,
        "scaling": True,
    }

    if config:
        default_config.update(config)

    # Initialize feature engineer
    engineer = TimeSeriesFeatureEngineer(
        target_col=target_col, date_col=date_col, **default_config
    )

    # Fit and transform
    features = engineer.fit_transform(data, external_data)

    return features, engineer


def analyze_feature_importance(
    engineer: TimeSeriesFeatureEngineer, top_k: int = 20
) -> pd.DataFrame:
    """
    Analyze and display feature importance.

    Parameters:
    -----------
    engineer : TimeSeriesFeatureEngineer
        Fitted feature engineer
    top_k : int
        Number of top features to show

    Returns:
    --------
    importance_df : DataFrame
        Feature importance analysis
    """
    importance_scores = engineer.get_feature_importance()

    if not importance_scores:
        print("No feature importance scores available")
        return pd.DataFrame()

    # Create importance dataframe
    importance_df = pd.DataFrame(
        [
            {"feature": feat, "importance": score}
            for feat, score in importance_scores.items()
        ]
    ).sort_values("importance", ascending=False)

    # Categorize features
    def categorize_feature(feature_name):
        if "lag_" in feature_name:
            return "Lag Features"
        elif "rolling_" in feature_name:
            return "Rolling Statistics"
        elif "trend" in feature_name:
            return "Trend Features"
        elif any(x in feature_name for x in ["sin_", "cos_", "month_", "quarter_"]):
            return "Seasonal Features"
        elif "holiday" in feature_name or "is_" in feature_name:
            return "Calendar Features"
        elif "ext_" in feature_name:
            return "External Features"
        elif "interact_" in feature_name:
            return "Interaction Features"
        else:
            return "Other Features"

    importance_df["category"] = importance_df["feature"].apply(categorize_feature)

    # Show top features
    top_features = importance_df.head(top_k)

    print(f"\nTop {top_k} Most Important Features:")
    print("=" * 50)
    for _, row in top_features.iterrows():
        print(f"{row['feature']:<30} {row['importance']:.4f} ({row['category']})")

    # Show category summary
    category_summary = (
        importance_df.groupby("category")
        .agg({"importance": ["count", "mean", "sum"]})
        .round(4)
    )

    print(f"\nFeature Importance by Category:")
    print("=" * 50)
    print(category_summary)

    return importance_df


def main():
    """Example usage of the feature engineering module."""
    print("=== Time Series Feature Engineering Demo ===")

    # Generate sample data
    from demos.sales_forecasting.data.generate_sales_data import SalesDataGenerator

    generator = SalesDataGenerator(
        start_date="2022-01-01",
        end_date="2023-12-31",
        stores=["Store_001", "Store_002"],
        product_categories=["Electronics", "Clothing"],
    )

    print("Generating sample sales data...")
    sales_data = generator.generate_sales_data()

    # Prepare data for a single store-category combination
    sample_data = sales_data[
        (sales_data["store_id"] == "Store_001")
        & (sales_data["product_category"] == "Electronics")
    ].copy()

    sample_data = sample_data.rename(columns={"daily_sales": "y", "date": "ds"})

    print(f"Sample data shape: {sample_data.shape}")

    # Create external factors data
    external_data = generator._create_external_factors_dataset()
    external_data = external_data.rename(columns={"date": "ds"})

    print("Engineering features...")

    # Create features
    features, engineer = create_forecasting_features(
        data=sample_data,
        target_col="y",
        date_col="ds",
        external_data=external_data,
        config={"feature_selection_k": 30},
    )

    print(f"Generated features shape: {features.shape}")
    print(f"Feature names: {len(engineer.get_feature_names())}")

    # Analyze feature importance
    importance_analysis = analyze_feature_importance(engineer, top_k=15)

    print("\nFeature engineering demonstration complete!")


# Alias for backward compatibility and simpler imports
FeatureEngineer = TimeSeriesFeatureEngineer


if __name__ == "__main__":
    main()
