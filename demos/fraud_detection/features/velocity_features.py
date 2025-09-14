"""
Velocity Feature Engineering

This module provides velocity-based feature engineering for fraud detection,
focusing on transaction frequency, timing patterns, and rapid-fire detection.

Key Features:
- Transaction velocity calculation across multiple time windows
- Rapid-fire transaction detection
- Customer spending pattern analysis
- Merchant transaction frequency analysis
- Time-based aggregation features
- Real-time velocity computation

Author: IntegratedML Pluggable Models Team
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import warnings

logger = logging.getLogger(__name__)


class VelocityFeatureEngineer:
    """
    Velocity-based feature engineering for fraud detection.

    This class computes various velocity features including transaction frequency,
    timing patterns, and rapid-fire detection across multiple time windows.
    """

    def __init__(
        self,
        time_windows: Optional[List[int]] = None,
        customer_velocity: bool = True,
        merchant_velocity: bool = True,
        amount_velocity: bool = True,
        location_velocity: bool = True,
        cache_size: int = 10000,
    ):
        """
        Initialize velocity feature engineer.

        Parameters:
        -----------
        time_windows : list of int, optional
            Time windows in minutes for velocity calculation
        customer_velocity : bool, default=True
            Whether to compute customer-level velocity features
        merchant_velocity : bool, default=True
            Whether to compute merchant-level velocity features
        amount_velocity : bool, default=True
            Whether to compute amount-based velocity features
        location_velocity : bool, default=True
            Whether to compute location-based velocity features
        cache_size : int, default=10000
            Size of transaction cache for real-time processing
        """
        self.time_windows = time_windows or [1, 5, 15, 60, 240, 1440]  # 1min to 1day
        self.customer_velocity = customer_velocity
        self.merchant_velocity = merchant_velocity
        self.amount_velocity = amount_velocity
        self.location_velocity = location_velocity
        self.cache_size = cache_size

        # Transaction caches for real-time processing
        self.customer_transactions = defaultdict(list)
        self.merchant_transactions = defaultdict(list)
        self.location_transactions = defaultdict(list)

        # Historical statistics for baseline computation
        self.customer_baselines = {}
        self.merchant_baselines = {}
        self.global_baselines = {}

        # Fitted status
        self.is_fitted = False

        logger.info("Initialized VelocityFeatureEngineer")

    def fit(
        self, X: pd.DataFrame, y: Optional[np.ndarray] = None
    ) -> "VelocityFeatureEngineer":
        """
        Fit the velocity feature engineer on training data.

        Parameters:
        -----------
        X : DataFrame
            Transaction data with timestamp, customer_id, etc.
        y : array-like, optional
            Fraud labels

        Returns:
        --------
        self : VelocityFeatureEngineer
            Fitted feature engineer
        """
        logger.info("Fitting VelocityFeatureEngineer...")

        if "timestamp" not in X.columns:
            raise ValueError("Timestamp column is required for velocity features")

        # Convert timestamp if needed
        if not pd.api.types.is_datetime64_any_dtype(X["timestamp"]):
            X = X.copy()
            X["timestamp"] = pd.to_datetime(X["timestamp"])

        # Compute baseline statistics from training data
        self._compute_baselines(X, y)

        self.is_fitted = True
        logger.info("VelocityFeatureEngineer fitted successfully")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform transaction data into velocity features.

        Parameters:
        -----------
        X : DataFrame
            Transaction data

        Returns:
        --------
        X_features : DataFrame
            Data with velocity features added
        """
        if not self.is_fitted:
            raise ValueError(
                "Velocity feature engineer must be fitted before transform"
            )

        logger.debug(f"Computing velocity features for {len(X)} transactions...")

        # Create copy to avoid modifying original data
        X_features = X.copy()

        # Convert timestamp if needed
        if not pd.api.types.is_datetime64_any_dtype(X_features["timestamp"]):
            X_features["timestamp"] = pd.to_datetime(X_features["timestamp"])

        # Sort by timestamp for proper velocity calculation
        X_features = X_features.sort_values("timestamp").reset_index(drop=True)

        # Customer velocity features
        if self.customer_velocity and "customer_id" in X_features.columns:
            X_features = self._add_customer_velocity_features(X_features)

        # Merchant velocity features
        if self.merchant_velocity and "merchant_name" in X_features.columns:
            X_features = self._add_merchant_velocity_features(X_features)

        # Amount velocity features
        if self.amount_velocity and "amount" in X_features.columns:
            X_features = self._add_amount_velocity_features(X_features)

        # Location velocity features
        if self.location_velocity and "customer_location" in X_features.columns:
            X_features = self._add_location_velocity_features(X_features)

        # Cross-entity velocity features
        X_features = self._add_cross_velocity_features(X_features)

        logger.debug(f"Generated velocity features")
        return X_features

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Parameters:
        -----------
        X : DataFrame
            Transaction data
        y : array-like, optional
            Fraud labels

        Returns:
        --------
        X_features : DataFrame
            Data with velocity features
        """
        return self.fit(X, y).transform(X)

    def transform_realtime(self, transaction: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute velocity features for a single transaction in real-time.

        Parameters:
        -----------
        transaction : dict
            Single transaction data

        Returns:
        --------
        velocity_features : dict
            Computed velocity features
        """
        if not self.is_fitted:
            raise ValueError(
                "Velocity feature engineer must be fitted before real-time transform"
            )

        features = {}

        # Parse transaction timestamp
        timestamp = pd.to_datetime(transaction["timestamp"])
        customer_id = transaction.get("customer_id")
        merchant_name = transaction.get("merchant_name")
        amount = transaction.get("amount", 0)
        location = transaction.get("customer_location")

        # Customer velocity features
        if customer_id and self.customer_velocity:
            customer_features = self._compute_realtime_customer_velocity(
                customer_id, timestamp, amount, location
            )
            features.update(customer_features)

        # Merchant velocity features
        if merchant_name and self.merchant_velocity:
            merchant_features = self._compute_realtime_merchant_velocity(
                merchant_name, timestamp, amount
            )
            features.update(merchant_features)

        # Update caches with new transaction
        self._update_transaction_caches(transaction, timestamp)

        return features

    def _compute_baselines(
        self, X: pd.DataFrame, y: Optional[np.ndarray] = None
    ) -> None:
        """
        Compute baseline statistics from training data.

        Parameters:
        -----------
        X : DataFrame
            Training transaction data
        y : array-like, optional
            Fraud labels
        """
        # Global baselines
        self.global_baselines = {
            "avg_transaction_count_per_hour": len(X)
            / max(
                1, (X["timestamp"].max() - X["timestamp"].min()).total_seconds() / 3600
            ),
            "avg_amount": X["amount"].mean() if "amount" in X.columns else 0,
            "std_amount": X["amount"].std() if "amount" in X.columns else 0,
        }

        # Customer baselines
        if "customer_id" in X.columns:
            customer_stats = (
                X.groupby("customer_id")
                .agg({"amount": ["count", "mean", "std"], "timestamp": ["min", "max"]})
                .reset_index()
            )

            customer_stats.columns = [
                "customer_id",
                "transaction_count",
                "avg_amount",
                "std_amount",
                "first_transaction",
                "last_transaction",
            ]
            customer_stats["days_active"] = (
                customer_stats["last_transaction"] - customer_stats["first_transaction"]
            ).dt.total_seconds() / 86400
            customer_stats["transactions_per_day"] = customer_stats[
                "transaction_count"
            ] / np.maximum(customer_stats["days_active"], 1)

            self.customer_baselines = customer_stats.set_index("customer_id").to_dict(
                "index"
            )

        # Merchant baselines
        if "merchant_name" in X.columns:
            merchant_stats = (
                X.groupby("merchant_name")
                .agg({"amount": ["count", "mean", "std"], "timestamp": ["min", "max"]})
                .reset_index()
            )

            merchant_stats.columns = [
                "merchant_name",
                "transaction_count",
                "avg_amount",
                "std_amount",
                "first_transaction",
                "last_transaction",
            ]
            merchant_stats["days_active"] = (
                merchant_stats["last_transaction"] - merchant_stats["first_transaction"]
            ).dt.total_seconds() / 86400
            merchant_stats["transactions_per_day"] = merchant_stats[
                "transaction_count"
            ] / np.maximum(merchant_stats["days_active"], 1)

            self.merchant_baselines = merchant_stats.set_index("merchant_name").to_dict(
                "index"
            )

    def _add_customer_velocity_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add customer-level velocity features.

        Parameters:
        -----------
        X : DataFrame
            Input data

        Returns:
        --------
        X : DataFrame
            Data with customer velocity features added
        """
        logger.debug("Computing customer velocity features...")

        # Initialize velocity features
        for window in self.time_windows:
            X[f"customer_tx_count_{window}m"] = 0
            X[f"customer_amount_sum_{window}m"] = 0.0
            X[f"customer_amount_avg_{window}m"] = 0.0
            X[f"customer_unique_merchants_{window}m"] = 0

        # Group by customer and compute rolling velocities
        for customer_id, group in X.groupby("customer_id"):
            group = group.sort_values("timestamp")

            for i, (idx, row) in enumerate(group.iterrows()):
                current_time = row["timestamp"]

                for window in self.time_windows:
                    # Define time window
                    start_time = current_time - timedelta(minutes=window)

                    # Get transactions in window (excluding current transaction)
                    window_mask = (group["timestamp"] >= start_time) & (
                        group["timestamp"] < current_time
                    )
                    window_txns = group[window_mask]

                    # Compute velocity features
                    X.loc[idx, f"customer_tx_count_{window}m"] = len(window_txns)

                    if len(window_txns) > 0:
                        X.loc[idx, f"customer_amount_sum_{window}m"] = window_txns[
                            "amount"
                        ].sum()
                        X.loc[idx, f"customer_amount_avg_{window}m"] = window_txns[
                            "amount"
                        ].mean()

                        if "merchant_name" in window_txns.columns:
                            X.loc[idx, f"customer_unique_merchants_{window}m"] = (
                                window_txns["merchant_name"].nunique()
                            )

        # Add derived velocity features
        for window in self.time_windows:
            # Transaction frequency (transactions per minute)
            X[f"customer_tx_frequency_{window}m"] = (
                X[f"customer_tx_count_{window}m"] / window
            )

            # Amount velocity (amount per minute)
            X[f"customer_amount_velocity_{window}m"] = (
                X[f"customer_amount_sum_{window}m"] / window
            )

            # Baseline comparison features
            if self.customer_baselines:
                baseline_key = f"customer_tx_count_{window}m"
                baseline_values = []

                for customer_id in X["customer_id"]:
                    baseline = self.customer_baselines.get(customer_id, {}).get(
                        "transactions_per_day", 1
                    )
                    expected_count = baseline * (
                        window / 1440
                    )  # Convert to expected count for window
                    baseline_values.append(expected_count)

                X[f"customer_velocity_vs_baseline_{window}m"] = X[
                    f"customer_tx_count_{window}m"
                ] / np.maximum(baseline_values, 0.1)

        # Rapid-fire detection features
        X["customer_rapid_fire_1m"] = (X["customer_tx_count_1m"] >= 3).astype(int)
        X["customer_rapid_fire_5m"] = (X["customer_tx_count_5m"] >= 5).astype(int)
        X["customer_rapid_fire_15m"] = (X["customer_tx_count_15m"] >= 10).astype(int)

        return X

    def _add_merchant_velocity_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add merchant-level velocity features.

        Parameters:
        -----------
        X : DataFrame
            Input data

        Returns:
        --------
        X : DataFrame
            Data with merchant velocity features added
        """
        logger.debug("Computing merchant velocity features...")

        # Initialize velocity features
        for window in self.time_windows:
            X[f"merchant_tx_count_{window}m"] = 0
            X[f"merchant_amount_sum_{window}m"] = 0.0
            X[f"merchant_unique_customers_{window}m"] = 0

        # Group by merchant and compute rolling velocities
        for merchant_name, group in X.groupby("merchant_name"):
            group = group.sort_values("timestamp")

            for i, (idx, row) in enumerate(group.iterrows()):
                current_time = row["timestamp"]

                for window in self.time_windows:
                    # Define time window
                    start_time = current_time - timedelta(minutes=window)

                    # Get transactions in window (excluding current transaction)
                    window_mask = (group["timestamp"] >= start_time) & (
                        group["timestamp"] < current_time
                    )
                    window_txns = group[window_mask]

                    # Compute velocity features
                    X.loc[idx, f"merchant_tx_count_{window}m"] = len(window_txns)

                    if len(window_txns) > 0:
                        X.loc[idx, f"merchant_amount_sum_{window}m"] = window_txns[
                            "amount"
                        ].sum()

                        if "customer_id" in window_txns.columns:
                            X.loc[idx, f"merchant_unique_customers_{window}m"] = (
                                window_txns["customer_id"].nunique()
                            )

        # Add derived merchant velocity features
        for window in self.time_windows:
            # Merchant transaction frequency
            X[f"merchant_tx_frequency_{window}m"] = (
                X[f"merchant_tx_count_{window}m"] / window
            )

            # Merchant amount velocity
            X[f"merchant_amount_velocity_{window}m"] = (
                X[f"merchant_amount_sum_{window}m"] / window
            )

            # Customer diversity ratio
            X[f"merchant_customer_diversity_{window}m"] = X[
                f"merchant_unique_customers_{window}m"
            ] / np.maximum(X[f"merchant_tx_count_{window}m"], 1)

        # Merchant rapid-fire detection
        X["merchant_rapid_fire_1m"] = (X["merchant_tx_count_1m"] >= 10).astype(int)
        X["merchant_rapid_fire_5m"] = (X["merchant_tx_count_5m"] >= 25).astype(int)

        return X

    def _add_amount_velocity_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add amount-based velocity features.

        Parameters:
        -----------
        X : DataFrame
            Input data

        Returns:
        --------
        X : DataFrame
            Data with amount velocity features added
        """
        # Amount change patterns
        X = X.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

        # Previous transaction amount for each customer
        X["prev_amount"] = X.groupby("customer_id")["amount"].shift(1)
        X["amount_change"] = X["amount"] - X["prev_amount"]
        X["amount_change_ratio"] = X["amount"] / np.maximum(X["prev_amount"], 0.01)
        X["amount_change_abs"] = np.abs(X["amount_change"])

        # Amount change categories
        X["amount_increase"] = (X["amount_change"] > 0).astype(int)
        X["amount_large_increase"] = (X["amount_change_ratio"] > 2.0).astype(int)
        X["amount_large_decrease"] = (X["amount_change_ratio"] < 0.5).astype(int)

        # Rolling amount statistics
        for window in [5, 15, 60]:  # Focus on shorter windows for amount patterns
            X[f"amount_rolling_mean_{window}m"] = 0.0
            X[f"amount_rolling_std_{window}m"] = 0.0
            X[f"amount_vs_rolling_mean_{window}m"] = 0.0

        # Compute rolling amount statistics
        for customer_id, group in X.groupby("customer_id"):
            group = group.sort_values("timestamp")

            for window in [5, 15, 60]:
                for i, (idx, row) in enumerate(group.iterrows()):
                    current_time = row["timestamp"]
                    start_time = current_time - timedelta(minutes=window)

                    # Get previous transactions in window
                    window_mask = (group["timestamp"] >= start_time) & (
                        group["timestamp"] < current_time
                    )
                    window_amounts = group[window_mask]["amount"]

                    if len(window_amounts) > 0:
                        rolling_mean = window_amounts.mean()
                        rolling_std = window_amounts.std()

                        X.loc[idx, f"amount_rolling_mean_{window}m"] = rolling_mean
                        X.loc[idx, f"amount_rolling_std_{window}m"] = (
                            rolling_std if not pd.isna(rolling_std) else 0
                        )
                        X.loc[idx, f"amount_vs_rolling_mean_{window}m"] = row[
                            "amount"
                        ] / max(rolling_mean, 0.01)

        return X

    def _add_location_velocity_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add location-based velocity features.

        Parameters:
        -----------
        X : DataFrame
            Input data

        Returns:
        --------
        X : DataFrame
            Data with location velocity features added
        """
        if "customer_location" not in X.columns:
            return X

        # Location change detection
        X = X.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
        X["prev_location"] = X.groupby("customer_id")["customer_location"].shift(1)
        X["location_changed"] = (X["customer_location"] != X["prev_location"]).astype(
            int
        )

        # Location velocity features
        for window in [15, 60, 240]:  # Focus on meaningful time windows for location
            X[f"location_changes_{window}m"] = 0
            X[f"unique_locations_{window}m"] = 0

        # Compute location velocities
        for customer_id, group in X.groupby("customer_id"):
            group = group.sort_values("timestamp")

            for window in [15, 60, 240]:
                for i, (idx, row) in enumerate(group.iterrows()):
                    current_time = row["timestamp"]
                    start_time = current_time - timedelta(minutes=window)

                    # Get transactions in window
                    window_mask = (group["timestamp"] >= start_time) & (
                        group["timestamp"] < current_time
                    )
                    window_txns = group[window_mask]

                    if len(window_txns) > 0:
                        # Count location changes
                        location_changes = (
                            window_txns["customer_location"]
                            != window_txns["customer_location"].shift(1)
                        ).sum()
                        X.loc[idx, f"location_changes_{window}m"] = location_changes

                        # Count unique locations
                        X.loc[idx, f"unique_locations_{window}m"] = window_txns[
                            "customer_location"
                        ].nunique()

        # Location velocity ratios
        X["location_velocity_15m"] = (
            X["location_changes_15m"] / 15 * 60
        )  # changes per hour
        X["location_velocity_60m"] = (
            X["location_changes_60m"] / 60 * 60
        )  # changes per hour

        # Impossible travel detection (simplified)
        X["potential_impossible_travel"] = (X["location_changes_15m"] >= 2).astype(int)

        return X

    def _add_cross_velocity_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add cross-entity velocity features.

        Parameters:
        -----------
        X : DataFrame
            Input data

        Returns:
        --------
        X : DataFrame
            Data with cross-velocity features added
        """
        # Velocity ratio features (customer vs merchant activity)
        if self.customer_velocity and self.merchant_velocity:
            for window in [5, 15, 60]:
                customer_col = f"customer_tx_count_{window}m"
                merchant_col = f"merchant_tx_count_{window}m"

                if customer_col in X.columns and merchant_col in X.columns:
                    X[f"customer_merchant_velocity_ratio_{window}m"] = X[
                        customer_col
                    ] / np.maximum(X[merchant_col], 1)

        # Velocity anomaly scores (how unusual is this velocity?)
        for window in [1, 5, 15]:
            if f"customer_tx_count_{window}m" in X.columns:
                # Z-score based on global statistics
                mean_count = X[f"customer_tx_count_{window}m"].mean()
                std_count = X[f"customer_tx_count_{window}m"].std()

                if std_count > 0:
                    X[f"customer_velocity_zscore_{window}m"] = (
                        X[f"customer_tx_count_{window}m"] - mean_count
                    ) / std_count
                else:
                    X[f"customer_velocity_zscore_{window}m"] = 0

        return X

    def _compute_realtime_customer_velocity(
        self, customer_id: str, timestamp: datetime, amount: float, location: str
    ) -> Dict[str, float]:
        """
        Compute real-time customer velocity features.

        Parameters:
        -----------
        customer_id : str
            Customer identifier
        timestamp : datetime
            Transaction timestamp
        amount : float
            Transaction amount
        location : str
            Customer location

        Returns:
        --------
        features : dict
            Real-time velocity features
        """
        features = {}

        # Get customer's recent transactions
        customer_txns = self.customer_transactions[customer_id]

        for window in self.time_windows:
            start_time = timestamp - timedelta(minutes=window)

            # Filter transactions in window
            window_txns = [
                txn for txn in customer_txns if txn["timestamp"] >= start_time
            ]

            # Compute velocity features
            features[f"customer_tx_count_{window}m"] = len(window_txns)
            features[f"customer_amount_sum_{window}m"] = sum(
                txn["amount"] for txn in window_txns
            )
            features[f"customer_tx_frequency_{window}m"] = len(window_txns) / window

            if window_txns:
                features[f"customer_amount_avg_{window}m"] = features[
                    f"customer_amount_sum_{window}m"
                ] / len(window_txns)
                features[f"customer_unique_merchants_{window}m"] = len(
                    set(txn.get("merchant_name", "") for txn in window_txns)
                )
            else:
                features[f"customer_amount_avg_{window}m"] = 0.0
                features[f"customer_unique_merchants_{window}m"] = 0

        # Rapid-fire detection
        features["customer_rapid_fire_1m"] = int(features["customer_tx_count_1m"] >= 3)
        features["customer_rapid_fire_5m"] = int(features["customer_tx_count_5m"] >= 5)

        return features

    def _compute_realtime_merchant_velocity(
        self, merchant_name: str, timestamp: datetime, amount: float
    ) -> Dict[str, float]:
        """
        Compute real-time merchant velocity features.

        Parameters:
        -----------
        merchant_name : str
            Merchant name
        timestamp : datetime
            Transaction timestamp
        amount : float
            Transaction amount

        Returns:
        --------
        features : dict
            Real-time merchant velocity features
        """
        features = {}

        # Get merchant's recent transactions
        merchant_txns = self.merchant_transactions[merchant_name]

        for window in self.time_windows:
            start_time = timestamp - timedelta(minutes=window)

            # Filter transactions in window
            window_txns = [
                txn for txn in merchant_txns if txn["timestamp"] >= start_time
            ]

            # Compute velocity features
            features[f"merchant_tx_count_{window}m"] = len(window_txns)
            features[f"merchant_amount_sum_{window}m"] = sum(
                txn["amount"] for txn in window_txns
            )
            features[f"merchant_tx_frequency_{window}m"] = len(window_txns) / window

            if window_txns:
                features[f"merchant_unique_customers_{window}m"] = len(
                    set(txn.get("customer_id", "") for txn in window_txns)
                )
            else:
                features[f"merchant_unique_customers_{window}m"] = 0

        # Merchant rapid-fire detection
        features["merchant_rapid_fire_1m"] = int(features["merchant_tx_count_1m"] >= 10)
        features["merchant_rapid_fire_5m"] = int(features["merchant_tx_count_5m"] >= 25)

        return features

    def _update_transaction_caches(
        self, transaction: Dict[str, Any], timestamp: datetime
    ) -> None:
        """
        Update transaction caches with new transaction.

        Parameters:
        -----------
        transaction : dict
            New transaction data
        timestamp : datetime
            Transaction timestamp
        """
        # Prepare transaction record
        txn_record = {
            "timestamp": timestamp,
            "amount": transaction.get("amount", 0),
            "merchant_name": transaction.get("merchant_name", ""),
            "customer_id": transaction.get("customer_id", ""),
            "location": transaction.get("customer_location", ""),
        }

        # Update customer cache
        customer_id = transaction.get("customer_id")
        if customer_id:
            self.customer_transactions[customer_id].append(txn_record)
            # Maintain cache size
            if len(self.customer_transactions[customer_id]) > self.cache_size:
                self.customer_transactions[customer_id] = self.customer_transactions[
                    customer_id
                ][-self.cache_size :]

        # Update merchant cache
        merchant_name = transaction.get("merchant_name")
        if merchant_name:
            self.merchant_transactions[merchant_name].append(txn_record)
            # Maintain cache size
            if len(self.merchant_transactions[merchant_name]) > self.cache_size:
                self.merchant_transactions[merchant_name] = self.merchant_transactions[
                    merchant_name
                ][-self.cache_size :]

        # Update location cache
        location = transaction.get("customer_location")
        if location:
            self.location_transactions[location].append(txn_record)
            # Maintain cache size
            if len(self.location_transactions[location]) > self.cache_size:
                self.location_transactions[location] = self.location_transactions[
                    location
                ][-self.cache_size :]

    def get_feature_names(self) -> List[str]:
        """
        Get names of all generated velocity features.

        Returns:
        --------
        feature_names : list of str
            List of velocity feature names
        """
        features = []

        # Customer velocity features
        if self.customer_velocity:
            for window in self.time_windows:
                features.extend(
                    [
                        f"customer_tx_count_{window}m",
                        f"customer_amount_sum_{window}m",
                        f"customer_amount_avg_{window}m",
                        f"customer_unique_merchants_{window}m",
                        f"customer_tx_frequency_{window}m",
                        f"customer_amount_velocity_{window}m",
                    ]
                )

            features.extend(
                [
                    "customer_rapid_fire_1m",
                    "customer_rapid_fire_5m",
                    "customer_rapid_fire_15m",
                ]
            )

        # Merchant velocity features
        if self.merchant_velocity:
            for window in self.time_windows:
                features.extend(
                    [
                        f"merchant_tx_count_{window}m",
                        f"merchant_amount_sum_{window}m",
                        f"merchant_unique_customers_{window}m",
                        f"merchant_tx_frequency_{window}m",
                        f"merchant_amount_velocity_{window}m",
                        f"merchant_customer_diversity_{window}m",
                    ]
                )

            features.extend(["merchant_rapid_fire_1m", "merchant_rapid_fire_5m"])

        # Amount velocity features
        if self.amount_velocity:
            features.extend(
                [
                    "prev_amount",
                    "amount_change",
                    "amount_change_ratio",
                    "amount_change_abs",
                    "amount_increase",
                    "amount_large_increase",
                    "amount_large_decrease",
                ]
            )

            for window in [5, 15, 60]:
                features.extend(
                    [
                        f"amount_rolling_mean_{window}m",
                        f"amount_rolling_std_{window}m",
                        f"amount_vs_rolling_mean_{window}m",
                    ]
                )

        # Location velocity features
        if self.location_velocity:
            features.extend(
                [
                    "prev_location",
                    "location_changed",
                    "location_velocity_15m",
                    "location_velocity_60m",
                    "potential_impossible_travel",
                ]
            )

            for window in [15, 60, 240]:
                features.extend(
                    [f"location_changes_{window}m", f"unique_locations_{window}m"]
                )

        return features

    def get_velocity_summary(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of velocity patterns in the data.

        Parameters:
        -----------
        X : DataFrame
            Data with velocity features

        Returns:
        --------
        summary : dict
            Velocity pattern summary
        """
        summary = {}

        # Customer velocity summary
        if self.customer_velocity:
            rapid_fire_1m = X.get("customer_rapid_fire_1m", pd.Series([])).sum()
            rapid_fire_5m = X.get("customer_rapid_fire_5m", pd.Series([])).sum()

            summary["customer_velocity"] = {
                "rapid_fire_1m_count": rapid_fire_1m,
                "rapid_fire_5m_count": rapid_fire_5m,
                "avg_tx_count_5m": X.get("customer_tx_count_5m", pd.Series([])).mean(),
                "max_tx_count_5m": X.get("customer_tx_count_5m", pd.Series([])).max(),
            }

        # Merchant velocity summary
        if self.merchant_velocity:
            merchant_rapid_fire_1m = X.get(
                "merchant_rapid_fire_1m", pd.Series([])
            ).sum()
            merchant_rapid_fire_5m = X.get(
                "merchant_rapid_fire_5m", pd.Series([])
            ).sum()

            summary["merchant_velocity"] = {
                "rapid_fire_1m_count": merchant_rapid_fire_1m,
                "rapid_fire_5m_count": merchant_rapid_fire_5m,
                "avg_tx_count_5m": X.get("merchant_tx_count_5m", pd.Series([])).mean(),
                "max_tx_count_5m": X.get("merchant_tx_count_5m", pd.Series([])).max(),
            }

        return summary
