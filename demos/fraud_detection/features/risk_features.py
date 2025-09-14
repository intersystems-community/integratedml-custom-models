"""
Risk Feature Engineering

This module provides risk scoring and aggregation features for fraud detection,
combining insights from multiple feature types into comprehensive risk assessments.

Key Features:
- Multi-dimensional risk scoring
- Risk factor aggregation and weighting
- Dynamic risk threshold adjustment
- Risk pattern recognition
- Ensemble risk scoring
- Real-time risk computation

Author: IntegratedML Pluggable Models Team
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)


class RiskFeatureEngineer:
    """
    Risk scoring and aggregation feature engineering for fraud detection.

    This class combines insights from transaction, velocity, location, and
    behavioral features to create comprehensive risk assessments and scores.
    """

    def __init__(
        self,
        enable_transaction_risk: bool = True,
        enable_velocity_risk: bool = True,
        enable_location_risk: bool = True,
        enable_behavioral_risk: bool = True,
        enable_ensemble_risk: bool = True,
        risk_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize risk feature engineer.

        Parameters:
        -----------
        enable_transaction_risk : bool, default=True
            Whether to compute transaction-level risk features
        enable_velocity_risk : bool, default=True
            Whether to compute velocity-based risk features
        enable_location_risk : bool, default=True
            Whether to compute location-based risk features
        enable_behavioral_risk : bool, default=True
            Whether to compute behavioral risk features
        enable_ensemble_risk : bool, default=True
            Whether to compute ensemble risk scores
        risk_weights : dict, optional
            Custom weights for different risk components
        """
        self.enable_transaction_risk = enable_transaction_risk
        self.enable_velocity_risk = enable_velocity_risk
        self.enable_location_risk = enable_location_risk
        self.enable_behavioral_risk = enable_behavioral_risk
        self.enable_ensemble_risk = enable_ensemble_risk

        # Default risk weights
        self.risk_weights = risk_weights or {
            "transaction": 0.25,
            "velocity": 0.25,
            "location": 0.25,
            "behavioral": 0.25,
        }

        # Risk thresholds and baselines
        self.risk_thresholds = {"low_risk": 0.3, "medium_risk": 0.6, "high_risk": 0.8}

        # Risk pattern templates
        self.fraud_patterns = {}

        # Historical risk statistics
        self.risk_statistics = {}

        # Fitted status
        self.is_fitted = False

        logger.info("Initialized RiskFeatureEngineer")

    def fit(
        self, X: pd.DataFrame, y: Optional[np.ndarray] = None
    ) -> "RiskFeatureEngineer":
        """
        Fit the risk feature engineer on training data.

        Parameters:
        -----------
        X : DataFrame
            Transaction data with other features already computed
        y : array-like, optional
            Fraud labels for risk pattern learning

        Returns:
        --------
        self : RiskFeatureEngineer
            Fitted feature engineer
        """
        logger.info("Fitting RiskFeatureEngineer...")

        # Learn fraud patterns if labels provided
        if y is not None:
            self._learn_fraud_patterns(X, y)

        # Compute risk statistics
        self._compute_risk_statistics(X)

        # Calibrate risk thresholds
        if y is not None:
            self._calibrate_risk_thresholds(X, y)

        self.is_fitted = True
        logger.info("RiskFeatureEngineer fitted successfully")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform transaction data into risk features.

        Parameters:
        -----------
        X : DataFrame
            Transaction data with other features

        Returns:
        --------
        X_features : DataFrame
            Data with risk features added
        """
        if not self.is_fitted:
            raise ValueError("Risk feature engineer must be fitted before transform")

        logger.debug(f"Computing risk features for {len(X)} transactions...")

        # Create copy to avoid modifying original data
        X_features = X.copy()

        # Transaction risk features
        if self.enable_transaction_risk:
            X_features = self._add_transaction_risk_features(X_features)

        # Velocity risk features
        if self.enable_velocity_risk:
            X_features = self._add_velocity_risk_features(X_features)

        # Location risk features
        if self.enable_location_risk:
            X_features = self._add_location_risk_features(X_features)

        # Behavioral risk features
        if self.enable_behavioral_risk:
            X_features = self._add_behavioral_risk_features(X_features)

        # Ensemble risk features
        if self.enable_ensemble_risk:
            X_features = self._add_ensemble_risk_features(X_features)

        # Risk pattern matching
        X_features = self._add_pattern_matching_features(X_features)

        # Final risk categorization
        X_features = self._add_risk_categorization(X_features)

        logger.debug(f"Generated risk features")
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
            Data with risk features
        """
        return self.fit(X, y).transform(X)

    def _learn_fraud_patterns(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Learn fraud patterns from training data.

        Parameters:
        -----------
        X : DataFrame
            Training data with features
        y : array-like
            Fraud labels
        """
        fraud_data = X[y == 1]
        legitimate_data = X[y == 0]

        # High-risk transaction patterns
        if "amount_vs_personal_avg" in X.columns:
            fraud_amount_stats = fraud_data["amount_vs_personal_avg"].describe()
            legit_amount_stats = legitimate_data["amount_vs_personal_avg"].describe()

            self.fraud_patterns["amount_ratio"] = {
                "fraud_median": fraud_amount_stats["50%"],
                "fraud_75th": fraud_amount_stats["75%"],
                "legit_median": legit_amount_stats["50%"],
                "risk_threshold": fraud_amount_stats["25%"],  # 25th percentile of fraud
            }

        # High-risk velocity patterns
        velocity_features = [
            col for col in X.columns if "velocity" in col or "rapid_fire" in col
        ]
        if velocity_features:
            fraud_velocity = fraud_data[velocity_features].mean()
            legit_velocity = legitimate_data[velocity_features].mean()

            self.fraud_patterns["velocity"] = {
                "fraud_means": fraud_velocity.to_dict(),
                "legit_means": legit_velocity.to_dict(),
                "risk_multipliers": (
                    fraud_velocity / (legit_velocity + 0.001)
                ).to_dict(),
            }

        # High-risk location patterns
        location_features = [
            col
            for col in X.columns
            if "location_risk" in col or "impossible_travel" in col
        ]
        if location_features:
            fraud_location = fraud_data[location_features].mean()
            legit_location = legitimate_data[location_features].mean()

            self.fraud_patterns["location"] = {
                "fraud_means": fraud_location.to_dict(),
                "legit_means": legit_location.to_dict(),
                "risk_indicators": (fraud_location > legit_location * 1.5).to_dict(),
            }

        # High-risk behavioral patterns
        behavioral_features = [
            col for col in X.columns if "behavioral" in col or "deviation" in col
        ]
        if behavioral_features:
            fraud_behavioral = fraud_data[behavioral_features].mean()
            legit_behavioral = legitimate_data[behavioral_features].mean()

            self.fraud_patterns["behavioral"] = {
                "fraud_means": fraud_behavioral.to_dict(),
                "legit_means": legit_behavioral.to_dict(),
                "deviation_thresholds": (
                    fraud_behavioral * 0.8
                ).to_dict(),  # Conservative threshold
            }

    def _compute_risk_statistics(self, X: pd.DataFrame) -> None:
        """
        Compute risk statistics from data.

        Parameters:
        -----------
        X : DataFrame
            Data with features
        """
        # Amount risk statistics
        if "amount" in X.columns:
            self.risk_statistics["amount"] = {
                "mean": X["amount"].mean(),
                "std": X["amount"].std(),
                "percentiles": X["amount"]
                .quantile([0.5, 0.75, 0.9, 0.95, 0.99])
                .to_dict(),
            }

        # Velocity statistics
        velocity_cols = [col for col in X.columns if "velocity" in col]
        if velocity_cols:
            self.risk_statistics["velocity"] = {}
            for col in velocity_cols:
                self.risk_statistics["velocity"][col] = {
                    "mean": X[col].mean(),
                    "std": X[col].std(),
                    "percentiles": X[col].quantile([0.9, 0.95, 0.99]).to_dict(),
                }

    def _calibrate_risk_thresholds(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Calibrate risk thresholds based on fraud data.

        Parameters:
        -----------
        X : DataFrame
            Training data
        y : array-like
            Fraud labels
        """
        # This would typically use more sophisticated threshold optimization
        # For now, we use simple percentile-based approach

        fraud_indices = y == 1
        if fraud_indices.sum() > 0:
            # Adjust thresholds based on fraud distribution
            self.risk_thresholds["low_risk"] = 0.2
            self.risk_thresholds["medium_risk"] = 0.5
            self.risk_thresholds["high_risk"] = 0.75

    def _add_transaction_risk_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add transaction-level risk features.

        Parameters:
        -----------
        X : DataFrame
            Input data

        Returns:
        --------
        X : DataFrame
            Data with transaction risk features added
        """
        # Amount-based risk scoring
        if "amount" in X.columns:
            amount_stats = self.risk_statistics.get("amount", {})
            amount_percentiles = amount_stats.get("percentiles", {})

            # High amount risk
            X["amount_risk_score"] = 0.0
            if amount_percentiles:
                conditions = [
                    (X["amount"] <= amount_percentiles.get(0.5, 0)),
                    (X["amount"] <= amount_percentiles.get(0.75, 0)),
                    (X["amount"] <= amount_percentiles.get(0.9, 0)),
                    (X["amount"] <= amount_percentiles.get(0.95, 0)),
                    (X["amount"] <= amount_percentiles.get(0.99, 0)),
                ]

                choices = [0.1, 0.3, 0.6, 0.8, 0.9]
                X["amount_risk_score"] = np.select(conditions, choices, default=1.0)

        # Round amount risk (suspicious patterns)
        if "is_round_amount" in X.columns:
            X["round_amount_risk"] = X["is_round_amount"] * 0.3

        # Payment method risk
        if "payment_method_risk_score" in X.columns:
            X["payment_method_risk"] = X["payment_method_risk_score"]
        else:
            X["payment_method_risk"] = 0.5  # Default neutral risk

        # Merchant risk
        if "merchant_risk_score" in X.columns:
            X["merchant_risk"] = X["merchant_risk_score"]
        else:
            X["merchant_risk"] = 0.5  # Default neutral risk

        # Transaction consistency risk
        consistency_features = [col for col in X.columns if "consistency" in col]
        if consistency_features:
            X["transaction_consistency_risk"] = 1.0 - X[consistency_features].mean(
                axis=1
            )
        else:
            X["transaction_consistency_risk"] = 0.0

        # Combined transaction risk
        transaction_risk_features = [
            "amount_risk_score",
            "round_amount_risk",
            "payment_method_risk",
            "merchant_risk",
            "transaction_consistency_risk",
        ]

        available_features = [f for f in transaction_risk_features if f in X.columns]
        if available_features:
            X["transaction_risk_score"] = X[available_features].mean(axis=1)

        return X

    def _add_velocity_risk_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add velocity-based risk features.

        Parameters:
        -----------
        X : DataFrame
            Input data

        Returns:
        --------
        X : DataFrame
            Data with velocity risk features added
        """
        # Rapid-fire transaction risk
        rapid_fire_features = [col for col in X.columns if "rapid_fire" in col]
        if rapid_fire_features:
            X["rapid_fire_risk_score"] = X[rapid_fire_features].max(axis=1)
        else:
            X["rapid_fire_risk_score"] = 0.0

        # High velocity risk
        velocity_features = [
            col for col in X.columns if "velocity" in col and "rapid_fire" not in col
        ]
        if velocity_features:
            velocity_risks = []

            for feature in velocity_features:
                if feature in self.risk_statistics.get("velocity", {}):
                    feature_stats = self.risk_statistics["velocity"][feature]
                    percentile_99 = feature_stats.get("percentiles", {}).get(0.99, 1.0)

                    # Risk score based on percentile position
                    risk_score = np.minimum(X[feature] / max(percentile_99, 0.001), 1.0)
                    velocity_risks.append(risk_score)

            if velocity_risks:
                X["velocity_risk_score"] = np.mean(velocity_risks, axis=0)
            else:
                X["velocity_risk_score"] = 0.0
        else:
            X["velocity_risk_score"] = 0.0

        # Transaction frequency anomaly
        frequency_features = [col for col in X.columns if "frequency" in col]
        if frequency_features:
            X["frequency_anomaly_risk"] = X[frequency_features].max(axis=1)
            # Normalize to 0-1 range
            max_freq = X["frequency_anomaly_risk"].max()
            if max_freq > 0:
                X["frequency_anomaly_risk"] = X["frequency_anomaly_risk"] / max_freq
        else:
            X["frequency_anomaly_risk"] = 0.0

        # Combined velocity risk
        velocity_risk_features = [
            "rapid_fire_risk_score",
            "velocity_risk_score",
            "frequency_anomaly_risk",
        ]

        available_features = [f for f in velocity_risk_features if f in X.columns]
        if available_features:
            X["velocity_risk_combined"] = X[available_features].mean(axis=1)

        return X

    def _add_location_risk_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add location-based risk features.

        Parameters:
        -----------
        X : DataFrame
            Input data

        Returns:
        --------
        X : DataFrame
            Data with location risk features added
        """
        # Geographic risk
        if "combined_location_risk" in X.columns:
            X["geographic_risk_score"] = X["combined_location_risk"]
        else:
            # Fallback to individual location risks
            location_risk_features = [
                col for col in X.columns if "location_risk" in col
            ]
            if location_risk_features:
                X["geographic_risk_score"] = X[location_risk_features].mean(axis=1)
            else:
                X["geographic_risk_score"] = 0.5  # Default neutral risk

        # Travel pattern risk
        if "impossible_travel" in X.columns:
            X["travel_risk_score"] = X["impossible_travel"].astype(float)

            # Add suspicious travel as medium risk
            if "suspicious_travel" in X.columns:
                X["travel_risk_score"] = np.maximum(
                    X["travel_risk_score"], X["suspicious_travel"] * 0.6
                )
        else:
            X["travel_risk_score"] = 0.0

        # Distance anomaly risk
        if "is_very_long_distance" in X.columns:
            X["distance_anomaly_risk"] = X["is_very_long_distance"] * 0.7
        elif "is_long_distance" in X.columns:
            X["distance_anomaly_risk"] = X["is_long_distance"] * 0.4
        else:
            X["distance_anomaly_risk"] = 0.0

        # Timezone inconsistency risk
        if "timezone_mismatch" in X.columns:
            X["timezone_risk_score"] = X["timezone_mismatch"] * 0.3

            # Enhance with timezone difference
            if "timezone_difference" in X.columns:
                normalized_tz_diff = np.minimum(X["timezone_difference"] / 12.0, 1.0)
                X["timezone_risk_score"] = np.maximum(
                    X["timezone_risk_score"], normalized_tz_diff * 0.5
                )
        else:
            X["timezone_risk_score"] = 0.0

        # Combined location risk
        location_risk_features = [
            "geographic_risk_score",
            "travel_risk_score",
            "distance_anomaly_risk",
            "timezone_risk_score",
        ]

        available_features = [f for f in location_risk_features if f in X.columns]
        if available_features:
            X["location_risk_combined"] = X[available_features].mean(axis=1)

        return X

    def _add_behavioral_risk_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add behavioral risk features.

        Parameters:
        -----------
        X : DataFrame
            Input data

        Returns:
        --------
        X : DataFrame
            Data with behavioral risk features added
        """
        # Behavioral deviation risk
        if "behavioral_deviation_score" in X.columns:
            X["behavioral_risk_score"] = X["behavioral_deviation_score"]
        else:
            X["behavioral_risk_score"] = 0.0

        # Pattern break risk
        if "pattern_break_count" in X.columns:
            max_breaks = X["pattern_break_count"].max()
            if max_breaks > 0:
                X["pattern_break_risk"] = X["pattern_break_count"] / max_breaks
            else:
                X["pattern_break_risk"] = 0.0
        else:
            X["pattern_break_risk"] = 0.0

        # New customer risk
        if "is_new_customer" in X.columns:
            X["new_customer_risk"] = X["is_new_customer"] * 0.4  # Moderate risk
        else:
            X["new_customer_risk"] = 0.0

        # Dormant reactivation risk
        if "is_dormant_reactivation" in X.columns:
            X["dormant_reactivation_risk"] = X["is_dormant_reactivation"] * 0.6
        else:
            X["dormant_reactivation_risk"] = 0.0

        # Spending pattern risk
        spending_risk_features = [
            col
            for col in X.columns
            if any(
                keyword in col
                for keyword in ["unusual_amount", "spending_burst", "very_large_amount"]
            )
        ]

        if spending_risk_features:
            X["spending_pattern_risk"] = X[spending_risk_features].max(axis=1)
        else:
            X["spending_pattern_risk"] = 0.0

        # Merchant pattern risk
        if "merchant_pattern_deviation" in X.columns:
            X["merchant_pattern_risk"] = X["merchant_pattern_deviation"]
        else:
            X["merchant_pattern_risk"] = 0.0

        # Combined behavioral risk
        behavioral_risk_features = [
            "behavioral_risk_score",
            "pattern_break_risk",
            "new_customer_risk",
            "dormant_reactivation_risk",
            "spending_pattern_risk",
            "merchant_pattern_risk",
        ]

        available_features = [f for f in behavioral_risk_features if f in X.columns]
        if available_features:
            X["behavioral_risk_combined"] = X[available_features].mean(axis=1)

        return X

    def _add_ensemble_risk_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add ensemble risk scoring features.

        Parameters:
        -----------
        X : DataFrame
            Input data

        Returns:
        --------
        X : DataFrame
            Data with ensemble risk features added
        """
        # Collect all component risk scores
        risk_components = {}

        if "transaction_risk_score" in X.columns:
            risk_components["transaction"] = X["transaction_risk_score"]

        if "velocity_risk_combined" in X.columns:
            risk_components["velocity"] = X["velocity_risk_combined"]

        if "location_risk_combined" in X.columns:
            risk_components["location"] = X["location_risk_combined"]

        if "behavioral_risk_combined" in X.columns:
            risk_components["behavioral"] = X["behavioral_risk_combined"]

        # Weighted ensemble risk score
        if risk_components:
            weighted_scores = []
            total_weight = 0

            for component, score in risk_components.items():
                weight = self.risk_weights.get(component, 0.25)
                weighted_scores.append(score * weight)
                total_weight += weight

            if total_weight > 0:
                X["ensemble_risk_score"] = sum(weighted_scores) / total_weight
            else:
                X["ensemble_risk_score"] = np.mean(
                    list(risk_components.values()), axis=0
                )
        else:
            X["ensemble_risk_score"] = 0.5  # Default neutral risk

        # Risk score variance (uncertainty measure)
        if len(risk_components) > 1:
            risk_matrix = np.column_stack(list(risk_components.values()))
            X["risk_score_variance"] = np.var(risk_matrix, axis=1)
            X["risk_score_consensus"] = 1.0 - X["risk_score_variance"]
        else:
            X["risk_score_variance"] = 0.0
            X["risk_score_consensus"] = 1.0

        # Maximum risk score (worst-case scenario)
        if risk_components:
            risk_matrix = np.column_stack(list(risk_components.values()))
            X["max_component_risk"] = np.max(risk_matrix, axis=1)
            X["min_component_risk"] = np.min(risk_matrix, axis=1)
            X["risk_score_range"] = X["max_component_risk"] - X["min_component_risk"]

        return X

    def _add_pattern_matching_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add fraud pattern matching features.

        Parameters:
        -----------
        X : DataFrame
            Input data

        Returns:
        --------
        X : DataFrame
            Data with pattern matching features added
        """
        # Initialize pattern matching scores
        X["fraud_pattern_similarity"] = 0.0
        X["known_fraud_pattern_match"] = 0

        # Amount pattern matching
        if (
            "amount_vs_personal_avg" in X.columns
            and "amount_ratio" in self.fraud_patterns
        ):
            amount_pattern = self.fraud_patterns["amount_ratio"]
            risk_threshold = amount_pattern.get("risk_threshold", 2.0)

            amount_risk_matches = (
                X["amount_vs_personal_avg"] >= risk_threshold
            ).astype(int)
            X["fraud_pattern_similarity"] += amount_risk_matches * 0.25

        # Velocity pattern matching
        if "velocity" in self.fraud_patterns:
            velocity_patterns = self.fraud_patterns["velocity"]["risk_multipliers"]

            pattern_score = 0
            matched_patterns = 0

            for feature, risk_multiplier in velocity_patterns.items():
                if feature in X.columns and risk_multiplier > 1.5:
                    feature_max = X[feature].max()
                    if feature_max > 0:
                        normalized_values = X[feature] / feature_max
                        pattern_score += (
                            normalized_values * (risk_multiplier - 1) / 5
                        )  # Normalize contribution
                        matched_patterns += 1

            if matched_patterns > 0:
                X["fraud_pattern_similarity"] += pattern_score / matched_patterns * 0.25

        # Location pattern matching
        if "location" in self.fraud_patterns:
            location_indicators = self.fraud_patterns["location"]["risk_indicators"]

            location_matches = 0
            total_indicators = 0

            for feature, is_risk_indicator in location_indicators.items():
                if feature in X.columns and is_risk_indicator:
                    location_matches += X[feature]
                    total_indicators += 1

            if total_indicators > 0:
                X["fraud_pattern_similarity"] += (
                    location_matches / total_indicators
                ) * 0.25

        # Behavioral pattern matching
        if "behavioral" in self.fraud_patterns:
            behavioral_thresholds = self.fraud_patterns["behavioral"][
                "deviation_thresholds"
            ]

            behavioral_matches = 0
            total_thresholds = 0

            for feature, threshold in behavioral_thresholds.items():
                if feature in X.columns:
                    feature_matches = (X[feature] >= threshold).astype(int)
                    behavioral_matches += feature_matches
                    total_thresholds += 1

            if total_thresholds > 0:
                X["fraud_pattern_similarity"] += (
                    behavioral_matches / total_thresholds
                ) * 0.25

        # Known fraud pattern detection
        X["known_fraud_pattern_match"] = (X["fraud_pattern_similarity"] > 0.6).astype(
            int
        )

        return X

    def _add_risk_categorization(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add final risk categorization features.

        Parameters:
        -----------
        X : DataFrame
            Input data

        Returns:
        --------
        X : DataFrame
            Data with risk categorization added
        """
        # Risk level categorization
        if "ensemble_risk_score" in X.columns:
            conditions = [
                (X["ensemble_risk_score"] <= self.risk_thresholds["low_risk"]),
                (X["ensemble_risk_score"] <= self.risk_thresholds["medium_risk"]),
                (X["ensemble_risk_score"] <= self.risk_thresholds["high_risk"]),
            ]

            choices = ["low_risk", "medium_risk", "high_risk"]
            X["risk_level"] = np.select(conditions, choices, default="very_high_risk")

        # Risk flags
        if "ensemble_risk_score" in X.columns:
            X["high_risk_flag"] = (
                X["ensemble_risk_score"] > self.risk_thresholds["high_risk"]
            ).astype(int)
            X["medium_risk_flag"] = (
                (X["ensemble_risk_score"] > self.risk_thresholds["medium_risk"])
                & (X["ensemble_risk_score"] <= self.risk_thresholds["high_risk"])
            ).astype(int)
            X["low_risk_flag"] = (
                X["ensemble_risk_score"] <= self.risk_thresholds["low_risk"]
            ).astype(int)

        # Confidence in risk assessment
        if "risk_score_consensus" in X.columns:
            X["risk_confidence"] = X["risk_score_consensus"]
            X["high_confidence_assessment"] = (X["risk_confidence"] > 0.8).astype(int)
            X["low_confidence_assessment"] = (X["risk_confidence"] < 0.5).astype(int)

        return X

    def get_feature_names(self) -> List[str]:
        """
        Get names of all generated risk features.

        Returns:
        --------
        feature_names : list of str
            List of risk feature names
        """
        features = []

        # Transaction risk features
        if self.enable_transaction_risk:
            features.extend(
                [
                    "amount_risk_score",
                    "round_amount_risk",
                    "payment_method_risk",
                    "merchant_risk",
                    "transaction_consistency_risk",
                    "transaction_risk_score",
                ]
            )

        # Velocity risk features
        if self.enable_velocity_risk:
            features.extend(
                [
                    "rapid_fire_risk_score",
                    "velocity_risk_score",
                    "frequency_anomaly_risk",
                    "velocity_risk_combined",
                ]
            )

        # Location risk features
        if self.enable_location_risk:
            features.extend(
                [
                    "geographic_risk_score",
                    "travel_risk_score",
                    "distance_anomaly_risk",
                    "timezone_risk_score",
                    "location_risk_combined",
                ]
            )

        # Behavioral risk features
        if self.enable_behavioral_risk:
            features.extend(
                [
                    "behavioral_risk_score",
                    "pattern_break_risk",
                    "new_customer_risk",
                    "dormant_reactivation_risk",
                    "spending_pattern_risk",
                    "merchant_pattern_risk",
                    "behavioral_risk_combined",
                ]
            )

        # Ensemble risk features
        if self.enable_ensemble_risk:
            features.extend(
                [
                    "ensemble_risk_score",
                    "risk_score_variance",
                    "risk_score_consensus",
                    "max_component_risk",
                    "min_component_risk",
                    "risk_score_range",
                ]
            )

        # Pattern matching features
        features.extend(["fraud_pattern_similarity", "known_fraud_pattern_match"])

        # Risk categorization features
        features.extend(
            [
                "risk_level",
                "high_risk_flag",
                "medium_risk_flag",
                "low_risk_flag",
                "risk_confidence",
                "high_confidence_assessment",
                "low_confidence_assessment",
            ]
        )

        return features

    def get_risk_summary(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of risk patterns in the data.

        Parameters:
        -----------
        X : DataFrame
            Data with risk features

        Returns:
        --------
        summary : dict
            Risk pattern summary
        """
        summary = {}

        # Risk level distribution
        if "risk_level" in X.columns:
            risk_distribution = X["risk_level"].value_counts(normalize=True) * 100
            summary["risk_distribution"] = risk_distribution.to_dict()

        # Risk score statistics
        if "ensemble_risk_score" in X.columns:
            summary["risk_score_stats"] = {
                "mean": X["ensemble_risk_score"].mean(),
                "std": X["ensemble_risk_score"].std(),
                "median": X["ensemble_risk_score"].median(),
                "max": X["ensemble_risk_score"].max(),
                "min": X["ensemble_risk_score"].min(),
            }

        # High-risk patterns
        if "known_fraud_pattern_match" in X.columns:
            summary["pattern_analysis"] = {
                "fraud_pattern_matches": X["known_fraud_pattern_match"].sum(),
                "fraud_pattern_match_rate": X["known_fraud_pattern_match"].mean() * 100,
            }

        # Risk component analysis
        risk_components = [
            "transaction_risk_score",
            "velocity_risk_combined",
            "location_risk_combined",
            "behavioral_risk_combined",
        ]

        component_stats = {}
        for component in risk_components:
            if component in X.columns:
                component_stats[component] = {
                    "mean": X[component].mean(),
                    "high_risk_pct": (X[component] > 0.7).mean() * 100,
                }

        if component_stats:
            summary["component_analysis"] = component_stats

        return summary

    def compute_realtime_risk(
        self, transaction_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute risk scores for a single transaction in real-time.

        Parameters:
        -----------
        transaction_features : dict
            Computed features for a single transaction

        Returns:
        --------
        risk_scores : dict
            Risk scores and categorization
        """
        if not self.is_fitted:
            raise ValueError(
                "Risk feature engineer must be fitted before real-time computation"
            )

        # Convert to DataFrame for processing
        df = pd.DataFrame([transaction_features])

        # Apply risk feature engineering
        df_with_risk = self.transform(df)

        # Extract risk scores
        risk_scores = {}

        risk_features = self.get_feature_names()
        for feature in risk_features:
            if feature in df_with_risk.columns:
                risk_scores[feature] = df_with_risk[feature].iloc[0]

        return risk_scores
