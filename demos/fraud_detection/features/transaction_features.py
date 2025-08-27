"""
Transaction Feature Engineering

This module provides comprehensive transaction-level feature engineering for fraud detection,
including monetary patterns, merchant categories, payment methods, and temporal features.

Key Features:
- Amount-based features and anomaly scoring
- Merchant category and risk analysis
- Payment method patterns
- Transaction timing features
- Cross-reference validation
- Real-time feature computation

Author: IntegratedML Pluggable Models Team
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

logger = logging.getLogger(__name__)


class TransactionFeatureEngineer:
    """
    Comprehensive transaction feature engineering for fraud detection.
    
    This class extracts and computes various transaction-level features
    that are crucial for fraud detection including monetary patterns,
    merchant analysis, and temporal characteristics.
    """
    
    def __init__(self,
                 amount_buckets: Optional[List[float]] = None,
                 merchant_risk_scoring: bool = True,
                 temporal_features: bool = True,
                 cross_validation_features: bool = True):
        """
        Initialize transaction feature engineer.
        
        Parameters:
        -----------
        amount_buckets : list of float, optional
            Custom amount buckets for discretization
        merchant_risk_scoring : bool, default=True
            Whether to compute merchant risk scores
        temporal_features : bool, default=True
            Whether to extract temporal features
        cross_validation_features : bool, default=True
            Whether to compute cross-validation features
        """
        self.amount_buckets = amount_buckets or [0, 10, 50, 100, 500, 1000, 5000, float('inf')]
        self.merchant_risk_scoring = merchant_risk_scoring
        self.temporal_features = temporal_features
        self.cross_validation_features = cross_validation_features
        
        # Initialize encoders and scalers
        self.amount_scaler = StandardScaler()
        self.merchant_encoder = LabelEncoder()
        self.payment_method_encoder = LabelEncoder()
        
        # Risk scoring components
        self.merchant_risk_scores = {}
        self.mcc_risk_scores = {}
        self.payment_method_risk_scores = {}
        
        # Fitted status
        self.is_fitted = False
        
        logger.info("Initialized TransactionFeatureEngineer")
    
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'TransactionFeatureEngineer':
        """
        Fit the feature engineer on training data.
        
        Parameters:
        -----------
        X : DataFrame
            Transaction data
        y : array-like, optional
            Fraud labels for risk scoring
            
        Returns:
        --------
        self : TransactionFeatureEngineer
            Fitted feature engineer
        """
        logger.info("Fitting TransactionFeatureEngineer...")
        
        # Fit amount scaler
        if 'amount' in X.columns:
            amounts = X['amount'].values.reshape(-1, 1)
            self.amount_scaler.fit(amounts)
        
        # Fit categorical encoders
        if 'merchant_name' in X.columns:
            self.merchant_encoder.fit(X['merchant_name'].fillna('unknown'))
        
        if 'payment_method' in X.columns:
            self.payment_method_encoder.fit(X['payment_method'].fillna('unknown'))
        
        # Compute risk scores if labels provided
        if y is not None and self.merchant_risk_scoring:
            self._compute_risk_scores(X, y)
        
        self.is_fitted = True
        logger.info("TransactionFeatureEngineer fitted successfully")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform transaction data into features.
        
        Parameters:
        -----------
        X : DataFrame
            Transaction data
            
        Returns:
        --------
        X_features : DataFrame
            Engineered features
        """
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transform")
        
        logger.debug(f"Transforming {len(X)} transactions...")
        
        # Create copy to avoid modifying original data
        X_features = X.copy()
        
        # Amount-based features
        X_features = self._add_amount_features(X_features)
        
        # Merchant features
        X_features = self._add_merchant_features(X_features)
        
        # Payment method features
        X_features = self._add_payment_method_features(X_features)
        
        # Temporal features
        if self.temporal_features:
            X_features = self._add_temporal_features(X_features)
        
        # Cross-validation features
        if self.cross_validation_features:
            X_features = self._add_cross_validation_features(X_features)
        
        # Risk scoring features
        if self.merchant_risk_scoring:
            X_features = self._add_risk_features(X_features)
        
        logger.debug(f"Generated {len(X_features.columns)} features")
        return X_features
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> pd.DataFrame:
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
            Engineered features
        """
        return self.fit(X, y).transform(X)
    
    def _add_amount_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add amount-based features.
        
        Parameters:
        -----------
        X : DataFrame
            Input data
            
        Returns:
        --------
        X : DataFrame
            Data with amount features added
        """
        if 'amount' not in X.columns:
            logger.warning("Amount column not found, skipping amount features")
            return X
        
        # Basic amount features
        X['amount_log'] = np.log1p(X['amount'])
        X['amount_sqrt'] = np.sqrt(X['amount'])
        
        # Normalized amount
        amounts_scaled = self.amount_scaler.transform(X['amount'].values.reshape(-1, 1))
        X['amount_normalized'] = amounts_scaled.flatten()
        
        # Amount buckets
        X['amount_bucket'] = pd.cut(X['amount'], bins=self.amount_buckets, labels=False, include_lowest=True)
        
        # Round number detection
        X['is_round_amount'] = (X['amount'] % 1 == 0).astype(int)
        X['is_round_10'] = (X['amount'] % 10 == 0).astype(int)
        X['is_round_100'] = (X['amount'] % 100 == 0).astype(int)
        
        # Amount patterns
        X['amount_decimal_places'] = X['amount'].apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
        
        return X
    
    def _add_merchant_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add merchant-based features.
        
        Parameters:
        -----------
        X : DataFrame
            Input data
            
        Returns:
        --------
        X : DataFrame
            Data with merchant features added
        """
        if 'merchant_name' not in X.columns:
            logger.warning("Merchant name column not found, skipping merchant features")
            return X
        
        # Encode merchant names
        merchant_filled = X['merchant_name'].fillna('unknown')
        
        # Handle unseen merchants during transform
        try:
            X['merchant_encoded'] = self.merchant_encoder.transform(merchant_filled)
        except ValueError:
            # Handle unseen merchants
            X['merchant_encoded'] = merchant_filled.apply(
                lambda x: self.merchant_encoder.transform([x])[0] 
                if x in self.merchant_encoder.classes_ else -1
            )
        
        # Merchant name features
        X['merchant_name_length'] = X['merchant_name'].fillna('').str.len()
        X['merchant_name_words'] = X['merchant_name'].fillna('').str.split().str.len()
        X['merchant_has_numbers'] = X['merchant_name'].fillna('').str.contains(r'\d').astype(int)
        X['merchant_has_special_chars'] = X['merchant_name'].fillna('').str.contains(r'[^a-zA-Z0-9\s]').astype(int)
        
        # MCC (Merchant Category Code) features if available
        if 'mcc' in X.columns:
            X['mcc_risk_category'] = X['mcc'].apply(self._get_mcc_risk_category)
            X['is_high_risk_mcc'] = (X['mcc_risk_category'] == 'high_risk').astype(int)
        
        return X
    
    def _add_payment_method_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add payment method features.
        
        Parameters:
        -----------
        X : DataFrame
            Input data
            
        Returns:
        --------
        X : DataFrame
            Data with payment method features added
        """
        if 'payment_method' not in X.columns:
            logger.warning("Payment method column not found, skipping payment method features")
            return X
        
        # Encode payment methods
        payment_filled = X['payment_method'].fillna('unknown')
        
        try:
            X['payment_method_encoded'] = self.payment_method_encoder.transform(payment_filled)
        except ValueError:
            # Handle unseen payment methods
            X['payment_method_encoded'] = payment_filled.apply(
                lambda x: self.payment_method_encoder.transform([x])[0] 
                if x in self.payment_method_encoder.classes_ else -1
            )
        
        # Payment method risk categories
        X['is_card_payment'] = X['payment_method'].fillna('').str.contains('card', case=False).astype(int)
        X['is_digital_payment'] = X['payment_method'].fillna('').str.contains('digital|paypal|venmo|zelle', case=False).astype(int)
        X['is_cash_equivalent'] = X['payment_method'].fillna('').str.contains('cash|check|wire', case=False).astype(int)
        
        return X
    
    def _add_temporal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features.
        
        Parameters:
        -----------
        X : DataFrame
            Input data
            
        Returns:
        --------
        X : DataFrame
            Data with temporal features added
        """
        if 'timestamp' not in X.columns:
            logger.warning("Timestamp column not found, skipping temporal features")
            return X
        
        # Convert timestamp if needed
        if not pd.api.types.is_datetime64_any_dtype(X['timestamp']):
            X['timestamp'] = pd.to_datetime(X['timestamp'])
        
        # Basic temporal features
        X['hour'] = X['timestamp'].dt.hour
        X['day_of_week'] = X['timestamp'].dt.dayofweek
        X['day_of_month'] = X['timestamp'].dt.day
        X['month'] = X['timestamp'].dt.month
        X['quarter'] = X['timestamp'].dt.quarter
        
        # Time categories
        X['is_weekend'] = (X['day_of_week'] >= 5).astype(int)
        X['is_business_hours'] = ((X['hour'] >= 9) & (X['hour'] <= 17)).astype(int)
        X['is_late_night'] = ((X['hour'] >= 23) | (X['hour'] <= 5)).astype(int)
        X['is_early_morning'] = ((X['hour'] >= 6) & (X['hour'] <= 9)).astype(int)
        
        # Holiday detection (simplified)
        X['is_month_end'] = (X['day_of_month'] >= 28).astype(int)
        X['is_month_start'] = (X['day_of_month'] <= 3).astype(int)
        
        # Cyclical encoding for temporal features
        X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
        X['day_of_week_sin'] = np.sin(2 * np.pi * X['day_of_week'] / 7)
        X['day_of_week_cos'] = np.cos(2 * np.pi * X['day_of_week'] / 7)
        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
        
        return X
    
    def _add_cross_validation_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add cross-validation features.
        
        Parameters:
        -----------
        X : DataFrame
            Input data
            
        Returns:
        --------
        X : DataFrame
            Data with cross-validation features added
        """
        # Amount vs merchant consistency
        if 'amount' in X.columns and 'merchant_name' in X.columns:
            # Check if amount is consistent with merchant type
            X['amount_merchant_consistency'] = self._check_amount_merchant_consistency(X)
        
        # Geographic consistency
        if all(col in X.columns for col in ['merchant_location', 'customer_location']):
            X['location_consistency'] = self._check_location_consistency(X)
        
        # Time consistency
        if 'timestamp' in X.columns and 'merchant_location' in X.columns:
            X['time_zone_consistency'] = self._check_timezone_consistency(X)
        
        return X
    
    def _add_risk_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add risk scoring features.
        
        Parameters:
        -----------
        X : DataFrame
            Input data
            
        Returns:
        --------
        X : DataFrame
            Data with risk features added
        """
        # Merchant risk scores
        if 'merchant_name' in X.columns and self.merchant_risk_scores:
            X['merchant_risk_score'] = X['merchant_name'].map(self.merchant_risk_scores).fillna(0.5)
        
        # MCC risk scores
        if 'mcc' in X.columns and self.mcc_risk_scores:
            X['mcc_risk_score'] = X['mcc'].map(self.mcc_risk_scores).fillna(0.5)
        
        # Payment method risk scores
        if 'payment_method' in X.columns and self.payment_method_risk_scores:
            X['payment_method_risk_score'] = X['payment_method'].map(self.payment_method_risk_scores).fillna(0.5)
        
        return X
    
    def _compute_risk_scores(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Compute risk scores for categorical variables.
        
        Parameters:
        -----------
        X : DataFrame
            Training data
        y : array-like
            Fraud labels
        """
        # Merchant risk scores
        if 'merchant_name' in X.columns:
            merchant_fraud_rates = X.groupby('merchant_name')['fraud'].mean() if 'fraud' in X.columns else \
                                 pd.DataFrame({'merchant_name': X['merchant_name'], 'fraud': y}).groupby('merchant_name')['fraud'].mean()
            self.merchant_risk_scores = merchant_fraud_rates.to_dict()
        
        # MCC risk scores
        if 'mcc' in X.columns:
            mcc_fraud_rates = X.groupby('mcc')['fraud'].mean() if 'fraud' in X.columns else \
                            pd.DataFrame({'mcc': X['mcc'], 'fraud': y}).groupby('mcc')['fraud'].mean()
            self.mcc_risk_scores = mcc_fraud_rates.to_dict()
        
        # Payment method risk scores
        if 'payment_method' in X.columns:
            payment_fraud_rates = X.groupby('payment_method')['fraud'].mean() if 'fraud' in X.columns else \
                                pd.DataFrame({'payment_method': X['payment_method'], 'fraud': y}).groupby('payment_method')['fraud'].mean()
            self.payment_method_risk_scores = payment_fraud_rates.to_dict()
    
    def _get_mcc_risk_category(self, mcc: str) -> str:
        """
        Categorize MCC by risk level.
        
        Parameters:
        -----------
        mcc : str
            Merchant category code
            
        Returns:
        --------
        risk_category : str
            Risk category (low_risk, medium_risk, high_risk)
        """
        if pd.isna(mcc):
            return 'unknown'
        
        # High-risk MCCs (gambling, adult, etc.)
        high_risk_mccs = ['7995', '5993', '7011', '7273', '7394']
        
        # Medium-risk MCCs (jewelry, electronics, etc.)
        medium_risk_mccs = ['5944', '5732', '5734', '5722']
        
        mcc_str = str(mcc)
        
        if mcc_str in high_risk_mccs:
            return 'high_risk'
        elif mcc_str in medium_risk_mccs:
            return 'medium_risk'
        else:
            return 'low_risk'
    
    def _check_amount_merchant_consistency(self, X: pd.DataFrame) -> np.ndarray:
        """
        Check consistency between transaction amount and merchant type.
        
        Parameters:
        -----------
        X : DataFrame
            Input data
            
        Returns:
        --------
        consistency_scores : ndarray
            Consistency scores (0-1)
        """
        # Simplified consistency check
        # In practice, this would use historical data and domain knowledge
        scores = np.ones(len(X)) * 0.8  # Default consistency
        
        # Flag potentially inconsistent patterns
        if 'amount' in X.columns and 'merchant_name' in X.columns:
            # Very high amounts at convenience stores
            convenience_mask = X['merchant_name'].str.contains('convenience|7-eleven|circle k', case=False, na=False)
            high_amount_mask = X['amount'] > 500
            scores[convenience_mask & high_amount_mask] = 0.3
            
            # Very low amounts at luxury stores
            luxury_mask = X['merchant_name'].str.contains('luxury|gucci|tiffany', case=False, na=False)
            low_amount_mask = X['amount'] < 50
            scores[luxury_mask & low_amount_mask] = 0.4
        
        return scores
    
    def _check_location_consistency(self, X: pd.DataFrame) -> np.ndarray:
        """
        Check consistency between customer and merchant locations.
        
        Parameters:
        -----------
        X : DataFrame
            Input data
            
        Returns:
        --------
        consistency_scores : ndarray
            Consistency scores (0-1)
        """
        # Simplified location consistency
        # In practice, this would use geolocation and distance calculations
        scores = np.ones(len(X)) * 0.9  # Default high consistency
        
        # Flag potentially inconsistent locations
        if 'customer_location' in X.columns and 'merchant_location' in X.columns:
            different_states = X['customer_location'] != X['merchant_location']
            scores[different_states] = 0.6
        
        return scores
    
    def _check_timezone_consistency(self, X: pd.DataFrame) -> np.ndarray:
        """
        Check consistency between transaction time and merchant location.
        
        Parameters:
        -----------
        X : DataFrame
            Input data
            
        Returns:
        --------
        consistency_scores : ndarray
            Consistency scores (0-1)
        """
        # Simplified timezone consistency
        scores = np.ones(len(X)) * 0.95  # Default high consistency
        
        # In practice, this would check if transaction time makes sense
        # for the merchant's timezone
        
        return scores
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all generated features.
        
        Returns:
        --------
        feature_names : list of str
            List of feature names
        """
        base_features = [
            'amount_log', 'amount_sqrt', 'amount_normalized', 'amount_bucket',
            'is_round_amount', 'is_round_10', 'is_round_100', 'amount_decimal_places'
        ]
        
        merchant_features = [
            'merchant_encoded', 'merchant_name_length', 'merchant_name_words',
            'merchant_has_numbers', 'merchant_has_special_chars'
        ]
        
        payment_features = [
            'payment_method_encoded', 'is_card_payment', 'is_digital_payment', 'is_cash_equivalent'
        ]
        
        temporal_features = [
            'hour', 'day_of_week', 'day_of_month', 'month', 'quarter',
            'is_weekend', 'is_business_hours', 'is_late_night', 'is_early_morning',
            'is_month_end', 'is_month_start',
            'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos'
        ]
        
        risk_features = [
            'merchant_risk_score', 'mcc_risk_score', 'payment_method_risk_score'
        ]
        
        cross_validation_features = [
            'amount_merchant_consistency', 'location_consistency', 'time_zone_consistency'
        ]
        
        all_features = base_features + merchant_features + payment_features
        
        if self.temporal_features:
            all_features.extend(temporal_features)
        
        if self.merchant_risk_scoring:
            all_features.extend(risk_features)
        
        if self.cross_validation_features:
            all_features.extend(cross_validation_features)
        
        return all_features
    
    def get_feature_importance_hints(self) -> Dict[str, str]:
        """
        Get hints about feature importance and interpretation.
        
        Returns:
        --------
        hints : dict
            Feature importance hints
        """
        return {
            'amount_log': 'Log-transformed amount helps with skewed distributions',
            'amount_normalized': 'Standardized amount for comparison across ranges',
            'is_round_amount': 'Round amounts can indicate manual/suspicious transactions',
            'merchant_risk_score': 'Historical fraud rate for this merchant',
            'is_business_hours': 'Transactions outside business hours are riskier',
            'is_weekend': 'Weekend transactions have different risk profiles',
            'amount_merchant_consistency': 'Unusual amounts for merchant type',
            'location_consistency': 'Geographic consistency between customer and merchant',
            'payment_method_risk_score': 'Risk level associated with payment method'
        }