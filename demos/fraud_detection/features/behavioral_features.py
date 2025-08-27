"""
Behavioral Feature Engineering

This module provides behavioral analysis feature engineering for fraud detection,
focusing on customer spending patterns, habits, and deviation analysis.

Key Features:
- Customer spending behavior profiling
- Deviation from normal patterns detection
- Shopping habit analysis
- Temporal behavior patterns
- Customer lifecycle features
- Behavioral anomaly scoring

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


class BehavioralFeatureEngineer:
    """
    Behavioral analysis feature engineering for fraud detection.
    
    This class analyzes customer behavior patterns and detects deviations
    that might indicate fraudulent activity, including spending habits,
    temporal patterns, and lifestyle changes.
    """
    
    def __init__(self,
                 lookback_days: int = 90,
                 min_transactions_for_profile: int = 10,
                 enable_spending_patterns: bool = True,
                 enable_temporal_patterns: bool = True,
                 enable_merchant_patterns: bool = True,
                 enable_lifecycle_features: bool = True):
        """
        Initialize behavioral feature engineer.
        
        Parameters:
        -----------
        lookback_days : int, default=90
            Number of days to look back for behavioral analysis
        min_transactions_for_profile : int, default=10
            Minimum transactions needed to build reliable behavioral profile
        enable_spending_patterns : bool, default=True
            Whether to analyze spending behavior patterns
        enable_temporal_patterns : bool, default=True
            Whether to analyze temporal behavior patterns
        enable_merchant_patterns : bool, default=True
            Whether to analyze merchant preference patterns
        enable_lifecycle_features : bool, default=True
            Whether to compute customer lifecycle features
        """
        self.lookback_days = lookback_days
        self.min_transactions_for_profile = min_transactions_for_profile
        self.enable_spending_patterns = enable_spending_patterns
        self.enable_temporal_patterns = enable_temporal_patterns
        self.enable_merchant_patterns = enable_merchant_patterns
        self.enable_lifecycle_features = enable_lifecycle_features
        
        # Customer behavioral profiles
        self.customer_profiles = {}
        
        # Behavioral baselines
        self.spending_baselines = {}
        self.temporal_baselines = {}
        self.merchant_baselines = {}
        
        # Fitted status
        self.is_fitted = False
        
        logger.info("Initialized BehavioralFeatureEngineer")
    
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'BehavioralFeatureEngineer':
        """
        Fit the behavioral feature engineer on training data.
        
        Parameters:
        -----------
        X : DataFrame
            Transaction data with customer_id, timestamp, amount, etc.
        y : array-like, optional
            Fraud labels
            
        Returns:
        --------
        self : BehavioralFeatureEngineer
            Fitted feature engineer
        """
        logger.info("Fitting BehavioralFeatureEngineer...")
        
        if 'customer_id' not in X.columns:
            raise ValueError("customer_id column is required for behavioral analysis")
        
        if 'timestamp' not in X.columns:
            raise ValueError("timestamp column is required for behavioral analysis")
        
        # Convert timestamp if needed
        if not pd.api.types.is_datetime64_any_dtype(X['timestamp']):
            X = X.copy()
            X['timestamp'] = pd.to_datetime(X['timestamp'])
        
        # Build customer behavioral profiles
        self._build_customer_profiles(X, y)
        
        # Compute behavioral baselines
        self._compute_behavioral_baselines(X)
        
        self.is_fitted = True
        logger.info("BehavioralFeatureEngineer fitted successfully")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform transaction data into behavioral features.
        
        Parameters:
        -----------
        X : DataFrame
            Transaction data
            
        Returns:
        --------
        X_features : DataFrame
            Data with behavioral features added
        """
        if not self.is_fitted:
            raise ValueError("Behavioral feature engineer must be fitted before transform")
        
        logger.debug(f"Computing behavioral features for {len(X)} transactions...")
        
        # Create copy to avoid modifying original data
        X_features = X.copy()
        
        # Convert timestamp if needed
        if not pd.api.types.is_datetime64_any_dtype(X_features['timestamp']):
            X_features['timestamp'] = pd.to_datetime(X_features['timestamp'])
        
        # Sort by customer and timestamp
        X_features = X_features.sort_values(['customer_id', 'timestamp']).reset_index(drop=True)
        
        # Spending pattern features
        if self.enable_spending_patterns:
            X_features = self._add_spending_pattern_features(X_features)
        
        # Temporal pattern features
        if self.enable_temporal_patterns:
            X_features = self._add_temporal_pattern_features(X_features)
        
        # Merchant pattern features
        if self.enable_merchant_patterns:
            X_features = self._add_merchant_pattern_features(X_features)
        
        # Customer lifecycle features
        if self.enable_lifecycle_features:
            X_features = self._add_lifecycle_features(X_features)
        
        # Behavioral deviation features
        X_features = self._add_deviation_features(X_features)
        
        logger.debug(f"Generated behavioral features")
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
            Data with behavioral features
        """
        return self.fit(X, y).transform(X)
    
    def _build_customer_profiles(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> None:
        """
        Build behavioral profiles for each customer.
        
        Parameters:
        -----------
        X : DataFrame
            Training transaction data
        y : array-like, optional
            Fraud labels
        """
        logger.debug("Building customer behavioral profiles...")
        
        for customer_id, group in X.groupby('customer_id'):
            if len(group) < self.min_transactions_for_profile:
                continue
            
            profile = {}
            
            # Spending behavior
            if 'amount' in group.columns:
                profile['spending'] = {
                    'avg_amount': group['amount'].mean(),
                    'std_amount': group['amount'].std(),
                    'median_amount': group['amount'].median(),
                    'min_amount': group['amount'].min(),
                    'max_amount': group['amount'].max(),
                    'total_spent': group['amount'].sum(),
                    'transaction_count': len(group)
                }
                
                # Amount distribution features
                profile['spending']['amount_percentiles'] = {
                    'p25': group['amount'].quantile(0.25),
                    'p75': group['amount'].quantile(0.75),
                    'p90': group['amount'].quantile(0.90)
                }
            
            # Temporal behavior
            if 'timestamp' in group.columns:
                group_sorted = group.sort_values('timestamp')
                
                # Transaction frequency
                days_active = (group_sorted['timestamp'].max() - group_sorted['timestamp'].min()).days
                profile['temporal'] = {
                    'days_active': max(days_active, 1),
                    'transactions_per_day': len(group) / max(days_active, 1),
                    'first_transaction': group_sorted['timestamp'].min(),
                    'last_transaction': group_sorted['timestamp'].max()
                }
                
                # Time patterns
                hours = group_sorted['timestamp'].dt.hour
                days_of_week = group_sorted['timestamp'].dt.dayofweek
                
                profile['temporal']['preferred_hours'] = hours.mode().tolist()
                profile['temporal']['preferred_days'] = days_of_week.mode().tolist()
                profile['temporal']['hour_distribution'] = hours.value_counts().to_dict()
                profile['temporal']['dow_distribution'] = days_of_week.value_counts().to_dict()
            
            # Merchant behavior
            if 'merchant_name' in group.columns:
                merchant_counts = group['merchant_name'].value_counts()
                
                profile['merchant'] = {
                    'unique_merchants': group['merchant_name'].nunique(),
                    'top_merchants': merchant_counts.head(5).to_dict(),
                    'merchant_loyalty': merchant_counts.max() / len(group),  # Fraction at most frequent merchant
                    'merchant_diversity': len(merchant_counts) / len(group)  # Merchant variety
                }
                
                # Merchant category behavior
                if 'mcc' in group.columns:
                    mcc_counts = group['mcc'].value_counts()
                    profile['merchant']['unique_categories'] = group['mcc'].nunique()
                    profile['merchant']['top_categories'] = mcc_counts.head(3).to_dict()
            
            # Location behavior
            if 'customer_location' in group.columns:
                location_counts = group['customer_location'].value_counts()
                
                profile['location'] = {
                    'unique_locations': group['customer_location'].nunique(),
                    'top_locations': location_counts.head(3).to_dict(),
                    'location_stability': location_counts.max() / len(group)
                }
            
            # Payment method behavior
            if 'payment_method' in group.columns:
                payment_counts = group['payment_method'].value_counts()
                
                profile['payment'] = {
                    'unique_payment_methods': group['payment_method'].nunique(),
                    'preferred_payment_methods': payment_counts.to_dict(),
                    'payment_method_consistency': payment_counts.max() / len(group)
                }
            
            self.customer_profiles[customer_id] = profile
        
        logger.debug(f"Built profiles for {len(self.customer_profiles)} customers")
    
    def _compute_behavioral_baselines(self, X: pd.DataFrame) -> None:
        """
        Compute behavioral baselines from all customers.
        
        Parameters:
        -----------
        X : DataFrame
            Training data
        """
        if 'amount' in X.columns:
            self.spending_baselines = {
                'global_avg_amount': X['amount'].mean(),
                'global_std_amount': X['amount'].std(),
                'global_median_amount': X['amount'].median()
            }
        
        if 'timestamp' in X.columns:
            hours = X['timestamp'].dt.hour
            days = X['timestamp'].dt.dayofweek
            
            self.temporal_baselines = {
                'global_hour_distribution': hours.value_counts(normalize=True).to_dict(),
                'global_dow_distribution': days.value_counts(normalize=True).to_dict()
            }
        
        if 'merchant_name' in X.columns:
            merchant_popularity = X['merchant_name'].value_counts(normalize=True)
            
            self.merchant_baselines = {
                'global_merchant_distribution': merchant_popularity.to_dict()
            }
    
    def _add_spending_pattern_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add spending pattern behavioral features.
        
        Parameters:
        -----------
        X : DataFrame
            Input data
            
        Returns:
        --------
        X : DataFrame
            Data with spending pattern features added
        """
        if 'amount' not in X.columns:
            return X
        
        # Initialize features
        X['amount_vs_personal_avg'] = 1.0
        X['amount_vs_personal_std'] = 0.0
        X['amount_percentile_personal'] = 0.5
        X['spending_burst_indicator'] = 0
        X['spending_pattern_deviation'] = 0.0
        
        # Compute for each customer
        for customer_id, group_indices in X.groupby('customer_id').groups.items():
            if customer_id not in self.customer_profiles:
                continue
            
            profile = self.customer_profiles[customer_id]
            spending_profile = profile.get('spending', {})
            
            if not spending_profile:
                continue
            
            personal_avg = spending_profile.get('avg_amount', 0)
            personal_std = spending_profile.get('std_amount', 1)
            
            for idx in group_indices:
                amount = X.loc[idx, 'amount']
                
                # Amount vs personal average
                if personal_avg > 0:
                    X.loc[idx, 'amount_vs_personal_avg'] = amount / personal_avg
                
                # Amount vs personal standard deviation
                if personal_std > 0:
                    X.loc[idx, 'amount_vs_personal_std'] = (amount - personal_avg) / personal_std
                
                # Personal percentile
                percentiles = spending_profile.get('amount_percentiles', {})
                if amount <= percentiles.get('p25', 0):
                    X.loc[idx, 'amount_percentile_personal'] = 0.25
                elif amount <= percentiles.get('p75', 0):
                    X.loc[idx, 'amount_percentile_personal'] = 0.75
                elif amount <= percentiles.get('p90', 0):
                    X.loc[idx, 'amount_percentile_personal'] = 0.90
                else:
                    X.loc[idx, 'amount_percentile_personal'] = 1.0
        
        # Spending burst detection (multiple large transactions in short time)
        X = X.sort_values(['customer_id', 'timestamp']).reset_index(drop=True)
        
        for customer_id, group in X.groupby('customer_id'):
            if customer_id not in self.customer_profiles:
                continue
            
            group = group.sort_values('timestamp')
            personal_avg = self.customer_profiles[customer_id].get('spending', {}).get('avg_amount', 0)
            
            for i, (idx, row) in enumerate(group.iterrows()):
                # Look at transactions in the last 1 hour
                current_time = row['timestamp']
                recent_window = current_time - timedelta(hours=1)
                
                recent_mask = (group['timestamp'] >= recent_window) & (group['timestamp'] < current_time)
                recent_txns = group[recent_mask]
                
                # Check for spending burst
                if len(recent_txns) >= 2:  # At least 2 recent transactions
                    recent_large = (recent_txns['amount'] > personal_avg * 1.5).sum()
                    if recent_large >= 2:
                        X.loc[idx, 'spending_burst_indicator'] = 1
        
        # Spending pattern deviation score
        X['is_unusual_amount'] = (X['amount_vs_personal_std'].abs() > 2).astype(int)
        X['is_very_large_amount'] = (X['amount_vs_personal_avg'] > 3).astype(int)
        X['is_very_small_amount'] = (X['amount_vs_personal_avg'] < 0.1).astype(int)
        
        return X
    
    def _add_temporal_pattern_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal pattern behavioral features.
        
        Parameters:
        -----------
        X : DataFrame
            Input data
            
        Returns:
        --------
        X : DataFrame
            Data with temporal pattern features added
        """
        if 'timestamp' not in X.columns:
            return X
        
        # Extract time features
        X['hour'] = X['timestamp'].dt.hour
        X['day_of_week'] = X['timestamp'].dt.dayofweek
        
        # Initialize features
        X['hour_deviation_score'] = 0.0
        X['dow_deviation_score'] = 0.0
        X['unusual_time_indicator'] = 0
        X['temporal_pattern_break'] = 0
        
        # Compute for each customer
        for customer_id, group_indices in X.groupby('customer_id').groups.items():
            if customer_id not in self.customer_profiles:
                continue
            
            profile = self.customer_profiles[customer_id]
            temporal_profile = profile.get('temporal', {})
            
            if not temporal_profile:
                continue
            
            preferred_hours = temporal_profile.get('preferred_hours', [])
            preferred_days = temporal_profile.get('preferred_days', [])
            hour_dist = temporal_profile.get('hour_distribution', {})
            dow_dist = temporal_profile.get('dow_distribution', {})
            
            for idx in group_indices:
                hour = X.loc[idx, 'hour']
                dow = X.loc[idx, 'day_of_week']
                
                # Hour deviation score
                if hour_dist:
                    max_hour_freq = max(hour_dist.values())
                    current_hour_freq = hour_dist.get(hour, 0)
                    X.loc[idx, 'hour_deviation_score'] = 1 - (current_hour_freq / max_hour_freq)
                
                # Day of week deviation score
                if dow_dist:
                    max_dow_freq = max(dow_dist.values())
                    current_dow_freq = dow_dist.get(dow, 0)
                    X.loc[idx, 'dow_deviation_score'] = 1 - (current_dow_freq / max_dow_freq)
                
                # Unusual time indicator
                if preferred_hours and hour not in preferred_hours:
                    X.loc[idx, 'unusual_time_indicator'] = 1
                
                if preferred_days and dow not in preferred_days:
                    X.loc[idx, 'unusual_time_indicator'] = 1
        
        # Temporal pattern consistency
        X['is_preferred_hour'] = 0
        X['is_preferred_day'] = 0
        
        for customer_id, group_indices in X.groupby('customer_id').groups.items():
            if customer_id not in self.customer_profiles:
                continue
            
            temporal_profile = self.customer_profiles[customer_id].get('temporal', {})
            preferred_hours = temporal_profile.get('preferred_hours', [])
            preferred_days = temporal_profile.get('preferred_days', [])
            
            for idx in group_indices:
                hour = X.loc[idx, 'hour']
                dow = X.loc[idx, 'day_of_week']
                
                if hour in preferred_hours:
                    X.loc[idx, 'is_preferred_hour'] = 1
                
                if dow in preferred_days:
                    X.loc[idx, 'is_preferred_day'] = 1
        
        return X
    
    def _add_merchant_pattern_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add merchant pattern behavioral features.
        
        Parameters:
        -----------
        X : DataFrame
            Input data
            
        Returns:
        --------
        X : DataFrame
            Data with merchant pattern features added
        """
        if 'merchant_name' not in X.columns:
            return X
        
        # Initialize features
        X['is_familiar_merchant'] = 0
        X['is_top_merchant'] = 0
        X['merchant_novelty_score'] = 1.0
        X['merchant_pattern_deviation'] = 0.0
        
        # Compute for each customer
        for customer_id, group_indices in X.groupby('customer_id').groups.items():
            if customer_id not in self.customer_profiles:
                continue
            
            profile = self.customer_profiles[customer_id]
            merchant_profile = profile.get('merchant', {})
            
            if not merchant_profile:
                continue
            
            top_merchants = merchant_profile.get('top_merchants', {})
            unique_merchants = merchant_profile.get('unique_merchants', 0)
            
            for idx in group_indices:
                merchant = X.loc[idx, 'merchant_name']
                
                # Is familiar merchant
                if merchant in top_merchants:
                    X.loc[idx, 'is_familiar_merchant'] = 1
                    
                    # Is top merchant (top 3)
                    top_3_merchants = list(top_merchants.keys())[:3]
                    if merchant in top_3_merchants:
                        X.loc[idx, 'is_top_merchant'] = 1
                    
                    # Merchant frequency score
                    merchant_freq = top_merchants[merchant]
                    total_transactions = merchant_profile.get('transaction_count', 1)
                    X.loc[idx, 'merchant_novelty_score'] = merchant_freq / total_transactions
                
                # Merchant diversity impact
                if unique_merchants > 0:
                    merchant_diversity = merchant_profile.get('merchant_diversity', 0)
                    if merchant not in top_merchants and merchant_diversity < 0.5:
                        # New merchant for a customer with low diversity
                        X.loc[idx, 'merchant_pattern_deviation'] = 1.0
        
        # Merchant category analysis
        if 'mcc' in X.columns:
            X['is_familiar_category'] = 0
            
            for customer_id, group_indices in X.groupby('customer_id').groups.items():
                if customer_id not in self.customer_profiles:
                    continue
                
                merchant_profile = self.customer_profiles[customer_id].get('merchant', {})
                top_categories = merchant_profile.get('top_categories', {})
                
                for idx in group_indices:
                    mcc = X.loc[idx, 'mcc']
                    if mcc in top_categories:
                        X.loc[idx, 'is_familiar_category'] = 1
        
        return X
    
    def _add_lifecycle_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add customer lifecycle behavioral features.
        
        Parameters:
        -----------
        X : DataFrame
            Input data
            
        Returns:
        --------
        X : DataFrame
            Data with lifecycle features added
        """
        # Initialize features
        X['days_since_first_transaction'] = 0
        X['days_since_last_transaction'] = 0
        X['customer_maturity_score'] = 0.0
        X['is_new_customer'] = 0
        X['is_dormant_reactivation'] = 0
        
        # Current timestamp for reference
        current_date = X['timestamp'].max()
        
        # Compute for each customer
        for customer_id, group_indices in X.groupby('customer_id').groups.items():
            if customer_id not in self.customer_profiles:
                continue
            
            profile = self.customer_profiles[customer_id]
            temporal_profile = profile.get('temporal', {})
            
            if not temporal_profile:
                continue
            
            first_transaction = temporal_profile.get('first_transaction')
            last_transaction = temporal_profile.get('last_transaction')
            days_active = temporal_profile.get('days_active', 0)
            transaction_count = profile.get('spending', {}).get('transaction_count', 0)
            
            for idx in group_indices:
                transaction_time = X.loc[idx, 'timestamp']
                
                # Days since first transaction
                if first_transaction:
                    days_since_first = (transaction_time - first_transaction).days
                    X.loc[idx, 'days_since_first_transaction'] = max(days_since_first, 0)
                    
                    # Customer maturity score
                    maturity = min(days_since_first / 365, 1.0)  # Normalized to 1 year
                    X.loc[idx, 'customer_maturity_score'] = maturity
                    
                    # New customer indicator (less than 30 days)
                    if days_since_first <= 30:
                        X.loc[idx, 'is_new_customer'] = 1
                
                # Days since last transaction (before current)
                customer_data = X[X['customer_id'] == customer_id]
                customer_data = customer_data[customer_data['timestamp'] < transaction_time]
                
                if len(customer_data) > 0:
                    last_prev_transaction = customer_data['timestamp'].max()
                    days_since_last = (transaction_time - last_prev_transaction).days
                    X.loc[idx, 'days_since_last_transaction'] = days_since_last
                    
                    # Dormant reactivation (more than 60 days gap)
                    if days_since_last > 60:
                        X.loc[idx, 'is_dormant_reactivation'] = 1
        
        # Customer activity level
        X['customer_activity_level'] = 'unknown'
        
        for customer_id, group_indices in X.groupby('customer_id').groups.items():
            if customer_id not in self.customer_profiles:
                continue
            
            temporal_profile = self.customer_profiles[customer_id].get('temporal', {})
            transactions_per_day = temporal_profile.get('transactions_per_day', 0)
            
            # Categorize activity level
            if transactions_per_day >= 1.0:
                activity_level = 'very_active'
            elif transactions_per_day >= 0.5:
                activity_level = 'active'
            elif transactions_per_day >= 0.1:
                activity_level = 'moderate'
            else:
                activity_level = 'low'
            
            for idx in group_indices:
                X.loc[idx, 'customer_activity_level'] = activity_level
        
        return X
    
    def _add_deviation_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add behavioral deviation summary features.
        
        Parameters:
        -----------
        X : DataFrame
            Input data
            
        Returns:
        --------
        X : DataFrame
            Data with deviation features added
        """
        # Composite behavioral deviation score
        deviation_features = [
            'amount_vs_personal_std', 'hour_deviation_score', 'dow_deviation_score',
            'merchant_pattern_deviation'
        ]
        
        available_features = [f for f in deviation_features if f in X.columns]
        
        if available_features:
            # Normalize deviation scores to 0-1 range
            for feature in available_features:
                if feature == 'amount_vs_personal_std':
                    # Convert z-score to 0-1 probability
                    X[f'{feature}_normalized'] = 1 / (1 + np.exp(-np.abs(X[feature])))
                else:
                    X[f'{feature}_normalized'] = X[feature]
            
            # Combined behavioral deviation score
            normalized_features = [f'{f}_normalized' for f in available_features]
            X['behavioral_deviation_score'] = X[normalized_features].mean(axis=1)
            
            # Behavioral anomaly flags
            X['high_behavioral_deviation'] = (X['behavioral_deviation_score'] > 0.7).astype(int)
            X['moderate_behavioral_deviation'] = (
                (X['behavioral_deviation_score'] > 0.4) & (X['behavioral_deviation_score'] <= 0.7)
            ).astype(int)
        
        # Pattern break indicators
        pattern_break_features = [
            'unusual_time_indicator', 'spending_burst_indicator', 'is_dormant_reactivation'
        ]
        
        available_breaks = [f for f in pattern_break_features if f in X.columns]
        if available_breaks:
            X['pattern_break_count'] = X[available_breaks].sum(axis=1)
            X['multiple_pattern_breaks'] = (X['pattern_break_count'] >= 2).astype(int)
        
        return X
    
    def get_customer_profile(self, customer_id: str) -> Dict[str, Any]:
        """
        Get behavioral profile for a specific customer.
        
        Parameters:
        -----------
        customer_id : str
            Customer identifier
            
        Returns:
        --------
        profile : dict
            Customer behavioral profile
        """
        return self.customer_profiles.get(customer_id, {})
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all generated behavioral features.
        
        Returns:
        --------
        feature_names : list of str
            List of behavioral feature names
        """
        features = []
        
        # Spending pattern features
        if self.enable_spending_patterns:
            features.extend([
                'amount_vs_personal_avg', 'amount_vs_personal_std', 'amount_percentile_personal',
                'spending_burst_indicator', 'spending_pattern_deviation',
                'is_unusual_amount', 'is_very_large_amount', 'is_very_small_amount'
            ])
        
        # Temporal pattern features
        if self.enable_temporal_patterns:
            features.extend([
                'hour_deviation_score', 'dow_deviation_score', 'unusual_time_indicator',
                'temporal_pattern_break', 'is_preferred_hour', 'is_preferred_day'
            ])
        
        # Merchant pattern features
        if self.enable_merchant_patterns:
            features.extend([
                'is_familiar_merchant', 'is_top_merchant', 'merchant_novelty_score',
                'merchant_pattern_deviation', 'is_familiar_category'
            ])
        
        # Lifecycle features
        if self.enable_lifecycle_features:
            features.extend([
                'days_since_first_transaction', 'days_since_last_transaction',
                'customer_maturity_score', 'is_new_customer', 'is_dormant_reactivation',
                'customer_activity_level'
            ])
        
        # Deviation features
        features.extend([
            'behavioral_deviation_score', 'high_behavioral_deviation',
            'moderate_behavioral_deviation', 'pattern_break_count', 'multiple_pattern_breaks'
        ])
        
        return features
    
    def get_behavioral_insights(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Get insights about behavioral patterns in the data.
        
        Parameters:
        -----------
        X : DataFrame
            Data with behavioral features
            
        Returns:
        --------
        insights : dict
            Behavioral pattern insights
        """
        insights = {}
        
        # Spending behavior insights
        if 'behavioral_deviation_score' in X.columns:
            insights['behavioral_analysis'] = {
                'avg_deviation_score': X['behavioral_deviation_score'].mean(),
                'high_deviation_pct': X['high_behavioral_deviation'].mean() * 100,
                'moderate_deviation_pct': X['moderate_behavioral_deviation'].mean() * 100
            }
        
        # Pattern break insights
        if 'pattern_break_count' in X.columns:
            insights['pattern_breaks'] = {
                'avg_pattern_breaks': X['pattern_break_count'].mean(),
                'multiple_breaks_pct': X['multiple_pattern_breaks'].mean() * 100,
                'max_pattern_breaks': X['pattern_break_count'].max()
            }
        
        # Customer lifecycle insights
        if 'is_new_customer' in X.columns:
            insights['lifecycle'] = {
                'new_customers_pct': X['is_new_customer'].mean() * 100,
                'dormant_reactivations_pct': X['is_dormant_reactivation'].mean() * 100,
                'avg_customer_maturity': X['customer_maturity_score'].mean()
            }
        
        # Activity level distribution
        if 'customer_activity_level' in X.columns:
            activity_dist = X['customer_activity_level'].value_counts(normalize=True) * 100
            insights['activity_distribution'] = activity_dist.to_dict()
        
        return insights