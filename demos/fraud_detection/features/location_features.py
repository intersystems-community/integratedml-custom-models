"""
Location Feature Engineering

This module provides location-based feature engineering for fraud detection,
including geographic analysis, travel patterns, and location risk scoring.

Key Features:
- Geographic distance and travel time analysis
- Location risk scoring based on fraud history
- Impossible travel detection
- Location consistency validation
- Timezone and business hours analysis
- Real-time location processing

Author: IntegratedML Pluggable Models Team
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import math
import warnings

logger = logging.getLogger(__name__)


class LocationFeatureEngineer:
    """
    Location-based feature engineering for fraud detection.
    
    This class extracts and computes various location-based features
    that are crucial for fraud detection including geographic patterns,
    travel analysis, and location risk assessment.
    """
    
    def __init__(self,
                 enable_distance_features: bool = True,
                 enable_travel_analysis: bool = True,
                 enable_location_risk: bool = True,
                 enable_timezone_features: bool = True,
                 max_travel_speed_kmh: float = 1000.0):  # Max reasonable travel speed
        """
        Initialize location feature engineer.
        
        Parameters:
        -----------
        enable_distance_features : bool, default=True
            Whether to compute distance-based features
        enable_travel_analysis : bool, default=True
            Whether to analyze travel patterns
        enable_location_risk : bool, default=True
            Whether to compute location risk scores
        enable_timezone_features : bool, default=True
            Whether to compute timezone-based features
        max_travel_speed_kmh : float, default=1000.0
            Maximum reasonable travel speed in km/h for impossible travel detection
        """
        self.enable_distance_features = enable_distance_features
        self.enable_travel_analysis = enable_travel_analysis
        self.enable_location_risk = enable_location_risk
        self.enable_timezone_features = enable_timezone_features
        self.max_travel_speed_kmh = max_travel_speed_kmh
        
        # Location risk scoring
        self.location_risk_scores = {}
        self.state_risk_scores = {}
        self.country_risk_scores = {}
        
        # Location statistics
        self.location_stats = {}
        
        # Common location coordinates (simplified mapping)
        self.location_coordinates = {}
        
        # Fitted status
        self.is_fitted = False
        
        logger.info("Initialized LocationFeatureEngineer")
    
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'LocationFeatureEngineer':
        """
        Fit the location feature engineer on training data.
        
        Parameters:
        -----------
        X : DataFrame
            Transaction data with location information
        y : array-like, optional
            Fraud labels for risk scoring
            
        Returns:
        --------
        self : LocationFeatureEngineer
            Fitted feature engineer
        """
        logger.info("Fitting LocationFeatureEngineer...")
        
        # Compute location risk scores if labels provided
        if y is not None and self.enable_location_risk:
            self._compute_location_risk_scores(X, y)
        
        # Build location coordinate mapping
        self._build_location_coordinates(X)
        
        # Compute location statistics
        self._compute_location_statistics(X)
        
        self.is_fitted = True
        logger.info("LocationFeatureEngineer fitted successfully")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform transaction data into location features.
        
        Parameters:
        -----------
        X : DataFrame
            Transaction data
            
        Returns:
        --------
        X_features : DataFrame
            Data with location features added
        """
        if not self.is_fitted:
            raise ValueError("Location feature engineer must be fitted before transform")
        
        logger.debug(f"Computing location features for {len(X)} transactions...")
        
        # Create copy to avoid modifying original data
        X_features = X.copy()
        
        # Basic location features
        X_features = self._add_basic_location_features(X_features)
        
        # Distance features
        if self.enable_distance_features:
            X_features = self._add_distance_features(X_features)
        
        # Travel analysis features
        if self.enable_travel_analysis:
            X_features = self._add_travel_features(X_features)
        
        # Location risk features
        if self.enable_location_risk:
            X_features = self._add_location_risk_features(X_features)
        
        # Timezone features
        if self.enable_timezone_features:
            X_features = self._add_timezone_features(X_features)
        
        logger.debug(f"Generated location features")
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
            Data with location features
        """
        return self.fit(X, y).transform(X)
    
    def _add_basic_location_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic location features.
        
        Parameters:
        -----------
        X : DataFrame
            Input data
            
        Returns:
        --------
        X : DataFrame
            Data with basic location features added
        """
        # Customer location features
        if 'customer_location' in X.columns:
            X['customer_location_encoded'] = self._encode_location(X['customer_location'])
            X['customer_state'] = X['customer_location'].apply(self._extract_state)
            X['customer_country'] = X['customer_location'].apply(self._extract_country)
        
        # Merchant location features
        if 'merchant_location' in X.columns:
            X['merchant_location_encoded'] = self._encode_location(X['merchant_location'])
            X['merchant_state'] = X['merchant_location'].apply(self._extract_state)
            X['merchant_country'] = X['merchant_location'].apply(self._extract_country)
        
        # Location consistency
        if 'customer_location' in X.columns and 'merchant_location' in X.columns:
            X['same_location'] = (X['customer_location'] == X['merchant_location']).astype(int)
            X['same_state'] = (X['customer_state'] == X['merchant_state']).astype(int)
            X['same_country'] = (X['customer_country'] == X['merchant_country']).astype(int)
        
        return X
    
    def _add_distance_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add distance-based features.
        
        Parameters:
        -----------
        X : DataFrame
            Input data
            
        Returns:
        --------
        X : DataFrame
            Data with distance features added
        """
        if 'customer_location' not in X.columns or 'merchant_location' not in X.columns:
            logger.warning("Customer or merchant location missing, skipping distance features")
            return X
        
        # Calculate distances
        distances = []
        for _, row in X.iterrows():
            customer_loc = row['customer_location']
            merchant_loc = row['merchant_location']
            
            distance = self._calculate_distance(customer_loc, merchant_loc)
            distances.append(distance)
        
        X['customer_merchant_distance_km'] = distances
        
        # Distance categories
        X['distance_category'] = pd.cut(
            X['customer_merchant_distance_km'],
            bins=[0, 5, 25, 100, 500, float('inf')],
            labels=['very_close', 'close', 'medium', 'far', 'very_far']
        )
        
        # Distance flags
        X['is_local_transaction'] = (X['customer_merchant_distance_km'] <= 25).astype(int)
        X['is_long_distance'] = (X['customer_merchant_distance_km'] > 500).astype(int)
        X['is_very_long_distance'] = (X['customer_merchant_distance_km'] > 1000).astype(int)
        
        # Log distance for modeling
        X['distance_log'] = np.log1p(X['customer_merchant_distance_km'])
        
        return X
    
    def _add_travel_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add travel pattern analysis features.
        
        Parameters:
        -----------
        X : DataFrame
            Input data
            
        Returns:
        --------
        X : DataFrame
            Data with travel features added
        """
        if 'timestamp' not in X.columns or 'customer_id' not in X.columns:
            logger.warning("Timestamp or customer_id missing, skipping travel features")
            return X
        
        # Sort by customer and timestamp
        X = X.sort_values(['customer_id', 'timestamp']).reset_index(drop=True)
        
        # Initialize travel features
        X['travel_distance_km'] = 0.0
        X['travel_time_hours'] = 0.0
        X['travel_speed_kmh'] = 0.0
        X['impossible_travel'] = 0
        X['suspicious_travel'] = 0
        
        # Compute travel features for each customer
        for customer_id, group in X.groupby('customer_id'):
            group = group.sort_values('timestamp')
            
            for i in range(1, len(group)):
                current_idx = group.index[i]
                prev_idx = group.index[i-1]
                
                current_row = group.iloc[i]
                prev_row = group.iloc[i-1]
                
                # Calculate travel distance
                if 'customer_location' in X.columns:
                    distance = self._calculate_distance(
                        prev_row['customer_location'],
                        current_row['customer_location']
                    )
                    X.loc[current_idx, 'travel_distance_km'] = distance
                    
                    # Calculate travel time
                    time_diff = (current_row['timestamp'] - prev_row['timestamp']).total_seconds() / 3600  # hours
                    X.loc[current_idx, 'travel_time_hours'] = time_diff
                    
                    # Calculate travel speed
                    if time_diff > 0:
                        speed = distance / time_diff
                        X.loc[current_idx, 'travel_speed_kmh'] = speed
                        
                        # Flag impossible travel
                        if speed > self.max_travel_speed_kmh:
                            X.loc[current_idx, 'impossible_travel'] = 1
                        
                        # Flag suspicious travel (very fast but possible)
                        if speed > 200 and speed <= self.max_travel_speed_kmh:
                            X.loc[current_idx, 'suspicious_travel'] = 1
        
        # Travel pattern features
        X['is_stationary'] = (X['travel_distance_km'] <= 1).astype(int)
        X['high_mobility'] = (X['travel_distance_km'] > 100).astype(int)
        
        return X
    
    def _add_location_risk_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add location risk scoring features.
        
        Parameters:
        -----------
        X : DataFrame
            Input data
            
        Returns:
        --------
        X : DataFrame
            Data with location risk features added
        """
        # Customer location risk
        if 'customer_location' in X.columns and self.location_risk_scores:
            X['customer_location_risk'] = X['customer_location'].map(
                self.location_risk_scores
            ).fillna(0.5)  # Default medium risk
        
        # Merchant location risk
        if 'merchant_location' in X.columns and self.location_risk_scores:
            X['merchant_location_risk'] = X['merchant_location'].map(
                self.location_risk_scores
            ).fillna(0.5)
        
        # State-level risk
        if 'customer_state' in X.columns and self.state_risk_scores:
            X['customer_state_risk'] = X['customer_state'].map(
                self.state_risk_scores
            ).fillna(0.5)
        
        if 'merchant_state' in X.columns and self.state_risk_scores:
            X['merchant_state_risk'] = X['merchant_state'].map(
                self.state_risk_scores
            ).fillna(0.5)
        
        # Country-level risk
        if 'customer_country' in X.columns and self.country_risk_scores:
            X['customer_country_risk'] = X['customer_country'].map(
                self.country_risk_scores
            ).fillna(0.5)
        
        # Combined location risk score
        risk_cols = [col for col in X.columns if col.endswith('_risk')]
        if risk_cols:
            X['combined_location_risk'] = X[risk_cols].mean(axis=1)
        
        # High-risk location flags
        if 'customer_location_risk' in X.columns:
            X['high_risk_customer_location'] = (X['customer_location_risk'] > 0.7).astype(int)
        
        if 'merchant_location_risk' in X.columns:
            X['high_risk_merchant_location'] = (X['merchant_location_risk'] > 0.7).astype(int)
        
        return X
    
    def _add_timezone_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add timezone and business hours features.
        
        Parameters:
        -----------
        X : DataFrame
            Input data
            
        Returns:
        --------
        X : DataFrame
            Data with timezone features added
        """
        if 'timestamp' not in X.columns:
            logger.warning("Timestamp missing, skipping timezone features")
            return X
        
        # Convert timestamp if needed
        if not pd.api.types.is_datetime64_any_dtype(X['timestamp']):
            X['timestamp'] = pd.to_datetime(X['timestamp'])
        
        # Extract time features
        X['hour'] = X['timestamp'].dt.hour
        X['day_of_week'] = X['timestamp'].dt.dayofweek
        
        # Business hours analysis based on merchant location
        if 'merchant_location' in X.columns:
            X['merchant_timezone'] = X['merchant_location'].apply(self._get_timezone)
            X['merchant_local_hour'] = X.apply(
                lambda row: self._convert_to_local_time(row['timestamp'], row['merchant_timezone']),
                axis=1
            )
            
            # Business hours flags
            X['merchant_business_hours'] = (
                (X['merchant_local_hour'] >= 9) & (X['merchant_local_hour'] <= 17)
            ).astype(int)
            
            X['merchant_after_hours'] = (
                (X['merchant_local_hour'] < 9) | (X['merchant_local_hour'] > 17)
            ).astype(int)
            
            X['merchant_late_night'] = (
                (X['merchant_local_hour'] >= 22) | (X['merchant_local_hour'] <= 5)
            ).astype(int)
        
        # Customer timezone analysis
        if 'customer_location' in X.columns:
            X['customer_timezone'] = X['customer_location'].apply(self._get_timezone)
            X['customer_local_hour'] = X.apply(
                lambda row: self._convert_to_local_time(row['timestamp'], row['customer_timezone']),
                axis=1
            )
            
            # Unusual timing flags
            X['customer_unusual_hour'] = (
                (X['customer_local_hour'] >= 2) & (X['customer_local_hour'] <= 6)
            ).astype(int)
        
        # Timezone consistency
        if 'merchant_timezone' in X.columns and 'customer_timezone' in X.columns:
            X['timezone_mismatch'] = (X['merchant_timezone'] != X['customer_timezone']).astype(int)
            X['timezone_difference'] = X.apply(
                lambda row: abs(self._get_timezone_offset(row['merchant_timezone']) - 
                              self._get_timezone_offset(row['customer_timezone'])),
                axis=1
            )
        
        return X
    
    def _compute_location_risk_scores(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Compute location risk scores based on fraud history.
        
        Parameters:
        -----------
        X : DataFrame
            Training data
        y : array-like
            Fraud labels
        """
        # Customer location risk scores
        if 'customer_location' in X.columns:
            location_fraud_rates = pd.DataFrame({
                'customer_location': X['customer_location'],
                'fraud': y
            }).groupby('customer_location')['fraud'].agg(['mean', 'count'])
            
            # Only consider locations with sufficient data
            min_transactions = 5
            reliable_locations = location_fraud_rates[location_fraud_rates['count'] >= min_transactions]
            
            self.location_risk_scores.update(reliable_locations['mean'].to_dict())
        
        # State-level risk scores
        if 'customer_location' in X.columns:
            states = X['customer_location'].apply(self._extract_state)
            state_fraud_rates = pd.DataFrame({
                'state': states,
                'fraud': y
            }).groupby('state')['fraud'].agg(['mean', 'count'])
            
            reliable_states = state_fraud_rates[state_fraud_rates['count'] >= 10]
            self.state_risk_scores.update(reliable_states['mean'].to_dict())
        
        # Country-level risk scores
        if 'customer_location' in X.columns:
            countries = X['customer_location'].apply(self._extract_country)
            country_fraud_rates = pd.DataFrame({
                'country': countries,
                'fraud': y
            }).groupby('country')['fraud'].agg(['mean', 'count'])
            
            reliable_countries = country_fraud_rates[country_fraud_rates['count'] >= 20]
            self.country_risk_scores.update(reliable_countries['mean'].to_dict())
    
    def _build_location_coordinates(self, X: pd.DataFrame) -> None:
        """
        Build a mapping of locations to coordinates (simplified).
        
        Parameters:
        -----------
        X : DataFrame
            Transaction data
        """
        # In a real implementation, this would use a geocoding service
        # For demo purposes, we'll use simplified coordinates
        
        unique_locations = set()
        for col in ['customer_location', 'merchant_location']:
            if col in X.columns:
                unique_locations.update(X[col].dropna().unique())
        
        # Generate simplified coordinates
        for location in unique_locations:
            self.location_coordinates[location] = self._get_simplified_coordinates(location)
    
    def _compute_location_statistics(self, X: pd.DataFrame) -> None:
        """
        Compute location statistics from training data.
        
        Parameters:
        -----------
        X : DataFrame
            Training data
        """
        # Customer location statistics
        if 'customer_location' in X.columns:
            customer_location_stats = X['customer_location'].value_counts().to_dict()
            self.location_stats['customer_locations'] = customer_location_stats
        
        # Merchant location statistics
        if 'merchant_location' in X.columns:
            merchant_location_stats = X['merchant_location'].value_counts().to_dict()
            self.location_stats['merchant_locations'] = merchant_location_stats
    
    def _encode_location(self, locations: pd.Series) -> pd.Series:
        """
        Encode locations as numeric values.
        
        Parameters:
        -----------
        locations : Series
            Location names
            
        Returns:
        --------
        encoded : Series
            Encoded location values
        """
        unique_locations = locations.dropna().unique()
        location_map = {loc: i for i, loc in enumerate(unique_locations)}
        return locations.map(location_map).fillna(-1)
    
    def _extract_state(self, location: str) -> str:
        """
        Extract state from location string.
        
        Parameters:
        -----------
        location : str
            Location string
            
        Returns:
        --------
        state : str
            Extracted state
        """
        if pd.isna(location):
            return 'unknown'
        
        # Simplified state extraction
        parts = str(location).split(',')
        if len(parts) >= 2:
            return parts[-2].strip()
        return 'unknown'
    
    def _extract_country(self, location: str) -> str:
        """
        Extract country from location string.
        
        Parameters:
        -----------
        location : str
            Location string
            
        Returns:
        --------
        country : str
            Extracted country
        """
        if pd.isna(location):
            return 'unknown'
        
        # Simplified country extraction
        parts = str(location).split(',')
        if len(parts) >= 1:
            return parts[-1].strip()
        return 'unknown'
    
    def _calculate_distance(self, loc1: str, loc2: str) -> float:
        """
        Calculate distance between two locations.
        
        Parameters:
        -----------
        loc1 : str
            First location
        loc2 : str
            Second location
            
        Returns:
        --------
        distance : float
            Distance in kilometers
        """
        if pd.isna(loc1) or pd.isna(loc2) or loc1 == loc2:
            return 0.0
        
        # Get coordinates
        coord1 = self.location_coordinates.get(loc1)
        coord2 = self.location_coordinates.get(loc2)
        
        if coord1 is None or coord2 is None:
            # Return estimated distance based on location names
            return self._estimate_distance_from_names(loc1, loc2)
        
        # Calculate haversine distance
        return self._haversine_distance(coord1[0], coord1[1], coord2[0], coord2[1])
    
    def _get_simplified_coordinates(self, location: str) -> Tuple[float, float]:
        """
        Get simplified coordinates for a location.
        
        Parameters:
        -----------
        location : str
            Location name
            
        Returns:
        --------
        coordinates : tuple
            (latitude, longitude)
        """
        # Simplified coordinate mapping for demo
        # In production, use a proper geocoding service
        
        location_lower = str(location).lower()
        
        # Major US cities (lat, lon)
        city_coords = {
            'new york': (40.7128, -74.0060),
            'los angeles': (34.0522, -118.2437),
            'chicago': (41.8781, -87.6298),
            'houston': (29.7604, -95.3698),
            'phoenix': (33.4484, -112.0740),
            'philadelphia': (39.9526, -75.1652),
            'san antonio': (29.4241, -98.4936),
            'san diego': (32.7157, -117.1611),
            'dallas': (32.7767, -96.7970),
            'san jose': (37.3382, -121.8863)
        }
        
        for city, coords in city_coords.items():
            if city in location_lower:
                return coords
        
        # Default to random coordinates in US
        import random
        lat = random.uniform(25.0, 49.0)  # US latitude range
        lon = random.uniform(-125.0, -66.0)  # US longitude range
        return (lat, lon)
    
    def _estimate_distance_from_names(self, loc1: str, loc2: str) -> float:
        """
        Estimate distance based on location names.
        
        Parameters:
        -----------
        loc1 : str
            First location
        loc2 : str
            Second location
            
        Returns:
        --------
        distance : float
            Estimated distance in kilometers
        """
        # Very simplified distance estimation
        state1 = self._extract_state(loc1)
        state2 = self._extract_state(loc2)
        
        if state1 == state2:
            return 100.0  # Same state
        else:
            return 500.0  # Different states
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate haversine distance between two coordinates.
        
        Parameters:
        -----------
        lat1, lon1, lat2, lon2 : float
            Coordinates in degrees
            
        Returns:
        --------
        distance : float
            Distance in kilometers
        """
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def _get_timezone(self, location: str) -> str:
        """
        Get timezone for a location.
        
        Parameters:
        -----------
        location : str
            Location name
            
        Returns:
        --------
        timezone : str
            Timezone identifier
        """
        if pd.isna(location):
            return 'UTC'
        
        location_lower = str(location).lower()
        
        # Simplified timezone mapping
        if any(city in location_lower for city in ['new york', 'philadelphia', 'atlanta']):
            return 'EST'
        elif any(city in location_lower for city in ['chicago', 'dallas', 'houston']):
            return 'CST'
        elif any(city in location_lower for city in ['denver', 'phoenix']):
            return 'MST'
        elif any(city in location_lower for city in ['los angeles', 'san francisco', 'seattle']):
            return 'PST'
        else:
            return 'UTC'
    
    def _convert_to_local_time(self, timestamp: datetime, timezone: str) -> int:
        """
        Convert timestamp to local hour.
        
        Parameters:
        -----------
        timestamp : datetime
            UTC timestamp
        timezone : str
            Target timezone
            
        Returns:
        --------
        local_hour : int
            Local hour (0-23)
        """
        offset = self._get_timezone_offset(timezone)
        local_time = timestamp + timedelta(hours=offset)
        return local_time.hour
    
    def _get_timezone_offset(self, timezone: str) -> int:
        """
        Get timezone offset from UTC.
        
        Parameters:
        -----------
        timezone : str
            Timezone identifier
            
        Returns:
        --------
        offset : int
            Hours offset from UTC
        """
        offsets = {
            'EST': -5,
            'CST': -6,
            'MST': -7,
            'PST': -8,
            'UTC': 0
        }
        return offsets.get(timezone, 0)
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all generated location features.
        
        Returns:
        --------
        feature_names : list of str
            List of location feature names
        """
        features = []
        
        # Basic location features
        features.extend([
            'customer_location_encoded', 'customer_state', 'customer_country',
            'merchant_location_encoded', 'merchant_state', 'merchant_country',
            'same_location', 'same_state', 'same_country'
        ])
        
        # Distance features
        if self.enable_distance_features:
            features.extend([
                'customer_merchant_distance_km', 'distance_category',
                'is_local_transaction', 'is_long_distance', 'is_very_long_distance',
                'distance_log'
            ])
        
        # Travel features
        if self.enable_travel_analysis:
            features.extend([
                'travel_distance_km', 'travel_time_hours', 'travel_speed_kmh',
                'impossible_travel', 'suspicious_travel', 'is_stationary', 'high_mobility'
            ])
        
        # Location risk features
        if self.enable_location_risk:
            features.extend([
                'customer_location_risk', 'merchant_location_risk',
                'customer_state_risk', 'merchant_state_risk',
                'customer_country_risk', 'combined_location_risk',
                'high_risk_customer_location', 'high_risk_merchant_location'
            ])
        
        # Timezone features
        if self.enable_timezone_features:
            features.extend([
                'merchant_timezone', 'merchant_local_hour', 'merchant_business_hours',
                'merchant_after_hours', 'merchant_late_night',
                'customer_timezone', 'customer_local_hour', 'customer_unusual_hour',
                'timezone_mismatch', 'timezone_difference'
            ])
        
        return features
    
    def get_location_insights(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Get insights about location patterns in the data.
        
        Parameters:
        -----------
        X : DataFrame
            Data with location features
            
        Returns:
        --------
        insights : dict
            Location pattern insights
        """
        insights = {}
        
        # Distance analysis
        if 'customer_merchant_distance_km' in X.columns:
            insights['distance_analysis'] = {
                'avg_distance_km': X['customer_merchant_distance_km'].mean(),
                'max_distance_km': X['customer_merchant_distance_km'].max(),
                'local_transactions_pct': X['is_local_transaction'].mean() * 100,
                'long_distance_pct': X['is_long_distance'].mean() * 100
            }
        
        # Travel analysis
        if 'impossible_travel' in X.columns:
            insights['travel_analysis'] = {
                'impossible_travel_count': X['impossible_travel'].sum(),
                'suspicious_travel_count': X['suspicious_travel'].sum(),
                'avg_travel_speed_kmh': X['travel_speed_kmh'].mean(),
                'max_travel_speed_kmh': X['travel_speed_kmh'].max()
            }
        
        # Location risk analysis
        if 'combined_location_risk' in X.columns:
            insights['risk_analysis'] = {
                'avg_location_risk': X['combined_location_risk'].mean(),
                'high_risk_locations_pct': (X['combined_location_risk'] > 0.7).mean() * 100
            }
        
        return insights