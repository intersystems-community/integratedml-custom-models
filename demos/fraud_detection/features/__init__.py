"""
Advanced Feature Engineering for Fraud Detection

This package provides comprehensive feature engineering capabilities for the
Fraud Detection Ensemble, including:

- Transaction velocity and frequency features
- Geographic and location-based analysis
- Behavioral deviation scoring
- Time-based pattern recognition
- Risk aggregation and scoring
- Real-time feature computation

Author: IntegratedML Pluggable Models Team
"""

from .transaction_features import TransactionFeatureEngineer
from .velocity_features import VelocityFeatureEngineer
from .location_features import LocationFeatureEngineer
from .behavioral_features import BehavioralFeatureEngineer
from .risk_features import RiskFeatureEngineer
from .realtime_features import RealTimeFeatureProcessor

__all__ = [
    'TransactionFeatureEngineer',
    'VelocityFeatureEngineer',
    'LocationFeatureEngineer',
    'BehavioralFeatureEngineer',
    'RiskFeatureEngineer',
    'RealTimeFeatureProcessor'
]