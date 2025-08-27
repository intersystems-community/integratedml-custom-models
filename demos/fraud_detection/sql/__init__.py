"""
IntegratedML SQL Integration for Fraud Detection

This package provides comprehensive SQL integration scripts for deploying
and managing the fraud detection ensemble system in IRIS IntegratedML.

Key Components:
- Model deployment and management
- Real-time prediction queries
- Performance monitoring
- Data preparation and feature extraction
- Batch processing and optimization

Author: IntegratedML Pluggable Models Team
"""

__version__ = "1.0.0"
__author__ = "IntegratedML Pluggable Models Team"

# SQL script categories
SQL_CATEGORIES = [
    "deployment",
    "prediction", 
    "monitoring",
    "optimization",
    "maintenance"
]

# Model types supported
SUPPORTED_MODEL_TYPES = [
    "ensemble_fraud_detector",
    "rule_based_detector", 
    "anomaly_detector",
    "neural_detector",
    "behavioral_detector"
]