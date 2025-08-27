"""
IntegratedML Flexible Model Integration Framework - Shared Components

This package provides common utilities, base classes, and testing infrastructure
for IntegratedML flexible model integration demonstrations.
"""

__version__ = "1.0.0"
__author__ = "InterSystems Corporation"

# Import key base classes for easy access
from .models.base import IntegratedMLBaseModel
from .models.classification import ClassificationModel
from .models.regression import RegressionModel
from .models.ensemble import EnsembleModel

__all__ = [
    "IntegratedMLBaseModel",
    "ClassificationModel", 
    "RegressionModel",
    "EnsembleModel"
]