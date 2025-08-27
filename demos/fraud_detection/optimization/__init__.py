"""
Performance Optimization for Real-time Fraud Detection

This package provides comprehensive performance optimization tools for the
Fraud Detection Ensemble, focusing on achieving sub-100ms prediction latency
while maintaining high accuracy.

Key Components:
- Model prediction optimization
- Feature computation caching
- Memory management utilities
- Batch processing optimization
- Latency monitoring and profiling
- Real-time performance tuning

Author: IntegratedML Pluggable Models Team
"""

from .model_optimization import ModelOptimizer, PredictionCache
from .memory_optimization import MemoryOptimizer, DataBuffer
from .latency_optimization import LatencyOptimizer, PerformanceMonitor
from .batch_optimization import BatchProcessor, StreamProcessor
from .caching_strategies import FeatureCache, ModelCache, IntelligentCache

__all__ = [
    'ModelOptimizer',
    'PredictionCache', 
    'MemoryOptimizer',
    'DataBuffer',
    'LatencyOptimizer',
    'PerformanceMonitor',
    'BatchProcessor',
    'StreamProcessor',
    'FeatureCache',
    'ModelCache',
    'IntelligentCache'
]