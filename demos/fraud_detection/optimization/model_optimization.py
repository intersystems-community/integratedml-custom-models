"""
Model Optimization for Real-time Fraud Detection

This module provides model-level optimizations to achieve sub-100ms prediction latency
including model quantization, pruning, prediction caching, and ensemble optimization.

Key Features:
- Model inference optimization
- Prediction result caching
- Ensemble model selection
- Model warm-up and pre-loading
- Memory-efficient model storage
- Dynamic model switching

Author: IntegratedML Pluggable Models Team
"""

import numpy as np
import pandas as pd
import logging
import time
import pickle
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
import hashlib
import warnings

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Model-level optimization for fraud detection ensemble.

    This class provides various optimization techniques to reduce model
    inference time while maintaining prediction accuracy.
    """

    def __init__(
        self,
        enable_prediction_caching: bool = True,
        enable_model_pruning: bool = True,
        enable_fast_path: bool = True,
        cache_size: int = 10000,
        cache_ttl_seconds: int = 300,
    ):
        """
        Initialize model optimizer.

        Parameters:
        -----------
        enable_prediction_caching : bool, default=True
            Whether to cache prediction results
        enable_model_pruning : bool, default=True
            Whether to enable model pruning optimizations
        enable_fast_path : bool, default=True
            Whether to enable fast path for obvious cases
        cache_size : int, default=10000
            Maximum size of prediction cache
        cache_ttl_seconds : int, default=300
            Time-to-live for cached predictions in seconds
        """
        self.enable_prediction_caching = enable_prediction_caching
        self.enable_model_pruning = enable_model_pruning
        self.enable_fast_path = enable_fast_path
        self.cache_size = cache_size
        self.cache_ttl_seconds = cache_ttl_seconds

        # Prediction cache
        self.prediction_cache = PredictionCache(cache_size, cache_ttl_seconds)

        # Model performance statistics
        self.model_performance = defaultdict(list)

        # Fast path rules
        self.fast_path_rules = []

        # Model selection strategy
        self.model_selection_strategy = "adaptive"

        # Optimization statistics
        self.optimization_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "fast_path_hits": 0,
            "total_predictions": 0,
            "avg_prediction_time_ms": 0.0,
        }

        logger.info("Initialized ModelOptimizer")

    def optimize_model(self, model, optimization_level: str = "medium") -> Any:
        """
        Optimize a model for faster inference.

        Parameters:
        -----------
        model : object
            Model to optimize
        optimization_level : str, default='medium'
            Optimization level ('light', 'medium', 'aggressive')

        Returns:
        --------
        optimized_model : object
            Optimized model
        """
        logger.info(f"Optimizing model with {optimization_level} optimization level...")

        optimized_model = model

        # Apply optimizations based on level
        if optimization_level in ["medium", "aggressive"]:
            optimized_model = self._apply_model_compression(optimized_model)

        if optimization_level == "aggressive":
            optimized_model = self._apply_model_pruning(optimized_model)

        # Warm up the model
        self._warm_up_model(optimized_model)

        logger.info("Model optimization completed")
        return optimized_model

    def create_fast_path_rules(
        self, training_data: pd.DataFrame, labels: np.ndarray
    ) -> None:
        """
        Create fast path rules for obvious fraud/legitimate cases.

        Parameters:
        -----------
        training_data : DataFrame
            Training data with features
        labels : array-like
            Training labels
        """
        logger.info("Creating fast path rules...")

        # Rule 1: Very high amount transactions
        high_amount_threshold = training_data["amount"].quantile(0.99)
        fraud_rate_high_amount = labels[
            training_data["amount"] > high_amount_threshold
        ].mean()

        if fraud_rate_high_amount > 0.8:
            self.fast_path_rules.append(
                {
                    "condition": lambda x: x.get("amount", 0) > high_amount_threshold,
                    "prediction": 1,  # Fraud
                    "confidence": fraud_rate_high_amount,
                    "description": f"Very high amount (>{high_amount_threshold:.2f})",
                }
            )

        # Rule 2: Impossible travel
        if "impossible_travel" in training_data.columns:
            impossible_travel_fraud_rate = labels[
                training_data["impossible_travel"] == 1
            ].mean()

            if impossible_travel_fraud_rate > 0.9:
                self.fast_path_rules.append(
                    {
                        "condition": lambda x: x.get("impossible_travel", 0) == 1,
                        "prediction": 1,  # Fraud
                        "confidence": impossible_travel_fraud_rate,
                        "description": "Impossible travel detected",
                    }
                )

        # Rule 3: Multiple rapid-fire transactions
        if "customer_rapid_fire_1m" in training_data.columns:
            rapid_fire_fraud_rate = labels[
                training_data["customer_rapid_fire_1m"] == 1
            ].mean()

            if rapid_fire_fraud_rate > 0.7:
                self.fast_path_rules.append(
                    {
                        "condition": lambda x: x.get("customer_rapid_fire_1m", 0) == 1,
                        "prediction": 1,  # Fraud
                        "confidence": rapid_fire_fraud_rate,
                        "description": "Rapid-fire transactions detected",
                    }
                )

        # Rule 4: Very low risk legitimate transactions
        if "ensemble_risk_score" in training_data.columns:
            low_risk_threshold = 0.1
            low_risk_fraud_rate = labels[
                training_data["ensemble_risk_score"] <= low_risk_threshold
            ].mean()

            if low_risk_fraud_rate < 0.01:
                self.fast_path_rules.append(
                    {
                        "condition": lambda x: x.get("ensemble_risk_score", 0.5)
                        <= low_risk_threshold,
                        "prediction": 0,  # Legitimate
                        "confidence": 1 - low_risk_fraud_rate,
                        "description": f"Very low risk score (<={low_risk_threshold})",
                    }
                )

        logger.info(f"Created {len(self.fast_path_rules)} fast path rules")

    def predict_optimized(
        self,
        model,
        features: Union[Dict[str, Any], pd.DataFrame],
        transaction_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Make optimized prediction with caching and fast paths.

        Parameters:
        -----------
        model : object
            Model for prediction
        features : dict or DataFrame
            Input features
        transaction_id : str, optional
            Transaction identifier for caching

        Returns:
        --------
        result : dict
            Prediction result with optimization metadata
        """
        start_time = time.time()

        # Convert features to dict if DataFrame
        if isinstance(features, pd.DataFrame):
            if len(features) == 1:
                features_dict = features.iloc[0].to_dict()
            else:
                raise ValueError(
                    "DataFrame must contain exactly one row for single prediction"
                )
        else:
            features_dict = features

        self.optimization_stats["total_predictions"] += 1

        # Check prediction cache first
        cache_result = None
        if self.enable_prediction_caching and transaction_id:
            cache_key = self._generate_cache_key(features_dict, transaction_id)
            cache_result = self.prediction_cache.get(cache_key)

            if cache_result is not None:
                self.optimization_stats["cache_hits"] += 1
                cache_result["optimization_metadata"] = {
                    "cache_hit": True,
                    "fast_path": False,
                    "prediction_time_ms": (time.time() - start_time) * 1000,
                }
                return cache_result

        self.optimization_stats["cache_misses"] += 1

        # Check fast path rules
        if self.enable_fast_path:
            fast_path_result = self._check_fast_path(features_dict)
            if fast_path_result is not None:
                self.optimization_stats["fast_path_hits"] += 1

                result = {
                    "prediction": fast_path_result["prediction"],
                    "confidence": fast_path_result["confidence"],
                    "explanation": fast_path_result["description"],
                    "optimization_metadata": {
                        "cache_hit": False,
                        "fast_path": True,
                        "fast_path_rule": fast_path_result["description"],
                        "prediction_time_ms": (time.time() - start_time) * 1000,
                    },
                }

                # Cache the result
                if self.enable_prediction_caching and transaction_id:
                    cache_key = self._generate_cache_key(features_dict, transaction_id)
                    self.prediction_cache.put(cache_key, result)

                return result

        # Full model prediction
        try:
            # Convert features to appropriate format for model
            if hasattr(model, "predict_with_explanations"):
                # Use enhanced prediction method
                df_features = pd.DataFrame([features_dict])
                model_result = model.predict_with_explanations(df_features)

                prediction = model_result.get("predictions", [0])[0]
                confidence = model_result.get("confidence_scores", [0.5])[0]
                explanation = "Full ensemble prediction"

            elif hasattr(model, "predict_proba"):
                # Standard scikit-learn interface
                df_features = pd.DataFrame([features_dict])
                probabilities = model.predict_proba(df_features)[0]

                prediction = 1 if probabilities[1] > 0.5 else 0
                confidence = max(probabilities)
                explanation = "Standard model prediction"

            else:
                # Basic predict method
                df_features = pd.DataFrame([features_dict])
                prediction = model.predict(df_features)[0]
                confidence = 0.7  # Default confidence
                explanation = "Basic model prediction"

            result = {
                "prediction": int(prediction),
                "confidence": float(confidence),
                "explanation": explanation,
                "optimization_metadata": {
                    "cache_hit": False,
                    "fast_path": False,
                    "prediction_time_ms": (time.time() - start_time) * 1000,
                },
            }

            # Cache the result
            if self.enable_prediction_caching and transaction_id:
                cache_key = self._generate_cache_key(features_dict, transaction_id)
                self.prediction_cache.put(cache_key, result)

            # Update performance statistics
            prediction_time = (time.time() - start_time) * 1000
            self.model_performance["prediction_times"].append(prediction_time)

            return result

        except Exception as e:
            logger.error(f"Model prediction failed: {e}")

            # Return safe default
            result = {
                "prediction": 1,  # Conservative: flag as fraud
                "confidence": 0.5,
                "explanation": f"Prediction failed: {str(e)}",
                "optimization_metadata": {
                    "cache_hit": False,
                    "fast_path": False,
                    "error": True,
                    "prediction_time_ms": (time.time() - start_time) * 1000,
                },
            }

            return result

    def _apply_model_compression(self, model) -> Any:
        """
        Apply model compression techniques.

        Parameters:
        -----------
        model : object
            Model to compress

        Returns:
        --------
        compressed_model : object
            Compressed model
        """
        # For scikit-learn models, we can't easily compress
        # but we can optimize by reducing precision for certain operations

        logger.debug("Applying model compression...")

        # For now, return the original model
        # In production, you might implement specific compression for each model type
        return model

    def _apply_model_pruning(self, model) -> Any:
        """
        Apply model pruning techniques.

        Parameters:
        -----------
        model : object
            Model to prune

        Returns:
        --------
        pruned_model : object
            Pruned model
        """
        logger.debug("Applying model pruning...")

        # For ensemble models, we could potentially remove less important components
        # For now, return the original model
        return model

    def _warm_up_model(self, model) -> None:
        """
        Warm up model by running sample predictions.

        Parameters:
        -----------
        model : object
            Model to warm up
        """
        logger.debug("Warming up model...")

        try:
            # Create sample features for warm-up
            sample_features = {
                "amount": 100.0,
                "customer_id": "sample_customer",
                "merchant_name": "sample_merchant",
                "timestamp": datetime.now(),
            }

            # Run a few warm-up predictions
            for _ in range(3):
                try:
                    df_sample = pd.DataFrame([sample_features])
                    if hasattr(model, "predict"):
                        model.predict(df_sample)
                    elif hasattr(model, "predict_proba"):
                        model.predict_proba(df_sample)
                except:
                    pass  # Ignore warm-up errors

            logger.debug("Model warm-up completed")

        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

    def _check_fast_path(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if transaction matches any fast path rules.

        Parameters:
        -----------
        features : dict
            Transaction features

        Returns:
        --------
        fast_path_result : dict or None
            Fast path result if rule matches
        """
        for rule in self.fast_path_rules:
            try:
                if rule["condition"](features):
                    return {
                        "prediction": rule["prediction"],
                        "confidence": rule["confidence"],
                        "description": rule["description"],
                    }
            except Exception as e:
                logger.warning(f"Fast path rule failed: {e}")
                continue

        return None

    def _generate_cache_key(self, features: Dict[str, Any], transaction_id: str) -> str:
        """
        Generate cache key for prediction.

        Parameters:
        -----------
        features : dict
            Transaction features
        transaction_id : str
            Transaction identifier

        Returns:
        --------
        cache_key : str
            Cache key
        """
        # Use relevant features for cache key
        key_features = ["amount", "customer_id", "merchant_name", "payment_method"]

        key_parts = [f"txn:{transaction_id}"]

        for feature in key_features:
            if feature in features:
                value = features[feature]
                key_parts.append(f"{feature}:{value}")

        key_string = "|".join(key_parts)

        # Hash for consistent length
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics.

        Returns:
        --------
        stats : dict
            Optimization statistics
        """
        stats = self.optimization_stats.copy()

        # Calculate derived statistics
        total_predictions = stats["total_predictions"]
        if total_predictions > 0:
            stats["cache_hit_rate"] = (stats["cache_hits"] / total_predictions) * 100
            stats["fast_path_rate"] = (
                stats["fast_path_hits"] / total_predictions
            ) * 100

        # Prediction time statistics
        if "prediction_times" in self.model_performance:
            times = self.model_performance["prediction_times"]
            if times:
                stats["avg_prediction_time_ms"] = np.mean(times)
                stats["median_prediction_time_ms"] = np.median(times)
                stats["p95_prediction_time_ms"] = np.percentile(times, 95)
                stats["max_prediction_time_ms"] = np.max(times)

        # Cache statistics
        stats["cache_size"] = len(self.prediction_cache.cache)
        stats["cache_utilization"] = (
            len(self.prediction_cache.cache) / self.cache_size
        ) * 100

        return stats

    def clear_cache(self) -> None:
        """Clear prediction cache."""
        self.prediction_cache.clear()
        logger.info("Prediction cache cleared")

    def reset_stats(self) -> None:
        """Reset optimization statistics."""
        self.optimization_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "fast_path_hits": 0,
            "total_predictions": 0,
            "avg_prediction_time_ms": 0.0,
        }
        self.model_performance.clear()
        logger.info("Optimization statistics reset")


class PredictionCache:
    """
    Thread-safe prediction cache with TTL support.
    """

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 300):
        """
        Initialize prediction cache.

        Parameters:
        -----------
        max_size : int, default=10000
            Maximum cache size
        ttl_seconds : int, default=300
            Time-to-live for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get prediction from cache.

        Parameters:
        -----------
        key : str
            Cache key

        Returns:
        --------
        result : dict or None
            Cached prediction result
        """
        with self.lock:
            if key not in self.cache:
                return None

            # Check TTL
            if self._is_expired(key):
                self._remove(key)
                return None

            # Move to end (LRU)
            value = self.cache[key]
            del self.cache[key]
            self.cache[key] = value

            return value.copy()

    def put(self, key: str, value: Dict[str, Any]) -> None:
        """
        Put prediction in cache.

        Parameters:
        -----------
        key : str
            Cache key
        value : dict
            Prediction result to cache
        """
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                del self.cache[key]

            # Add new entry
            self.cache[key] = value.copy()
            self.timestamps[key] = datetime.now()

            # Maintain size limit
            while len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                self._remove(oldest_key)

    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.timestamps:
            return True

        timestamp = self.timestamps[key]
        return (datetime.now() - timestamp).total_seconds() > self.ttl_seconds

    def _remove(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]

    def cleanup_expired(self) -> int:
        """
        Clean up expired entries.

        Returns:
        --------
        removed_count : int
            Number of entries removed
        """
        with self.lock:
            expired_keys = []

            for key in list(self.cache.keys()):
                if self._is_expired(key):
                    expired_keys.append(key)

            for key in expired_keys:
                self._remove(key)

            return len(expired_keys)


class ModelSelector:
    """
    Dynamic model selection for optimal performance.
    """

    def __init__(self, models: Dict[str, Any], selection_strategy: str = "adaptive"):
        """
        Initialize model selector.

        Parameters:
        -----------
        models : dict
            Available models {name: model}
        selection_strategy : str, default='adaptive'
            Selection strategy ('fastest', 'most_accurate', 'adaptive')
        """
        self.models = models
        self.selection_strategy = selection_strategy

        # Performance tracking
        self.model_performance = defaultdict(
            lambda: {"prediction_times": [], "accuracy_estimates": [], "usage_count": 0}
        )

        # Current model selection
        self.current_model = None
        self.selection_timestamp = None

        logger.info(f"Initialized ModelSelector with {len(models)} models")

    def select_model(self, features: Dict[str, Any]) -> str:
        """
        Select optimal model for given features.

        Parameters:
        -----------
        features : dict
            Transaction features

        Returns:
        --------
        model_name : str
            Selected model name
        """
        if self.selection_strategy == "fastest":
            return self._select_fastest_model()
        elif self.selection_strategy == "most_accurate":
            return self._select_most_accurate_model()
        elif self.selection_strategy == "adaptive":
            return self._select_adaptive_model(features)
        else:
            # Default to first available model
            return list(self.models.keys())[0]

    def _select_fastest_model(self) -> str:
        """Select model with lowest average prediction time."""
        fastest_model = None
        min_time = float("inf")

        for model_name, perf in self.model_performance.items():
            if perf["prediction_times"]:
                avg_time = np.mean(perf["prediction_times"])
                if avg_time < min_time:
                    min_time = avg_time
                    fastest_model = model_name

        return fastest_model or list(self.models.keys())[0]

    def _select_most_accurate_model(self) -> str:
        """Select model with highest estimated accuracy."""
        most_accurate_model = None
        max_accuracy = 0

        for model_name, perf in self.model_performance.items():
            if perf["accuracy_estimates"]:
                avg_accuracy = np.mean(perf["accuracy_estimates"])
                if avg_accuracy > max_accuracy:
                    max_accuracy = avg_accuracy
                    most_accurate_model = model_name

        return most_accurate_model or list(self.models.keys())[0]

    def _select_adaptive_model(self, features: Dict[str, Any]) -> str:
        """Select model adaptively based on features and performance."""
        # For now, use fastest model
        # In production, this could be more sophisticated
        return self._select_fastest_model()

    def update_performance(
        self,
        model_name: str,
        prediction_time: float,
        accuracy_estimate: Optional[float] = None,
    ) -> None:
        """
        Update model performance statistics.

        Parameters:
        -----------
        model_name : str
            Model name
        prediction_time : float
            Prediction time in milliseconds
        accuracy_estimate : float, optional
            Estimated accuracy (0-1)
        """
        perf = self.model_performance[model_name]

        perf["prediction_times"].append(prediction_time)
        perf["usage_count"] += 1

        if accuracy_estimate is not None:
            perf["accuracy_estimates"].append(accuracy_estimate)

        # Keep only recent performance data
        max_history = 1000
        if len(perf["prediction_times"]) > max_history:
            perf["prediction_times"] = perf["prediction_times"][-max_history:]

        if len(perf["accuracy_estimates"]) > max_history:
            perf["accuracy_estimates"] = perf["accuracy_estimates"][-max_history:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for all models.

        Returns:
        --------
        summary : dict
            Performance summary
        """
        summary = {}

        for model_name, perf in self.model_performance.items():
            model_summary = {"usage_count": perf["usage_count"]}

            if perf["prediction_times"]:
                model_summary.update(
                    {
                        "avg_prediction_time_ms": np.mean(perf["prediction_times"]),
                        "median_prediction_time_ms": np.median(
                            perf["prediction_times"]
                        ),
                        "p95_prediction_time_ms": np.percentile(
                            perf["prediction_times"], 95
                        ),
                    }
                )

            if perf["accuracy_estimates"]:
                model_summary.update(
                    {
                        "avg_accuracy": np.mean(perf["accuracy_estimates"]),
                        "min_accuracy": np.min(perf["accuracy_estimates"]),
                        "max_accuracy": np.max(perf["accuracy_estimates"]),
                    }
                )

            summary[model_name] = model_summary

        return summary
