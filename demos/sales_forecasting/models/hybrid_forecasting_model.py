import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

from shared.models.regression import RegressionModel

_VALID_STRATEGIES = {"simple_average", "horizon_weighted", "confidence_weighted"}
_MIN_TRAINING_SAMPLES = 30


class HybridForecastingModel(RegressionModel):

    def __init__(
        self,
        forecast_horizon: int = 30,
        prophet_params: Optional[Dict[str, Any]] = None,
        lightgbm_params: Optional[Dict[str, Any]] = None,
        lightgbm_config: Optional[Dict[str, Any]] = None,
        ensemble_strategy: str = "simple_average",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            forecast_horizon=forecast_horizon,
            ensemble_strategy=ensemble_strategy,
            **kwargs,
        )
        self.forecast_horizon = forecast_horizon
        self.prophet_params = prophet_params or {}
        self.lightgbm_params = lightgbm_params or lightgbm_config or {}
        self.ensemble_strategy = ensemble_strategy

        if ensemble_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"ensemble_strategy must be one of {_VALID_STRATEGIES}, got {ensemble_strategy!r}"
            )

        self._lgbm_model: Optional[Any] = None
        self._feature_names: Optional[list] = None

    @property
    def is_fitted(self) -> bool:
        return self._is_trained

    def _fit_impl(self, X: np.ndarray, y: np.ndarray, **fit_params: Any) -> None:
        if len(X) < _MIN_TRAINING_SAMPLES:
            raise ValueError(
                f"Need at least {_MIN_TRAINING_SAMPLES} training samples, got {len(X)}"
            )

        try:
            import lightgbm as lgb

            params = {
                "objective": "regression",
                "metric": "rmse",
                "verbose": -1,
                "n_estimators": 100,
                "num_leaves": 31,
                "learning_rate": 0.1,
                "random_state": 42,
            }
            params.update(self.lightgbm_params)
            n_estimators = params.pop("n_estimators", 100)
            random_state = params.pop("random_state", 42)
            self._lgbm_model = lgb.LGBMRegressor(
                n_estimators=n_estimators, random_state=random_state, **params
            )
            self._lgbm_model.fit(X, y)
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor

            self._lgbm_model = GradientBoostingRegressor(
                n_estimators=50, random_state=42
            )
            self._lgbm_model.fit(X, y)

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        raw = self._lgbm_model.predict(X)
        return np.clip(raw, 0, None)

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "HybridForecastingModel":
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
        y_arr = np.asarray(y, dtype=float)

        if len(X_arr) != len(y_arr):
            raise ValueError("X and y must have the same number of rows")

        self._fit_impl(X_arr, y_arr, **fit_params)
        self._is_trained = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        self._require_trained()
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
        return self._predict_impl(X_arr)

    def predict_with_confidence(
        self, X: Any, confidence_level: float = 0.95
    ) -> Dict[str, np.ndarray]:
        predictions = self.predict(X)
        margin = predictions * 0.1 * (1.0 - confidence_level + 0.05)
        return {
            "lower": np.clip(predictions - margin, 0, None),
            "upper": predictions + margin,
        }

    def get_feature_importance(self) -> Optional[pd.Series]:
        if not self._is_trained or self._lgbm_model is None:
            return None
        if hasattr(self._lgbm_model, "feature_importances_"):
            imp = self._lgbm_model.feature_importances_
            if self._feature_names and len(self._feature_names) == len(imp):
                return pd.Series(imp, index=self._feature_names)
            return pd.Series(imp)
        return None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "forecast_horizon": self.forecast_horizon,
            "prophet_params": self.prophet_params,
            "lightgbm_params": self.lightgbm_params,
            "ensemble_strategy": self.ensemble_strategy,
        }

    def set_params(self, **params: Any) -> "HybridForecastingModel":
        for k, v in params.items():
            setattr(self, k, v)
            self.parameters[k] = v
        return self
