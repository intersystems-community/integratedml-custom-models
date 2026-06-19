"""
IRISModel wrapper for Gaia Variable Star detection.

Deploy this file to the pathtoclassifiers directory referenced in:
    CREATE MODEL GaiaVariability PREDICTING (is_variable)
    FROM GaiaObservationStats
    USING {"pathtoclassifiers": "/path/to/iris_models", "iscmodelsdisabled": 1}

# Feedback: The IRISModel contract is simple — just fit/predict/predict_proba.
# The threshold parameter flows in from the USING clause JSON.
# IntegratedML handles serialization automatically after TRAIN MODEL.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class IRISModel:
    name = "gaia_variability_detector"

    def __init__(self, threshold_pct: float = 10.0, **kwargs):
        self.threshold_pct = float(threshold_pct)
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
            )),
        ])

    def get_params(self, deep=True):
        return {"threshold_pct": self.threshold_pct}

    def set_params(self, **params):
        if "threshold_pct" in params:
            self.threshold_pct = float(params["threshold_pct"])
        return self

    def _features(self, X):
        X = np.asarray(X, dtype=float)
        mean_mag = X[:, 0]
        std_mag = X[:, 1]
        n_obs = X[:, 4]
        mag_range = X[:, 5]
        cv = std_mag / (np.abs(mean_mag) + 1e-9)
        rom = mag_range / (np.abs(mean_mag) + 1e-9)
        log_n = np.log1p(n_obs)
        return np.column_stack([X, cv, rom, log_n])

    def fit(self, X, y, **kwargs):
        self._pipeline.fit(self._features(X), y)
        return self

    def predict(self, X):
        return self._pipeline.predict(self._features(X))

    def predict_proba(self, X):
        return self._pipeline.predict_proba(self._features(X))
