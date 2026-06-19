"""
Gaia Variable Star Detector — IntegratedML Custom Model

Detects brightness variability in Gaia DR3 time-series photometry.
Computes per-source statistics and classifies as variable if pct_change >= threshold.

Features passed from IRIS SQL:
  mean_mag, std_mag, min_mag, max_mag, n_obs, mag_range

Model predicts: is_variable (1 = variable, 0 = stable)

# Feedback on using Gaia/astroquery API:
#   astroquery.gaia works well for ADQL queries against the Gaia archive.
#   The epoch_photometry tables are large; cone search + source filter is essential.
#   phot_g_mean_mag is per-transit flux — variability is captured by std_over_mean.
#   Using IntegratedML for this is unconventional but powerful: the feature
#   engineering (std/range statistics) lives in SQL, the classifier in Python.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from shared.models.classification import ClassificationModel


class GaiaVariabilityDetector(ClassificationModel):
    """
    Classifies Gaia sources as variable stars based on photometric statistics.

    Input features (columns from GaiaObservationStats table):
        mean_mag  - mean G-band magnitude
        std_mag   - std dev of G-band magnitude across observations
        min_mag   - minimum (brightest) magnitude
        max_mag   - maximum (faintest) magnitude
        n_obs     - number of observations
        mag_range - max_mag - min_mag

    Target: is_variable (1 = variable by threshold, 0 = stable)
    """

    def __init__(self, threshold_pct: float = 10.0, n_estimators: int = 100, **kwargs):
        super().__init__(threshold_pct=threshold_pct, n_estimators=n_estimators, **kwargs)
        self.threshold_pct = threshold_pct
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            )),
        ])

    def _engineer_features(self, X, is_training=False):
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X = X.copy()
        else:
            cols = ["mean_mag", "std_mag", "min_mag", "max_mag", "n_obs", "mag_range"]
            X = pd.DataFrame(X, columns=cols[:X.shape[1]])

        # Derived variability features
        X["cv"] = X["std_mag"] / (X["mean_mag"].abs() + 1e-9)          # coefficient of variation
        X["range_over_mean"] = X["mag_range"] / (X["mean_mag"].abs() + 1e-9)
        X["log_n_obs"] = np.log1p(X["n_obs"])
        return X.values

    def fit(self, X, y, **kwargs):
        X_feat = self._engineer_features(X, is_training=True)
        self._pipeline.fit(X_feat, y)
        self._is_trained = True
        return self

    def predict(self, X):
        self._require_trained()
        return self._pipeline.predict(self._engineer_features(X))

    def predict_proba(self, X):
        self._require_trained()
        return self._pipeline.predict_proba(self._engineer_features(X))
