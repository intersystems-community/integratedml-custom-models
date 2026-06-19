"""Unit tests for GaiaVariabilityDetector — no IRIS required."""

import numpy as np
import pandas as pd
import pytest
from demos.gaia_variable_stars.models.variability_detector import GaiaVariabilityDetector


def _make_features(n=50, seed=0):
    rng = np.random.default_rng(seed)
    mean_mag  = rng.uniform(10, 20, n)
    std_mag   = rng.uniform(0.01, 2.0, n)
    min_mag   = mean_mag - 3 * std_mag
    max_mag   = mean_mag + 3 * std_mag
    n_obs     = rng.integers(20, 200, n).astype(float)
    mag_range = max_mag - min_mag
    return pd.DataFrame({
        "mean_mag": mean_mag, "std_mag": std_mag,
        "min_mag": min_mag,   "max_mag": max_mag,
        "n_obs": n_obs,       "mag_range": mag_range,
    })


def _make_labels(df, threshold_pct=10.0):
    pct = df["mag_range"] / df["mean_mag"].abs() * 100.0
    return (pct >= threshold_pct).astype(int).values


class TestGaiaVariabilityDetector:

    def test_fit_predict_shape(self):
        X = _make_features(100)
        y = _make_labels(X)
        model = GaiaVariabilityDetector(threshold_pct=10.0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_before_fit_raises(self):
        model = GaiaVariabilityDetector()
        X = _make_features(10)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            model.predict(X)

    def test_predict_proba_shape(self):
        X = _make_features(50)
        y = _make_labels(X)
        model = GaiaVariabilityDetector()
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (50, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_threshold_respected_in_training_labels(self):
        """High-variability sources should mostly be predicted as variable."""
        rng = np.random.default_rng(42)
        n = 200
        mean_mag = rng.uniform(12, 18, n)
        # Make half clearly variable (large range) and half clearly stable (tiny range)
        std_mag = np.concatenate([
            rng.uniform(1.5, 3.0, n // 2),    # variable: high std
            rng.uniform(0.001, 0.01, n // 2),  # stable: tiny std
        ])
        mag_range = 6 * std_mag
        min_mag = mean_mag - 3 * std_mag
        max_mag = mean_mag + 3 * std_mag
        n_obs = np.full(n, 80.0)
        X = pd.DataFrame({
            "mean_mag": mean_mag, "std_mag": std_mag,
            "min_mag": min_mag,   "max_mag": max_mag,
            "n_obs": n_obs,       "mag_range": mag_range,
        })
        y = (mag_range / mean_mag * 100 >= 10.0).astype(int)

        model = GaiaVariabilityDetector(threshold_pct=10.0)
        model.fit(X, y)
        preds = model.predict(X)
        # Variable half (first 100) should be mostly predicted 1
        assert preds[:100].mean() > 0.7
        # Stable half (last 100) should be mostly predicted 0
        assert preds[100:].mean() < 0.3

    def test_numpy_array_input(self):
        X = _make_features(30)
        y = _make_labels(X)
        model = GaiaVariabilityDetector()
        model.fit(X.values, y)
        preds = model.predict(X.values)
        assert preds.shape == (30,)

    def test_state_serialization_roundtrip(self):
        X = _make_features(50)
        y = _make_labels(X)
        model = GaiaVariabilityDetector()
        model.fit(X, y)
        state = model._get_model_state()
        model2 = GaiaVariabilityDetector()
        model2._set_model_state(state)
        np.testing.assert_array_equal(model.predict(X), model2.predict(X))
