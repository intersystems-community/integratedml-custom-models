"""Tests for gaia_variability_iris_model — verifies the IRISModel contract."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "iris_models"))

from gaia_variability_iris_model import IRISModel


def _make_X_y(n=40, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    y = (rng.random(n) > 0.5).astype(int)
    return X, y


class TestIRISModelContract:

    def test_has_required_attributes(self):
        m = IRISModel(random_state=0)
        assert hasattr(m, "name"), "IRISModel must have .name"
        assert hasattr(m, "model"), "IRISModel must have .model (sklearn estimator)"
        assert isinstance(m.name, str)

    def test_model_is_sklearn_estimator(self):
        from sklearn.base import BaseEstimator
        m = IRISModel(random_state=0)
        assert isinstance(m.model, BaseEstimator)

    def test_model_fit_predict(self):
        m = IRISModel(random_state=0)
        X, y = _make_X_y()
        m.model.fit(X, y)
        preds = m.model.predict(X)
        assert preds.shape == (len(y),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_model_predict_proba(self):
        m = IRISModel(random_state=0)
        X, y = _make_X_y()
        m.model.fit(X, y)
        proba = m.model.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_kwargs_forwarded(self):
        m = IRISModel(random_state=7, n_estimators=50, max_depth=2)
        assert m.model.random_state == 7
        assert m.model.n_estimators == 50
        assert m.model.max_depth == 2

    def test_sparse_input_not_needed(self):
        """IntegratedML densifies before calling fit — plain arrays must work."""
        m = IRISModel(random_state=0)
        X, y = _make_X_y()
        m.model.fit(X, y)
        preds = m.model.predict(X)
        assert len(preds) == len(y)
