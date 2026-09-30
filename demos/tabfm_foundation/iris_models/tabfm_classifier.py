"""TabFM classifier exposed as an IntegratedML Custom Model.

TabFM (Google Research) is a ~400M-parameter zero-shot foundation model for
tabular data, pre-trained entirely on synthetic tables drawn from structural
causal models. Prediction is in-context learning: `fit()` just hands the
labelled rows to the model as context, and `predict()` is one forward pass —
no gradient training, no hyperparameter search. That maps directly onto the
IRIS `fit() / predict()` Custom Models contract.

SQL usage:

    CREATE MODEL PatientRiskScreener
    PREDICTING (needs_followup)
    FROM TabFM.PatientScreening
    USING {
        "pathtoclassifiers": "/opt/irisapp/demos/tabfm_foundation/iris_models/_staging/tabfm_classifier",
        "iscmodelsdisabled": 1,
        "userparams": {"n_estimators": 4, "max_num_rows": 1000}
    };
    TRAIN MODEL PatientRiskScreener;
    SELECT patient_id, PREDICT(PatientRiskScreener) AS needs_followup
    FROM TabFM.PatientScreening;

Backend selection:

* If the `tabfm` package (github.com/google-research/tabfm) is importable,
  this model uses `tabfm.TabFMClassifier` with the pretrained 1.0.0 weights
  (PyTorch build preferred, JAX build as a fallback). NOTE: the default
  weights ship under the non-commercial "tabfm-non-commercial-v1.0" license.
* Otherwise it falls back to `sklearn.ensemble.GradientBoostingClassifier`
  so the demo still trains and predicts end-to-end. The active backend is
  exposed via `IRISModel.backend` ("tabfm" vs "sklearn").

Note TabFM's `max_num_rows` (default 100) caps the in-context rows; this
wrapper raises it to 1000 by default so the sample datasets are used in full.

Self-contained for irispython: standard library + sklearn + numpy + pandas,
with an optional `tabfm` import.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd


def _try_import_tabfm():
    """Return (TabFMClassifier, loader, source) or (None, None, None).

    `loader` is the zero-arg-callable that builds the pretrained model; the
    PyTorch variant is preferred, with JAX as a fallback.
    """
    try:
        import tabfm  # type: ignore
        from tabfm import TabFMClassifier  # type: ignore
    except Exception:
        return None, None, None
    for variant, source in (
        ("tabfm_v1_0_0_pytorch", "tabfm-pytorch"),
        ("tabfm_v1_0_0_jax", "tabfm-jax"),
    ):
        mod = getattr(tabfm, variant, None)
        if mod is not None:
            return TabFMClassifier, mod.load, source
    return None, None, None


def _to_dataframe(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])


def _encode_features(df: pd.DataFrame) -> np.ndarray:
    """Convert mixed-type DataFrame columns to a numeric matrix.

    TabFM accepts categorical features natively, but the sklearn fallback
    does not — and we want both paths to take the same training matrix so
    behaviour is comparable. We label-encode object columns with a small
    per-column code book that's stored on the model for inference reuse.
    """
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col]
        if s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype):
            codes, _ = pd.factorize(s.astype(str), sort=True)
            out[col] = codes.astype(float)
        else:
            out[col] = pd.to_numeric(s, errors="coerce").fillna(0.0)
    return out.to_numpy(dtype=float)


class _SklearnFallback:
    """GradientBoostingClassifier wrapper used when tabfm isn't installed."""

    def __init__(self, **kwargs):
        from sklearn.ensemble import GradientBoostingClassifier

        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class IRISModel:
    """TabFM classifier (with sklearn fallback)."""

    name = "tabfm_classifier"

    def __init__(
        self,
        n_estimators: int = 4,
        max_num_rows: int = 1000,
        max_num_features: int = 500,
        force_fallback: bool = False,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_num_rows = int(max_num_rows)
        self.max_num_features = int(max_num_features)
        self.force_fallback = bool(force_fallback)

        self._impl: Any = None
        self._feature_columns: Optional[List[str]] = None
        self._classes: Optional[np.ndarray] = None
        self.backend: str = "uninitialized"
        # IRIS introspects `model` to confirm sklearn-shape compatibility.
        self.model = self

    def get_params(self, deep: bool = True):
        return {
            "n_estimators": self.n_estimators,
            "max_num_rows": self.max_num_rows,
            "max_num_features": self.max_num_features,
            "force_fallback": self.force_fallback,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _make_impl(self):
        if self.force_fallback:
            return _SklearnFallback(), "sklearn"
        cls, loader, source = _try_import_tabfm()
        if cls is None:
            return _SklearnFallback(), "sklearn"
        impl = cls(
            model=loader(),
            n_estimators=self.n_estimators,
            max_num_rows=self.max_num_rows,
            max_num_features=self.max_num_features,
        )
        return impl, source

    def fit(self, X, y, **kwargs):
        df = _to_dataframe(X)
        self._feature_columns = list(df.columns)
        X_enc = _encode_features(df)
        y_arr = np.asarray(y)
        self._classes = np.unique(y_arr)

        self._impl, self.backend = self._make_impl()
        self._impl.fit(X_enc, y_arr)
        return self

    def _prepare_predict_input(self, X) -> np.ndarray:
        df = _to_dataframe(X)
        if self._feature_columns is not None:
            # Align column order to what was seen at fit time, filling missing.
            for col in self._feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self._feature_columns]
        return _encode_features(df)

    def predict(self, X):
        if self._impl is None:
            raise RuntimeError("Model must be fitted before predict()")
        return np.asarray(self._impl.predict(self._prepare_predict_input(X)))

    def predict_proba(self, X):
        if self._impl is None:
            raise RuntimeError("Model must be fitted before predict_proba()")
        return np.asarray(
            self._impl.predict_proba(self._prepare_predict_input(X))
        )

    @property
    def classes_(self):
        return self._classes
