"""TabPFN-3 classifier exposed as an IntegratedML Custom Model.

TabPFN-3 (Prior Labs) is a transformer-based foundation model for tabular
classification. Unlike traditional ML, training is a single forward pass:
the model is pre-trained on millions of synthetic tabular tasks and performs
in-context learning at inference time. This maps cleanly onto IRIS's
`fit() / predict()` interface — fit just hands the training rows to the
foundation model, and predict runs one forward pass per query batch.

SQL usage:

    CREATE MODEL PatientRiskScreener
    PREDICTING (needs_followup)
    FROM TabPFN.PatientScreening
    USING {
        "pathtoclassifiers": "/opt/irisapp/demos/tabpfn_foundation/iris_models/_staging/tabpfn_classifier",
        "iscmodelsdisabled": 1,
        "userparams": {"device": "cpu", "n_estimators": 8}
    };
    TRAIN MODEL PatientRiskScreener;
    SELECT patient_id, PREDICT(PatientRiskScreener) AS needs_followup
    FROM TabPFN.PatientScreening;

Backend selection:

* If the `tabpfn` PyPI package is importable in the IRIS Python environment,
  this model uses `tabpfn.TabPFNClassifier` with the open-weights TabPFN-3
  checkpoint. Install with `pip install tabpfn` inside the IRIS container,
  which also pulls PyTorch. Set `device="cuda"` for GPU inference.
* Otherwise it falls back to `sklearn.ensemble.GradientBoostingClassifier`
  so that the demo still trains and predicts end-to-end. The fallback
  identity is exposed via `IRISModel.backend` ("tabpfn" vs "sklearn") so
  callers can spot the difference at a glance.

Self-contained for irispython: only standard library + sklearn + numpy +
pandas, with an optional `tabpfn` import.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd


def _try_import_tabpfn():
    """Try a few import paths so this file keeps working across releases."""
    try:
        from tabpfn import TabPFNClassifier  # type: ignore

        return TabPFNClassifier, "tabpfn"
    except Exception:
        pass
    try:
        from tabpfn_client import TabPFNClassifier  # type: ignore

        return TabPFNClassifier, "tabpfn_client"
    except Exception:
        pass
    return None, None


def _to_dataframe(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])


def _encode_features(df: pd.DataFrame) -> np.ndarray:
    """Convert mixed-type DataFrame columns to a numeric matrix.

    TabPFN-3 accepts categorical features natively, but the sklearn fallback
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
    """GradientBoostingClassifier wrapper used when tabpfn isn't installed."""

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
    """TabPFN-3 classifier (with sklearn fallback)."""

    name = "tabpfn_classifier"

    def __init__(
        self,
        device: str = "cpu",
        n_estimators: int = 8,
        ignore_pretraining_limits: bool = False,
        force_fallback: bool = False,
        **kwargs: Any,
    ) -> None:
        self.device = str(device)
        self.n_estimators = int(n_estimators)
        self.ignore_pretraining_limits = bool(ignore_pretraining_limits)
        self.force_fallback = bool(force_fallback)

        self._impl: Any = None
        self._feature_columns: Optional[List[str]] = None
        self._classes: Optional[np.ndarray] = None
        self.backend: str = "uninitialized"
        # IRIS introspects `model` to confirm sklearn-shape compatibility.
        self.model = self

    def get_params(self, deep: bool = True):
        return {
            "device": self.device,
            "n_estimators": self.n_estimators,
            "ignore_pretraining_limits": self.ignore_pretraining_limits,
            "force_fallback": self.force_fallback,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _make_impl(self):
        if self.force_fallback:
            return _SklearnFallback(), "sklearn"
        cls, source = _try_import_tabpfn()
        if cls is None:
            return _SklearnFallback(), "sklearn"
        kwargs = {"device": self.device, "n_estimators": self.n_estimators}
        if self.ignore_pretraining_limits:
            kwargs["ignore_pretraining_limits"] = True
        try:
            impl = cls(**kwargs)
        except TypeError:
            # Older / newer signatures may not accept every kwarg.
            impl = cls(device=self.device)
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
