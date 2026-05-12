"""TabPFN-3 regressor exposed as an IntegratedML Custom Model.

Companion to `tabpfn_classifier.py` — wraps `tabpfn.TabPFNRegressor` (Prior
Labs' TabPFN-3 open-weights tabular foundation model) behind the IRIS
IntegratedML Custom Models interface.

SQL usage:

    CREATE MODEL EnergyConsumption
    PREDICTING (kwh_day)
    FROM TabPFN.BuildingEnergy
    USING {
        "pathtoregressors": "/opt/irisapp/demos/tabpfn_foundation/iris_models/_staging/tabpfn_regressor",
        "iscmodelsdisabled": 1,
        "userparams": {"device": "cpu", "n_estimators": 8}
    };
    TRAIN MODEL EnergyConsumption;
    SELECT building_id, PREDICT(EnergyConsumption) AS kwh_day
    FROM TabPFN.BuildingEnergy;

Backend selection mirrors the classifier:

* `tabpfn` package present  →  `TabPFNRegressor` (TabPFN-3 default checkpoint)
* `tabpfn` package missing  →  `sklearn.ensemble.GradientBoostingRegressor`

The active backend is exposed on `IRISModel.backend` ("tabpfn" or "sklearn")
so downstream tooling can spot the fallback.

Self-contained for irispython: only standard library + sklearn + numpy +
pandas, with an optional `tabpfn` import.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd


def _try_import_tabpfn():
    try:
        from tabpfn import TabPFNRegressor  # type: ignore

        return TabPFNRegressor, "tabpfn"
    except Exception:
        pass
    try:
        from tabpfn_client import TabPFNRegressor  # type: ignore

        return TabPFNRegressor, "tabpfn_client"
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
    """GradientBoostingRegressor wrapper used when tabpfn isn't installed."""

    def __init__(self, **kwargs):
        from sklearn.ensemble import GradientBoostingRegressor

        self.model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class IRISModel:
    """TabPFN-3 regressor (with sklearn fallback)."""

    name = "tabpfn_regressor"

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
        self.backend: str = "uninitialized"
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
            impl = cls(device=self.device)
        return impl, source

    def fit(self, X, y, **kwargs):
        df = _to_dataframe(X)
        self._feature_columns = list(df.columns)
        X_enc = _encode_features(df)
        y_arr = np.asarray(y, dtype=float)

        self._impl, self.backend = self._make_impl()
        self._impl.fit(X_enc, y_arr)
        return self

    def _prepare_predict_input(self, X) -> np.ndarray:
        df = _to_dataframe(X)
        if self._feature_columns is not None:
            for col in self._feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self._feature_columns]
        return _encode_features(df)

    def predict(self, X):
        if self._impl is None:
            raise RuntimeError("Model must be fitted before predict()")
        return np.asarray(
            self._impl.predict(self._prepare_predict_input(X)), dtype=float
        )
