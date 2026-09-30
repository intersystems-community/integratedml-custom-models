"""TabFM regressor exposed as an IntegratedML Custom Model.

Companion to `tabfm_classifier.py` — wraps `tabfm.TabFMRegressor` (Google
Research's zero-shot tabular foundation model) behind the IRIS IntegratedML
Custom Models interface. `fit()` supplies in-context rows; `predict()` is a
single forward pass with no per-dataset training.

SQL usage:

    CREATE MODEL EnergyConsumption
    PREDICTING (kwh_day)
    FROM TabFM.BuildingEnergy
    USING {
        "pathtoregressors": "/opt/irisapp/demos/tabfm_foundation/iris_models/_staging/tabfm_regressor",
        "iscmodelsdisabled": 1,
        "userparams": {"n_estimators": 4, "max_num_rows": 1000}
    };
    TRAIN MODEL EnergyConsumption;
    SELECT building_id, PREDICT(EnergyConsumption) AS kwh_day
    FROM TabFM.BuildingEnergy;

Backend selection mirrors the classifier:

* `tabfm` package present  →  `TabFMRegressor` (pretrained 1.0.0 regression
  weights via `load(model_type="regression")`; non-commercial license)
* `tabfm` package missing  →  `sklearn.ensemble.GradientBoostingRegressor`

The active backend is exposed on `IRISModel.backend` ("tabfm-pytorch",
"tabfm-jax" or "sklearn").
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


def _try_import_tabfm():
    """Return (TabFMRegressor, loader, source) or (None, None, None).

    PyTorch build preferred, JAX build as a fallback.
    """
    try:
        import tabfm  # type: ignore
        from tabfm import TabFMRegressor  # type: ignore
    except Exception:
        return None, None, None
    for variant, source in (
        ("tabfm_v1_0_0_pytorch", "tabfm-pytorch"),
        ("tabfm_v1_0_0_jax", "tabfm-jax"),
    ):
        mod = getattr(tabfm, variant, None)
        if mod is not None:
            return TabFMRegressor, (lambda m=mod: m.load(model_type="regression")), source
    return None, None, None


def _to_dataframe(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    if hasattr(X, "toarray"):
        # IntegratedML's AutoML data prep passes scipy sparse matrices.
        X = X.toarray()
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
    """GradientBoostingRegressor wrapper used when tabfm isn't installed."""

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


class IRISModel(RegressorMixin, BaseEstimator):
    """TabFM regressor (with sklearn fallback)."""

    name = "tabfm_regressor"

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
        self.backend: str = "uninitialized"
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
