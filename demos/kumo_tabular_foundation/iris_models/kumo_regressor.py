"""NVIDIA Kumo Tabular regressor exposed as an IntegratedML Custom Model.

Companion to `kumo_classifier.py`: wraps `sdm.models.KumoTabular` with
`task="regression"` (huggingface.co/nvidia/Kumo-Tabular) behind the IRIS
IntegratedML Custom Models interface. `fit()` keeps the labelled rows as
in-context examples; `predict()` is a single forward pass with no
per-dataset training.

SQL usage:

    CREATE MODEL KumoBuildingEnergyForecaster
    PREDICTING (kwh_day)
    FROM KumoTab.BuildingEnergy
    USING {
        "pathtoregressors": "/opt/irisapp/demos/kumo_tabular_foundation/iris_models/_staging/kumo_regressor",
        "iscmodelsdisabled": 1,
        "userparams": {"num_estimators": 8, "max_context_rows": 10000}
    };
    TRAIN MODEL KumoBuildingEnergyForecaster;
    SELECT building_id, PREDICT(KumoBuildingEnergyForecaster) AS kwh_day
    FROM KumoTab.BuildingEnergy;

Backend selection mirrors the classifier:

* `sdm` (structured-data-models) importable -> Kumo Tabular regressor
* otherwise -> `sklearn.ensemble.GradientBoostingRegressor`

The active backend is exposed on `IRISModel.backend` ("kumo-tabular" or
"sklearn"). As in the classifier, the network is not pickled with the IRIS
model and is reloaded on first use.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

TARGET = "__target__"


def _try_import_sdm():
    """Return the `sdm` module, or None if structured-data-models is absent."""
    try:
        import sdm  # type: ignore
    except Exception:
        return None
    if not hasattr(getattr(sdm, "models", None), "KumoTabular"):
        return None
    return sdm


def _resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _to_numpy(out) -> np.ndarray:
    if hasattr(out, "detach"):  # torch.Tensor
        out = out.detach().float().cpu().numpy()
    return np.asarray(out)


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


def _is_categorical(s: pd.Series) -> bool:
    return (
        s.dtype == object
        or pd.api.types.is_string_dtype(s.dtype)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )


def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Categorical columns as str, everything else as float (NaN allowed)."""
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col]
        if _is_categorical(s):
            out[col] = s.astype(str)
        else:
            out[col] = pd.to_numeric(s, errors="coerce").astype(float)
    return out


def _fit_codebook(df: pd.DataFrame) -> Dict[str, List[str]]:
    return {
        col: sorted(df[col].astype(str).unique())
        for col in df.columns
        if _is_categorical(df[col])
    }


def _encode_features(df: pd.DataFrame, codebook: Dict[str, List[str]]):
    """Numeric matrix for the sklearn fallback.

    Categoricals are label-encoded with the categories seen at fit time, so a
    value gets the same code in every prediction batch (unseen -> -1).
    """
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col]
        if col in codebook:
            cat = pd.Categorical(s.astype(str), categories=codebook[col])
            out[col] = cat.codes.astype(float)
        else:
            out[col] = pd.to_numeric(s, errors="coerce").fillna(0.0)
    return out.to_numpy(dtype=float)


class _SklearnFallback:
    """GradientBoostingRegressor used when structured-data-models is absent."""

    def __init__(self):
        from sklearn.ensemble import GradientBoostingRegressor

        self.model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        )
        self.codebook: Dict[str, List[str]] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.codebook = _fit_codebook(X)
        self.model.fit(_encode_features(X, self.codebook), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(_encode_features(X, self.codebook))


class _KumoBackend:
    """In-context regressor on top of sdm.models.KumoTabular."""

    def __init__(self, sdm, device: str, num_estimators: int,
                 model_kwargs: Dict[str, Any]):
        self.sdm = sdm
        self.device = device
        self.num_estimators = num_estimators
        self.model_kwargs = model_kwargs
        self._model = None
        self._X: Optional[pd.DataFrame] = None
        self._y: Optional[np.ndarray] = None

    def __getstate__(self):
        # Don't pickle the module or the network weights into the IRIS model.
        state = self.__dict__.copy()
        state["sdm"] = None
        state["_model"] = None
        return state

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self._X = X.reset_index(drop=True)
        self._y = y
        return self

    def _ensure_model(self):
        if self.sdm is None:
            self.sdm = _try_import_sdm()
            if self.sdm is None:
                raise RuntimeError(
                    "Model was fitted with structured-data-models (sdm), "
                    "which is not importable here"
                )
        if self._model is None:
            self._model = self.sdm.models.KumoTabular(
                task="regression",
                device=_resolve_device(self.device),
                **self.model_kwargs,
            )
        return self._model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        model = self._ensure_model()
        device = _resolve_device(self.device)
        n_ctx = len(self._X)
        # Context and query rows go into one table so they share the same
        # column types; query rows have a missing target.
        df = pd.concat([self._X, X.reset_index(drop=True)], ignore_index=True)
        target = np.full(len(df), np.nan)
        target[:n_ctx] = self._y
        df[TARGET] = target
        table = self.sdm.TableTensor.from_pandas(
            df=df,
            stypes=self.sdm.infer_stypes(df, overrides={TARGET: "numerical"}),
            device=device,
        )
        with torch.amp.autocast(
            device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            pred = model(
                x_context=table[:n_ctx].drop_columns(TARGET),
                y_context=table[:n_ctx, TARGET],
                x_query=table[n_ctx:].drop_columns(TARGET),
                num_estimators=self.num_estimators,
            )
        return _to_numpy(pred).reshape(-1)


class IRISModel(RegressorMixin, BaseEstimator):
    """Kumo Tabular regressor (with sklearn fallback)."""

    name = "kumo_regressor"

    def __init__(
        self,
        num_estimators: int = 8,
        max_context_rows: int = 10000,
        device: str = "auto",
        model_kwargs: Optional[Dict[str, Any]] = None,
        force_fallback: bool = False,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        # Parameters are stored as passed (sklearn convention): clone(), which
        # AutoML uses, checks identity, and AutoML passes random_state=None.
        self.num_estimators = num_estimators
        self.max_context_rows = max_context_rows
        self.device = device
        self.model_kwargs = model_kwargs
        self.force_fallback = force_fallback
        self.random_state = random_state

        self._impl: Any = None
        self._feature_columns: Optional[List[str]] = None
        self.backend: str = "uninitialized"
        self.model = self

    def get_params(self, deep: bool = True):
        return {
            "num_estimators": self.num_estimators,
            "max_context_rows": self.max_context_rows,
            "device": self.device,
            "model_kwargs": self.model_kwargs,
            "force_fallback": self.force_fallback,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _make_impl(self):
        if self.force_fallback:
            return _SklearnFallback(), "sklearn"
        sdm = _try_import_sdm()
        if sdm is None:
            return _SklearnFallback(), "sklearn"
        impl = _KumoBackend(
            sdm,
            str(self.device),
            int(self.num_estimators),
            dict(self.model_kwargs or {}),
        )
        return impl, "kumo-tabular"

    def fit(self, X, y, **kwargs):
        df = _clean_features(_to_dataframe(X))
        self._feature_columns = list(df.columns)
        y_arr = np.asarray(y, dtype=float)

        max_rows = int(self.max_context_rows)
        if len(df) > max_rows:
            seed = 0 if self.random_state is None else self.random_state
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(len(df), max_rows, replace=False))
            df, y_arr = df.iloc[keep], y_arr[keep]

        self._impl, self.backend = self._make_impl()
        self._impl.fit(df, y_arr)
        return self

    def _prepare_predict_input(self, X) -> pd.DataFrame:
        df = _to_dataframe(X)
        if self._feature_columns is not None:
            for col in self._feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self._feature_columns]
        return _clean_features(df)

    def predict(self, X):
        if self._impl is None:
            raise RuntimeError("Model must be fitted before predict()")
        return np.asarray(
            self._impl.predict(self._prepare_predict_input(X)), dtype=float
        )
