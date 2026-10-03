"""NVIDIA Kumo Tabular classifier exposed as an IntegratedML Custom Model.

Kumo Tabular (huggingface.co/nvidia/Kumo-Tabular, 28M-215M parameters) is a
pretrained tabular foundation model. Prediction is in-context learning: the
labelled rows are the context and the label of each new row comes out of one
forward pass, with no training or tuning. `fit()` therefore just keeps the
labelled rows, and `predict()` runs the model with them as context.

SQL usage:

    CREATE MODEL KumoPatientRiskScreener
    PREDICTING (needs_followup)
    FROM KumoTab.PatientScreening
    USING {
        "pathtoclassifiers": "/opt/irisapp/demos/kumo_tabular_foundation/iris_models/_staging/kumo_classifier",
        "iscmodelsdisabled": 1,
        "userparams": {"num_estimators": 8, "max_context_rows": 10000}
    };
    TRAIN MODEL KumoPatientRiskScreener;
    SELECT patient_id, PREDICT(KumoPatientRiskScreener) AS needs_followup
    FROM KumoTab.PatientScreening;

Backend selection:

* If NVIDIA's `structured-data-models` library (import name `sdm`,
  github.com/NVIDIA/structured-data-models) is importable, this model calls
  `sdm.models.KumoTabular(task="classification")`, following the usage on the
  Hugging Face model card. A GPU is recommended; it also runs on CPU. The
  weights are released under OpenMDW 1.1.
* Otherwise it falls back to `sklearn.ensemble.GradientBoostingClassifier`
  so the demo still trains and predicts end to end. The active backend is
  exposed via `IRISModel.backend` ("kumo-tabular" vs "sklearn").

Kumo Tabular handles numerical and categorical columns and up to 10 classes
natively; more classes raise a ValueError on the real backend.

The loaded network is never pickled with the IRIS model (see
`__getstate__`); it is reloaded on first use after unpickling, so the stored
model stays small.

Self-contained for irispython: standard library + sklearn + numpy + pandas,
with an optional `sdm` (and `torch`) import.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

MAX_KUMO_CLASSES = 10
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
    """GradientBoostingClassifier used when structured-data-models is absent."""

    def __init__(self):
        from sklearn.ensemble import GradientBoostingClassifier

        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        )
        self.codebook: Dict[str, List[str]] = {}

    def fit(self, X: pd.DataFrame, y_codes: np.ndarray):
        self.codebook = _fit_codebook(X)
        self.model.fit(_encode_features(X, self.codebook), y_codes)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(_encode_features(X, self.codebook))


class _KumoBackend:
    """In-context classifier on top of sdm.models.KumoTabular."""

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

    def fit(self, X: pd.DataFrame, y_codes: np.ndarray):
        self._X = X.reset_index(drop=True)
        self._y = y_codes
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
                task="classification",
                device=_resolve_device(self.device),
                **self.model_kwargs,
            )
        return self._model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
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
            stypes=self.sdm.infer_stypes(df, overrides={TARGET: "categorical"}),
            device=device,
        )
        with torch.amp.autocast(
            device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            probs = model(
                x_context=table[:n_ctx].drop_columns(TARGET),
                y_context=table[:n_ctx, TARGET],
                x_query=table[n_ctx:].drop_columns(TARGET),
                num_estimators=self.num_estimators,
            )
        return _to_numpy(probs)


class IRISModel(ClassifierMixin, BaseEstimator):
    """Kumo Tabular classifier (with sklearn fallback)."""

    name = "kumo_classifier"

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
        self._classes: Optional[np.ndarray] = None
        self.backend: str = "uninitialized"
        # IRIS introspects `model` to confirm sklearn-shape compatibility.
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
        if len(self._classes) > MAX_KUMO_CLASSES:
            raise ValueError(
                f"Kumo Tabular supports up to {MAX_KUMO_CLASSES} classes, "
                f"got {len(self._classes)}; set force_fallback to use sklearn"
            )
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
        y_arr = np.asarray(y)
        # Encode labels as 0..k-1 so probability column j is classes_[j].
        self._classes, y_codes = np.unique(y_arr, return_inverse=True)

        max_rows = int(self.max_context_rows)
        if len(df) > max_rows:
            seed = 0 if self.random_state is None else self.random_state
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(len(df), max_rows, replace=False))
            df, y_codes = df.iloc[keep], y_codes[keep]

        self._impl, self.backend = self._make_impl()
        self._impl.fit(df, y_codes)
        return self

    def _prepare_predict_input(self, X) -> pd.DataFrame:
        df = _to_dataframe(X)
        if self._feature_columns is not None:
            # Align column order to what was seen at fit time, filling missing.
            for col in self._feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self._feature_columns]
        return _clean_features(df)

    def predict_proba(self, X):
        if self._impl is None:
            raise RuntimeError("Model must be fitted before predict_proba()")
        return np.asarray(
            self._impl.predict_proba(self._prepare_predict_input(X)),
            dtype=float,
        )

    def predict(self, X):
        proba = self.predict_proba(X)
        return self._classes[np.argmax(proba, axis=1)]

    @property
    def classes_(self):
        return self._classes
