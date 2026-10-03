"""Minimal stand-ins for `sdm` (structured-data-models) and `torch`.

They record how the IRISModel wrappers call the Kumo Tabular API from the
Hugging Face model card, without needing the library, a GPU or the weights.
"""

import contextlib
import sys
import types

import numpy as np


class StubTable:
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, col = key
            return self.df.iloc[rows][col].to_numpy()
        return StubTable(self.df.iloc[key])

    def drop_columns(self, col):
        return StubTable(self.df.drop(columns=col))


def install(monkeypatch, n_classes=None):
    """Install stub `sdm` and `torch` modules; return the call log."""
    calls = {"models": [], "forward": [], "tables": [], "stypes": []}

    class KumoTabular:
        def __init__(self, task, device, **kwargs):
            calls["models"].append({"task": task, "device": device, **kwargs})
            self.task = task

        def __call__(self, x_context, y_context, x_query, num_estimators):
            calls["forward"].append({
                "x_context": x_context.df,
                "y_context": y_context,
                "x_query": x_query.df,
                "num_estimators": num_estimators,
            })
            n = len(x_query)
            if self.task == "classification":
                k = n_classes or len(np.unique(y_context))
                probs = np.zeros((n, k))
                probs[:, -1] = 1.0  # always predict the last class
                return probs
            return np.full((n, 1), float(np.mean(y_context)))

    def from_pandas(df, stypes, device):
        calls["tables"].append(df.copy())
        return StubTable(df)

    def infer_stypes(df, overrides=None):
        calls["stypes"].append(dict(overrides or {}))
        return {"overrides": overrides}

    sdm = types.ModuleType("sdm")
    sdm.models = types.SimpleNamespace(KumoTabular=KumoTabular)
    sdm.TableTensor = types.SimpleNamespace(from_pandas=from_pandas)
    sdm.infer_stypes = infer_stypes

    torch = types.ModuleType("torch")
    torch.device = lambda name: types.SimpleNamespace(type=name)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.float16 = "float16"
    torch.amp = types.SimpleNamespace(
        autocast=lambda *a, **kw: contextlib.nullcontext()
    )

    monkeypatch.setitem(sys.modules, "sdm", sdm)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls
