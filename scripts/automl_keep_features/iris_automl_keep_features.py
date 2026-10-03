"""Turn off IntegratedML AutoML's automatic feature elimination.

Before AutoML hands training data to any model, including Custom Models, it
always drops columns with `SelectFpr(alpha=0.2)` (the `f_classif` ANOVA test)
in `iris_automl.automl_data_prep._calc_feature_elimination_fit`. `USING`
offers no switch to turn this off, and for regression targets `f_classif` is
a poor fit: in the TabFM and Kumo Tabular demos it dropped the strongest
numeric drivers of the target.

This module replaces that one function with a pass-through that keeps every
feature, leaving the rest of AutoML's data prep (type handling, one-hot
encoding, NULL handling, scaling) unchanged. It contains no AutoML code: it
patches the installed package at import time.

Activation: install this file into IRIS's Python path and import it from
`sitecustomize` (see install.py next to it). The import hook waits until
AutoML's data-prep module is first imported and then patches it, so nothing
is imported eagerly in other IRIS Python processes.

The patch applies to every AutoML model in the instance. Remove it with
`install.py --uninstall`. If a future AutoML release renames the function,
the hook logs a warning to stderr and leaves AutoML unchanged.
"""

from __future__ import annotations

import importlib.abc
import sys

TARGET_MODULE = "iris_automl.automl_data_prep"
TARGET_FUNCTION = "_calc_feature_elimination_fit"
PATCHED_FLAG = "_keep_features_patched"


def _keep_all_features_fit(df, y, log):
    """Drop-in replacement: same signature, keeps every column."""
    n = df.shape[1]
    log("\nfeature elimination disabled (iris_automl_keep_features)")
    log("  keeping all", n, "features")

    def transform_fn(df):
        if hasattr(df, "columns"):  # DataFrame
            return df
        # The original returns a CSC matrix for sparse input; match it.
        return df.tocsc() if hasattr(df, "tocsc") else df

    return transform_fn


def patch(module) -> bool:
    """Patch an imported automl_data_prep module. Returns True if patched."""
    if getattr(module, PATCHED_FLAG, False):
        return True
    if not callable(getattr(module, TARGET_FUNCTION, None)):
        sys.stderr.write(
            f"iris_automl_keep_features: {TARGET_MODULE}.{TARGET_FUNCTION} "
            "not found; AutoML left unchanged\n"
        )
        return False
    setattr(module, f"_original{TARGET_FUNCTION}", getattr(module, TARGET_FUNCTION))
    setattr(module, TARGET_FUNCTION, _keep_all_features_fit)
    setattr(module, PATCHED_FLAG, True)
    return True


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def create_module(self, spec):
        return self.wrapped.create_module(spec)

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    """Wraps the loader of TARGET_MODULE so it is patched right after import."""

    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET_MODULE:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is not None:
                    spec.loader = _PatchingLoader(spec.loader)
                return spec
        return None


def install_hook() -> None:
    """Patch now if AutoML is already imported, else when it is imported."""
    if TARGET_MODULE in sys.modules:
        patch(sys.modules[TARGET_MODULE])
        return
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install_hook()
