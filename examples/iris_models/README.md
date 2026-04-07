# IRISModel Examples

Working examples of the IRISModel interface that IRIS loads during `TRAIN MODEL`.

## The IRISModel Interface

Each `.py` file you place in `pathtoclassifiers` or `pathtoregressors` must define a class called **`IRISModel`** with:

| Attribute / Method | Required | Notes |
|--------------------|----------|-------|
| `name` | Yes | String identifier, shown in training logs |
| `model` | Recommended | sklearn-compatible object (IRIS inspects it) |
| `fit(X, y, **kwargs)` | Yes | Train on data |
| `predict(X)` | Yes | Return predictions |
| `predict_proba(X)` | Classifiers | Required for classification models |
| `get_params(deep)` | Yes | sklearn-style param getter |
| `set_params(**params)` | Yes | sklearn-style param setter |

IRIS loads every `.py` file in the directory, imports the `IRISModel` class, and creates an instance. If `iscmodelsdisabled=1`, built-in ISC models are skipped and only your custom files run.

## Files

| File | Problem type | Base model |
|------|-------------|-----------|
| `example_classifier.py` | Classification | LogisticRegression |
| `example_regressor.py` | Regression | LinearRegression |

## Usage

1. Copy one of these files to your classifier/regressor directory
2. Rename it (the filename becomes the module name in training logs)
3. Run `TRAIN MODEL` with the appropriate path:

```sql
-- Classification
CREATE MODEL MyModel PREDICTING (label) FROM MyTable
USING {
    "pathtoclassifiers": "/usr/irissys/mgr/python/my_classifiers",
    "iscmodelsdisabled": 1
}
TRAIN MODEL MyModel

-- Regression
CREATE MODEL MyModel PREDICTING (amount) FROM MyTable
USING {
    "pathtoregressors": "/usr/irissys/mgr/python/my_regressors",
    "iscmodelsdisabled": 1
}
TRAIN MODEL MyModel
```

## Embedded Python (irispython) constraint — critical

IRIS loads these files in its **embedded Python interpreter** (`irispython`), not in your local CPython environment. This means:

- **Only import packages installed into IRIS** — e.g. via `pip_iris install <pkg>` or `docker exec ... pip3 install`. Packages in your local venv are invisible to IRIS.
- **Do not import from this repo** — `from shared.models import ...` will fail because `shared/` is not on irispython's `sys.path`. IRISModel files must be fully self-contained.
- **sklearn, numpy, pandas** are available — they ship with `intersystems-iris-automl`.
- **tensorflow, prophet, lightgbm** must be explicitly installed into IRIS if you need them.

The examples in this directory import only `sklearn.linear_model`, which is always available.

## Difference from `shared.models`

`shared.models` (`ClassificationModel`, `EnsembleModel`, etc.) are base classes for local development and testing in this repo only. They are **not** part of the IRISModel interface and **cannot** be used inside files deployed to IRIS.

The distinction:
- `shared.models` → for unit-testing demo models in this repo's test suite (runs in your local Python)
- `IRISModel` class in a deployment file → what IRIS's embedded Python actually loads and trains

Do not inherit from `shared.models` inside a file you intend to deploy to IRIS.

## Parameter naming (critical)

JSON keys in the `USING {}` block use **concatenated names** (no underscores):

| Correct | Wrong (silently ignored) |
|---------|--------------------------|
| `"iscmodelsdisabled"` | `"isc_models_disabled"` |
| `"pathtoclassifiers"` | `"path_to_classifiers"` |
| `"pathtoregressors"` | `"path_to_regressors"` |

IRIS silently ignores keys it does not recognize, so passing the underscored names causes no error but has no effect.
