# AutoML keep-all-features patch

Before IntegratedML's AutoML provider (`intersystems-iris-automl`) hands
training data to any model, including Custom Models, it always runs a
feature-elimination step: `SelectFpr(alpha=0.2)` with the default `f_classif`
ANOVA test (`iris_automl.automl_data_prep._calc_feature_elimination_fit`).
`CREATE MODEL ... USING` offers no option to turn it off.

For foundation models such as TabFM and Kumo Tabular, which are built to use
every column, this costs accuracy. It is worst for regression, where
`f_classif` (a classification test) is a poor fit for a continuous target.

This patch replaces that one step with a pass-through that keeps every
feature. The rest of AutoML's data prep (type handling, one-hot encoding,
NULL handling, scaling) is unchanged.

## Not a copy of AutoML

`intersystems-iris-automl` is proprietary, so this is not a fork of its
source. `iris_automl_keep_features.py` contains no AutoML code. It patches
the installed package at import time, through an import hook that
`sitecustomize` loads when IRIS starts Python. The hook waits until AutoML's
data-prep module is first imported, so other IRIS Python work imports
nothing extra.

## Install

Run with IRIS's Python, inside the IRIS container (the repo is mounted at
`/opt/irisapp` in the demo setup):

```bash
docker exec iris /usr/irissys/bin/irispython \
    /opt/irisapp/scripts/automl_keep_features/install.py            # install
docker exec iris /usr/irissys/bin/irispython \
    /opt/irisapp/scripts/automl_keep_features/install.py --status   # check
docker exec iris /usr/irissys/bin/irispython \
    /opt/irisapp/scripts/automl_keep_features/install.py --uninstall
```

The installer copies the module into `/usr/irissys/mgr/python` and adds a
marked block to `sitecustomize.py` there, leaving any other content in that
file alone. Use `--target` for a different directory.

Things to know:

* It affects **every AutoML model** in the instance, built-in providers
  included, not only Custom Models.
* It applies to IRIS processes started after installing; a process that has
  already imported AutoML keeps the old behaviour.
* With the patch active, the `TRAIN MODEL` log shows
  `feature elimination disabled (iris_automl_keep_features)` and
  `keeping all N features` instead of `reducing features from N to M`.
* If a future AutoML release renames the function, the hook writes a warning
  to stderr and leaves AutoML unchanged.

Written against `intersystems-iris-automl` 1.0.3.

## Effect on the foundation-model demos

IRIS 2026.1, `intersystems-iris-automl` 1.0.3, 75-row test split, sklearn
fallback backend (the real foundation models weren't installed in IRIS).
TabFM and Kumo Tabular give identical results here because both fall back to
the same scikit-learn models:

| Model      | AutoML default           | With this patch       | Local, all features |
|------------|--------------------------|-----------------------|---------------------|
| Classifier | accuracy 0.747 / 0.787\* | accuracy 0.800        | accuracy 0.800      |
| Regressor  | MAE 128.9 kWh/day        | MAE 28.7 kWh/day      | MAE 29–31 kWh/day   |

\* The two default-AutoML runs differ although the model is the same. A
likely reason, not verified, is that AutoML shuffles the training rows with
`random_state=None`.

## Tests

```bash
pytest scripts/automl_keep_features/tests -v
```

The tests run the hook against a fake `iris_automl` package and check the
installer's handling of `sitecustomize.py`.
