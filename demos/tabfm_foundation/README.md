# TabFM (Google Research) Zero-Shot Foundation Model for IRIS IntegratedML

> Google Research's [TabFM](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)
> tabular foundation model exposed as IRIS IntegratedML Custom Models.

## What this demo is

TabFM is a ~400M-parameter foundation model for tabular data, pre-trained
entirely on synthetic tables generated from structural causal models. It
treats supervised prediction as **in-context learning**: the labelled training
rows are the prompt, and predictions come out of a single forward pass — no
gradient training, no hyperparameter tuning, no feature engineering. Google
reports it ranking first among default tabular foundation models on
[TabArena](https://tabarena.ai/) (38 classification + 13 regression datasets).

This demo wraps TabFM in two self-contained `IRISModel` classes so you can
call it from IRIS SQL:

```sql
CREATE MODEL PatientRiskScreener PREDICTING (needs_followup)
FROM (SELECT age, sex, bmi, systolic_bp, diastolic_bp, fasting_glucose, hdl, ldl,
             smoker, family_history, exercise_days, needs_followup
      FROM TabFM.PatientScreening WHERE split = 'train')
USING {
    "pathtoclassifiers": "/opt/irisapp/demos/tabfm_foundation/iris_models/_staging/tabfm_classifier",
    "iscmodelsdisabled": 1,
    "userparams": {"n_estimators": 4, "max_num_rows": 1000}
};
TRAIN MODEL PatientRiskScreener;

SELECT patient_id, PREDICT(PatientRiskScreener) AS needs_followup
FROM TabFM.PatientScreening WHERE split = 'test';
```

`TRAIN MODEL` is effectively instant: "training" just hands the rows to TabFM
as context.

## What's in the box

| IRISModel file                     | Wraps                     | Default backend |
|------------------------------------|---------------------------|-----------------|
| `iris_models/tabfm_classifier.py`  | `tabfm.TabFMClassifier`   | TabFM 1.0.0     |
| `iris_models/tabfm_regressor.py`   | `tabfm.TabFMRegressor`    | TabFM 1.0.0     |

Both fall back to scikit-learn gradient boosting when the `tabfm` package
isn't importable, so the demo runs offline. The active backend is exposed on
`IRISModel.backend` (`"tabfm-pytorch"`, `"tabfm-jax"` or `"sklearn"`).

Datasets are the same two synthetic sets used by the TabPFN demo
(`patient_screening.csv` — classification, `building_energy.csv` —
regression), so the two foundation models are directly comparable.

## Quick start

```bash
make install
pytest demos/tabfm_foundation/tests/ -v
python run_tabfm_demo.py        # or: make demo-tabfm
```

## Using the real TabFM model

```bash
git clone https://github.com/google-research/tabfm.git
cd tabfm
pip install -e .[pytorch]      # or .[jax] / .[jax,cuda]; needs Python >= 3.11
```

Inside the IRIS container, install into the IRIS Python environment and make
sure it can reach Hugging Face to download the weights
(`google/tabfm-1.0.0-pytorch`).

Things to know:

* **License** — the code is Apache-2.0, but the default pretrained weights are
  under `tabfm-non-commercial-v1.0` (non-commercial / non-production use).
* **Context size** — TabFM defaults to `max_num_rows=100` in-context rows and
  `max_num_features=500`. This wrapper raises rows to 1000; larger tables need
  sampling or splitting. Both are settable via `userparams`.
* **IRIS run status** — verified end to end on IRIS 2026.1
  (`intersystemsdc/iris-community`) with `intersystems-iris-automl` 1.0.3:
  `CREATE MODEL`, `TRAIN MODEL`, `PREDICT()` and every query in
  `sql/03_evaluation.sql` run. That run used the sklearn fallback, since
  `tabfm` was not installed in IRIS.
* **AutoML feature selection** — before calling the custom model, the AutoML
  provider always drops features with `SelectFpr(alpha=0.2)` and its default
  `f_classif` test, and USING offers no switch to turn this off. On the test
  split this cost accuracy on the classifier (in IRIS 0.747, 23→8 features;
  local, all features, 0.800) and much more on the regressor, where
  `f_classif` is a poor fit for a continuous target (19→5 features, leaving
  mostly `hvac_type`: in IRIS MAE 128.9 kWh/day; local 30.9). The
  [keep-all-features patch](../../scripts/automl_keep_features/README.md)
  turns this step off; with it, IRIS gives accuracy 0.800 and MAE
  28.7 kWh/day, in line with the local all-feature run.
* **Verification** — this wrapper was written from the public repo README. The
  sandbox it was built in couldn't install TabFM, so the real-backend path is
  covered by a stubbed-package unit test, and the `requires_tabfm` tests run
  only where `tabfm` is installed. Check the `load()` / constructor calls
  against your installed version.

## Deploying to IRIS

Container setup, the AutoML install, the healthcheck fix and how to run the
SQL are in [docker/IRIS_COMMUNITY_SETUP.md](../../docker/IRIS_COMMUNITY_SETUP.md).
With that container running, the steps for this demo are:

```bash
python demos/tabfm_foundation/scripts/deploy_models.py
docker exec iris cp \
    /opt/irisapp/demos/tabfm_foundation/iris_models/tabfm_classifier.py \
    /opt/irisapp/demos/tabfm_foundation/iris_models/tabfm_regressor.py \
    /usr/irissys/mgr/python/

# create tables + load the CSVs (from the host over DB-API)
python demos/tabfm_foundation/scripts/load_data.py

# CREATE / TRAIN MODEL: run inside the container, because TRAIN MODEL over
# DB-API currently ends the server process (see the setup guide)
docker exec -e IRISNAMESPACE=USER iris /usr/irissys/bin/irispython \
    /opt/irisapp/scripts/run_sql.py \
    /opt/irisapp/demos/tabfm_foundation/sql/02_create_models.sql

# PREDICT() / evaluation queries (from the host over DB-API)
python scripts/run_sql.py demos/tabfm_foundation/sql/03_evaluation.sql
```

`load_data.py` assigns the `split` column with the same seeded 75/25 shuffle
as `run_tabfm_demo.py`, so SQL results are directly comparable with
the local demo.

## TabFM vs TabPFN

See [`../tabpfn_foundation`](../tabpfn_foundation/README.md) for Prior Labs'
TabPFN-3 version of the same demo. Both expose identical `IRISModel`
interfaces, so swapping `pathtoclassifiers` between them is a one-line change
for an A/B comparison.
