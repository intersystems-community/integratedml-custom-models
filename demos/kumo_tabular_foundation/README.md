# NVIDIA Kumo Tabular Foundation Model for IRIS IntegratedML

> NVIDIA's [Kumo Tabular](https://huggingface.co/blog/nvidia/kumo-tabular)
> tabular foundation model ([`nvidia/Kumo-Tabular`](https://huggingface.co/nvidia/Kumo-Tabular))
> exposed as IRIS IntegratedML Custom Models.

## What this demo is

Kumo Tabular is a pretrained foundation model for tabular classification and
regression, released in three sizes (28M–215M parameters). It works by
**in-context learning**: given a table of labelled rows, it predicts the
labels of new rows in a single forward pass, with no training, tuning or
feature engineering. NVIDIA reports it ranking first on TabArena,
BeyondArena, TALENT and ScoringBench, and running 17× faster than LimiX-2.

This demo wraps it in two self-contained `IRISModel` classes so you can call
it from IRIS SQL:

```sql
CREATE MODEL KumoPatientRiskScreener PREDICTING (needs_followup)
FROM (SELECT age, sex, bmi, systolic_bp, diastolic_bp, fasting_glucose, hdl, ldl,
             smoker, family_history, exercise_days, needs_followup
      FROM KumoTab.PatientScreening WHERE split = 'train')
USING {
    "pathtoclassifiers": "/opt/irisapp/demos/kumo_tabular_foundation/iris_models/_staging/kumo_classifier",
    "iscmodelsdisabled": 1,
    "userparams": {"num_estimators": 8, "max_context_rows": 10000}
};
TRAIN MODEL KumoPatientRiskScreener;

SELECT patient_id, PREDICT(KumoPatientRiskScreener) AS needs_followup
FROM KumoTab.PatientScreening WHERE split = 'test';
```

`TRAIN MODEL` just stores the labelled rows. Each `PREDICT()` call sends them
to Kumo Tabular as context, together with the rows being scored.

## What's in the box

| IRISModel file                    | Wraps                                              |
|-----------------------------------|----------------------------------------------------|
| `iris_models/kumo_classifier.py`  | `sdm.models.KumoTabular(task="classification")`    |
| `iris_models/kumo_regressor.py`   | `sdm.models.KumoTabular(task="regression")`        |

Both fall back to scikit-learn gradient boosting when NVIDIA's
`structured-data-models` library (import name `sdm`) isn't importable, so the
demo runs offline. The active backend is exposed on `IRISModel.backend`
(`"kumo-tabular"` or `"sklearn"`).

The datasets are the same synthetic sets as the TabPFN and TabFM demos
(`patient_screening.csv` for classification, `building_energy.csv` for
regression), so all three foundation models can be compared directly. The
models are named with a `Kumo` prefix and live in the `KumoTab` schema, so
they don't replace the other demos' models in the same namespace.

`userparams`:

| Key                | Default  | Meaning                                                        |
|--------------------|----------|----------------------------------------------------------------|
| `num_estimators`   | 8        | ensemble size passed to Kumo Tabular at inference              |
| `max_context_rows` | 10000    | labelled rows kept as context; larger training sets are sampled |
| `device`           | `"auto"` | `"cuda"` if available, else `"cpu"`                             |
| `model_kwargs`     | `{}`     | extra kwargs for `sdm.models.KumoTabular(...)`, e.g. model size |
| `force_fallback`   | false    | use the sklearn fallback even when `sdm` is installed          |

## Quick start

```bash
make install
pytest demos/kumo_tabular_foundation/tests/ -v
python run_kumo_tabular_demo.py        # or: make demo-kumo
```

## Using the real Kumo Tabular model

Install [`structured-data-models`](https://github.com/NVIDIA/structured-data-models)
into the Python that runs the model (for IRIS, `/usr/irissys/mgr/python`). It
needs PyTorch, and the weights are downloaded from Hugging Face
(`nvidia/Kumo-Tabular`). A CUDA GPU is recommended; the model card's example
also runs on CPU. Under IRIS, also copy the wrapper modules into
`mgr/python` (see below).

Things to know:

* **Install source**: as of this writing, `structured-data-models` on PyPI
  is a `0.0.0a0` placeholder with no code, although the model card says
  `pip install structured-data-models`. Install from the GitHub repository
  until a real release is published.
* **License**: the weights are under OpenMDW 1.1.
* **Limits**: numerical and categorical columns only. Classification supports
  up to 10 classes natively; the wrapper raises a ValueError above that unless
  `force_fallback` is set. Training data covered tables of up to 100 columns
  and 60,000 rows, so accuracy may drop well outside that range.
* **Model storage**: the wrapper never pickles the network into the IRIS
  model. Only the context rows are stored, and the network is reloaded on the
  first `PREDICT()` in each process.
* **Verification**: the wrapper follows the API on the Hugging Face model
  card. This environment couldn't install the library (PyPI has only the
  placeholder, and GitHub was blocked), so the real backend has **not been
  run**. The calls the wrapper makes are covered by tests against a stub
  `sdm`, and the `requires_sdm` tests run only where the library is
  installed. Two assumptions to check against the real library:
  * probability column `j` corresponds to class code `j` (labels are
    encoded to `0..k-1` in sorted order);
  * query rows in the combined table can carry a missing (NaN) target.

## IRIS run status

Verified on IRIS 2026.1 (`intersystemsdc/iris-community`) with
`intersystems-iris-automl` 1.0.3, using the sklearn fallback: `CREATE MODEL`,
`TRAIN MODEL`, `PREDICT()` and every query in `sql/03_evaluation.sql` run.

On the 75-row test split:

| Model      | In IRIS, AutoML default | In IRIS, keep-all-features patch | Local, all features        |
|------------|-------------------------|----------------------------------|----------------------------|
| Classifier | accuracy 0.787          | accuracy 0.800                   | accuracy 0.800, AUC 0.827  |
| Regressor  | MAE 128.9 kWh/day       | MAE 28.7 kWh/day                 | MAE 29.0 kWh/day, R² 0.941 |

With AutoML's defaults the gap comes from the provider, not the wrapper.
Before calling the custom model, AutoML always drops features with
`SelectFpr(alpha=0.2)` and its default `f_classif` test, and `USING` offers
no switch to turn this off. The classifier keeps 8 of 23 features; the
regressor keeps 5 of 19, mostly `hvac_type`, because `f_classif` suits a
continuous target poorly. The
[keep-all-features patch](../../scripts/automl_keep_features/README.md)
turns that step off for the whole instance, which brings the IRIS results in
line with the local run.

Running the wrapper under AutoML also showed what an `IRISModel` needs there,
all built into these wrappers and covered by tests:

* subclass `ClassifierMixin`/`RegressorMixin` + `BaseEstimator`, because
  AutoML cross-validates with scikit-learn;
* store constructor arguments unchanged, because `sklearn.base.clone()`
  checks identity and AutoML passes `random_state=None`;
* accept scipy sparse matrices, which AutoML's data prep produces.

## Deploying to IRIS

Container setup, the AutoML install, the healthcheck fix and how to run the
SQL are in [docker/IRIS_COMMUNITY_SETUP.md](../../docker/IRIS_COMMUNITY_SETUP.md).
With that container running, the steps for this demo are:

```bash
python demos/kumo_tabular_foundation/scripts/deploy_models.py
docker exec iris cp \
    /opt/irisapp/demos/kumo_tabular_foundation/iris_models/kumo_classifier.py \
    /opt/irisapp/demos/kumo_tabular_foundation/iris_models/kumo_regressor.py \
    /usr/irissys/mgr/python/

# create tables + load the CSVs (from the host over DB-API)
python demos/kumo_tabular_foundation/scripts/load_data.py

# CREATE / TRAIN MODEL: run inside the container, because TRAIN MODEL over
# DB-API currently ends the server process (see the setup guide)
docker exec -e IRISNAMESPACE=USER iris /usr/irissys/bin/irispython \
    /opt/irisapp/scripts/run_sql.py \
    /opt/irisapp/demos/kumo_tabular_foundation/sql/02_create_models.sql

# PREDICT() / evaluation queries (from the host over DB-API)
python scripts/run_sql.py demos/kumo_tabular_foundation/sql/03_evaluation.sql
```

`load_data.py` assigns the `split` column with the same seeded 75/25 shuffle
as `run_kumo_tabular_demo.py`, so SQL results are directly comparable with
the local demo.

## Kumo Tabular vs TabPFN vs TabFM

All three demos expose the same `IRISModel` interface on the same data, so an
A/B comparison is a matter of pointing `pathtoclassifiers` /
`pathtoregressors` at a different staging directory.
