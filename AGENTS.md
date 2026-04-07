# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-07
**Commit:** 5b823ab
**Branch:** main

## OVERVIEW
EAP demo portfolio for IntegratedML Custom Models — Python ML models executed directly within IRIS SQL (`CREATE MODEL ... USING`, `PREDICT()`). Stack: Python 3.8+, scikit-learn, IRIS 2025.2, Docker, pytest.

## CRITICAL: shared/models/ DOES NOT EXIST ON DISK
CLAUDE.md and README reference `shared/models/base.py`, `ClassificationModel`, `EnsembleModel` — these base classes are **imported by demos but the `shared/models/` directory is missing**. Models in `demos/*/models/` will fail to import unless this is created or the imports are fixed.

## STRUCTURE
```
integratedml-custom-models/
├── shared/              # DB connection, model manager, utils — NO shared/models/ yet
├── demos/
│   ├── credit_risk/     # Loan default prediction, custom feature engineering
│   ├── fraud_detection/ # Ensemble (rule+anomaly+neural), sub-100ms latency target
│   ├── sales_forecasting/ # Prophet + LightGBM hybrid time-series
│   └── dna_similarity/  # K-mer vectorization, configurable algorithm selection
├── docs/                # EAP guides (EAP_GUIDE.md, INSTALLATION.md, KNOWN_ISSUES)
├── specs/               # Feature spec markdown folders (001/002/003)
├── docker/              # Dockerfile.iris + iris-init/*.sql bootstrap scripts
├── tests/               # Root E2E: test_all_demos_e2e.py, test_real_iris_integration.py
├── scripts/             # setup_integratedml_complete.py, generate_ml_datasets.py
└── docker-compose.yml   # container: integratedml_iris
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| IRIS DB connection | `shared/database/connection.py` | `IRISConnection`, fallback HTTP via Atelier REST |
| Model lifecycle (CREATE/TRAIN/PREDICT) | `shared/database/model_manager.py` | Wraps IntegratedML SQL |
| Fraud ensemble | `demos/fraud_detection/models/ensemble_fraud_detector.py` | 820 lines, extends `EnsembleModel` |
| DNA classification | `demos/dna_similarity/models/dna_classifier.py` | Extends `ClassificationModel` |
| Credit risk model | `demos/credit_risk/models/` | **directory may not exist** — check CLAUDE.md discrepancy |
| IntegratedML SQL syntax | `demos/*/sql/*.sql` | `CREATE MODEL ... USING {...}`, `PREDICT()` |
| IRIS DB init scripts | `docker/iris-init/*.sql` | Numbered 01–99, run on container startup |
| E2E tests | `tests/test_all_demos_e2e.py` | Hard-codes `IRIS_PORT=1974` |
| Test fixtures | `demos/fraud_detection/tests/conftest.py` | Only conftest in project |
| EAP participant docs | `docs/EAP_*.md` | GA target: IRIS 2026.1 |

## CONVENTIONS

### Model authoring
- Models extend base classes from `shared.models.*` (classification/regression/ensemble)
- Config via `**kwargs` → `self.parameters` dict (passed as JSON from SQL `USING` block)
- `_engineer_features()` for domain preprocessing; `_get_model_state()` / `_set_model_state()` for serialization
- Scikit-learn API: `fit(X, y)` + `predict(X)` required; `predict_proba()` optional

### SQL patterns
```sql
CREATE MODEL ModelName PREDICTING (target) FROM Table
USING {"model_name": "ClassName", "path_to_classifiers": "/opt/iris/mgr/python/..."}
TRAIN MODEL ModelName
SELECT PREDICT(ModelName) FROM Table
VALIDATE MODEL ModelName
```

### Testing
- Markers: `unit`, `integration`, `performance`, `slow`, `requires_data`
- `test_unit_*` files auto-get `@unit`; `test_integration.py` auto-gets `@integration`
- Integration tests need live IRIS — `IRIS_PORT=1974` (E2E) or env var
- Only `demos/fraud_detection/tests/` has a `conftest.py`

### Formatting / Lint
- Black line-length=88; `flake8 --max-line-length=88 --extend-ignore=E203,W503`
- `mypy shared/ --ignore-missing-imports` (not demos/)

## ANTI-PATTERNS
- **Don't hardcode IRIS ports** — use `IRIS_PORT` env var (note: E2E test hard-codes 1974, a known wart)
- **Don't import `shared.models.*` without verifying the directory exists** — currently missing
- **Don't use `requests` for SQL** when native `iris` package is available — HTTP fallback only
- `shared/database/connection.py` does naive string-replace for SQL params in HTTP path — don't rely on it for untrusted input

## COMMANDS
```bash
make setup          # install deps + start IRIS
make test           # pytest demos/*/tests/
make demo-fraud     # run fraud detection demo
make demo-credit    # run credit risk demo
make demo-sales     # run sales forecasting demo
make demo-dna       # run DNA similarity demo
make lint           # flake8 + mypy
make format         # black .
make logs           # docker logs integratedml_iris
make clean          # remove containers + volumes
pytest -m "not integration" demos/fraud_detection/tests/  # unit only
pytest -m integration  # integration tests (needs IRIS running)
```

## NOTES
- `uv` preferred over pip (`make install-uv`)
- `shared/database/connection.py:152` uses `IRIS_WEB_PORT=52776` for HTTP fallback (Atelier REST endpoint)
- `INFORMATION_SCHEMA.ML_MODELS` and `ML_PROVIDERS` are IntegratedML-specific system tables
- `sentence_transformers` + `prophet` + `tensorflow` + `lightgbm` are heavy deps — cold install is slow
- EAP status: repo is the primary doc source until GA (2026.1)
