# SHARED MODULE

## OVERVIEW
Cross-demo infrastructure: IRIS connection management, IntegratedML model lifecycle, shared utilities. No `shared/models/` base classes exist yet — they are referenced but unimplemented.

## STRUCTURE
```
shared/
├── database/
│   ├── connection.py      # IRISConnection class, global get_connection()
│   ├── model_manager.py   # ModelManager — CREATE/TRAIN/PREDICT/VALIDATE SQL wrappers
│   ├── data_loader.py     # Bulk data loading utilities
│   ├── setup_database.py  # Database initialization
│   └── tools/check_tables.py  # Diagnostic tool
├── utils/
│   ├── logging.py         # setup_logger() — used by demos
│   └── __init__.py
├── testing/__init__.py    # Shared test helpers (minimal)
└── data/__init__.py       # Data utilities placeholder
```

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Connect to IRIS | `database/connection.py:29` | `IRISConnection(**env_vars)` |
| Execute IntegratedML SQL | `database/connection.py:99` | `execute_sql(sql)` — native iris preferred, HTTP fallback |
| CREATE/TRAIN/PREDICT wrappers | `database/model_manager.py` | `ModelManager.create_model()`, `.train_model()` |
| HTTP fallback endpoint | `database/connection.py:152` | `IRIS_WEB_PORT=52776`, Atelier REST `/api/atelier/v1/{ns}/action/query` |
| Logger setup | `utils/logging.py` | `setup_logger(__name__)` — used in all demo models |

## CRITICAL GAP
`shared/models/` does NOT exist. `demos/dna_similarity/models/dna_classifier.py` and `demos/fraud_detection/models/ensemble_fraud_detector.py` import from `shared.models.classification` and `shared.models.ensemble` — these imports will fail at runtime. Any new demo model must either create these base classes or restructure imports.

## CONVENTIONS
- `IRISConnection` reads all config from env vars: `IRIS_HOST`, `IRIS_PORT`, `IRIS_USERNAME`, `IRIS_PASSWORD`, `IRIS_NAMESPACE`
- `get_connection()` returns a module-level singleton — not thread-safe for concurrent tests
- `ModelManager` stores serialized models at `IML_MODEL_PATH` env var (default `/app/models`)
- HTTP fallback uses naive string-replace for params — unsafe for untrusted input

## ANTI-PATTERNS
- Don't call `get_connection()` without env vars set — silently defaults to `localhost:1972/demo:demo/USER`
- Don't import `shared.models.*` — directory missing
- Don't use `requests`-based SQL path when native `iris` package available
