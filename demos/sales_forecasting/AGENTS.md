# SALES FORECASTING DEMO

## OVERVIEW
Hybrid Prophet + LightGBM time-series forecasting demo — most analytically complex, with business intelligence layer and forecast evaluation utilities.

## STRUCTURE
```
sales_forecasting/
├── analytics/
│   ├── business_intelligence.py  # 1152 lines — BI dashboards, KPI calculations
│   └── forecast_evaluator.py     # 1085 lines — MAPE, RMSE, multi-model comparison
├── data/generate_sales_data.py   # 955 lines — seasonal/holiday patterns
├── scripts/feature_engineering.py # 1029 lines — lag features, rolling stats
├── sql/
│   ├── create_tables.sql
│   ├── data_integration.sql
│   └── model_training.sql
├── tests/
│   ├── test_hybrid_forecasting_model.py  # 914 lines
│   ├── test_components.py                # 839 lines
│   └── test_integration.py
└── notebooks/ (placeholder __init__.py only)
```

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Main model | NOT ON DISK | `HybridForecastingModel` referenced in CLAUDE.md but no `models/` dir |
| Feature engineering | `scripts/feature_engineering.py` | Lag, rolling window, seasonality transforms |
| Forecast metrics | `analytics/forecast_evaluator.py` | MAPE=26.9% is the demo target |
| BI reporting | `analytics/business_intelligence.py` | Multi-store aggregation, trend detection |
| Test components | `tests/test_components.py` | 839 lines, tests analytics + data pipeline |

## CONVENTIONS
- Test data: 365 days × 5 stores; training time target ~0.4s
- Heavy deps: `prophet`, `lightgbm` — slow cold import
- No conftest.py — no shared fixtures; tests are self-contained

## ANTI-PATTERNS
- `HybridForecastingModel` class is not implemented — no `models/` directory exists
- Don't import `prophet` at module level — lazy import to avoid startup cost
