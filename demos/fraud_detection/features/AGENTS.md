# FRAUD DETECTION — FEATURES

## OVERVIEW
6 feature engineering modules for real-time fraud scoring. All are large (860–1010 lines), standalone, and consumed by `EnsembleFraudDetector` and `RealTimeFeatureProcessor`.

## FILES
| File | Class/Focus | Lines |
|------|-------------|-------|
| `behavioral_features.py` | Customer spend/pattern history | 921 |
| `location_features.py` | Merchant geo, IP, device location | 978 |
| `realtime_features.py` | `RealTimeFeatureProcessor` — streaming aggregation | 864 |
| `risk_features.py` | Merchant/customer risk scoring | 1009 |
| `transaction_features.py` | Core amount/time/category features | — |
| `velocity_features.py` | Frequency/velocity windows (1h, 24h) | 959 |

## CONVENTIONS
- All processors accept `enable_caching`, `cache_ttl_seconds` constructor params
- `RealTimeFeatureProcessor(enable_parallel_processing=False)` in tests — never enable in test fixtures
- Features output pandas DataFrames with consistent column naming

## ANTI-PATTERNS
- Don't enable `parallel_processing=True` during test execution
- Don't instantiate `SentenceTransformer` in feature code — those belong in model layer
