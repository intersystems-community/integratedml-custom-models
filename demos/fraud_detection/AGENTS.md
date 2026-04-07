# FRAUD DETECTION DEMO

## OVERVIEW
Most complex demo — ensemble of 4 sub-models (rule-based, anomaly, neural, behavioral) with sub-100ms latency target, 6 feature engineering modules, and 5 optimization modules.

## STRUCTURE
```
fraud_detection/
├── models/
│   └── ensemble_fraud_detector.py   # EnsembleFraudDetector(EnsembleModel) — 820 lines
├── features/                        # 6 feature extractors, all ~900-1000 lines
│   ├── behavioral_features.py       # Customer behavior patterns
│   ├── location_features.py         # Geo/merchant location signals
│   ├── realtime_features.py         # RealTimeFeatureProcessor — streaming
│   ├── risk_features.py             # Risk scoring features
│   ├── transaction_features.py      # Core transaction signals
│   └── velocity_features.py        # Transaction velocity/frequency
├── optimization/                    # 5 performance modules, each ~900-1200 lines
│   ├── batch_optimization.py        # Batch prediction pipeline
│   ├── caching_strategies.py        # FraudDetectionCacheManager
│   ├── latency_optimization.py      # Sub-100ms optimization
│   ├── memory_optimization.py       # Memory pooling
│   └── model_optimization.py        # Model compression/quantization
├── data/generate_transaction_data.py # TransactionDataGenerator
├── sql/deployment/                  # CREATE MODEL, full deployment SQL
├── sql/prediction/realtime_prediction.sql
├── tests/conftest.py                # Only conftest.py in project — all fixtures here
└── tests/test_unit_*.py             # test_unit_data_generator, test_unit_ensemble
```

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Main ensemble class | `models/ensemble_fraud_detector.py:25` | `EnsembleFraudDetector(EnsembleModel)` |
| Ensemble init params | `models/ensemble_fraud_detector.py:52` | voting, confidence_threshold, enable_* flags |
| Test fixtures (ALL) | `tests/conftest.py` | Session-scoped: data, ensemble, timer |
| Streaming features | `features/realtime_features.py` | `RealTimeFeatureProcessor` |
| Cache layer | `optimization/caching_strategies.py` | `FraudDetectionCacheManager` |
| IntegratedML SQL | `sql/deployment/01_create_model.sql` | Full CREATE MODEL syntax example |

## CONVENTIONS
- Sub-model flags: `enable_rule_engine`, `enable_anomaly_detection`, `enable_neural_classifier`, `enable_behavioral_analysis`
- Voting strategies: `'hard'`, `'soft'`, `'weighted'`, `'confidence'`
- Performance target: `performance_target_ms=100.0` (constructor param)
- Test config fixture: `max_latency_ms=100`, `min_accuracy=0.85`, `test_data_size=100`
- `test_unit_*` files auto-marked `@unit` by conftest `pytest_collection_modifyitems`

## ANTI-PATTERNS
- Don't skip conftest.py — all test fixtures live there, not in individual test files
- `EnsembleFraudDetector` imports from `shared.models.ensemble` which doesn't exist
- Don't enable `parallel_processing=True` in test fixtures — conftest explicitly disables it
