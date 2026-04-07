# FRAUD DETECTION — OPTIMIZATION

## OVERVIEW
5 performance optimization modules targeting sub-100ms prediction latency. Each is large (830–1232 lines). These are production-hardening utilities, not core ML logic.

## FILES
| File | Focus | Lines |
|------|-------|-------|
| `batch_optimization.py` | Batch prediction pipeline, throughput | 1232 |
| `caching_strategies.py` | `FraudDetectionCacheManager` — prediction cache | 1174 |
| `latency_optimization.py` | P95 <100ms target, async scoring | 874 |
| `memory_optimization.py` | Model memory pooling, GC tuning | 1105 |
| `model_optimization.py` | Model compression, quantization | 831 |

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Prediction cache | `caching_strategies.py` | `FraudDetectionCacheManager()` — used in conftest fixture |
| Latency benchmarking | `latency_optimization.py` | Ties to `performance_target_ms=100.0` |
| Batch scoring | `batch_optimization.py` | For 100+ concurrent requests target |

## CONVENTIONS
- `FraudDetectionCacheManager` is a session-scoped fixture in conftest
- Cache is cleared in `setup_test_environment` autouse fixture via `RealTimeFeatureProcessor._cache.clear()`

## ANTI-PATTERNS
- Don't import these modules in unit tests — they pull heavy ML deps
- Don't run latency tests without a warm JVM/Python process — cold-start skews results
