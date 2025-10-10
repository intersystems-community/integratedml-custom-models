# Implementation Plan: IntegratedML Custom Models Platform

**Branch**: `001-use-the-current` | **Date**: 2025-10-10 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-use-the-current/spec.md`

**Note**: This template is filled in by the `/plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

The IntegratedML Custom Models Platform enables data scientists to deploy custom Python machine learning models with domain-specific feature engineering directly within InterSystems IRIS SQL. The primary requirement is executing fit/predict operations entirely in-database without data export, supporting four demonstration use cases: credit risk assessment, ensemble fraud detection, time-series sales forecasting, and DNA sequence similarity analysis. The technical approach leverages scikit-learn compatibility, extensible base model classes, and IRIS 2025.2's JSON USING clause syntax for parameter passing. Performance targets include sub-50ms prediction latency and 100% state persistence across database restarts.

## Technical Context

**Language/Version**: Python 3.8+ (compatible with IRIS 2025.2 Python runtime)
**Primary Dependencies**: scikit-learn 1.3+, pandas 2.0+, numpy 1.24+, InterSystems IRIS IntegratedML (intersystems-iris-automl), LightGBM 4.0+ (forecasting), Prophet 1.1+ (forecasting)
**Storage**: InterSystems IRIS 2025.2+ database with IntegratedML extension; model state persisted via IRIS mechanisms; test data in IRIS tables
**Testing**: pytest 7.4+ with IRIS database integration; performance benchmarks measuring training time and prediction latency; E2E test suite (tests/test_all_demos_e2e.py)
**Target Platform**: Docker containerized IRIS database on Linux/macOS; Python models execute within IRIS embedded Python runtime
**Project Type**: Single project with multiple domain-specific demo modules (demos/{credit_risk,fraud_detection,sales_forecasting,dna_similarity})
**Performance Goals**: <50ms p95 prediction latency per record; training completion in seconds for 1K-100K record datasets; support 1000 concurrent prediction requests
**Constraints**: Sub-50ms prediction latency (p95); 100% state persistence across IRIS restarts; scikit-learn interface compatibility required; no data export allowed
**Scale/Scope**: 4 demonstration applications; 1,000-100,000 training records per demo; custom feature engineering producing 10-20 engineered features from 5-10 base features

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: In-Database ML
✅ **PASS** - All models execute within IRIS SQL via CREATE MODEL and PREDICT() commands. No data export required. Parameters passed via JSON USING clause (IRIS 2025.2 syntax).

### Principle II: Scikit-learn Compatibility
✅ **PASS** - All custom models inherit from IntegratedMLBaseModel and implement fit/predict/get_params/set_params interface. Four base model classes provide scikit-learn compatibility.

### Principle III: Test-Driven Development (NON-NEGOTIABLE)
✅ **PASS** - Project includes unit tests (demos/*/tests/), integration tests with IRIS, performance benchmarks, and E2E test suite (tests/test_all_demos_e2e.py). Test data generators produce >1000 records per demo.

### Principle IV: Low-Latency Performance
✅ **PASS** - Performance goal explicitly set at <50ms p95 prediction latency. Integration tests measure and validate latency. Feature engineering optimized for repeated predictions.

### Principle V: Model State Management
✅ **PASS** - Base models implement _get_model_state() and _set_model_state() for serialization. Model state includes pipelines, encoders, and learned parameters. Persistence verified across IRIS restarts.

**Constitution Compliance**: All 5 principles PASS. No violations to justify. Ready for Phase 0 research.

## Project Structure

### Documentation (this feature)

```
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```
demos/
├── credit_risk/
│   ├── models/
│   │   └── credit_risk_classifier.py    # Custom feature engineering (debt ratios, risk scoring)
│   ├── data/
│   │   └── generate_sample_data.py      # Test data generator (10,000 records)
│   └── tests/
│       ├── test_integration.py          # SQL CREATE MODEL / PREDICT integration
│       └── test_components.py           # Unit tests for feature engineering
├── fraud_detection/
│   ├── models/
│   │   ├── ensemble_fraud_detector.py   # Multi-model ensemble with voting
│   │   ├── neural_detector.py           # Neural network sub-model
│   │   ├── rules_detector.py            # Rule-based sub-model
│   │   └── behavioral_detector.py       # Behavioral analysis sub-model
│   ├── data/
│   │   └── generate_transaction_data.py # Transaction generator (25,000 records)
│   └── tests/
│       ├── test_integration.py
│       └── test_ensemble.py
├── sales_forecasting/
│   ├── models/
│   │   ├── hybrid_forecasting_model.py  # Prophet + LightGBM hybrid
│   │   └── components/
│   │       ├── prophet_component.py     # Seasonality detection
│   │       └── lightgbm_component.py    # Feature-based predictions
│   ├── data/
│   │   └── generate_sales_data.py       # Sales data (365 days × 5 stores)
│   └── tests/
│       ├── test_integration.py
│       └── test_hybrid_forecasting_model.py
└── dna_similarity/
    ├── models/
    │   └── dna_classifier.py            # K-NN with custom Hamming distance
    ├── data/
    │   └── generate_dna_sequences.py    # DNA sequence generator (5,000 sequences)
    └── tests/
        ├── test_integration.py
        └── test_dna_classifier.py

shared/
├── models/
│   ├── base.py                          # IntegratedMLBaseModel abstract class
│   ├── classification.py                # ClassificationModel base class
│   ├── regression.py                    # RegressionModel base class
│   └── ensemble.py                      # EnsembleModel base class
├── database/
│   └── connection.py                    # IRIS database connection utilities
└── utils/
    ├── preprocessing.py                 # Common preprocessing functions
    └── validation.py                    # Parameter validation helpers

tests/
├── test_all_demos_e2e.py               # End-to-end test suite for all 4 demos
└── conftest.py                         # pytest configuration and fixtures

docker/
├── Dockerfile                          # IRIS + IntegratedML environment
└── docker-compose.yml                  # Database service definition

scripts/
└── setup_iris.sh                       # IRIS configuration and symlink creation
```

**Structure Decision**: Single project structure with domain-specific demo modules. Each demo is independently testable with its own models/, data/, and tests/ directories. Shared utilities (base classes, database connections) centralized in shared/ directory. Docker configuration provides reproducible IRIS environment for development and testing.

## Complexity Tracking

*No constitutional violations identified. Section intentionally left empty.*
