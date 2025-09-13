# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository demonstrates **IntegratedML's Custom Models feature**, which allows Python ML models to be executed directly within InterSystems IRIS SQL commands. This is a groundbreaking capability that enables:

- Custom Python preprocessing and model training code within SQL `CREATE MODEL` statements
- Model validation using `VALIDATE MODEL` syntax
- Real-time predictions using `SELECT ... PREDICT()` without data movement
- Integration of any scikit-learn compatible model into database workflows

**Key Innovation**: Data scientists can now bring their custom Python models directly into SQL, eliminating the need for data export/import cycles and enabling real-time ML on live data.

See [PRD.md](PRD.md) for the complete product vision and feature documentation.

## Common Development Commands

### Environment Setup
```bash
# Install dependencies with uv (recommended) or pip
make install

# Start IRIS database (required for demos)
make start

# Complete setup (dependencies + IRIS)
make setup
```

### Development Workflow
```bash
# Run all tests
make test
# or directly with pytest
pytest demos/*/tests/ -v --tb=short

# Format code
make format
# or directly
black .

# Run linting
make lint
# or directly
flake8 . --max-line-length=88 --extend-ignore=E203,W503
mypy shared/ --ignore-missing-imports

# Open notebooks in VS Code
make notebooks
```

### Running Demos
```bash
# Individual demos
make demo-credit
make demo-fraud
make demo-sales
make demo-dna

# All demos
make demos

# Or run directly
python run_credit_risk_demo.py
python run_fraud_detection_demo.py
python run_sales_forecasting_demo.py
python run_dna_similarity_demo.py
```

### Database Management
```bash
# Check status
make status

# View logs
make logs

# Initialize database with sample data
make init-db

# Clean up (removes containers and volumes)
make clean
```

## High-Level Architecture

### Model Integration Pattern
All ML models follow a standardized integration pattern based on scikit-learn compatibility:

1. **Base Model Inheritance**: All models inherit from `IntegratedMLBaseModel` (shared/models/base.py:20), which provides:
   - IntegratedML parameter serialization/deserialization
   - Model state persistence
   - Input validation and preprocessing hooks
   - Consistent fit/predict interfaces

2. **Model Types**: Three specialized base classes extend the core pattern:
   - `ClassificationModel` (shared/models/classification.py) - Binary/multi-class classification
   - `RegressionModel` (shared/models/regression.py) - Continuous value prediction
   - `EnsembleModel` (shared/models/ensemble.py) - Multi-model voting/averaging

3. **Demo Model Structure**: Each demo implements a custom model:
   - `CustomCreditRiskClassifier` (demos/credit_risk/models/credit_risk_classifier.py:22) - Feature engineering for financial data
   - `EnsembleFraudDetector` (demos/fraud_detection/models/ensemble_fraud_detector.py) - Multiple sub-models with weighted voting
   - `HybridForecastingModel` (demos/sales_forecasting/models/hybrid_forecasting_model.py) - Prophet + LightGBM combination
   - `DNASimilarityAnalyzer` (demos/dna_similarity/models/dna_classifier.py) - Sequence analysis algorithms

### Key Architectural Patterns

1. **Feature Engineering Pipeline**: Models implement custom preprocessing in `_engineer_features()` methods, allowing domain-specific transformations before training/prediction.

2. **Model State Management**: Models use `_get_model_state()` and `_set_model_state()` for serialization, enabling persistence across database sessions.

3. **Ensemble Architecture**: The fraud detection demo shows how to combine multiple models (neural, rule-based, behavioral) with weighted voting and confidence thresholds.

4. **Database Integration**: Models are designed to work within IRIS database constraints:
   - Parameters passed via JSON from SQL
   - Models execute in-database for data locality
   - Results returned directly to SQL queries

### Database Connection
The project uses InterSystems IRIS with IntegratedML. Connection details are configured via environment variables (see .env.example). The default setup uses:
- Host: localhost
- Port: 1972
- Namespace: USER
- Default credentials in docker-compose.yml

### Testing Strategy
Tests are organized by demo with shared utilities:
- Unit tests for individual components
- Integration tests with IRIS database
- Performance benchmarks for latency requirements
- Test data generators for reproducible scenarios

Run specific test suites:
```bash
pytest demos/credit_risk/tests/
pytest demos/fraud_detection/tests/
pytest demos/sales_forecasting/tests/
```