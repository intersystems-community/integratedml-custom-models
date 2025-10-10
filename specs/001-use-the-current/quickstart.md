# Quickstart: IntegratedML Custom Models Platform

**Feature**: IntegratedML Custom Models Platform
**Date**: 2025-10-10
**Audience**: Data Scientists, ML Engineers

## Overview

This quickstart guides you through deploying your first custom ML model to InterSystems IRIS in under 5 minutes. You'll create a credit risk classifier with custom feature engineering that executes entirely within SQL.

---

## Prerequisites

Before starting, ensure you have:

- ✅ InterSystems IRIS 2025.2+ installed (or Docker with docker-compose)
- ✅ Python 3.8+ with pip or uv
- ✅ Basic SQL knowledge
- ✅ Basic Python knowledge (for custom models)

---

## Step 1: Environment Setup (2 minutes)

### Option A: Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/intersystems/integratedml-custom-models.git
cd integratedml-custom-models

# Start IRIS database with IntegratedML
make setup
# This runs: make install && make start
# - Installs Python dependencies
# - Starts IRIS Docker container
# - Creates iris_automl symlink

# Verify IRIS is running
make status
# Expected output: Container "iris" is running on port 1972
```

### Option B: Local IRIS Installation

```bash
# Install Python dependencies
pip install -r requirements.txt
# or with uv (faster):
uv pip install -r requirements.txt

# Configure IRIS connection (edit .env file)
cp .env.example .env
# Set: IRIS_HOST, IRIS_PORT, IRIS_NAMESPACE, IRIS_USERNAME, IRIS_PASSWORD

# Install IntegratedML to IRIS Python runtime
python -m pip install --index-url https://registry.intersystems.com/pypi/simple \
  --no-cache-dir --target /usr/irissys/mgr/python intersystems-iris-automl

# Create required symlink
ln -sf /usr/irissys/mgr/python/iris_automl /opt/irisapp/data/mgr/python/iris_automl
```

---

## Step 2: Prepare Training Data (30 seconds)

```bash
# Generate sample credit application data (10,000 records)
python demos/credit_risk/data/generate_sample_data.py

# Output:
# ✓ Generated 10,000 credit applications
# ✓ Inserted into IRIS table: USER.CreditApplications
# ✓ Table schema: age, credit_amount, duration, employment_duration,
#                 existing_credits, credit_history, purpose, default_risk
```

---

## Step 3: Create Custom Model (1 minute)

### Python Model (already implemented in demos/credit_risk/)

The `CustomCreditRiskClassifier` model includes:
- Debt-to-income ratio calculations
- Custom risk scoring features
- Interaction terms between age and credit amount
- Preprocessing pipeline (scaling, encoding, imputation)

**File**: `demos/credit_risk/models/credit_risk_classifier.py`

Key features:
```python
class CustomCreditRiskClassifier(ClassificationModel):
    def __init__(self,
                 enable_debt_ratio: bool = True,
                 enable_risk_scoring: bool = True,
                 decision_threshold: float = 0.5):
        # Initialize with custom parameters

    def _engineer_features(self, X):
        # Add debt-to-income ratios
        # Add interaction terms
        # Add risk scores
        return X_engineered
```

---

## Step 4: Train Model via SQL (30 seconds)

```sql
-- Connect to IRIS SQL interface (via Management Portal or SQL shell)

CREATE MODEL CreditRiskModel
PREDICTING (default_risk)
FROM CreditApplications
USING {
    "model_name": "CustomCreditRiskClassifier",
    "path_to_classifiers": "/opt/iris/mgr/python/custom_models/classifiers",
    "user_params": {
        "enable_debt_ratio": 1,
        "enable_risk_scoring": 1,
        "decision_threshold": 0.5
    }
}

-- Expected output:
-- Model 'CreditRiskModel' created successfully
-- Training time: 2.3 seconds
-- Training samples: 10,000
-- Features: 8 → 23 (after engineering)
```

**What happened**:
1. IRIS loaded CustomCreditRiskClassifier Python class
2. Model validated parameters (enable_debt_ratio, decision_threshold)
3. Feature engineering created 23 features from 8 base features
4. Preprocessing pipeline fitted (imputers, scalers, encoders)
5. Logistic regression model fitted on engineered features
6. Model state serialized and saved to IRIS repository

---

## Step 5: Make Predictions (10 seconds)

```sql
-- Predict class labels (0=good credit, 1=bad credit)
SELECT customer_id,
       age,
       credit_amount,
       PREDICT(CreditRiskModel) AS risk_prediction
FROM CreditApplications
WHERE customer_id <= 100
LIMIT 10;

-- Output:
-- customer_id | age | credit_amount | risk_prediction
-- 1           | 28  | 5000.00       | 0
-- 2           | 45  | 12000.00      | 0
-- 3           | 22  | 25000.00      | 1
-- ...
```

```sql
-- Predict probabilities (risk score 0.0-1.0)
SELECT customer_id,
       PREDICT(CreditRiskModel PROBABILITIES) AS risk_probability
FROM CreditApplications
WHERE customer_id <= 100
ORDER BY risk_probability DESC
LIMIT 10;

-- Output:
-- customer_id | risk_probability
-- 543         | 0.9234
-- 128         | 0.8876
-- 901         | 0.8542
-- ...
```

**Performance check**:
```sql
-- Verify sub-50ms latency
SELECT AVG(prediction_time_ms) as avg_latency_ms
FROM (
    SELECT PREDICT(CreditRiskModel),
           DATEDIFF('ms', start_time, CURRENT_TIMESTAMP) as prediction_time_ms
    FROM CreditApplications
    LIMIT 1000
);

-- Expected: avg_latency_ms < 50.0
```

---

## Step 6: Validate Model (10 seconds)

```sql
-- Evaluate accuracy on test data
VALIDATE MODEL CreditRiskModel
FROM CreditApplications
WHERE customer_id > 9000;  -- Hold-out 10% for validation

-- Output:
-- Accuracy:  0.8523
-- Precision: 0.8234
-- Recall:    0.7892
-- F1-Score:  0.8058
```

---

## Next Steps

### Explore Other Demos

**Fraud Detection (Ensemble Model)**:
```bash
python demos/fraud_detection/data/generate_transaction_data.py
make demo-fraud
```

**Sales Forecasting (Regression Model)**:
```bash
python demos/sales_forecasting/data/generate_sales_data.py
make demo-sales
```

**DNA Similarity (Custom Distance Metrics)**:
```bash
python demos/dna_similarity/data/generate_dna_sequences.py
make demo-dna
```

### Run All Demos

```bash
# Execute end-to-end test for all 4 demos
python tests/test_all_demos_e2e.py

# Expected output:
# ✓ Credit Risk: 10,000 records trained in 2.3s, 100% accuracy
# ✓ Fraud Detection: 25,000 transactions, 192 flagged, <50ms latency
# ✓ Sales Forecasting: 365 days × 5 stores, 26.9% MAPE
# ✓ DNA Similarity: 5,000 sequences, 50.5% accuracy
```

---

## Common Operations

### List Available Models

```sql
SHOW MODELS;

-- Output:
-- Model Name         | Type           | Created Date | Training Rows
-- CreditRiskModel    | Classification | 2025-10-10   | 10000
-- FraudDetector      | Classification | 2025-10-09   | 25000
-- SalesForecastModel | Regression     | 2025-10-08   | 1825
```

### Update Model (Retrain with Different Parameters)

```sql
DROP MODEL CreditRiskModel;

CREATE MODEL CreditRiskModel
PREDICTING (default_risk)
FROM CreditApplications
USING {
    "model_name": "CustomCreditRiskClassifier",
    "path_to_classifiers": "/opt/iris/mgr/python/custom_models/classifiers",
    "user_params": {
        "enable_debt_ratio": 1,
        "enable_interaction_terms": 0,  -- Disable interactions
        "decision_threshold": 0.6        -- Higher threshold
    }
}
```

### Batch Predictions

```sql
-- Efficient batch processing
SELECT customer_id,
       PREDICT(CreditRiskModel PROBABILITIES) AS risk_score
INTO HighRiskCustomers
FROM CreditApplications
WHERE PREDICT(CreditRiskModel PROBABILITIES) > 0.8;

-- Check results
SELECT COUNT(*) FROM HighRiskCustomers;
-- Output: 543 high-risk customers identified
```

---

## Troubleshooting

### Error: Cannot import model class

```
ERROR: Cannot import 'CustomCreditRiskClassifier' from path
```

**Solution**:
- Verify model file exists: `ls demos/credit_risk/models/credit_risk_classifier.py`
- Check path_to_classifiers in USING clause matches model directory
- Ensure Python can import: `python -c "from custom_models.credit_risk_classifier import CustomCreditRiskClassifier"`

### Error: Parameter validation failed

```
ERROR: Invalid parameter 'decision_threshold': must be in range [0.0, 1.0]
```

**Solution**:
- Check parameter type (boolean should be 0/1, not true/false in SQL)
- Verify parameter range constraints in model._validate_parameters()
- Review model documentation for valid parameter values

### Warning: Low training data

```
WARNING: Only 50 records found; recommend ≥1000 for reliable training
```

**Solution**:
- Generate more training data using data generators
- Model will train but may overfit with insufficient data
- Validation accuracy may be unreliable with small datasets

### Performance: Prediction latency >50ms

```
WARNING: Prediction latency 125ms exceeds target 50ms (p95)
```

**Solution**:
- Disable expensive feature engineering: `"enable_interaction_terms": 0`
- Use simpler base model: LogisticRegression instead of RandomForest
- Profile model with: `python -m cProfile -s cumtime demos/credit_risk/tests/test_integration.py`
- Optimize feature engineering loops with NumPy vectorization

---

## Development Workflow

### Create Your Own Custom Model

**Step 1**: Create new demo directory
```bash
mkdir -p demos/my_model/{models,data,tests}
```

**Step 2**: Implement model class
```python
# demos/my_model/models/my_classifier.py
from shared.models.classification import ClassificationModel

class MyCustomClassifier(ClassificationModel):
    def __init__(self, my_param: bool = True, **kwargs):
        self.my_param = my_param
        super().__init__(**kwargs)

    def _validate_parameters(self):
        super()._validate_parameters()
        if not isinstance(self.my_param, bool):
            raise ValueError("my_param must be boolean")

    def _fit_model(self, X, y):
        # Your training logic
        pass

    def _predict_proba_model(self, X):
        # Your prediction logic
        pass
```

**Step 3**: Write tests
```python
# demos/my_model/tests/test_integration.py
import pytest
from shared.database.connection import connect_iris

def test_my_model_integration():
    conn = connect_iris()

    # CREATE MODEL
    conn.execute("""
        CREATE MODEL MyModel
        PREDICTING (target)
        FROM MyTable
        USING {"model_name": "MyCustomClassifier", ...}
    """)

    # PREDICT
    results = conn.execute("SELECT PREDICT(MyModel) FROM TestData")
    assert results.latency_ms < 50
```

**Step 4**: Run tests
```bash
pytest demos/my_model/tests/ -v --tb=short
```

### Format and Lint Code

```bash
# Auto-format with Black
make format

# Run linters
make lint

# Expected: All checks pass
```

---

## Architecture Quick Reference

**Base Model Hierarchy**:
```
IntegratedMLBaseModel
├── ClassificationModel (binary/multi-class)
│   └── CustomCreditRiskClassifier
├── RegressionModel (continuous values)
│   └── HybridForecastingModel
└── EnsembleModel (multi-model voting)
    └── EnsembleFraudDetector
```

**Project Structure**:
```
demos/
├── credit_risk/        # Classification with feature engineering
├── fraud_detection/    # Ensemble with weighted voting
├── sales_forecasting/  # Regression with Prophet + LightGBM
└── dna_similarity/     # Custom distance metrics

shared/
├── models/            # Base classes
├── database/          # IRIS connections
└── utils/             # Common helpers

tests/
└── test_all_demos_e2e.py  # Comprehensive validation
```

**Key Files**:
- `shared/models/base.py` - Core IntegratedMLBaseModel
- `shared/models/classification.py` - ClassificationModel base
- `demos/{domain}/models/*.py` - Custom domain models
- `demos/{domain}/tests/test_integration.py` - SQL integration tests

---

## Performance Benchmarks

| Demo | Training Time | Prediction Latency (p95) | Accuracy |
|------|---------------|-------------------------|----------|
| Credit Risk | 2.3s (10K records) | <50ms | 85.2% |
| Fraud Detection | 11.7s (25K transactions) | <50ms | 192 flagged |
| Sales Forecasting | 0.4s (1,825 records) | <50ms | 26.9% MAPE |
| DNA Similarity | 1.7s (5K sequences) | <50ms | 50.5% |

---

## Support Resources

- **Documentation**: [docs/QUICK_GUIDE_CUSTOM_MODELS.md](../../docs/QUICK_GUIDE_CUSTOM_MODELS.md)
- **Constitution**: [.specify/memory/constitution.md](../../.specify/memory/constitution.md)
- **Code Examples**: `demos/*/models/*.py`
- **Test Examples**: `demos/*/tests/test_integration.py`
- **Makefile Commands**: `make help`

---

## Summary Checklist

After completing this quickstart, you should have:

- ✅ IRIS database running with IntegratedML installed
- ✅ Training data loaded (CreditApplications table)
- ✅ Custom model trained (CreditRiskModel)
- ✅ Predictions generated via SQL PREDICT()
- ✅ Model validated (accuracy metrics)
- ✅ Understanding of 4 demo applications
- ✅ Familiarity with base model classes
- ✅ Awareness of performance targets (<50ms latency)

**Next**: Explore other demos or create your own custom model!
