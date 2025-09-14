# Quick Guide to IntegratedML Custom Models

## Overview

IntegratedML Custom Models allow you to bring your own Python machine learning models directly into IRIS SQL workflows. This enables in-database machine learning without data movement.

## Basic SQL Syntax

### Creating a Custom Model

```sql
CREATE MODEL YourModelName
PREDICTING (target_column)
FROM YourTable
USING YourCustomModelClass
WITH (parameter1=value1, parameter2=value2)
```

### Making Predictions

```sql
SELECT id, feature1, feature2,
       PREDICT(YourModelName) as prediction
FROM NewData
```

### Validating Model Performance

```sql
VALIDATE MODEL YourModelName
FROM TestData
```

## Python Model Requirements

Your Python model must:

1. **Inherit from IntegratedML base classes**:
   - `ClassificationModel` for classification tasks
   - `RegressionModel` for regression tasks
   - `EnsembleModel` for ensemble approaches

2. **Implement required methods**:
   - `fit(X, y)` - Train the model
   - `predict(X)` - Make predictions
   - `_validate_parameters()` - Validate configuration

3. **Be scikit-learn compatible** for integration with IRIS

## Example Implementation

```python
from shared.models.classification import ClassificationModel

class CustomCreditRiskClassifier(ClassificationModel):
    def __init__(self, enable_debt_ratio=True, decision_threshold=0.5):
        super().__init__()
        self.enable_debt_ratio = enable_debt_ratio
        self.decision_threshold = decision_threshold
        self.model = None

    def fit(self, X, y):
        # Custom feature engineering
        X_engineered = self._engineer_features(X)

        # Train your model
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier()
        self.model.fit(X_engineered, y)
        return self

    def predict(self, X):
        X_engineered = self._engineer_features(X)
        probabilities = self.model.predict_proba(X_engineered)[:, 1]
        return (probabilities > self.decision_threshold).astype(int)

    def _engineer_features(self, X):
        # Your custom feature engineering logic
        return X  # Simplified for example
```

## Model Registration and Deployment

**TODO**: Add specific details about:
- How models are registered with IRIS
- Deployment process and requirements
- Model versioning and lifecycle management
- Security considerations
- Performance optimization tips

## Complete Examples

This repository provides four complete examples:

1. **[Credit Risk Assessment](../demos/credit_risk/)** - Financial risk scoring
2. **[Fraud Detection](../demos/fraud_detection/)** - Real-time fraud detection
3. **[Sales Forecasting](../demos/sales_forecasting/)** - Time series forecasting
4. **[DNA Similarity](../demos/dna_similarity/)** - Sequence analysis

## Getting Help

- See [PRD.md](../PRD.md) for complete feature documentation
- Check [CLAUDE.md](../CLAUDE.md) for development guidance
- Run `python run_all_demos.py --quick` to see examples in action

---

*Note: This guide needs to be updated with the actual IntegratedML Custom Models syntax and implementation details from the official documentation.*