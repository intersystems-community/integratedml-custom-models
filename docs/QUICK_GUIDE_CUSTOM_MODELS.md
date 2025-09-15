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

### Model Registration

Custom models are registered with IRIS through the following process:

1. **Model Placement**: Place your Python model files in the IRIS container at:
   ```
   /usr/irissys/mgr/python/custom_models/
   ```
   Models must be organized by type (classifiers, regressors, etc.)

2. **Model Discovery**: IRIS automatically discovers models that:
   - Inherit from `sklearn.base.BaseEstimator`
   - Implement required `fit()` and `predict()` methods
   - Are placed in the correct directory structure

3. **SQL Registration**: Register the model using the JSON USING clause:
   ```sql
   CREATE MODEL YourModelName
   PREDICTING (target_column)
   FROM YourTable
   USING {
       "model_name": "YourCustomModelClass",
       "path_to_classifiers": "/path/to/models",
       "isc_models_disabled": 1,
       "user_params": {
           "param1": value1,
           "param2": value2
       }
   }
   ```

### Deployment Process

1. **Development**:
   - Develop your model following scikit-learn conventions
   - Test locally with sample data
   - Ensure all dependencies are available in IRIS Python environment

2. **Container Deployment**:
   - Copy model files to the IRIS container
   - Install any additional Python dependencies
   - Create required directory symlinks if needed

3. **Model Training**:
   ```sql
   TRAIN MODEL YourModelName
   ```
   - IRIS loads your Python class
   - Executes the `fit()` method with training data
   - Serializes the trained model for persistence

4. **Production Use**:
   ```sql
   SELECT PREDICT(YourModelName) as prediction
   FROM ProductionData
   ```

### Model Versioning and Lifecycle

1. **Version Control**:
   - Models are versioned through the file system
   - Use semantic versioning in model class names (e.g., `ModelV1`, `ModelV2`)
   - IRIS maintains model state between training and prediction

2. **Model Updates**:
   - To update a model, create a new version with a different name
   - Train the new model version
   - Update SQL queries to use the new model name
   - Old models remain available until explicitly dropped

3. **Model Retirement**:
   ```sql
   DROP MODEL OldModelName
   ```

### Security Considerations

1. **Code Execution**:
   - Models execute with IRIS process privileges
   - Ensure models don't access unauthorized resources
   - Validate all model inputs to prevent injection attacks

2. **Data Access**:
   - Models only access data provided through SQL
   - No direct file system or network access recommended
   - Use IRIS security features to control data access

3. **Dependency Management**:
   - Audit all Python dependencies for vulnerabilities
   - Use only trusted packages from official repositories
   - Keep dependencies updated with security patches

### Performance Optimization

1. **Model Design**:
   - Keep models lightweight for low-latency predictions
   - Implement efficient feature engineering in `_engineer_features()`
   - Use vectorized operations with NumPy/pandas

2. **Caching Strategies**:
   - Cache computed features when possible
   - Use model warm-up for initial predictions
   - Consider batch predictions for bulk operations

3. **Resource Management**:
   - Monitor memory usage during training
   - Implement proper cleanup in model destructors
   - Use IRIS monitoring tools to track performance

4. **Best Practices**:
   - Test prediction latency before production deployment
   - Profile model performance with realistic data volumes
   - Optimize feature engineering pipelines
   - Consider model complexity vs. accuracy trade-offs

## Tutorial: Customizing and Updating Models

### Step 1: Understanding Model Customization Points

Every custom model can be tailored through:

1. **Constructor Parameters** - Control model behavior
2. **Feature Engineering** - Domain-specific transformations
3. **Algorithm Selection** - Choose ML algorithms
4. **Ensemble Strategies** - Combine multiple models

### Step 2: Customizing an Existing Model

Let's customize the Credit Risk model as an example:

```python
# Original model in demos/credit_risk/models/credit_risk_classifier.py
class CustomCreditRiskClassifier(ClassificationModel):
    def __init__(self, enable_debt_ratio=True, decision_threshold=0.5):
        # Add new parameters for customization
        super().__init__()
        self.enable_debt_ratio = enable_debt_ratio
        self.decision_threshold = decision_threshold
        self.enable_age_groups = True  # NEW: Age-based risk groups
        self.use_ensemble = False       # NEW: Option for ensemble

    def _engineer_features(self, X):
        X_engineered = X.copy()

        # NEW: Add age group features
        if self.enable_age_groups:
            X_engineered['age_group'] = pd.cut(
                X['age'],
                bins=[0, 25, 35, 50, 100],
                labels=['young', 'adult', 'senior', 'elderly']
            )

        # Existing feature engineering...
        return X_engineered
```

### Step 3: Deploying Updated Models to Container

#### Method 1: Direct Container Update (Development)

```bash
# Copy updated model to running container
docker cp demos/credit_risk/models/credit_risk_classifier.py \
  iris-community:/opt/iris/mgr/python/custom_models/classifiers/

# Restart IRIS to reload models (optional)
docker exec iris-community iris restart iris quietly
```

#### Method 2: Rebuild with Updated Models (Production)

```bash
# Update Dockerfile to include new models
# In docker/Dockerfile.iris:
COPY demos/*/models/*.py /opt/iris/mgr/python/custom_models/classifiers/

# Rebuild and restart
make clean
make setup
```

#### Method 3: Volume Mount (Development)

```yaml
# In docker-compose.yml:
services:
  iris:
    volumes:
      - ./demos:/opt/iris/demos:ro
      - ./custom_models:/opt/iris/mgr/python/custom_models:ro
```

### Step 4: Using Customized Models in SQL

```sql
-- Drop existing model
DROP MODEL IF EXISTS CreditRiskModel;

-- Create model with new parameters
CREATE MODEL CreditRiskModelV2
PREDICTING (default_risk)
FROM CreditApplications
USING {
    "model_name": "CustomCreditRiskClassifier",
    "path_to_classifiers": "/opt/iris/mgr/python/custom_models/classifiers",
    "user_params": {
        "enable_debt_ratio": 1,
        "enable_age_groups": 1,  -- NEW parameter
        "use_ensemble": 0,       -- NEW parameter
        "decision_threshold": 0.45
    }
}

-- Train the updated model
TRAIN MODEL CreditRiskModelV2;

-- Use in production
SELECT customer_id,
       PREDICT(CreditRiskModelV2) as risk_score
FROM NewApplications;
```

### Step 5: A/B Testing Models

```sql
-- Keep both models active
SELECT
    customer_id,
    PREDICT(CreditRiskModel) as model_v1_score,
    PREDICT(CreditRiskModelV2) as model_v2_score,
    ABS(PREDICT(CreditRiskModel) - PREDICT(CreditRiskModelV2)) as difference
FROM TestApplications
WHERE difference > 0.1;  -- Find cases where models disagree
```

### Common Customization Patterns

1. **Feature Engineering Pipeline**:
   ```python
   def _engineer_features(self, X):
       # Add interaction terms
       X['income_to_amount'] = X['income'] / X['credit_amount']

       # Create polynomial features
       X['age_squared'] = X['age'] ** 2

       # Binning continuous variables
       X['income_bracket'] = pd.qcut(X['income'], q=5)

       return X
   ```

2. **Algorithm Swapping**:
   ```python
   def __init__(self, algorithm='random_forest'):
       self.algorithm = algorithm

   def fit(self, X, y):
       if self.algorithm == 'random_forest':
           self.model = RandomForestClassifier()
       elif self.algorithm == 'xgboost':
           self.model = XGBClassifier()
       elif self.algorithm == 'neural':
           self.model = MLPClassifier()
   ```

3. **Hyperparameter Tuning**:
   ```python
   def __init__(self, auto_tune=False, **kwargs):
       self.auto_tune = auto_tune
       self.hyperparams = kwargs

   def fit(self, X, y):
       if self.auto_tune:
           # Grid search for best parameters
           param_grid = {
               'n_estimators': [100, 200, 300],
               'max_depth': [5, 10, 15]
           }
           self.model = GridSearchCV(
               RandomForestClassifier(),
               param_grid
           )
   ```

### Deployment Checklist

- [ ] Test model locally with sample data
- [ ] Verify scikit-learn compatibility
- [ ] Copy model to container
- [ ] Create/update symlinks if needed
- [ ] Update SQL CREATE MODEL statement
- [ ] Train model with production data
- [ ] Validate model performance
- [ ] Monitor prediction latency

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