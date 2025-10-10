# SQL Interface Contract: IntegratedML Custom Models

**Feature**: IntegratedML Custom Models Platform
**Date**: 2025-10-10
**Version**: 1.0

## Overview

This document defines the SQL interface contracts for creating, training, validating, and using custom ML models within InterSystems IRIS. All contracts follow IRIS 2025.2 IntegratedML syntax.

---

## Contract 1: CREATE MODEL

**Purpose**: Deploy and train a custom Python model using SQL.

**Syntax**:
```sql
CREATE MODEL <model_name>
PREDICTING (<target_column>)
FROM <training_table>
USING {
    "model_name": "<PythonClassName>",
    "path_to_classifiers": "<python_module_path>",
    "user_params": {
        "<param1>": <value1>,
        "<param2>": <value2>
    }
}
```

**Parameters**:
- `model_name` (identifier): SQL name for the trained model
- `target_column` (identifier): Column to predict
- `training_table` (identifier): Table containing training data
- `PythonClassName` (string): Python class implementing scikit-learn interface
- `python_module_path` (string): Directory containing Python model file
- `user_params` (JSON object): Model-specific configuration parameters

**Preconditions**:
- Training table MUST exist in current namespace
- Target column MUST exist in training table
- Python class MUST be importable from path
- Python class MUST inherit from IntegratedMLBaseModel
- User params MUST pass model._validate_parameters()

**Postconditions**:
- Model trained and persisted in IRIS model repository
- Model state includes fitted pipeline and preprocessing artifacts
- Model marked as is_fitted=True
- Training duration logged

**Returns**: Success message with training summary

**Example**:
```sql
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
```

**Error Responses**:
- Table not found: `ERROR: Table 'CreditApplications' does not exist`
- Column not found: `ERROR: Column 'default_risk' not found in table`
- Class import failed: `ERROR: Cannot import 'CustomCreditRiskClassifier' from path`
- Parameter validation failed: `ERROR: Invalid parameter 'decision_threshold': must be in range [0.0, 1.0]`
- Insufficient training data: `WARNING: Only 50 records found; recommend ≥1000 for reliable training`

**Performance Contract**:
- Training SHALL complete for 10,000 records within 10 seconds (p95)
- Training duration SHOULD scale linearly with dataset size
- Memory usage SHALL NOT exceed 2GB for 100,000 record datasets

---

## Contract 2: PREDICT (Classification)

**Purpose**: Generate class label predictions from a trained classification model.

**Syntax**:
```sql
SELECT <columns>, PREDICT(<model_name>) AS <prediction_alias>
FROM <input_table>
[WHERE <conditions>]
```

**Parameters**:
- `model_name` (identifier): Name of trained classification model
- `input_table` (identifier): Table containing features for prediction
- `prediction_alias` (identifier): Column name for predictions

**Preconditions**:
- Model MUST exist and be fitted (is_fitted=True)
- Input table feature count MUST match model.n_features_in_
- Input table feature names MUST match model.feature_names_in_ (order-independent)
- Model type MUST be ClassificationModel or subclass

**Postconditions**:
- One prediction per input row
- Predictions are class labels (0/1 for binary, categorical for multi-class)
- Missing features imputed according to model configuration
- Prediction latency logged

**Returns**: Result set with original columns plus prediction column

**Example**:
```sql
SELECT customer_id,
       age,
       credit_amount,
       PREDICT(CreditRiskModel) AS risk_prediction
FROM NewApplications
WHERE application_date >= CURRENT_DATE
```

**Performance Contract**:
- Prediction latency SHALL be <50ms (p95) per record
- Batch predictions (1000 records) SHALL complete within 5 seconds
- Concurrent predictions SHALL scale to 100 simultaneous requests

---

## Contract 3: PREDICT (Probability)

**Purpose**: Generate class probability predictions from a trained classification model.

**Syntax**:
```sql
SELECT <columns>, PREDICT(<model_name> PROBABILITIES) AS <probability_alias>
FROM <input_table>
[WHERE <conditions>]
```

**Parameters**:
- `model_name` (identifier): Name of trained classification model
- `PROBABILITIES` (keyword): Request probability output instead of class labels
- `input_table` (identifier): Table containing features for prediction
- `probability_alias` (identifier): Column name for probabilities

**Preconditions**:
- Same as Contract 2 (PREDICT Classification)
- Model MUST implement predict_proba() method

**Postconditions**:
- One probability per input row
- Probabilities in range [0.0, 1.0]
- For binary classification: probability of positive class
- For multi-class: array of probabilities (one per class)

**Returns**: Result set with original columns plus probability column

**Example**:
```sql
SELECT customer_id,
       PREDICT(CreditRiskModel PROBABILITIES) AS risk_probability
FROM NewApplications
WHERE risk_probability > 0.7  -- High risk threshold
```

**Performance Contract**:
- Same latency requirements as Contract 2
- Probability calculation adds <5ms overhead vs. class label prediction

---

## Contract 4: PREDICT (Regression)

**Purpose**: Generate continuous value predictions from a trained regression model.

**Syntax**:
```sql
SELECT <columns>, PREDICT(<model_name>) AS <prediction_alias>
FROM <input_table>
[WHERE <conditions>]
```

**Parameters**:
- `model_name` (identifier): Name of trained regression model
- `input_table` (identifier): Table containing features for prediction

**Preconditions**:
- Model MUST exist and be fitted
- Model type MUST be RegressionModel or subclass
- Input table features MUST match training features

**Postconditions**:
- One numeric prediction per input row
- Predictions are continuous values (not constrained to [0,1])
- Missing features imputed according to model configuration

**Returns**: Result set with original columns plus prediction column

**Example**:
```sql
SELECT store_id,
       date,
       PREDICT(SalesForecastModel) AS predicted_sales
FROM FutureDates
WHERE date BETWEEN '2025-01-01' AND '2025-01-31'
```

**Performance Contract**:
- Same latency requirements as Contract 2

---

## Contract 5: VALIDATE MODEL

**Purpose**: Evaluate model performance and return accuracy metrics.

**Syntax**:
```sql
VALIDATE MODEL <model_name>
FROM <validation_table>
```

**Parameters**:
- `model_name` (identifier): Name of trained model
- `validation_table` (identifier): Table with features and known target values

**Preconditions**:
- Model MUST exist and be fitted
- Validation table MUST include target column
- Validation table features MUST match training features

**Postconditions**:
- Metrics calculated by comparing predictions to actual values
- Metrics vary by model type (classification vs. regression)

**Returns**: Metrics table with performance statistics

**Classification Metrics**:
```
| Metric    | Value |
|-----------|-------|
| Accuracy  | 0.8523 |
| Precision | 0.8234 |
| Recall    | 0.7892 |
| F1-Score  | 0.8058 |
```

**Regression Metrics**:
```
| Metric | Value   |
|--------|---------|
| R²     | 0.7812  |
| RMSE   | 123.45  |
| MAE    | 98.32   |
| MAPE   | 0.2693  |
```

**Example**:
```sql
VALIDATE MODEL CreditRiskModel
FROM TestApplications
```

**Performance Contract**:
- Validation SHALL complete within 2× training time for same dataset size
- Metrics calculation adds <10% overhead to prediction time

---

## Contract 6: DROP MODEL

**Purpose**: Delete a trained model from IRIS model repository.

**Syntax**:
```sql
DROP MODEL <model_name>
```

**Parameters**:
- `model_name` (identifier): Name of model to delete

**Preconditions**:
- Model MUST exist in repository

**Postconditions**:
- Model state removed from repository
- Subsequent PREDICT queries fail with "model not found" error

**Returns**: Success message

**Example**:
```sql
DROP MODEL OldCreditRiskModel
```

**Error Responses**:
- Model not found: `ERROR: Model 'OldCreditRiskModel' does not exist`

---

## Contract 7: SHOW MODELS

**Purpose**: List all trained models in current namespace.

**Syntax**:
```sql
SHOW MODELS
[WHERE model_name LIKE '<pattern>']
```

**Parameters**:
- Optional pattern filter for model names

**Preconditions**: None

**Postconditions**: None (read-only query)

**Returns**: Table of model metadata

**Result Schema**:
```
| Model Name         | Model Type       | Created Date | Training Rows | Status |
|--------------------|------------------|--------------|---------------|--------|
| CreditRiskModel    | Classification   | 2025-10-10   | 10000         | Fitted |
| FraudDetector      | Classification   | 2025-10-09   | 25000         | Fitted |
| SalesForecastModel | Regression       | 2025-10-08   | 1825          | Fitted |
```

**Example**:
```sql
SHOW MODELS
WHERE model_name LIKE 'Credit%'
```

---

## Python Model Interface Contract

**Purpose**: Define required methods for custom Python models to integrate with IntegratedML.

**Required Methods**:

### Method: `__init__(self, **params)`
- **Purpose**: Initialize model with user_params from SQL USING clause
- **Parameters**: Arbitrary keyword arguments from JSON
- **Returns**: None
- **Side Effects**: Store params, initialize instance variables

### Method: `fit(self, X, y, **params)`
- **Purpose**: Train model on dataset
- **Parameters**:
  - `X` (DataFrame): Feature matrix from SQL table
  - `y` (Series): Target values from PREDICTING column
  - `params` (dict): Additional training parameters from IntegratedML
- **Returns**: self (for method chaining)
- **Side Effects**: Fit preprocessing pipeline and base model, set is_fitted=True

### Method: `predict(self, X)`
- **Purpose**: Generate predictions for new data
- **Parameters**: `X` (DataFrame): Feature matrix from SQL SELECT
- **Returns**: ndarray of predictions (class labels or regression values)
- **Preconditions**: Model MUST be fitted (is_fitted=True)
- **Performance**: MUST complete within 50ms (p95) for single record

### Method: `predict_proba(self, X)` (ClassificationModel only)
- **Purpose**: Generate class probability predictions
- **Parameters**: `X` (DataFrame): Feature matrix
- **Returns**: ndarray of shape (n_samples, n_classes) with probabilities
- **Validation**: Probabilities MUST sum to 1.0 per sample (within 1e-6)

### Method: `get_params(self, deep=True)`
- **Purpose**: Return model configuration for introspection
- **Parameters**: `deep` (bool): Include nested estimator params
- **Returns**: dict of parameter name → value mappings
- **Required**: Scikit-learn BaseEstimator interface compliance

### Method: `set_params(self, **params)`
- **Purpose**: Update model configuration
- **Parameters**: Arbitrary keyword arguments
- **Returns**: self (for method chaining)
- **Side Effects**: Update parameters, re-run _validate_parameters()

### Method: `_validate_parameters(self)` (protected)
- **Purpose**: Check parameter validity before training
- **Returns**: None
- **Raises**: ValueError if parameters invalid with descriptive message

### Method: `_get_model_state(self)` (protected)
- **Purpose**: Serialize model for persistence
- **Returns**: dict containing all trained components
- **Required Contents**: fitted pipeline, preprocessors, base model, feature metadata

### Method: `_set_model_state(self, state)` (protected)
- **Purpose**: Deserialize model from persistence
- **Parameters**: `state` (dict): Previously serialized components
- **Side Effects**: Restore fitted pipeline and preprocessing artifacts

---

## Error Handling Contract

**Common Error Codes**:

| Error Code | Description | User Action |
|------------|-------------|-------------|
| ERR_MODEL_NOT_FOUND | Model name does not exist | Check SHOW MODELS for available models |
| ERR_TABLE_NOT_FOUND | Training/input table missing | Verify table name and namespace |
| ERR_COLUMN_NOT_FOUND | Target/feature column missing | Check table schema |
| ERR_FEATURE_MISMATCH | Input features don't match training | Ensure input has same columns as training data |
| ERR_MODEL_NOT_FITTED | Predict called before training | CREATE MODEL first, then PREDICT |
| ERR_PARAM_INVALID | Parameter fails validation | Check parameter type, range, and format |
| ERR_IMPORT_FAILED | Cannot load Python class | Verify model file in path_to_classifiers |
| ERR_LATENCY_EXCEEDED | Prediction took >50ms | Disable expensive feature engineering options |
| WARN_LOW_DATA | Training data <1000 records | Add more training examples for better accuracy |

**Error Message Format**:
```
ERROR [ERR_MODEL_NOT_FOUND]: Model 'NonExistentModel' does not exist in namespace USER.
Suggestion: Run SHOW MODELS to list available models.
```

---

## Performance Service Level Agreements (SLAs)

| Operation | Dataset Size | P50 Latency | P95 Latency | P99 Latency |
|-----------|--------------|-------------|-------------|-------------|
| CREATE MODEL | 10,000 rows | 2.5s | 5.0s | 8.0s |
| CREATE MODEL | 100,000 rows | 15s | 30s | 45s |
| PREDICT (single) | 1 row | 10ms | 50ms | 100ms |
| PREDICT (batch) | 1,000 rows | 500ms | 2s | 5s |
| VALIDATE MODEL | 10,000 rows | 3s | 6s | 10s |

**Throughput SLAs**:
- Concurrent predictions: 100 simultaneous requests without degradation
- Batch prediction throughput: 200 predictions/second (1000-row batches)
- Training throughput: 5 concurrent model training jobs

---

## Versioning and Compatibility

**IntegratedML API Version**: 1.0
**IRIS Version Requirement**: 2025.2+
**Python Version Requirement**: 3.8+
**Scikit-learn Version**: 1.3+

**Backward Compatibility**:
- Models trained on IRIS 2025.2 MUST work on IRIS 2025.3+
- Python 3.8 models MUST work on Python 3.9+ runtimes
- Pickle protocol compatibility maintained across minor versions

**Breaking Changes (Future)**:
- Major IRIS version upgrades may change JSON USING syntax
- Python runtime upgrades may affect pickle deserialization
- Deprecation warnings issued 2 releases before removal

---

## Security and Access Control

**Model Access**:
- CREATE MODEL requires: `%SQL_CREATE_MODEL` privilege
- PREDICT requires: `%SQL_PREDICT` privilege
- VALIDATE MODEL requires: `%SQL_VALIDATE_MODEL` privilege
- DROP MODEL requires: `%SQL_DROP_MODEL` privilege

**Data Access**:
- Model training inherits table SELECT permissions
- Predictions inherit input table SELECT permissions
- Models do not bypass row-level security

**Code Execution**:
- Python models execute in IRIS embedded Python runtime
- Models run with IRIS process user permissions
- File system access limited to model directory and temp files

---

## Summary

All SQL interface contracts defined for:
- ✅ CREATE MODEL (FR-001, FR-004)
- ✅ PREDICT (classification, probability, regression) (FR-003, FR-015)
- ✅ VALIDATE MODEL (FR-007)
- ✅ DROP MODEL, SHOW MODELS (operational management)
- ✅ Python model interface (FR-002, FR-009)
- ✅ Error handling (FR-012)
- ✅ Performance SLAs (SC-002, SC-003)

Ready for quickstart.md generation.
