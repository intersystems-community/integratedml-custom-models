# Data Model: IntegratedML Custom Models Platform

**Feature**: IntegratedML Custom Models Platform
**Date**: 2025-10-10
**Status**: Design Complete

## Overview

This document defines the key entities and their relationships for the IntegratedML Custom Models Platform. The data model describes both in-memory Python objects and persisted IRIS database structures.

---

## Entity Definitions

### 1. Custom Model

**Description**: A Python class implementing the scikit-learn estimator interface with custom preprocessing logic. Deployed to IRIS and invoked via SQL.

**Attributes**:
- `model_class_name` (string): Fully qualified class name (e.g., "CustomCreditRiskClassifier")
- `parameters` (dict): Model configuration passed from SQL USING clause
  - Keys: parameter names (e.g., "enable_debt_ratio", "decision_threshold")
  - Values: JSON-serializable types (bool, int, float, string, dict, list)
- `is_fitted` (boolean): Training status flag
- `feature_names_in_` (list[string]): Column names from training data
- `n_features_in_` (integer): Count of input features
- `trained_state` (dict): Serialized model components
  - `pipeline` (sklearn.pipeline.Pipeline): Preprocessing + base model
  - `preprocessor` (ColumnTransformer): Feature transformers
  - `base_model` (sklearn estimator): Underlying algorithm (Logistic Regression, RandomForest, etc.)
  - `feature_metadata` (dict): Numeric/categorical feature lists

**Relationships**:
- Inherits from: IntegratedMLBaseModel → ClassificationModel/RegressionModel/EnsembleModel
- Trained on: Training Dataset (via CREATE MODEL...FROM)
- Produces: Prediction Request results (via PREDICT function)
- Persisted as: Model State (pickle serialization)

**Validation Rules**:
- `model_class_name` MUST reference an importable Python class
- `parameters` MUST pass _validate_parameters() checks before training
- `is_fitted` MUST be True before predict() can execute
- `n_features_in_` MUST match input feature count during prediction
- All parameters MUST be JSON-serializable for IRIS marshaling

**State Transitions**:
1. **Initialized**: Model created with parameters, `is_fitted=False`
2. **Training**: fit() called with data, preprocessor and base_model fitted
3. **Fitted**: Training complete, `is_fitted=True`, state serialized to IRIS
4. **Predicting**: predict() called with new data, uses fitted pipeline
5. **Persisted**: Model state saved via _get_model_state() to IRIS storage

---

### 2. Training Dataset

**Description**: IRIS database table containing features (X) and target variable (y) for model training.

**Attributes**:
- `table_name` (string): IRIS table name (e.g., "CreditApplications")
- `feature_columns` (list[string]): Column names used as input features
- `target_column` (string): Column name for prediction target
- `row_count` (integer): Number of training records
- `schema` (dict): Column name → data type mapping
  - Numeric types: INTEGER, DECIMAL, FLOAT
  - Categorical types: VARCHAR, DATE
- `sample_data` (DataFrame): First 5 rows for validation

**Relationships**:
- Used by: Custom Model (via CREATE MODEL...FROM table_name)
- Contains: Training examples with features and labels

**Validation Rules**:
- `table_name` MUST exist in IRIS database before CREATE MODEL
- `target_column` MUST be present in table schema
- `feature_columns` MUST NOT include target_column
- `row_count` SHOULD be ≥1000 for reliable training (warning if <100)
- Numeric columns MUST NOT contain non-numeric values
- Categorical columns with >100 unique values generate warning (high cardinality)

**Example**:
```sql
-- CreditApplications table structure
TABLE CreditApplications (
    customer_id INTEGER PRIMARY KEY,
    age INTEGER,
    credit_amount DECIMAL(10,2),
    duration INTEGER,
    employment_duration INTEGER,
    existing_credits INTEGER,
    credit_history VARCHAR(50),
    purpose VARCHAR(50),
    default_risk INTEGER  -- Target: 0=good, 1=bad
)
```

---

### 3. Model Parameters

**Description**: JSON configuration passed from SQL to Python model, controlling behavior and feature engineering.

**Attributes**:
- `parameter_name` (string): Identifier (e.g., "enable_debt_ratio")
- `value` (any): JSON-serializable value
- `type` (string): Data type (boolean, integer, float, string, dict, array)
- `validation_rule` (string): Constraint description
- `default_value` (any): Fallback if not specified

**Relationships**:
- Passed to: Custom Model via USING clause
- Validated by: Model._validate_parameters() method

**Examples**:

| Parameter Name | Type | Validation Rule | Default | Description |
|----------------|------|-----------------|---------|-------------|
| enable_debt_ratio | boolean | Must be 0 or 1 | True | Calculate debt-to-income features |
| decision_threshold | float | Range [0.0, 1.0] | 0.5 | Classification decision boundary |
| ensemble_weights | array | Sum must equal 1.0 | [0.33, 0.33, 0.34] | Sub-model voting weights |
| max_depth | integer | Range [1, 50] | 10 | Tree depth for ensemble sub-models |

**Validation Rules**:
- Boolean params: Accept 0/1 (SQL) or true/false (JSON)
- Float params: Reject infinity/NaN values
- Array params: All elements must be same type
- Dict params: Keys must be strings (JSON requirement)
- Invalid params trigger ValueError before training

**SQL Usage Example**:
```sql
CREATE MODEL CreditRiskModel
PREDICTING (default_risk)
FROM CreditApplications
USING {
    "model_name": "CustomCreditRiskClassifier",
    "user_params": {
        "enable_debt_ratio": 1,
        "enable_risk_scoring": 1,
        "decision_threshold": 0.5
    }
}
```

---

### 4. Prediction Request

**Description**: SQL query using PREDICT() function to generate predictions from a trained model.

**Attributes**:
- `model_name` (string): Name of trained model (from CREATE MODEL)
- `input_features` (DataFrame): Features from SELECT clause
- `output_format` (string): Result type ("class_label", "probability", "regression_value")
- `prediction_timestamp` (datetime): When prediction executed
- `latency_ms` (float): Time to compute prediction

**Relationships**:
- Uses: Custom Model (via model_name reference)
- Produces: Prediction results (class labels, probabilities, or regression values)

**SQL Usage Examples**:
```sql
-- Binary classification: Get predicted class label
SELECT customer_id,
       PREDICT(CreditRiskModel) as risk_prediction
FROM NewApplications

-- Classification with probabilities: Get risk probability
SELECT customer_id,
       PREDICT(CreditRiskModel PROBABILITIES) as risk_probability
FROM NewApplications

-- Regression: Get forecasted sales value
SELECT store_id, date,
       PREDICT(SalesForecastModel) as predicted_sales
FROM FutureDates
```

**Validation Rules**:
- `model_name` MUST reference a fitted model (is_fitted=True)
- Input feature count MUST match `n_features_in_`
- Input feature names MUST match `feature_names_in_` (order-independent)
- Missing features trigger imputation (median/mode) or validation error
- Prediction must complete within 50ms (p95) or log performance warning

---

### 5. Model State

**Description**: Serialized representation of trained model for persistence across IRIS sessions.

**Attributes**:
- `model_id` (string): Unique identifier (IRIS-generated)
- `pickle_data` (binary): Pickled Python objects (pipeline, encoders, weights)
- `version` (string): Semantic version (e.g., "1.0.0")
- `creation_timestamp` (datetime): When model was trained
- `fit_status` (boolean): Training completion flag
- `metadata` (dict): Model-specific information
  - `feature_engineering_enabled` (dict): Which transformations active
  - `original_features` (int): Input feature count
  - `engineered_features` (int): Post-transformation feature count
  - `training_duration_seconds` (float): Time to fit model
  - `training_dataset_rows` (int): Number of training examples

**Relationships**:
- Serialized from: Custom Model via _get_model_state()
- Deserialized to: Custom Model via _set_model_state()
- Stored in: IRIS IntegratedML model repository

**Validation Rules**:
- Pickle protocol version MUST be compatible with Python runtime
- Deserialization MUST restore model to identical prediction behavior
- State MUST include all fitted preprocessing components
- Missing state components trigger retraining requirement

**Serialization Example**:
```python
def _get_model_state(self) -> Dict[str, Any]:
    return {
        'pipeline': self._pipeline,              # sklearn Pipeline object
        'preprocessor': self._preprocessor,      # ColumnTransformer
        'base_model': self._base_model,          # Fitted estimator
        'numeric_features': self._numeric_features,
        'categorical_features': self._categorical_features
    }
```

---

### 6. Base Model Classes

**Description**: Abstract classes providing common interfaces for all custom models.

**Class Hierarchy**:
```
IntegratedMLBaseModel (shared/models/base.py)
├── ClassificationModel (shared/models/classification.py)
│   ├── CustomCreditRiskClassifier
│   ├── DNASimilarityAnalyzer
│   └── [Ensemble sub-models]
├── RegressionModel (shared/models/regression.py)
│   └── HybridForecastingModel
└── EnsembleModel (shared/models/ensemble.py)
    └── EnsembleFraudDetector
```

**Common Attributes (IntegratedMLBaseModel)**:
- `parameters` (dict): Model configuration
- `is_fitted` (boolean): Training status
- `feature_names_in_` (list): Training feature names
- `n_features_in_` (int): Training feature count
- `_model_metadata` (dict): Diagnostic information

**Common Methods**:
- `fit(X, y)`: Train the model
- `predict(X)`: Generate predictions
- `get_params(deep=True)`: Return parameters dict
- `set_params(**params)`: Update parameters
- `_validate_parameters()`: Check parameter validity
- `_get_model_state()`: Serialize for persistence
- `_set_model_state(state)`: Deserialize from persistence

**ClassificationModel Extensions**:
- `predict_proba(X)`: Return class probabilities
- `decision_threshold` (float): Binary classification boundary
- `classes_` (list): Unique class labels

**RegressionModel Extensions**:
- `score(X, y)`: Calculate R² score
- `residuals(X, y)`: Compute prediction errors

**EnsembleModel Extensions**:
- `sub_models` (list): Collection of base estimators
- `voting_weights` (list): Sub-model importance
- `voting_strategy` (string): "soft" (probability) or "hard" (majority)

---

### 7. Feature Engineering Pipeline

**Description**: Sequence of transformations applied to raw data before model training/prediction.

**Attributes**:
- `transformation_steps` (list[tuple]): Ordered (name, transformer) pairs
- `fitted_encoders` (dict): One-hot/label encoders by column name
- `fitted_scalers` (dict): StandardScaler/MinMaxScaler by column name
- `feature_names_mapping` (dict): Original → engineered feature names
- `numeric_features` (list): Columns receiving numeric transformations
- `categorical_features` (list): Columns receiving encoding

**Transformation Types**:
1. **Imputation**: Fill missing values (median for numeric, mode for categorical)
2. **Scaling**: StandardScaler (mean=0, std=1) for numeric features
3. **Encoding**: OneHotEncoder for categorical features
4. **Custom Engineering**: Domain-specific calculations
   - Debt-to-income ratios (credit risk)
   - Transaction velocity (fraud detection)
   - Seasonal decomposition (sales forecasting)
   - GC content analysis (DNA similarity)

**Relationships**:
- Applied by: Custom Model._engineer_features() method
- Persisted in: Model State pickle data
- Configured via: Model Parameters

**Pipeline Example (Credit Risk)**:
```python
ColumnTransformer([
    ('numeric', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('categorical', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_features)
])
```

**Validation Rules**:
- Pipelines MUST be fitted during training before prediction
- Feature names MUST be consistent between fit and predict
- Encoders MUST handle unknown categories (handle_unknown='ignore')
- Imputers MUST have strategy compatible with column type

---

### 8. Ensemble Sub-Model

**Description**: Individual model within an ensemble, with its own architecture and voting weight.

**Attributes**:
- `model_instance` (BaseEstimator): Fitted sklearn-compatible model
- `voting_weight` (float): Importance in ensemble decision (range [0, 1])
- `confidence_threshold` (float): Minimum confidence for vote participation
- `sub_model_name` (string): Identifier (e.g., "neural_detector")
- `sub_model_type` (string): Architecture category ("neural", "rules", "statistical")

**Relationships**:
- Contained in: EnsembleModel.sub_models list
- Trained on: Same Training Dataset as ensemble
- Contributes to: Final ensemble prediction via voting

**Voting Strategies**:

**Hard Voting (Classification)**:
- Each sub-model votes for one class
- Final prediction: majority vote
- Ties broken by voting_weight sum

**Soft Voting (Classification)**:
- Each sub-model outputs class probabilities
- Weighted average of probabilities
- Final prediction: argmax of averaged probabilities

**Averaging (Regression)**:
- Each sub-model outputs numeric prediction
- Weighted average of predictions
- Final prediction: weighted sum

**Example (Fraud Detection Ensemble)**:
```python
sub_models = [
    ('neural', MLPClassifier(), 0.4),      # 40% weight - complex patterns
    ('rules', RuleBased Classifier(), 0.3),  # 30% weight - known fraud rules
    ('behavioral', IsolationForest(), 0.3)   # 30% weight - anomaly detection
]
```

**Validation Rules**:
- Voting weights MUST sum to 1.0 (within tolerance 1e-6)
- All sub-models MUST implement fit/predict interface
- All sub-models MUST be fitted before ensemble prediction
- Confidence threshold range: [0.0, 1.0]

---

## Entity Relationships Diagram

```
┌─────────────────────┐
│  Training Dataset   │
│  (IRIS Table)       │
└──────────┬──────────┘
           │ CREATE MODEL...FROM
           ▼
┌─────────────────────┐         ┌────────────────────┐
│   Model Parameters  │────────>│   Custom Model     │
│   (JSON USING)      │ configure│  (Python Class)    │
└─────────────────────┘         └──────────┬─────────┘
                                           │ inherit
                                           ▼
                                ┌────────────────────────┐
                                │  Base Model Classes    │
                                │  - IntegratedMLBase    │
                                │  - Classification      │
                                │  - Regression          │
                                │  - Ensemble            │
                                └──────────┬─────────────┘
                                           │ use
                                           ▼
┌─────────────────────┐         ┌────────────────────────┐
│   Model State       │<────────│ Feature Engineering    │
│   (Pickle)          │ persist │ Pipeline               │
└──────────┬──────────┘         └────────────────────────┘
           │ load
           ▼
┌─────────────────────┐         ┌────────────────────┐
│  Fitted Model       │<────────│ Prediction Request │
│  (IRIS Storage)     │ PREDICT │ (SQL Query)        │
└─────────────────────┘         └────────────────────┘
           │
           │ (for ensembles)
           ▼
┌─────────────────────┐
│ Ensemble Sub-Models │
│ (Multiple Fitted)   │
└─────────────────────┘
```

---

## Data Flow Sequences

### Sequence 1: Model Training

1. User executes SQL: `CREATE MODEL name PREDICTING (target) FROM table USING {params}`
2. IRIS validates table existence and parameter JSON syntax
3. IntegratedML instantiates Custom Model class with parameters
4. Model._validate_parameters() checks parameter validity
5. Model.fit(X, y) called with table data
6. Feature engineering applied via _engineer_features()
7. Preprocessing pipeline fitted (imputers, scalers, encoders)
8. Base model (LogisticRegression, RandomForest, etc.) fitted
9. Model state serialized via _get_model_state()
10. Pickle data stored in IRIS IntegratedML repository
11. is_fitted flag set to True
12. Success message returned to user

### Sequence 2: Prediction

1. User executes SQL: `SELECT PREDICT(model_name) FROM input_table`
2. IRIS retrieves model state from repository
3. Model state deserialized via _set_model_state()
4. Fitted pipeline and transformers restored
5. Input features extracted from SELECT clause
6. Feature count/names validated against model expectations
7. Feature engineering applied (identical to training)
8. Fitted pipeline.predict() generates predictions
9. Predictions returned to SQL result set
10. Latency measured and logged (warning if >50ms)

### Sequence 3: Ensemble Voting

1. Ensemble.predict(X) called with input features
2. Feature engineering applied once to X
3. For each sub-model in ensemble:
   - sub_model.predict_proba(X_engineered) generates probabilities
   - If confidence > threshold, include in vote
   - Multiply probabilities by voting_weight
4. Sum weighted probabilities across sub-models
5. Normalize to ensure probabilities sum to 1.0
6. Return final prediction: argmax(weighted_probabilities)

---

## Summary

The data model supports:
- ✅ In-database ML execution (Training Dataset → Custom Model → Predictions)
- ✅ Scikit-learn compatibility (Base Model Classes implement estimator interface)
- ✅ State persistence (Model State serialization across IRIS restarts)
- ✅ Custom feature engineering (Feature Engineering Pipeline)
- ✅ Ensemble modeling (Ensemble Sub-Models with weighted voting)

All entities validated against constitutional principles. Ready for contracts/ generation (Phase 1 continued).
