# API Reference

## 📚 Complete IntegratedML API Documentation

This comprehensive API reference documents all classes, methods, and configuration options available in the IntegratedML Pluggable Models Demo. Each section includes detailed parameter descriptions, return values, usage examples, and best practices.

---

## 🏗️ Core Base Classes

### IntegratedMLBaseModel

The foundational abstract base class for all IntegratedML pluggable models.

```python
class IntegratedMLBaseModel(BaseEstimator, ABC)
```

#### Constructor

```python
def __init__(self, **kwargs)
```

**Parameters:**
- `**kwargs` (dict): Model-specific parameters passed from IntegratedML

**Attributes:**
- `parameters` (dict): All model parameters
- `is_fitted` (bool): Whether the model has been trained
- `feature_names_in_` (list): Feature names from training data
- `n_features_in_` (int): Number of features from training data
- `_model_metadata` (dict): Internal model metadata storage

#### Abstract Methods

##### fit()
```python
@abstractmethod
def fit(X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series]) -> 'IntegratedMLBaseModel'
```

Train the model on provided data.

**Parameters:**
- `X` (array-like): Training data of shape (n_samples, n_features)
- `y` (array-like): Target values of shape (n_samples,)

**Returns:**
- `self`: Returns self for method chaining

**Raises:**
- `ValueError`: If input data is invalid or incompatible

**Example:**
```python
model = CustomModel(param1=value1, param2=value2)
model.fit(X_train, y_train)
```

##### predict()
```python
@abstractmethod
def predict(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray
```

Make predictions on provided data.

**Parameters:**
- `X` (array-like): Input data of shape (n_samples, n_features)

**Returns:**
- `predictions` (ndarray): Predicted values of shape (n_samples,)

**Raises:**
- `ValueError`: If model is not fitted or input is invalid

**Example:**
```python
predictions = model.predict(X_test)
```

##### _validate_parameters()
```python
@abstractmethod
def _validate_parameters() -> None
```

Validate model parameters and raise appropriate errors.

**Raises:**
- `ValueError`: If any parameter is invalid
- `TypeError`: If parameter types are incorrect

#### Utility Methods

##### get_params()
```python
def get_params(deep: bool = True) -> Dict[str, Any]
```

Get model parameters (inherited from BaseEstimator).

**Parameters:**
- `deep` (bool): Whether to return parameters of sub-estimators

**Returns:**
- `params` (dict): Model parameters

##### set_params()
```python
def set_params(**params) -> 'IntegratedMLBaseModel'
```

Set model parameters (inherited from BaseEstimator).

**Parameters:**
- `**params`: Parameter names and values to set

**Returns:**
- `self`: Returns self for method chaining

##### save_model()
```python
def save_model(path: str) -> None
```

Save model to disk with complete state preservation.

**Parameters:**
- `path` (str): Directory path to save the model

**Example:**
```python
model.save_model('models/my_model')
```

##### load_model()
```python
@classmethod
def load_model(cls, path: str) -> 'IntegratedMLBaseModel'
```

Load model from disk with full state restoration.

**Parameters:**
- `path` (str): Directory path containing saved model

**Returns:**
- `model`: Loaded model instance

**Example:**
```python
model = CustomModel.load_model('models/my_model')
```

##### get_model_info()
```python
def get_model_info() -> Dict[str, Any]
```

Get comprehensive model information and metadata.

**Returns:**
- `info` (dict): Model information including parameters, metadata, and statistics

**Example:**
```python
info = model.get_model_info()
print(f"Model type: {info['model_type']}")
print(f"Training samples: {info['n_samples']}")
```

---

## 🎯 Classification Models

### ClassificationModel

Base class for classification models with IntegratedML integration.

```python
class ClassificationModel(IntegratedMLBaseModel, ClassifierMixin)
```

#### Constructor

```python
def __init__(self, decision_threshold: float = 0.5, **kwargs)
```

**Parameters:**
- `decision_threshold` (float): Decision threshold for binary classification (default: 0.5)
- `**kwargs`: Additional model-specific parameters

**Additional Attributes:**
- `decision_threshold` (float): Binary classification threshold
- `_label_encoder` (LabelEncoder): Internal label encoder
- `classes_` (ndarray): Unique class labels
- `n_classes_` (int): Number of classes

#### Methods

##### predict_proba()
```python
def predict_proba(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray
```

Predict class probabilities.

**Parameters:**
- `X` (array-like): Input data of shape (n_samples, n_features)

**Returns:**
- `probabilities` (ndarray): Class probabilities of shape (n_samples, n_classes)

**Example:**
```python
probabilities = model.predict_proba(X_test)
# For binary classification: [:, 0] = class 0 prob, [:, 1] = class 1 prob
```

##### predict_log_proba()
```python
def predict_log_proba(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray
```

Predict log class probabilities.

**Parameters:**
- `X` (array-like): Input data of shape (n_samples, n_features)

**Returns:**
- `log_probabilities` (ndarray): Log probabilities of shape (n_samples, n_classes)

##### decision_function()
```python
def decision_function(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray
```

Calculate decision function values.

**Parameters:**
- `X` (array-like): Input data of shape (n_samples, n_features)

**Returns:**
- `scores` (ndarray): Decision scores of shape (n_samples,) for binary or (n_samples, n_classes) for multiclass

##### predict_with_confidence()
```python
def predict_with_confidence(X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]
```

Predict with confidence scores.

**Parameters:**
- `X` (array-like): Input data of shape (n_samples, n_features)

**Returns:**
- `predictions` (ndarray): Predicted class labels
- `confidence` (ndarray): Confidence scores for predictions

**Example:**
```python
predictions, confidence = model.predict_with_confidence(X_test)
high_confidence_mask = confidence > 0.8
```

---

## 📈 Regression Models

### RegressionModel

Base class for regression models with IntegratedML integration.

```python
class RegressionModel(IntegratedMLBaseModel, RegressorMixin)
```

#### Constructor

```python
def __init__(self, confidence_level: float = 0.95, **kwargs)
```

**Parameters:**
- `confidence_level` (float): Confidence level for prediction intervals (default: 0.95)
- `**kwargs`: Additional model-specific parameters

**Additional Attributes:**
- `confidence_level` (float): Confidence level for intervals
- `_residual_std` (float): Standard deviation of residuals
- `_prediction_std` (float): Standard deviation of predictions

#### Methods

##### predict_with_interval()
```python
def predict_with_interval(X: Union[np.ndarray, pd.DataFrame], 
                         confidence_level: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Predict with confidence intervals.

**Parameters:**
- `X` (array-like): Input data of shape (n_samples, n_features)
- `confidence_level` (float, optional): Override default confidence level

**Returns:**
- `predictions` (ndarray): Point predictions
- `lower_bounds` (ndarray): Lower confidence bounds
- `upper_bounds` (ndarray): Upper confidence bounds

**Example:**
```python
predictions, lower, upper = model.predict_with_interval(X_test, confidence_level=0.95)
prediction_width = upper - lower
```

##### get_residuals()
```python
def get_residuals(X: Union[np.ndarray, pd.DataFrame], 
                  y: Union[np.ndarray, pd.Series]) -> np.ndarray
```

Calculate residuals for given data.

**Parameters:**
- `X` (array-like): Input data
- `y` (array-like): True target values

**Returns:**
- `residuals` (ndarray): Residuals (actual - predicted)

##### score()
```python
def score(X: Union[np.ndarray, pd.DataFrame], 
          y: Union[np.ndarray, pd.Series]) -> float
```

Calculate R² score.

**Parameters:**
- `X` (array-like): Input data
- `y` (array-like): True target values

**Returns:**
- `r2_score` (float): R² coefficient of determination

---

## 🎼 Ensemble Models

### EnsembleModel

Base class for ensemble models with multiple component orchestration.

```python
class EnsembleModel(IntegratedMLBaseModel)
```

#### Constructor

```python
def __init__(self, voting: str = 'hard', weights: List[float] = None, **kwargs)
```

**Parameters:**
- `voting` (str): Voting strategy ('hard', 'soft', 'weighted')
- `weights` (list, optional): Component weights for weighted voting
- `**kwargs`: Additional ensemble parameters

**Attributes:**
- `voting` (str): Voting strategy
- `weights` (list): Component weights
- `_components` (dict): Component models
- `_component_weights` (dict): Dynamic component weights

#### Methods

##### add_component()
```python
def add_component(name: str, model: IntegratedMLBaseModel, weight: float = 1.0) -> None
```

Add a component model to the ensemble.

**Parameters:**
- `name` (str): Component name/identifier
- `model` (IntegratedMLBaseModel): Model instance
- `weight` (float): Component weight (default: 1.0)

**Example:**
```python
ensemble = EnsembleModel()
ensemble.add_component('rf', RandomForestModel())
ensemble.add_component('gb', GradientBoostingModel())
```

##### remove_component()
```python
def remove_component(name: str) -> None
```

Remove a component model from the ensemble.

**Parameters:**
- `name` (str): Component name to remove

##### set_voting_weights()
```python
def set_voting_weights(weights: Dict[str, float]) -> None
```

Set voting weights for components.

**Parameters:**
- `weights` (dict): Component names and their weights

**Example:**
```python
ensemble.set_voting_weights({
    'random_forest': 0.4,
    'gradient_boosting': 0.6
})
```

##### get_component_predictions()
```python
def get_component_predictions(X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, np.ndarray]
```

Get predictions from all components.

**Parameters:**
- `X` (array-like): Input data

**Returns:**
- `component_predictions` (dict): Predictions from each component

##### get_component_info()
```python
def get_component_info() -> Dict[str, Dict[str, Any]]
```

Get information about all components.

**Returns:**
- `component_info` (dict): Information for each component

---

## 🛠️ Demo-Specific Models

### CustomCreditRiskClassifier

Credit risk assessment classifier with financial feature engineering.

```python
class CustomCreditRiskClassifier(ClassificationModel)
```

#### Constructor

```python
def __init__(self,
             enable_debt_ratio: bool = True,
             enable_interaction_terms: bool = True,
             enable_risk_scoring: bool = True,
             decision_threshold: float = 0.5,
             **kwargs)
```

**Parameters:**
- `enable_debt_ratio` (bool): Enable debt-to-income ratio features
- `enable_interaction_terms` (bool): Enable feature interaction terms
- `enable_risk_scoring` (bool): Enable custom risk scoring
- `decision_threshold` (float): Classification decision threshold
- `**kwargs`: Additional classification parameters

**Example:**
```python
model = CustomCreditRiskClassifier(
    enable_debt_ratio=True,
    enable_interaction_terms=True,
    decision_threshold=0.7
)
```

#### Methods

##### get_feature_importance()
```python
def get_feature_importance() -> Dict[str, float]
```

Get feature importance scores for credit risk factors.

**Returns:**
- `importance` (dict): Feature names and importance scores

##### get_risk_factors()
```python
def get_risk_factors(X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, np.ndarray]
```

Get detailed risk factor analysis for predictions.

**Parameters:**
- `X` (array-like): Input data

**Returns:**
- `risk_factors` (dict): Risk factor contributions

### EnsembleFraudDetector

Ensemble fraud detection system with multiple specialized detectors.

```python
class EnsembleFraudDetector(EnsembleModel)
```

#### Constructor

```python
def __init__(self,
             voting: str = 'weighted',
             confidence_threshold: float = 0.8,
             enable_rule_engine: bool = True,
             enable_anomaly_detection: bool = True,
             enable_neural_classifier: bool = True,
             enable_behavioral_analysis: bool = True,
             **kwargs)
```

**Parameters:**
- `voting` (str): Ensemble voting strategy
- `confidence_threshold` (float): Minimum confidence for fraud decisions
- `enable_rule_engine` (bool): Enable rule-based detection
- `enable_anomaly_detection` (bool): Enable anomaly detection
- `enable_neural_classifier` (bool): Enable neural network classifier
- `enable_behavioral_analysis` (bool): Enable behavioral analysis
- `**kwargs`: Additional ensemble parameters

#### Methods

##### predict_with_confidence()
```python
def predict_with_confidence(X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]
```

Predict fraud with confidence scores.

**Returns:**
- `predictions` (ndarray): Fraud predictions (0=legitimate, 1=fraud)
- `confidence` (ndarray): Confidence scores

##### get_component_contributions()
```python
def get_component_contributions(X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, np.ndarray]
```

Get individual component contributions to final decisions.

**Parameters:**
- `X` (array-like): Input data

**Returns:**
- `contributions` (dict): Component-wise fraud scores

### HybridForecastingModel

Sales forecasting model combining Prophet and LightGBM.

```python
class HybridForecastingModel(RegressionModel)
```

#### Constructor

```python
def __init__(self,
             trend_model: str = 'prophet',
             ml_model: str = 'lightgbm',
             forecast_horizon: int = 12,
             seasonal_periods: List[str] = None,
             external_regressors: List[str] = None,
             **kwargs)
```

**Parameters:**
- `trend_model` (str): Trend/seasonality model ('prophet', 'arima')
- `ml_model` (str): ML model ('lightgbm', 'xgboost', 'rf')
- `forecast_horizon` (int): Number of periods to forecast
- `seasonal_periods` (list): Seasonal periods to model
- `external_regressors` (list): External variables to include
- `**kwargs`: Additional forecasting parameters

#### Methods

##### predict_with_components()
```python
def predict_with_components(X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, np.ndarray]
```

Predict with trend/seasonal component decomposition.

**Parameters:**
- `X` (array-like): Input data

**Returns:**
- `components` (dict): Forecast components (trend, seasonal, residual)

##### make_future_dataframe()
```python
def make_future_dataframe(periods: int, freq: str = 'D') -> pd.DataFrame
```

Create future dataframe for forecasting.

**Parameters:**
- `periods` (int): Number of future periods
- `freq` (str): Frequency string ('D', 'M', 'Y')

**Returns:**
- `future_df` (DataFrame): Future periods dataframe

---

## ⚙️ Configuration Reference

### Model Configuration Parameters

#### Common Parameters

All models support these base parameters:

```yaml
# Base model configuration
model_name: "CustomModel"
version: "1.0.0"

# Training parameters
random_state: 42
verbose: true
debug_mode: false

# Performance parameters  
n_jobs: -1
memory_limit_gb: 8
batch_size: 1000

# Validation parameters
validation_split: 0.2
cross_validation_folds: 5
stratify: true
```

#### Classification Parameters

```yaml
# Classification-specific parameters
decision_threshold: 0.5
class_weight: "balanced"
probability_calibration: true

# Multi-class parameters
multi_class: "ovr"  # 'ovr', 'multinomial'
average: "weighted"  # for metrics

# Imbalanced data handling
sampling_strategy: "auto"
resampling_method: "smote"
```

#### Regression Parameters

```yaml
# Regression-specific parameters
confidence_level: 0.95
prediction_intervals: true
residual_analysis: true

# Uncertainty quantification
uncertainty_method: "bootstrap"  # 'bootstrap', 'bayesian'
n_bootstrap_samples: 1000
```

#### Ensemble Parameters

```yaml
# Ensemble configuration
voting: "weighted"  # 'hard', 'soft', 'weighted'
component_weights:
  model_1: 0.4
  model_2: 0.6

# Meta-learning
enable_stacking: true
meta_model: "linear_regression"
cross_validation_meta: true

# Component selection
dynamic_selection: true
selection_strategy: "confidence_based"
```

### Feature Engineering Configuration

```yaml
# Feature engineering settings
feature_engineering:
  scaling:
    method: "standard"  # 'standard', 'minmax', 'robust'
    per_feature: false
    
  selection:
    method: "mutual_info"  # 'mutual_info', 'f_score', 'chi2'
    k_best: 50
    threshold: 0.01
    
  transformation:
    polynomial_features: false
    interaction_terms: true
    log_transform: ["feature_1", "feature_2"]
    
  encoding:
    categorical_method: "onehot"  # 'onehot', 'label', 'target'
    handle_unknown: "ignore"
    drop_first: true
```

### Performance Configuration

```yaml
# Performance settings
performance:
  caching:
    enable: true
    cache_size_mb: 1000
    ttl_seconds: 3600
    
  parallelization:
    backend: "threading"  # 'threading', 'multiprocessing'
    max_workers: 4
    chunk_size: 100
    
  memory:
    low_memory_mode: false
    memory_map_features: true
    garbage_collection: "auto"
    
  monitoring:
    enable_profiling: false
    log_predictions: true
    track_drift: true
```

---

## 🔧 Utility Functions

### Data Validation

```python
def validate_input_data(X: Union[np.ndarray, pd.DataFrame], 
                       feature_names: List[str] = None,
                       check_finite: bool = True) -> np.ndarray
```

Validate input data for model training/prediction.

**Parameters:**
- `X` (array-like): Input data to validate
- `feature_names` (list, optional): Expected feature names
- `check_finite` (bool): Check for infinite/NaN values

**Returns:**
- `X_validated` (ndarray): Validated input data

**Raises:**
- `ValueError`: If validation fails

### Model Serialization

```python
def serialize_model_state(model: IntegratedMLBaseModel) -> Dict[str, Any]
```

Serialize model state for storage.

**Parameters:**
- `model` (IntegratedMLBaseModel): Model to serialize

**Returns:**
- `state` (dict): Serializable model state

```python
def deserialize_model_state(model_class: Type[IntegratedMLBaseModel], 
                           state: Dict[str, Any]) -> IntegratedMLBaseModel
```

Deserialize model state from storage.

**Parameters:**
- `model_class` (type): Model class to instantiate
- `state` (dict): Serialized model state

**Returns:**
- `model` (IntegratedMLBaseModel): Restored model instance

### Performance Utilities

```python
def benchmark_model_performance(model: IntegratedMLBaseModel,
                              X_test: np.ndarray,
                              n_iterations: int = 100) -> Dict[str, float]
```

Benchmark model prediction performance.

**Parameters:**
- `model` (IntegratedMLBaseModel): Model to benchmark
- `X_test` (ndarray): Test data for predictions
- `n_iterations` (int): Number of benchmark iterations

**Returns:**
- `metrics` (dict): Performance metrics (latency, throughput, etc.)

### Metric Calculators

```python
def calculate_classification_metrics(y_true: np.ndarray, 
                                   y_pred: np.ndarray,
                                   y_prob: np.ndarray = None) -> Dict[str, float]
```

Calculate comprehensive classification metrics.

**Parameters:**
- `y_true` (ndarray): True labels
- `y_pred` (ndarray): Predicted labels  
- `y_prob` (ndarray, optional): Predicted probabilities

**Returns:**
- `metrics` (dict): Classification metrics

```python
def calculate_regression_metrics(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_lower: np.ndarray = None,
                               y_upper: np.ndarray = None) -> Dict[str, float]
```

Calculate comprehensive regression metrics.

**Parameters:**
- `y_true` (ndarray): True values
- `y_pred` (ndarray): Predicted values
- `y_lower` (ndarray, optional): Lower confidence bounds
- `y_upper` (ndarray, optional): Upper confidence bounds

**Returns:**
- `metrics` (dict): Regression metrics

---

## 🚨 Error Handling

### Exception Classes

#### ModelNotFittedError
```python
class ModelNotFittedError(ValueError):
    """Raised when prediction is attempted on unfitted model."""
    pass
```

#### ParameterValidationError
```python
class ParameterValidationError(ValueError):
    """Raised when model parameters are invalid."""
    pass
```

#### FeatureMismatchError
```python
class FeatureMismatchError(ValueError):
    """Raised when feature count/names don't match training data."""
    pass
```

#### SerializationError
```python
class SerializationError(Exception):
    """Raised when model serialization/deserialization fails."""
    pass
```

### Error Handling Patterns

```python
try:
    model = CustomModel(invalid_param="bad_value")
except ParameterValidationError as e:
    print(f"Parameter validation failed: {e}")

try:
    predictions = model.predict(X_test)
except ModelNotFittedError:
    print("Model must be fitted before prediction")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

try:
    model.save_model("models/my_model")
except SerializationError as e:
    print(f"Failed to save model: {e}")
```

---

## 📋 Best Practices

### Parameter Validation

```python
def _validate_parameters(self):
    """Best practices for parameter validation."""
    super()._validate_parameters()
    
    # Type validation with clear messages
    if not isinstance(self.learning_rate, (int, float)):
        raise ParameterValidationError(
            f"learning_rate must be numeric, got {type(self.learning_rate)}"
        )
    
    # Range validation with specific bounds
    if not 0 < self.learning_rate <= 1:
        raise ParameterValidationError(
            f"learning_rate must be in range (0, 1], got {self.learning_rate}"
        )
    
    # Cross-parameter validation
    if self.max_depth is not None and self.n_estimators < 10:
        raise ParameterValidationError(
            "n_estimators should be >= 10 when max_depth is specified"
        )
```

### Input Validation

```python
def _validate_input(self, X, check_fitted=False):
    """Best practices for input validation."""
    if check_fitted and not self.is_fitted:
        raise ModelNotFittedError("Model must be fitted before prediction")
    
    # Handle different input types
    if hasattr(X, 'values'):  # pandas DataFrame
        feature_names = X.columns.tolist()
        X = X.values
    else:
        feature_names = None
    
    X = np.asarray(X)
    
    # Comprehensive validation
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D array")
    
    if X.shape[0] == 0:
        raise ValueError("Empty input array")
    
    if hasattr(self, 'n_features_in_') and X.shape[1] != self.n_features_in_:
        raise FeatureMismatchError(
            f"Expected {self.n_features_in_} features, got {X.shape[1]}"
        )
    
    # Check for problematic values
    if not np.isfinite(X).all():
        raise ValueError("Input contains non-finite values (inf/nan)")
    
    return X
```

### Model Lifecycle

```python
def fit(self, X, y):
    """Best practices for model training."""
    # Always validate inputs first
    X = self._validate_input(X)
    y = self._validate_target(y)
    
    # Store feature metadata
    self.n_features_in_ = X.shape[1]
    if hasattr(X, 'columns'):
        self.feature_names_in_ = X.columns.tolist()
    
    # Log training start
    logger.info("Training started", 
                model_class=self.__class__.__name__,
                n_samples=len(X),
                n_features=X.shape[1])
    
    try:
        # Actual model training
        self._fit_model(X, y)
        
        # Mark as fitted and store metadata
        self.is_fitted = True
        self._model_metadata = self._create_metadata(X, y)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error("Training failed", error=str(e))
        raise
    
    return self
```

This comprehensive API reference provides complete documentation for all classes, methods, and configuration options in the IntegratedML Pluggable Models Demo, enabling developers to effectively build, deploy, and maintain custom ML models in database environments.