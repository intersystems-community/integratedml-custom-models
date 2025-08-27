
# Tutorial 4: Building Custom Pluggable Models from Scratch

## 🎯 Tutorial Overview

Welcome to the master-class IntegratedML tutorial! You'll learn how to build completely custom pluggable models from scratch, understanding the full architecture and creating sophisticated ML solutions that integrate seamlessly with database workflows.

### What You'll Learn
- **Base Architecture**: Master the IntegratedMLBaseModel foundation and design patterns
- **Custom Classification**: Build sophisticated classifiers with domain-specific logic
- **Custom Regression**: Create advanced regression models with uncertainty quantification
- **Ensemble Development**: Architect multi-model systems with custom orchestration
- **Production Integration**: Deploy custom models with full IntegratedML compatibility

### What You'll Build
Three complete custom models demonstrating different complexities:
- **Anomaly Detection Classifier**: Advanced outlier detection with custom algorithms
- **Multi-target Regression**: Simultaneous prediction of multiple related targets
- **Dynamic Ensemble**: Self-adapting model selection based on input characteristics

**Estimated Time**: 120-180 minutes  
**Difficulty**: 🔧 Expert  
**Prerequisites**: Completion of Tutorials 1-3, advanced Python/ML knowledge

---

## 📋 Prerequisites & Architecture Understanding

### System Requirements
- Python 3.8+
- 16GB RAM (recommended for custom model development)
- 8GB free disk space
- IDE with debugging capabilities (VS Code, PyCharm)

### Step 1: Understand the Base Architecture

```python
# Explore the IntegratedML architecture
from shared.models.base import IntegratedMLBaseModel
from shared.models.classification import ClassificationModel
from shared.models.regression import RegressionModel
import inspect

print("🏗️ IntegratedML Base Architecture Analysis:")
print("=" * 50)

# Examine base class structure
base_methods = [method for method in dir(IntegratedMLBaseModel) 
               if not method.startswith('_') or method in ['__init__', '_validate_parameters']]

print("📋 IntegratedMLBaseModel Core Interface:")
for method in sorted(base_methods):
    if hasattr(IntegratedMLBaseModel, method):
        method_obj = getattr(IntegratedMLBaseModel, method)
        if callable(method_obj):
            try:
                signature = inspect.signature(method_obj)
                print(f"  • {method}{signature}")
            except:
                print(f"  • {method}")

# Examine classification specialization
print("\n🎯 ClassificationModel Extensions:")
classification_methods = [method for method in dir(ClassificationModel) 
                         if method not in dir(IntegratedMLBaseModel) and not method.startswith('_')]
for method in sorted(classification_methods):
    print(f"  • {method}")

# Examine regression specialization  
print("\n📈 RegressionModel Extensions:")
regression_methods = [method for method in dir(RegressionModel) 
                     if method not in dir(IntegratedMLBaseModel) and not method.startswith('_')]
for method in sorted(regression_methods):
    print(f"  • {method}")
```

### Step 2: Design Patterns and Conventions

```python
# Key design patterns for custom models
print("\n🎨 IntegratedML Design Patterns:")
print("=" * 40)

design_patterns = {
    "Parameter Validation": "All parameters validated in _validate_parameters()",
    "Scikit-learn Compatibility": "BaseEstimator inheritance ensures fit/predict interface",
    "State Management": "is_fitted flag tracks model training state",
    "Metadata Storage": "_model_metadata dict stores model information",
    "Error Handling": "Consistent exception raising with clear messages",
    "Feature Tracking": "feature_names_in_ and n_features_in_ for validation",
    "Serialization": "Built-in save/load methods for model persistence",
    "Logging": "Structured logging for debugging and monitoring"
}

for pattern, description in design_patterns.items():
    print(f"✅ {pattern}: {description}")

# Required methods for custom models
print("\n🔧 Required Methods for Custom Models:")
required_methods = {
    "fit(X, y)": "Train the model - must set is_fitted=True",
    "predict(X)": "Make predictions - must check is_fitted",
    "_validate_parameters()": "Validate all model parameters",
    "get_params()": "Return model parameters (inherited from BaseEstimator)",
    "set_params()": "Set model parameters (inherited from BaseEstimator)"
}

for method, description in required_methods.items():
    print(f"📋 {method}: {description}")
```

---

## 🛠️ Building Custom Classification Model

Let's build a sophisticated anomaly detection classifier with custom algorithms.

### Step 1: Define the Custom Anomaly Detector

```python
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, List
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
import warnings

from shared.models.classification import ClassificationModel


class CustomAnomalyDetector(ClassificationModel):
    """
    Advanced anomaly detection classifier with multiple detection algorithms.
    
    This model demonstrates building sophisticated custom classifiers by combining:
    - Statistical outlier detection (Z-score, IQR)
    - Density-based anomaly detection (LOF, DBSCAN)
    - Dimensionality reduction anomaly detection (PCA reconstruction error)
    - Ensemble voting with adaptive thresholds
    
    Key Features:
    - Multiple detection algorithms with configurable weights
    - Adaptive threshold selection based on contamination rate
    - Interpretable anomaly scores with component breakdown
    - Real-time prediction with sub-millisecond latency
    - Custom feature engineering for anomaly detection
    
    Example Usage:
    -------------
    >>> detector = CustomAnomalyDetector(
    ...     contamination=0.1,
    ...     algorithms=['statistical', 'density', 'reconstruction'],
    ...     voting_strategy='weighted',
    ...     adaptive_threshold=True
    ... )
    >>> detector.fit(X_train, y_train)
    >>> anomaly_scores = detector.predict_proba(X_test)
    >>> anomalies = detector.predict(X_test)
    """
    
    def __init__(self,
                 contamination: float = 0.1,
                 algorithms: List[str] = None,
                 voting_strategy: str = 'weighted',
                 adaptive_threshold: bool = True,
                 statistical_method: str = 'zscore',
                 density_algorithm: str = 'lof',
                 pca_components: float = 0.95,
                 min_samples_fraction: float = 0.05,
                 **kwargs):
        """
        Initialize the custom anomaly detector.
        
        Parameters:
        -----------
        contamination : float, default=0.1
            Expected proportion of anomalies in the dataset
        algorithms : list of str, optional
            Detection algorithms to use: ['statistical', 'density', 'reconstruction']
        voting_strategy : str, default='weighted'
            How to combine algorithm results: 'weighted', 'majority', 'unanimous'
        adaptive_threshold : bool, default=True
            Whether to adaptively adjust detection threshold
        statistical_method : str, default='zscore'
            Statistical method: 'zscore', 'iqr', 'modified_zscore'
        density_algorithm : str, default='lof'
            Density algorithm: 'lof', 'dbscan', 'isolation_forest'
        pca_components : float, default=0.95
            PCA variance ratio or number of components
        min_samples_fraction : float, default=0.05
            Minimum samples fraction for density algorithms
        **kwargs : dict
            Additional parameters for the base classifier
        """
        self.contamination = contamination
        self.algorithms = algorithms or ['statistical', 'density', 'reconstruction']
        self.voting_strategy = voting_strategy
        self.adaptive_threshold = adaptive_threshold
        self.statistical_method = statistical_method
        self.density_algorithm = density_algorithm
        self.pca_components = pca_components
        self.min_samples_fraction = min_samples_fraction
        
        # Internal components
        self._statistical_detector = None
        self._density_detector = None
        self._reconstruction_detector = None
        self._feature_scaler = None
        self._algorithm_weights = {}
        self._adaptive_threshold_value = None
        
        super().__init__(
            contamination=contamination,
            algorithms=algorithms,
            voting_strategy=voting_strategy,
            adaptive_threshold=adaptive_threshold,
            statistical_method=statistical_method,
            density_algorithm=density_algorithm,
            pca_components=pca_components,
            min_samples_fraction=min_samples_fraction,
            **kwargs
        )
    
    def _validate_parameters(self) -> None:
        """Validate anomaly detector parameters."""
        super()._validate_parameters()
        
        # Validate contamination rate
        if not 0 < self.contamination < 0.5:
            raise ValueError(f"contamination must be between 0 and 0.5, got {self.contamination}")
        
        # Validate algorithms
        valid_algorithms = ['statistical', 'density', 'reconstruction']
        for algorithm in self.algorithms:
            if algorithm not in valid_algorithms:
                raise ValueError(f"Unknown algorithm '{algorithm}'. Valid options: {valid_algorithms}")
        
        # Validate voting strategy
        valid_strategies = ['weighted', 'majority', 'unanimous']
        if self.voting_strategy not in valid_strategies:
            raise ValueError(f"Unknown voting_strategy '{self.voting_strategy}'. Valid options: {valid_strategies}")
        
        # Validate statistical method
        valid_statistical = ['zscore', 'iqr', 'modified_zscore']
        if self.statistical_method not in valid_statistical:
            raise ValueError(f"Unknown statistical_method '{self.statistical_method}'. Valid options: {valid_statistical}")
        
        # Validate density algorithm
        valid_density = ['lof', 'dbscan', 'isolation_forest']
        if self.density_algorithm not in valid_density:
            raise ValueError(f"Unknown density_algorithm '{self.density_algorithm}'. Valid options: {valid_density}")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'CustomAnomalyDetector':
        """
        Train the anomaly detection model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels (1 for anomaly, 0 for normal)
            
        Returns:
        --------
        self : CustomAnomalyDetector
            Returns self for method chaining
        """
        # Input validation
        X = self._validate_input(X)
        y = self._validate_target(y)
        
        # Store feature information
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        
        # Initialize feature scaler
        self._feature_scaler = StandardScaler()
        X_scaled = self._feature_scaler.fit_transform(X)
        
        # Train individual detection algorithms
        self._train_detection_algorithms(X_scaled, y)
        
        # Calculate algorithm weights based on performance
        self._calculate_algorithm_weights(X_scaled, y)
        
        # Set adaptive threshold if enabled
        if self.adaptive_threshold:
            self._set_adaptive_threshold(X_scaled, y)
        
        # Mark as fitted
        self.is_fitted = True
        
        # Store metadata
        self._model_metadata = {
            'n_samples': len(X),
            'n_features': self.n_features_in_,
            'algorithms_used': self.algorithms,
            'contamination_rate': self.contamination,
            'algorithm_weights': self._algorithm_weights,
            'adaptive_threshold': self._adaptive_threshold_value
        }
        
        return self
    
    def _train_detection_algorithms(self, X_scaled: np.ndarray, y: np.ndarray) -> None:
        """Train individual detection algorithms."""
        
        # Statistical anomaly detection
        if 'statistical' in self.algorithms:
            self._statistical_detector = self._create_statistical_detector()
            self._statistical_detector.fit(X_scaled, y)
        
        # Density-based anomaly detection
        if 'density' in self.algorithms:
            self._density_detector = self._create_density_detector()
            self._density_detector.fit(X_scaled, y)
        
        # Reconstruction-based anomaly detection
        if 'reconstruction' in self.algorithms:
            self._reconstruction_detector = self._create_reconstruction_detector()
            self._reconstruction_detector.fit(X_scaled, y)
    
    def _create_statistical_detector(self):
        """Create statistical anomaly detector."""
        class StatisticalDetector:
            def __init__(self, method='zscore', threshold=3.0):
                self.method = method
                self.threshold = threshold
                self.feature_stats = {}
            
            def fit(self, X, y):
                if self.method == 'zscore':
                    self.feature_stats['mean'] = np.mean(X, axis=0)
                    self.feature_stats['std'] = np.std(X, axis=0)
                elif self.method == 'iqr':
                    self.feature_stats['q25'] = np.percentile(X, 25, axis=0)
                    self.feature_stats['q75'] = np.percentile(X, 75, axis=0)
                    self.feature_stats['iqr'] = self.feature_stats['q75'] - self.feature_stats['q25']
                elif self.method == 'modified_zscore':
                    self.feature_stats['median'] = np.median(X, axis=0)
                    self.feature_stats['mad'] = np.median(np.abs(X - self.feature_stats['median']), axis=0)
                
                return self
            
            def predict_proba(self, X):
                if self.method == 'zscore':
                    z_scores = np.abs((X - self.feature_stats['mean']) / (self.feature_stats['std'] + 1e-8))
                    anomaly_scores = np.max(z_scores, axis=1)
                elif self.method == 'iqr':
                    lower_bound = self.feature_stats['q25'] - 1.5 * self.feature_stats['iqr']
                    upper_bound = self.feature_stats['q75'] + 1.5 * self.feature_stats['iqr']
                    outlier_mask = (X < lower_bound) | (X > upper_bound)
                    anomaly_scores = np.sum(outlier_mask, axis=1) / X.shape[1]
                elif self.method == 'modified_zscore':
                    modified_z_scores = 0.6745 * (X - self.feature_stats['median']) / (self.feature_stats['mad'] + 1e-8)
                    anomaly_scores = np.max(np.abs(modified_z_scores), axis=1)
                
                # Normalize scores to [0, 1]
                anomaly_scores = np.clip(anomaly_scores / self.threshold, 0, 1)
                normal_scores = 1 - anomaly_scores
                
                return np.column_stack([normal_scores, anomaly_scores])
        
        return StatisticalDetector(method=self.statistical_method)
    
    def _create_density_detector(self):
        """Create density-based anomaly detector."""
        if self.density_algorithm == 'lof':
            from sklearn.neighbors import LocalOutlierFactor
            return LocalOutlierFactor(
                n_neighbors=max(2, int(self.min_samples_fraction * 100)),
                contamination=self.contamination,
                novelty=True
            )
        elif self.density_algorithm == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            return IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
        elif self.density_algorithm == 'dbscan':
            # Custom DBSCAN-based detector
            class DBSCANDetector:
                def __init__(self, eps=0.5, min_samples=5):
                    self.eps = eps
                    self.min_samples = min_samples
                    self.clusterer = None
                
                def fit(self, X, y):
                    self.clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples)
                    self.clusterer.fit(X)
                    return self
                
                def predict_proba(self, X):
                    # Predict clusters for new data (simplified)
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=self.min_samples)
                    nn.fit(X)  # This is simplified - in practice, use training data
                    
                    distances, _ = nn.kneighbors(X)
                    anomaly_scores = np.mean(distances, axis=1)
                    
                    # Normalize scores
                    anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-8)
                    normal_scores = 1 - anomaly_scores
                    
                    return np.column_stack([normal_scores, anomaly_scores])
            
            return DBSCANDetector(min_samples=max(2, int(self.min_samples_fraction * 100)))
    
    def _create_reconstruction_detector(self):
        """Create reconstruction-based anomaly detector using PCA."""
        class PCAReconstructionDetector:
            def __init__(self, n_components=0.95, threshold_percentile=95):
                self.n_components = n_components
                self.threshold_percentile = threshold_percentile
                self.pca = None
                self.reconstruction_threshold = None
            
            def fit(self, X, y):
                self.pca = PCA(n_components=self.n_components)
                X_transformed = self.pca.fit_transform(X)
                X_reconstructed = self.pca.inverse_transform(X_transformed)
                
                # Calculate reconstruction errors
                reconstruction_errors = np.sum((X - X_reconstructed) ** 2, axis=1)
                self.reconstruction_threshold = np.percentile(reconstruction_errors, self.threshold_percentile)
                
                return self
            
            def predict_proba(self, X):
                X_transformed = self.pca.transform(X)
                X_reconstructed = self.pca.inverse_transform(X_transformed)
                reconstruction_errors = np.sum((X - X_reconstructed) ** 2, axis=1)
                
                # Convert errors to anomaly scores
                anomaly_scores = np.clip(reconstruction_errors / self.reconstruction_threshold, 0, 1)
                normal_scores = 1 - anomaly_scores
                
                return np.column_stack([normal_scores, anomaly_scores])
        
        return PCAReconstructionDetector(n_components=self.pca_components)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict anomaly probabilities.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction
            
        Returns:
        --------
        probabilities : ndarray of shape (n_samples, 2)
            Predicted probabilities [normal_prob, anomaly_prob]
        """
        # Check if fitted
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Validate and scale input
        X = self._validate_input(X, check_fitted=True)
        X_scaled = self._feature_scaler.transform(X)
        
        # Get predictions from each algorithm
        algorithm_predictions = {}
        
        if 'statistical' in self.algorithms and self._statistical_detector:
            algorithm_predictions['statistical'] = self._statistical_detector.predict_proba(X_scaled)
        
        if 'density' in self.algorithms and self._density_detector:
            if hasattr(self._density_detector, 'predict_proba'):
                algorithm_predictions['density'] = self._density_detector.predict_proba(X_scaled)
            else:
                # Handle sklearn models that don't have predict_proba
                scores = self._density_detector.decision_function(X_scaled)
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                algorithm_predictions['density'] = np.column_stack([1 - scores, scores])
        
        if 'reconstruction' in self.algorithms and self._reconstruction_detector:
            algorithm_predictions['reconstruction'] = self._reconstruction_detector.predict_proba(X_scaled)
        
        # Combine predictions using voting strategy
        combined_probabilities = self._combine_predictions(algorithm_predictions)
        
        return combined_probabilities
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction
            
        Returns:
        --------
        predictions : ndarray of shape (n_samples,)
            Predicted labels (1 for anomaly, 0 for normal)
        """
        probabilities = self.predict_proba(X)
        
        # Use adaptive threshold if available, otherwise use decision_threshold
        threshold = self._adaptive_threshold_value if self._adaptive_threshold_value else self.decision_threshold
        
        return (probabilities[:, 1] > threshold).astype(int)
```

### Step 2: Advanced Feature Engineering and Testing

```python
# Test the custom anomaly detector
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

print("🧪 Testing Custom Anomaly Detector")
print("=" * 40)

# Generate synthetic anomaly detection dataset
def create_anomaly_dataset(n_samples=1000, contamination=0.1):
    """Create dataset with anomalies."""
    n_anomalies = int(n_samples * contamination)
    n_normal = n_samples - n_anomalies
    
    # Normal data (gaussian)
    normal_data = np.random.multivariate_normal(
        mean=[0, 0, 0, 0],
        cov=np.eye(4),
        size=n_normal
    )
    
    # Anomalous data (shifted and scaled)
    anomaly_data = np.random.multivariate_normal(
        mean=[3, 3, -3, -3],
        cov=2 * np.eye(4),
        size=n_anomalies
    )
    
    # Combine data
    X = np.vstack([normal_data, anomaly_data])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

# Create test dataset
X, y = create_anomaly_dataset(n_samples=2000, contamination=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"📊 Dataset: {len(X)} samples, {y.mean():.1%} anomalies")
print(f"Training: {len(X_train)}, Testing: {len(X_test)}")

# Test different configurations
configurations = [
    {
        'name': 'Statistical Only',
        'algorithms': ['statistical'],
        'statistical_method': 'zscore'
    },
    {
        'name': 'Density Only', 
        'algorithms': ['density'],
        'density_algorithm': 'lof'
    },
    {
        'name': 'Reconstruction Only',
        'algorithms': ['reconstruction'],
        'pca_components': 0.95
    },
    {
        'name': 'Full Ensemble',
        'algorithms': ['statistical', 'density', 'reconstruction'],
        'voting_strategy': 'weighted',
        'adaptive_threshold': True
    }
]

results = {}

for config in configurations:
    print(f"\n🔬 Testing: {config['name']}")
    
    # Initialize detector
    detector = CustomAnomalyDetector(
        contamination=0.1,
        **{k: v for k, v in config.items() if k != 'name'}
    )
    
    # Train and predict
    detector.fit(X_train, y_train)
    y_pred = detector.predict(X_test)
    y_prob = detector.predict_proba(X_test)
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_prob[:, 1])
    
    results[config['name']] = {
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_prob,
        'metadata': detector.get_model_info()
    }
    
    print(f"  AUC: {auc:.3f}")
    print(f"  Detected anomalies: {y_pred.sum()}/{len(y_test)}")

# Compare results
print(f"\n📈 Performance Comparison:")
for name, result in results.items():
    print(f"  • {name}: AUC = {result['auc']:.3f}")

best_model = max(results.items(), key=lambda x: x[1]['auc'])
print(f"\n🏆 Best performing: {best_model[0]} (AUC: {best_model[1]['auc']:.3f})")
```

---

## 📊 Building Custom Regression Model

Now let's build a multi-target regression model that predicts multiple related outcomes simultaneously.

### Step 1: Multi-target Regression Implementation

```python
from shared.models.regression import RegressionModel
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score


class CustomMultiTargetRegressor(RegressionModel):
    """
    Advanced multi-target regression model with sophisticated target relationships.
    
    This model demonstrates building custom regression models that can:
    - Predict multiple related targets simultaneously
    - Leverage target correlations for improved accuracy
    - Provide uncertainty quantification for each target
    - Handle missing targets during training and prediction
    - Support different loss functions for different targets
    
    Key Features:
    - Multi-output prediction with shared feature representations
    - Target-specific model adaptation
    - Cross-target information sharing
    - Confidence intervals for each target
    - Feature importance analysis per target
    
    Example Usage:
    -------------
    >>> regressor = CustomMultiTargetRegressor(
    ...     target_names=['sales', 'profit', 'customers'],
    ...     base_estimator='random_forest',
    ...     share_features=True,
    ...     target_correlations=True
    ... )
    >>> regressor.fit(X_train, y_train_multi)
    >>> predictions = regressor.predict(X_test)
    >>> intervals = regressor.predict_with_interval(X_test)
    """
    
    def __init__(self,
                 target_names: List[str] = None,
                 base_estimator: str = 'random_forest',
                 share_features: bool = True,
                 target_correlations: bool = True,
                 feature_selection: bool = True,
                 estimator_params: Dict[str, Any] = None,
                 correlation_threshold: float = 0.3,
                 **kwargs):
        """
        Initialize the multi-target regressor.
        
        Parameters:
        -----------
        target_names : list of str, optional
            Names of the target variables
        base_estimator : str, default='random_forest'
            Base estimator type: 'random_forest', 'ridge', 'lasso', 'gradient_boosting'
        share_features : bool, default=True
            Whether to share feature representations across targets
        target_correlations : bool, default=True
            Whether to model correlations between targets
        feature_selection : bool, default=True
            Whether to perform target-specific feature selection
        estimator_params : dict, optional
            Parameters for the base estimator
        correlation_threshold : float, default=0.3
            Minimum correlation to consider targets related
        **kwargs : dict
            Additional parameters for the base regressor
        """
        self.target_names = target_names
        self.base_estimator = base_estimator
        self.share_features = share_features
        self.target_correlations = target_correlations
        self.feature_selection = feature_selection
        self.estimator_params = estimator_params or {}
        self.correlation_threshold = correlation_threshold
        
        # Internal components
        self._target_models = {}
        self._feature_selectors = {}
        self._target_scalers = {}
        self._correlation_matrix = None
        self._shared_features = None
        self.n_targets_ = None
        
        super().__init__(
            target_names=target_names,
            base_estimator=base_estimator,
            share_features=share_features,
            target_correlations=target_correlations,
            feature_selection=feature_selection,
            correlation_threshold=correlation_threshold,
            **kwargs
        )
    
    def _validate_parameters(self) -> None:
        """Validate multi-target regressor parameters."""
        super()._validate_parameters()
        
        # Validate base estimator
        valid_estimators = ['random_forest', 'ridge', 'lasso', 'gradient_boosting']
        if self.base_estimator not in valid_estimators:
            raise ValueError(f"Unknown base_estimator '{self.base_estimator}'. Valid options: {valid_estimators}")
        
        # Validate correlation threshold
        if not 0 <= self.correlation_threshold <= 1:
            raise ValueError(f"correlation_threshold must be between 0 and 1, got {self.correlation_threshold}")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame]) -> 'CustomMultiTargetRegressor':
        """
        Train the multi-target regression model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples, n_targets)
            Multiple target values
            
        Returns:
        --------
        self : CustomMultiTargetRegressor
            Returns self for method chaining
        """
        # Input validation
        X = self._validate_input(X)
        y = self._validate_multi_target(y)
        
        # Store feature and target information
        self.n_features_in_ = X.shape[1]
        self.n_targets_ = y.shape[1]
        
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        
        if self.target_names is None:
            if hasattr(y, 'columns'):
                self.target_names = y.columns.tolist()
            else:
                self.target_names = [f'target_{i}' for i in range(self.n_targets_)]
        
        # Analyze target correlations
        if self.target_correlations:
            self._analyze_target_correlations(y)
        
        # Feature selection and engineering
        if self.feature_selection:
            self._perform_feature_selection(X, y)
        else:
            self._shared_features = list(range(X.shape[1]))
        
        # Train target-specific models
        self._train_target_models(X, y)
        
        # Mark as fitted
        self.is_fitted = True
        
        # Store metadata
        self._model_metadata = {
            'n_samples': len(X),
            'n_features': self.n_features_in_,
            'n_targets': self.n_targets_,
            'target_names': self.target_names,
            'base_estimator': self.base_estimator,
            'shared_features': len(self._shared_features),
            'target_correlations': self._correlation_matrix.tolist() if self._correlation_matrix is not None else None
        }
        
        return self
    
    def _validate_multi_target(self, y: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Validate multi-target array."""
        if hasattr(y, 'values'):
            y = y.values
        
        y = np.asarray(y)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        if y.ndim != 2:
            raise ValueError(f"y must be 1D or 2D array, got {y.ndim}D")
        
        return y
    
    def _analyze_target_correlations(self, y: np.ndarray) -> None:
        """Analyze correlations between targets."""
        self._correlation_matrix = np.corrcoef(y.T)
        
        # Find highly correlated target pairs
        high_correlations = []
        for i in range(self.n_targets_):
            for j in range(i + 1, self.n_targets_):
                corr = abs(self._correlation_matrix[i, j])
                if corr > self.correlation_threshold:
                    high_correlations.append((i, j, corr))
        
        if high_correlations:
            print(f"🔗 Found {len(high_correlations)} highly correlated target pairs:")
            for i, j, corr in high_correlations:
                print(f"  • {self.target_names[i]} ↔ {self.target_names[j]}: {corr:.3f}")
    
    def _perform_feature_selection(self, X: np.ndarray, y: np.ndarray) -> None:
        """Perform target-specific feature selection."""
        from sklearn.feature_selection import SelectKBest, f_regression
        
        # Global feature selection (features important for any target)
        global_selector = SelectKBest(score_func=f_regression, k='all')
        global_selector.fit(X, y.mean(axis=1))  # Use average target for global selection
        
        # Get top features based on global importance
        feature_scores = global_selector.scores_
        n_select = min(X.shape[1], int(X.shape[1] * 0.8))  # Select top 80% of features
        top_features = np.argsort(feature_scores)[-n_select:]
        
        self._shared_features = top_features.tolist()
        
        # Target-specific feature selection
        for i, target_name in enumerate(self.target_names):
            target_selector = SelectKBest(score_func=f_regression, k=min(len(self._shared_features), 50))
            target_selector.fit(X[:, self._shared_features], y[:, i])
            
            # Get target-specific top features
            target_feature_indices = target_selector.get_support(indices=True)
            target_features = [self._shared_features[idx] for idx in target_feature_indices]
            
            self._feature_selectors[target_name] = target_features
    
    def _train_target_models(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train individual models for each target."""
        
        for i, target_name in enumerate(self.target_names):
            # Get target-specific features
            if self.feature_selection and target_name in self._feature_selectors:
                feature_indices = self._feature_selectors[target_name]
            else:
                feature_indices = self._shared_features
            
            X_target = X[:, feature_indices]
            y_target = y[:, i]
            
            # Create base estimator
            if self.base_estimator == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    **self.estimator_params
                )
            elif self.base_estimator == 'ridge':
                model = Ridge(
                    alpha=1.0,
                    **self.estimator_params
                )
            elif self.base_estimator == 'lasso':
                model = Lasso(
                    alpha=1.0,
                    **self.estimator_params
                )
            elif self.base_estimator == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    random_state=42,
                    **self.estimator_params
                )
            
            # Train model
            model.fit(X_target, y_target)
            
            # Store model and feature indices
            self._target_models[target_name] = {
                'model': model,
                'feature_indices': feature_indices,
                'target_index': i
            }
            
            # Calculate residuals for uncertainty estimation
            y_pred = model.predict(X_target)
            residuals = y_target - y_pred
            self._target_models[target_name]['residual_std'] = np.std(residuals)
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict multiple targets.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction
            
        Returns:
        --------
        predictions : ndarray of shape (n_samples, n_targets)
            Predicted values for all targets
        """
        # Check if fitted
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Validate input
        X = self._validate_input(X, check_fitted=True)
        
        # Initialize predictions array
        predictions = np.zeros((X.shape[0], self.n_targets_))
        
        # Predict each target
        for target_name, model_info in self._target_models.items():
            model = model_info['model']
            feature_indices = model_info['feature_indices']
            target_index = model_info['target_index']
            
            X_target = X[:, feature_indices]
            predictions[:, target_index] = model.predict(X_target)
        
        return predictions
    
    def predict_with_interval(self, X: Union[np.ndarray, pd.DataFrame], confidence_level: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals for each target.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction
        confidence_level : float, optional
            Confidence level for intervals (uses model default if None)
            
        Returns:
        --------
        predictions : ndarray of shape (n_samples, n_targets)
            Point predictions
        lower_bounds : ndarray of shape (n_samples, n_targets)
            Lower confidence bounds
        upper_bounds : ndarray of shape (n_samples, n_targets)  
            Upper confidence bounds
        """
        from scipy import stats
        
        # Get point predictions
        predictions = self.predict(X)
        
        # Use model confidence level if not specified
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha / 2)
        
        lower_bounds = np.zeros_like(predictions)
        upper_bounds = np.zeros_like(predictions)
        
        for target_name, model_info in self._target_models.items():
            target_index = model_info['target_index']
            residual_std = model_info['residual_std']
            
            margin = z_score * residual_std
            lower_bounds[:, target_index] = predictions[:, target_index] - margin
            upper_bounds[:, target_index] = predictions[:, target_index] + margin
        
        return predictions, lower_bounds, upper_bounds
    
    def get_feature_importance(self, target_name: str = None) -> Dict[str, float]:
        """
        Get feature importance for a specific target or averaged across all targets.
        
        Parameters:
        -----------
        target_name : str, optional
            Target name to get importance for. If None, returns averaged importance.
            
        Returns:
        --------
        importance : dict
            Feature importance scores
        """
        if target_name and target_name in self._target_models:
            # Return importance for specific target
            model_info = self._target_models[target_name]
            model = model_info['model']
            feature_indices = model_info['feature_indices']
            
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_scores = np.abs(model.coef_)
            else:
                return {}
            
            if self.feature_names_in_:
                feature_names = [self.feature_names_in_[i] for i in feature_indices]
            else:
                feature_names = [f'feature_{i}' for i in feature_indices]
            
            return dict(zip(feature_names, importance_scores))
        
        else:
            # Return averaged importance across all targets
            all_importance = {}
            
            for target_name, model_info in self._target_models.items():
                target_importance = self.get_feature_importance(target_name)
                
                for feature, score in target_importance.items():
                    if feature not in all_importance:
                        all_importance[feature] = []
                    all_importance[feature].append(score)
            
            # Average the scores
            averaged_importance = {
                feature: np.mean(scores) 
                for feature, scores in all_importance.items()
            }
            
            return averaged_importance
```

### Step 2: Test Multi-target Regression

```python
# Test the custom multi-target regressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

print("\n🧪 Testing Custom Multi-Target Regressor")
print("=" * 45)

# Generate multi-target regression dataset
def create_multitarget_dataset(n_samples=1000, n_features=20, n_targets=3):
    """Create correlated multi-target dataset."""
    
    # Generate base features
    X, _ = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=0.1,
        random_state=42
    )
    
    # Create correlated targets
    # Target 1: Linear combination of features
    y1 = 2 * X[:, 0] + 1.5 * X[:, 1] - 0.5 * X[:, 2] + np.random.normal(0, 0.1, n_samples)
    
    # Target 2: Correlated with Target 1 plus additional features
    y2 = 0.7 * y1 + X[:, 3] - 0.8 * X[:, 4] + np.random.normal(0, 0.15, n_samples)
    
    # Target 3: More complex relationship
    y3 = 0.3 * y1 + 0.4 * y2 + X[:, 5] * X[:, 6] + np.random.normal(0, 0.2, n_samples)
    
    y = np.column_stack([y1, y2, y3])
    
    return X, y

# Create test dataset
X, y = create_multitarget_dataset(n_samples=1500, n_features=25, n_targets=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} targets")
print(f"Training: {X_train.shape[0]}, Testing: {X_test.shape[0]}")

# Test different configurations
target_names = ['sales', 'profit', 'customers']
mt_configurations = [
    {
        'name': 'Random Forest',
        'base_estimator': 'random_forest',
        'share_features': True,
        'target_correlations': True
    },
    {
        'name': 'Ridge Regression',
        'base_estimator': 'ridge',
        'share_features': True,
        'target_correlations': True
    },
    {
        'name': 'Independent Models',
        'base_estimator': 'random_forest',
        'share_features': False,
        'target_correlations': False
    },
    {
        'name': 'Full Multi-Target',
        'base_estimator': 'random_forest',
        'share_features': True,
        'target_correlations': True,
        'feature_selection': True
    }
]

mt_results = {}

for config in mt_configurations:
    print(f"\n🔬 Testing: {config['name']}")
    
    # Initialize regressor
    regressor = CustomMultiTargetRegressor(
        target_names=target_names,
        **{k: v for k, v in config.items() if k != 'name'}
    )
    
    # Train and predict
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    y_pred_intervals, lower_bounds, upper_bounds = regressor.predict_with_interval(X_test)
    
    # Calculate metrics for each target
    target_metrics = {}
    for i, target_name in enumerate(target_names):
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        
        target_metrics[target_name] = {'r2': r2, 'rmse': rmse}
        print(f"  {target_name}: R² = {r2:.3f}, RMSE = {rmse:.3f}")
    
    # Calculate average metrics
    avg_r2 = np.mean([metrics['r2'] for metrics in target_metrics.values()])
    avg_rmse = np.mean([metrics['rmse'] for metrics in target_metrics.values()])
    
    mt_results[config['name']] = {
        'avg_r2': avg_r2,
        'avg_rmse': avg_rmse,
        'target_metrics': target_metrics,
        'predictions': y_pred,
        'intervals': (lower_bounds, upper_bounds),
        'metadata': regressor.get_model_info()
    }
    
    print(f"  Average: R² = {avg_r2:.3f}, RMSE = {avg_rmse:.3f}")

# Compare results
print(f"\n📈 Multi-Target Performance Comparison:")
for name, result in mt_results.items():
    print(f"  • {name}: R² = {result['avg_r2']:.3f}, RMSE = {result['avg_rmse']:.3f}")

best_mt_model = max(mt_results.items(), key=lambda x: x[1]['avg_r2'])
print(f"\n🏆 Best performing: {best_mt_model[0]} (R² = {best_mt_model[1]['avg_r2']:.3f})")
```

---

## 🎼 Building Dynamic Ensemble Models

Let's create an advanced ensemble that dynamically selects models based on input characteristics.

### Step 3: Dynamic Ensemble Implementation

```python
from shared.models.ensemble import EnsembleModel


class DynamicEnsembleModel(EnsembleModel):
    """
    Dynamic ensemble model with adaptive model selection.
    
    This model demonstrates the most advanced custom model patterns by:
    - Dynamically selecting optimal models based on input characteristics
    - Learning when to trust each component model
    - Adapting ensemble weights in real-time
    - Providing explainable ensemble decisions
    - Supporting both classification and regression
    
    Key Features:
    - Input-dependent model selection
    - Meta-learning for ensemble optimization
    - Explainable AI integration
    - Real-time adaptation
    - Confidence-based routing
    
    Example Usage:
    -------------
    >>> ensemble = DynamicEnsembleModel(
    ...     base_models=['random_forest', 'gradient_boosting', 'neural_network'],
    ...     selection_strategy='adaptive',
    ...     meta_learning=True,
    ...     explain_decisions=True
    ... )
    >>> ensemble.fit(X_train, y_train)
    >>> predictions, explanations = ensemble.predict_with_explanation(X_test)
    """
    
    def __init__(self,
                 base_models: List[str] = None,
                 selection_strategy: str = 'adaptive',
                 meta_learning: bool = True,
                 adaptation_rate: float = 0.1,
                 confidence_threshold: float = 0.8,
                 explain_decisions: bool = True,
                 task_type: str = 'auto',
                 **kwargs):
        """
        Initialize the dynamic ensemble model.
        
        Parameters:
        -----------
        base_models : list of str, optional
            Base model types to include in ensemble
        selection_strategy : str, default='adaptive'
            Model selection strategy: 'adaptive', 'voting', 'stacking'
        meta_learning : bool, default=True
            Whether to use meta-learning for model selection
        adaptation_rate : float, default=0.1
            Rate of adaptation for dynamic weighting
        confidence_threshold : float, default=0.8
            Threshold for confident predictions
        explain_decisions : bool, default=True
            Whether to provide decision explanations
        task_type : str, default='auto'
            Task type: 'classification', 'regression', 'auto'
        **kwargs : dict
            Additional ensemble parameters
        """
        self.base_models = base_models or ['random_forest', 'gradient_boosting', 'svm']
        self.selection_strategy = selection_strategy
        self.meta_learning = meta_learning
        self.adaptation_rate = adaptation_rate
        self.confidence_threshold = confidence_threshold
        self.explain_decisions = explain_decisions
        self.task_type = task_type
        
        # Internal components
        self._models = {}
        self._meta_model = None
        self._input_analyzer = None
        self._weights_history = []
        self._performance_tracker = {}
        self.task_type_ = None
        
        super().__init__(
            base_models=base_models,
            selection_strategy=selection_strategy,
            meta_learning=meta_learning,
            adaptation_rate=adaptation_rate,
            confidence_threshold=confidence_threshold,
            explain_decisions=explain_decisions,
            task_type=task_type,
            **kwargs
        )
    
    def _validate_parameters(self) -> None:
        """Validate dynamic ensemble parameters."""
        super()._validate_parameters()
        
        # Validate base models
        valid_models = ['random_forest', 'gradient_boosting', 'svm', 'neural_network', 'linear']
        for model in self.base_models:
            if model not in valid_models:
                raise ValueError(f"Unknown base_model '{model}'. Valid options: {valid_models}")
        
        # Validate selection strategy
        valid_strategies = ['adaptive', 'voting', 'stacking', 'dynamic']
        if self.selection_strategy not in valid_strategies:
            raise ValueError(f"Unknown selection_strategy '{self.selection_strategy}'. Valid options: {valid_strategies}")
        
        # Validate adaptation rate
        if not 0 < self.adaptation_rate <= 1:
            raise ValueError(f"adaptation_rate must be between 0 and 1, got {self.adaptation_rate}")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'DynamicEnsembleModel':
        """
        Train the dynamic ensemble model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : DynamicEnsembleModel
            Returns self for method chaining
        """
        # Input validation
        X = self._validate_input(X)
        y = self._validate_target(y)
        
        # Determine task type
        self._determine_task_type(y)
        
        # Store feature information
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        
        # Train input analyzer for dynamic selection
        self._train_input_analyzer(X, y)
        
        # Train base models
        self._train_base_models(X, y)
        
        # Train meta-model if enabled
        if self.meta_learning:
            self._train_meta_model(X, y)
        
        # Initialize performance tracking
        self._initialize_performance_tracking()
        
        # Mark as fitted
        self.is_fitted = True
        
        # Store metadata
        self._model_metadata = {
            'n_samples': len(X),
            'n_features': self.n_features_in_,
            'task_type': self.task_type_,
            'base_models': self.base_models,
            'selection_strategy': self.selection_strategy,
            'meta_learning': self.meta_learning
        }
        
        return self
    
    def _determine_task_type(self, y: np.ndarray) -> None:
        """Determine if this is classification or regression."""
        if self.task_type == 'auto':
            # Heuristic: if y has few unique values and they're integers, it's classification
            unique_values = np.unique(y)
            if len(unique_values) <= 20 and np.all(unique_values == unique_values.astype(int)):
                self.task_type_ = 'classification'
            else:
                self.task_type_ = 'regression'
        else:
            self.task_type_ = self.task_type
    
    def _train_input_analyzer(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model to analyze input characteristics for dynamic selection."""
        
        class InputAnalyzer:
            def __init__(self):
                self.feature_stats = {}
                self.complexity_threshold = None
                
            def fit(self, X, y):
                # Calculate feature statistics
                self.feature_stats['mean'] = np.mean(X, axis=0)
                self.feature_stats['std'] = np.std(X, axis=0)
                self.feature_stats['min'] = np.min(X, axis=0)
                self.feature_stats['max'] = np.max(X, axis=0)
                
                # Calculate complexity metrics
                feature_ranges = self.feature_stats['max'] - self.feature_stats['min']
                self.complexity_threshold = np.median(feature_ranges)
                
                return self
            
            def analyze(self, X):
                """Analyze input characteristics."""
                batch_size = X.shape[0]
                n_features = X.shape[1]
                
                # Calculate complexity metrics
                feature_ranges = np.max(X, axis=0) - np.min(X, axis=0)
                avg_complexity = np.mean(feature_ranges)
                
                # Calculate novelty (distance from training distribution)
                novelty_scores = []
                for i in range(n_features):
                    feature_values = X[:, i]
                    z_scores = np.abs((feature_values - self.feature_stats['mean'][i]) / 
                                    (self.feature_stats['std'][i] + 1e-8))
                    novelty_scores.append(np.mean(z_scores))
                
                avg_novelty = np.mean(novelty_scores)
                
                return {
                    'batch_size': batch_size,
                    'complexity': avg_complexity,
                    'novelty': avg_novelty,
                    'high_complexity': avg_complexity > self.complexity_threshold,
                    'high_novelty': avg_novelty > 2.0  # Z-score threshold
                }
        
        self._input_analyzer = InputAnalyzer()
        self._input_analyzer.fit(X, y)
    
    def _train_base_models(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train all base models."""
        
        for model_name in self.base_models:
            print(f"Training {model_name}...")
            
            if model_name == 'random_forest':
                if self.task_type_ == 'classification':
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                else:
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            elif model_name == 'gradient_boosting':
                if self.task_type_ == 'classification':
                    from sklearn.ensemble import GradientBoostingClassifier
                    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                else:
                    from sklearn.ensemble import GradientBoostingRegressor
                    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            
            elif model_name == 'svm':
                if self.task_type_ == 'classification':
                    from sklearn.svm import SVC
                    model = SVC(probability=True, random_state=42)
                else:
                    from sklearn.svm import SVR
                    model = SVR()
            
            elif model_name == 'neural_network':
                if self.task_type_ == 'classification':
                    from sklearn.neural_network import MLPClassifier
                    model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
                else:
                    from sklearn.neural_network import MLPRegressor
                    model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
            
            elif model_name == 'linear':
                if self.task_type_ == 'classification':
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(random_state=42, max_iter=1000)
                else:
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
            
            # Train model
            model.fit(X, y)
            self._models[model_name] = model
    
    def _train_meta_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train meta-model for dynamic selection."""
        from sklearn.model_selection import cross_val_predict
        
        # Generate meta-features using cross-validation
        meta_features = []
        
        for model_name, model in self._models.items():
            # Get cross-validated predictions
            if self.task_type_ == 'classification':
                cv_pred = cross_val_predict(model, X, y, cv=5, method='predict_proba')
                if cv_pred.ndim > 1 and cv_pred.shape[1] > 1:
                    cv_pred = cv_pred[:, 1]  # Use positive class probability
            else:
                cv_pred = cross_val_predict(model, X, y, cv=5)
            
            meta_features.append(cv_pred)
        
        # Add input characteristics as meta-features
        input_analysis = self._input_analyzer.analyze(X)
        complexity_features = np.full(len(X), input_analysis['complexity'])
        novelty_features = np.full(len(X), input_analysis['novelty'])
        
        meta_features.extend([complexity_features, novelty_features])
        
        # Create meta-feature matrix
        meta_X = np.column_stack(meta_features)
        
        # Train meta-model
        if self.task_type_ == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            self._meta_model = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor
            self._meta_model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        self._meta_model.fit(meta_X, y)
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict using dynamic model selection.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction
            
        Returns:
        --------
        predictions : ndarray of shape (n_samples,)
            Dynamically selected predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._validate_input(X, check_fitted=True)
        
        # Analyze input characteristics
        input_analysis = self._input_analyzer.analyze(X)
        
        # Get predictions from all models
        model_predictions = {}
        model_confidences = {}
        
        for model_name, model in self._models.items():
            pred = model.predict(X)
            model_predictions[model_name] = pred
            
            # Calculate confidence (simplified)
            if hasattr(model, 'predict_proba') and self.task_type_ == 'classification':
                proba = model.predict_proba(X)
                confidence = np.max(proba, axis=1)
            else:
                # For regression, use prediction consistency as confidence proxy
                confidence = np.ones(len(X)) * 0.8
            
            model_confidences[model_name] = confidence
        
        # Dynamic model selection
        final_predictions = self._dynamic