# Research: IntegratedML Custom Models Platform

**Feature**: IntegratedML Custom Models Platform
**Date**: 2025-10-10
**Status**: Complete

## Overview

This document consolidates research findings for implementing a custom machine learning model platform within InterSystems IRIS database. All technical decisions have been validated against existing codebase patterns and constitutional principles.

## Technology Stack Decisions

### Decision: Python 3.8+ as Model Implementation Language

**Rationale**:
- IRIS 2025.2 embedded Python runtime requires Python 3.8+ compatibility
- Scikit-learn 1.3+ ecosystem provides mature ML libraries
- Pandas/NumPy integration essential for data manipulation
- Python enables rapid model development with extensive ML library support

**Alternatives Considered**:
- **Java-based ML**: Rejected due to limited IntegratedML support and verbose syntax for feature engineering
- **R integration**: Rejected due to lack of native IRIS embedded R runtime
- **ObjectScript ML**: Rejected due to limited ML library ecosystem compared to Python

**Implementation Notes**:
- Use type hints for shared/models/ base classes (mypy validation)
- Black formatting with 88-character line length for consistency
- flake8 linting with E203/W503 exceptions for Black compatibility

---

### Decision: Scikit-learn Interface as Base Model Contract

**Rationale**:
- IntegratedML requires fit/predict interface for model lifecycle management
- get_params/set_params enable parameter introspection and grid search compatibility
- Standardized interface simplifies testing and validation
- Enables interchangeability between custom and built-in scikit-learn models

**Alternatives Considered**:
- **Custom interface**: Rejected because IntegratedML explicitly requires scikit-learn compatibility
- **TensorFlow/Keras interface**: Rejected because IntegratedML uses scikit-learn conventions
- **PyTorch interface**: Rejected for same reason as TensorFlow

**Implementation Notes**:
- IntegratedMLBaseModel extends sklearn.base.BaseEstimator
- Abstract methods enforce fit/predict implementation in subclasses
- Parameter validation in _validate_parameters() hook
- State serialization via _get_model_state()/_set_model_state()

---

### Decision: JSON USING Clause for Parameter Passing (IRIS 2025.2 Syntax)

**Rationale**:
- IRIS 2025.2 introduces JSON-based parameter syntax replacing quoted Python paths
- Enables structured parameter validation before model instantiation
- Supports complex nested parameters (dicts, arrays) not possible with quoted strings
- Aligns with modern SQL standards for JSON data handling

**Alternatives Considered**:
- **Old quoted syntax**: Rejected as deprecated in IRIS 2025.2; less type-safe
- **Environment variables**: Rejected due to lack of per-model isolation
- **Config files**: Rejected because parameters must be specified in SQL for audit trails

**Implementation Example**:
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

---

### Decision: Four-Tiered Base Model Class Hierarchy

**Rationale**:
- IntegratedMLBaseModel: Abstract foundation with serialization, validation, preprocessing hooks
- ClassificationModel: Binary/multi-class classification with predict_proba support
- RegressionModel: Continuous value prediction with scoring metrics
- EnsembleModel: Multi-model composition with weighted voting

**Alternatives Considered**:
- **Single base class**: Rejected because classification and regression have different validation/output requirements
- **Flat structure (no inheritance)**: Rejected due to code duplication across 30+ shared methods
- **Deep hierarchy (>4 levels)**: Rejected to avoid over-engineering and maintain simplicity

**Implementation Notes**:
- Base class handles: parameter management, state serialization, input validation, logging
- Classification adds: decision thresholds, probability predictions, class label handling
- Regression adds: continuous output validation, residual calculations
- Ensemble adds: sub-model management, voting strategies, confidence thresholds

---

### Decision: Domain-Specific Demo Structure (demos/{domain}/)

**Rationale**:
- Each demo independently testable (credit risk, fraud, forecasting, DNA)
- Domain-specific feature engineering isolated in demo model classes
- Enables parallel development of demos without conflicts
- Clear separation of concerns: shared/ for common code, demos/ for specializations

**Alternatives Considered**:
- **Monolithic models/ directory**: Rejected due to lack of domain organization
- **Per-model repositories**: Rejected because demos share common base classes
- **Flat structure**: Rejected due to mixing unrelated domain logic

**Implementation Pattern**:
```
demos/credit_risk/
├── models/credit_risk_classifier.py     # Domain logic
├── data/generate_sample_data.py         # Test data specific to domain
└── tests/test_integration.py            # Domain-specific integration tests
```

---

### Decision: Docker Compose for IRIS Environment

**Rationale**:
- Reproducible database environment across development machines
- IntegratedML installation scriptable in Dockerfile
- Simplifies iris_automl symlink creation (required workaround)
- Enables CI/CD integration with consistent test environment

**Alternatives Considered**:
- **Local IRIS installation**: Rejected due to developer setup complexity and version drift
- **Kubernetes**: Rejected as over-engineered for development/demo environment
- **Virtual machines**: Rejected due to resource overhead and slower iteration

**Implementation Requirements**:
- Dockerfile installs intersystems-iris-automl from InterSystems registry
- Symlink created: `/usr/irissys/mgr/python/iris_automl` → `/opt/irisapp/data/mgr/python/iris_automl`
- Environment variables (.env) configure connection: host, port, namespace, credentials
- Volume mounts preserve database state across container restarts

---

### Decision: pytest with IRIS Integration for Testing

**Rationale**:
- pytest fixtures manage IRIS connection lifecycle
- Performance benchmarks integrated as test assertions (latency <50ms)
- Supports parameterized tests for multiple model configurations
- E2E test suite validates all 4 demos in single run

**Alternatives Considered**:
- **unittest**: Rejected due to less expressive fixture system
- **nose2**: Rejected due to declining ecosystem support
- **Manual test scripts**: Rejected due to lack of structured reporting

**Implementation Pattern**:
```python
# tests/conftest.py provides IRIS connection fixture
@pytest.fixture
def iris_connection():
    conn = connect_iris(...)
    yield conn
    conn.close()

# Integration tests use fixture
def test_credit_risk_prediction(iris_connection):
    iris_connection.execute("CREATE MODEL ...")
    result = iris_connection.execute("SELECT PREDICT(...)")
    assert result.latency_ms < 50
```

---

### Decision: Pickle-Based Model State Serialization

**Rationale**:
- IRIS IntegratedML persists models via pickle serialization
- Supports complex Python objects (pipelines, encoders, fitted transformers)
- Built-in versioning via pickle protocol versions
- Efficient binary format for trained model state

**Alternatives Considered**:
- **JSON serialization**: Rejected because scikit-learn objects not JSON-serializable
- **ONNX format**: Rejected due to loss of preprocessing pipeline state
- **Custom binary format**: Rejected to avoid reinventing pickle functionality

**Implementation Requirements**:
- _get_model_state() returns dict with all trained components
- _set_model_state() restores components from dict
- Version compatibility handled via try/except with fallback defaults
- Model metadata includes pickle protocol version for debugging

---

### Decision: Feature Engineering Within Model Methods

**Rationale**:
- Ensures identical transformations during fit and predict
- Prevents feature name mismatches between training and prediction
- Enables domain-specific transformations not available in sklearn
- Centralizes preprocessing logic with model definition

**Alternatives Considered**:
- **External preprocessing scripts**: Rejected due to fit/predict inconsistency risk
- **SQL-based feature engineering**: Rejected to maintain Python-centric workflow
- **Separate transformer classes**: Rejected due to increased complexity for single-model use case

**Implementation Pattern**:
```python
def _engineer_features(self, X):
    """Apply custom domain logic"""
    X_engineered = X.copy()
    if self.enable_debt_ratio:
        X_engineered = self._add_debt_ratio_features(X_engineered)
    if self.enable_interaction_terms:
        X_engineered = self._add_interaction_terms(X_engineered)
    return X_engineered

def fit(self, X, y):
    X_engineered = self._engineer_features(X)  # Consistent transformation
    self._pipeline.fit(X_engineered, y)

def predict(self, X):
    X_engineered = self._engineer_features(X)  # Identical transformation
    return self._pipeline.predict(X_engineered)
```

---

## Performance Optimization Research

### Decision: Sub-50ms Prediction Latency Target

**Rationale**:
- Production fraud detection requires real-time responses
- Credit risk decisions happen during customer interactions
- 50ms allows 20 predictions/second per model
- Aligns with industry standards for interactive ML applications

**Optimization Strategies**:
- Minimize feature engineering computation in predict path
- Use efficient NumPy operations for vectorized calculations
- Avoid disk I/O during predictions (preload fitted pipelines)
- Profile predict() methods to identify bottlenecks

**Measurement Approach**:
- pytest benchmarks with `pytest-benchmark` plugin
- p95 latency tracked across 1000 prediction requests
- Integration tests fail if latency exceeds 50ms threshold

---

## Testing Strategy Research

### Decision: Three-Tier Testing Approach

**Tier 1 - Unit Tests**:
- Validate individual model methods (feature engineering, validation)
- Mock IRIS connections to test model logic in isolation
- Fast execution (<1s per test suite)

**Tier 2 - Integration Tests**:
- Validate SQL CREATE MODEL / PREDICT integration
- Require running IRIS database (via Docker Compose)
- Measure training time and prediction latency
- Verify state persistence across model saves/loads

**Tier 3 - End-to-End Tests**:
- Single test file (test_all_demos_e2e.py) validates all 4 demos
- Generates test data, trains models, validates predictions
- Runs as final gate before releases
- Documents actual performance metrics in test output

**Implementation Notes**:
- Test data generators produce realistic volumes (1K-100K records)
- Fixtures manage IRIS connection lifecycle
- Performance assertions fail tests if latency exceeds thresholds

---

## Integration Pattern Research

### Decision: Shared Base Classes with Demo-Specific Extensions

**Pattern**:
1. shared/models/base.py: Core IntegratedMLBaseModel with common functionality
2. shared/models/{classification,regression,ensemble}.py: Type-specific extensions
3. demos/{domain}/models/*.py: Domain-specific models extending base classes

**Benefits**:
- 70% code reuse across demos (serialization, validation, preprocessing hooks)
- Domain experts extend base classes without modifying shared code
- New demos added by creating new demos/{domain}/ directory
- Base class improvements benefit all demos automatically

**Trade-offs**:
- Requires understanding base class contracts (fit/predict/get_params/set_params)
- Breaking changes to base classes affect all demos
- Mitigated by: comprehensive base class tests, semantic versioning, deprecation warnings

---

## Dependency Management Research

### Decision: pyproject.toml with uv or pip

**Rationale**:
- pyproject.toml: Modern Python standard (PEP 518/621)
- uv: Fast dependency resolution (recommended for development)
- pip: Fallback for environments without uv
- Lockfile (requirements.txt) ensures reproducible builds

**Core Dependencies**:
- scikit-learn 1.3+ (base estimator interface)
- pandas 2.0+ (data manipulation)
- numpy 1.24+ (numerical operations)
- intersystems-iris-automl (IntegratedML integration)
- LightGBM 4.0+ (sales forecasting demo)
- Prophet 1.1+ (time-series seasonality)

**Development Dependencies**:
- pytest 7.4+ (testing framework)
- black (code formatting)
- flake8 (linting)
- mypy (type checking for shared/)

---

## Summary

All technical decisions documented above are implemented in the existing codebase. No unresolved clarifications remain. The architecture supports:

1. **In-Database ML**: Models execute within IRIS via SQL commands
2. **Scikit-learn Compatibility**: All models follow estimator interface
3. **Test-Driven Development**: Unit, integration, and E2E tests in place
4. **Low-Latency Performance**: Sub-50ms prediction target with benchmarks
5. **State Management**: Pickle-based serialization for model persistence

**Ready for Phase 1: Design & Contracts generation**
