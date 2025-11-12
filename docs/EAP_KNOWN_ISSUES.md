# IntegratedML Custom Models - Known Issues & Limitations (EAP)

**Program Status**: Early Access Program (EAP)
**Last Updated**: 2025-01-12
**Target GA Release**: IRIS 2026.1

---

## Purpose

This document lists **current limitations, known bugs, and workarounds** for IntegratedML Custom Models during the Early Access Program. Please review this document before reporting issues to avoid duplicate reports.

**Before reporting a bug**, check:
1. ✅ Is it listed in this document?
2. ✅ Is there a documented workaround?
3. ✅ Have you tried the troubleshooting guide?

If the answer is "no" to all three, please report via [feedback channels](EAP_GUIDE.md#how-to-provide-feedback).

---

## Table of Contents

- [Known Limitations](#known-limitations)
  - [Model Type Limitations](#model-type-limitations)
  - [Platform Limitations](#platform-limitations)
  - [Development Workflow Limitations](#development-workflow-limitations)
  - [Performance Limitations](#performance-limitations)
  - [Documentation Gaps](#documentation-gaps)
- [Known Bugs](#known-bugs)
  - [Installation Issues](#installation-issues)
  - [Runtime Issues](#runtime-issues)
  - [SQL Integration Issues](#sql-integration-issues)
  - [Demo-Specific Issues](#demo-specific-issues)
- [Workarounds](#workarounds)
- [Fixed in Future Releases](#fixed-in-future-releases)
- [How to Report New Issues](#how-to-report-new-issues)

---

## Known Limitations

These are **intentional design limitations** in the current EAP release. Some may be addressed in future releases based on user feedback.

### Model Type Limitations

#### 1. Timeseries Models Not Fully Supported

**Issue**: Direct timeseries model integration (e.g., ARIMA, Prophet as standalone models) is not yet fully supported in the pluggable models architecture.

**Impact**: You cannot directly plug in pure timeseries models that expect sequential data without additional wrapper logic.

**Workaround**:
- ✅ Use the **Sales Forecasting demo** as a reference - it shows how to use Prophet within a hybrid model
- ✅ Create a wrapper class that converts IRIS tabular data to timeseries format
- ✅ Combine timeseries models with traditional ML models (as shown in `HybridForecastingModel`)

**Status**: Under investigation for GA release. Feedback welcome on desired timeseries model integration patterns.

**Example Workaround** (from Sales Forecasting demo):
```python
class HybridForecastingModel(RegressionModel):
    """Combines Prophet (timeseries) with LightGBM (regression)"""

    def fit(self, X, y):
        # Convert tabular data to Prophet format
        prophet_data = self._prepare_prophet_data(X, y)
        self.prophet_model.fit(prophet_data)

        # Use Prophet predictions as features for LightGBM
        prophet_predictions = self.prophet_model.predict(...)
        enhanced_features = self._add_prophet_features(X, prophet_predictions)
        self.lgbm_model.fit(enhanced_features, y)
```

---

#### 2. Model Name Uniqueness Requirement

**Issue**: Model names must be globally unique across all custom models (classifiers and regressors).

**Impact**: If two models have the same class name (even in different files), the last one loaded will override the first.

**Workaround**:
- ✅ Use descriptive, unique class names: `CreditRiskClassifier`, `FraudEnsembleDetector`, etc.
- ✅ Include domain or use case in model name: `SalesForecastHybridModel`
- ❌ Avoid generic names: `Model`, `Classifier`, `Predictor`

**Status**: This is a current architecture constraint. May be relaxed in GA with namespace support.

---

#### 3. Model Interface Requirements Strictly Enforced

**Issue**: Custom models **must** implement all required methods of the scikit-learn-like interface:

Required methods:
- `fit(X, y)` - Train the model
- `predict(X)` - Make predictions
- `predict_proba(X)` - Predict class probabilities (classification only)
- `get_params(deep=True)` - Get model parameters
- `set_params(**params)` - Set model parameters

**Impact**: Models missing any required method will fail at training or prediction time with unclear error messages.

**Workaround**:
- ✅ Inherit from `ClassificationModel`, `RegressionModel`, or `EnsembleModel` base classes (recommended)
- ✅ Review base class implementations in `shared/models/` for correct patterns
- ✅ Test all required methods before deploying to IRIS

**Status**: Working as designed. Enhanced error messages planned for GA.

---

### Platform Limitations

#### 4. Primary Support for macOS

**Issue**: EAP testing has been primarily conducted on macOS. Linux and Windows support is secondary.

**Impact**:
- Installation may require platform-specific troubleshooting on Linux/Windows
- Some demo data generation scripts may have platform-specific path issues
- Docker setup is most reliable across platforms

**Workaround**:
- ✅ **Recommended**: Use Docker setup (`make setup`) for most reliable cross-platform experience
- ✅ Linux users: Generally works well, minor path issues possible
- ⚠️ Windows users: Use WSL2 or Docker for best results

**Status**: Full multi-platform testing planned before GA. Please report platform-specific issues!

**Tested Platforms**:
- ✅ macOS 13+ (Ventura, Sonoma) - Primary
- ⚠️ Ubuntu 22.04 LTS - Secondary testing
- ⚠️ Windows 11 + WSL2 - Limited testing
- ⚠️ Windows 11 + Docker Desktop - Limited testing

---

#### 5. Python Version Requirements

**Issue**: Requires Python 3.8 or later. Python 3.11+ recommended for full AutoML compatibility.

**Impact**: Older Python installations (3.6, 3.7) are not supported.

**Workaround**:
- ✅ Use `pyenv` or `conda` to install Python 3.8+
- ✅ Docker setup includes correct Python version automatically

**Status**: This is a dependency requirement and will not change. Python 3.8+ is required.

---

### Development Workflow Limitations

#### 6. Terminal/IRIS Restart Required After Model Changes

**Issue**: After modifying a custom model Python file, you must **restart the IRIS terminal** (or IRIS instance) for changes to take effect.

**Impact**: Iterative development is slower - each model change requires a restart.

**Workaround**:
- ✅ Develop and test models in standard Python/Jupyter environment first
- ✅ Unit test your model with pytest before deploying to IRIS
- ✅ Only deploy to IRIS once model logic is working
- ⚠️ For IRIS testing, restart the terminal after each model update:
  ```bash
  # In IRIS terminal
  halt
  # Then reconnect or restart IRIS container
  docker restart iml-custom-models-iris
  ```

**Status**: This is a current architecture limitation. Hot-reload functionality is being investigated for GA.

**Example Development Workflow**:
```bash
# 1. Develop model locally
cd demos/my_demo/
pytest tests/test_my_model.py  # Unit test outside IRIS

# 2. Deploy to IRIS
cp models/my_model.py /path/to/iris/mgr/python/custom_models/

# 3. Restart IRIS
docker restart iml-custom-models-iris

# 4. Test in SQL
# ... run SQL commands ...
```

---

#### 7. InterSystems AutoML Models Can Be Modified

**Issue**: InterSystems' built-in AutoML models are exposed as `.py` files in `/iris/Mgr/python/AutoML/`. You can modify or remove these files.

**Impact**:
- ⚠️ Modifying built-in models can break AutoML functionality
- ⚠️ Removing built-in models will cause AutoML to fail
- ⚠️ Changes persist and affect all users of the IRIS instance

**Workaround**:
- ❌ **Do NOT modify files in** `/iris/Mgr/python/AutoML/Classifiers/` or `Regressors/`
- ✅ Place custom models in a separate directory (e.g., `/iris/Mgr/python/custom_models/`)
- ✅ Use `pathtoclassifiers` and `pathtoregressors` parameters to point to custom directories

**Status**: This is a current architecture design. Better isolation planned for GA.

**Recommended Directory Structure**:
```
/iris/Mgr/python/
├── AutoML/                     # DO NOT MODIFY
│   ├── Classifiers/            # Built-in AutoML classifiers
│   └── Regressors/             # Built-in AutoML regressors
└── custom_models/              # Your custom models
    ├── classifiers/
    │   ├── credit_risk_classifier.py
    │   └── fraud_detector.py
    └── regressors/
        └── sales_forecaster.py
```

---

### Performance Limitations

#### 8. Large Model Training May Time Out

**Issue**: Training very large models (>1GB in memory, >1 hour training time) may exceed SQL timeout limits.

**Impact**: SQL `TRAIN MODEL` command may time out before training completes.

**Workaround**:
- ✅ Pre-train large models outside IRIS using standard Python
- ✅ Save trained model using pickle/joblib
- ✅ Load pre-trained model in custom model's `__init__()` method
- ✅ Override `fit()` to skip training if model is already trained
- ⚠️ For production: Consider incremental learning or model updates outside IRIS

**Status**: This is a SQL execution timeout limitation. Async training is being considered for GA.

**Example Pre-Trained Model Pattern**:
```python
class PreTrainedClassifier(ClassificationModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load pre-trained model
        model_path = kwargs.get('pretrained_model_path')
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self._is_fitted = True
        else:
            self.model = MyLargeModel()
            self._is_fitted = False

    def fit(self, X, y):
        if self._is_fitted:
            return self  # Skip training if already fitted
        # Otherwise train as normal
        self.model.fit(X, y)
        return self
```

---

#### 9. Model File Size Limits

**Issue**: Very large custom model files (>100MB) may cause slow load times or memory issues.

**Impact**: IRIS may take a long time to load large model files, affecting `TRAIN MODEL` performance.

**Workaround**:
- ✅ Keep model code separate from model weights
- ✅ Load model weights from external files in `__init__()`
- ✅ Use efficient serialization (joblib, pickle) for model weights
- ✅ Consider model compression techniques

**Status**: This is a general Python module loading limitation. Best practices documentation will be expanded for GA.

---

### Documentation Gaps

#### 10. Limited Advanced Examples

**Issue**: Current demos cover common use cases, but advanced patterns (model ensembles, custom metrics, complex preprocessing) have limited documentation.

**Impact**: Users attempting advanced use cases may need to reverse-engineer patterns from demo code.

**Workaround**:
- ✅ Review demo source code in `demos/*/models/` for patterns
- ✅ Check `shared/models/` for base class implementations
- ✅ Consult API reference documentation
- ✅ Contact support for specific advanced use case guidance

**Status**: Expanding advanced examples based on EAP feedback. Please share your advanced use cases!

**Areas We Want Feedback On**:
- Custom loss functions
- Multi-output models
- Streaming/incremental learning
- Model interpretability (SHAP, LIME integration)
- A/B testing patterns
- Model monitoring and drift detection

---

#### 11. Production Deployment Documentation Incomplete

**Issue**: Security best practices, performance tuning, and operational considerations are documented but not comprehensive.

**Impact**: Users may miss important production considerations.

**Workaround**:
- ✅ Review `deployment.md` for current guidance
- ✅ Use EAP period to evaluate production readiness
- ✅ Provide feedback on missing operational considerations

**Status**: Production documentation will be expanded in Phase 2 based on EAP feedback.

---

## Known Bugs

These are **confirmed bugs** that will be fixed before GA or have documented workarounds.

### Installation Issues

#### BUG-001: Docker Volume Permissions on Linux

**Severity**: Medium

**Description**: On some Linux distributions, Docker volume mounts may have incorrect permissions, preventing IRIS from writing to `/opt/irisapp/data`.

**Symptoms**:
- IRIS container fails to start
- Error: "Permission denied" in IRIS logs
- Models cannot be loaded

**Workaround**:
```bash
# Option 1: Fix volume permissions
sudo chown -R 51773:51773 ./data

# Option 2: Use docker-compose with user override
docker-compose run --user root iris bash
chown -R 51773:51773 /opt/irisapp/data
exit
docker-compose up -d
```

**Status**: Investigating fix for GA. Docker-compose configuration will be updated.

**Tracking**: Related to JIRA tickets on Docker deployment

---

#### BUG-002: IntegratedML Symlink Issue After Fresh Install

**Severity**: Low

**Description**: On fresh IRIS installations, the required symlink from `/usr/irissys/mgr/python/iris_automl` to `/opt/irisapp/data/mgr/python/iris_automl` may not exist.

**Symptoms**:
- `TRAIN MODEL` fails with "Module not found: iris_automl"
- AutoML provider not available

**Workaround**:
```bash
# Connect to IRIS container
docker exec -it iml-custom-models-iris bash

# Create symlink
ln -sf /usr/irissys/mgr/python/iris_automl /opt/irisapp/data/mgr/python/iris_automl

# Restart IRIS
exit
docker restart iml-custom-models-iris
```

**Status**: Will be automated in installation scripts for GA.

**Tracking**: Installation automation improvements

---

### Runtime Issues

#### BUG-003: Unhelpful Error Messages for Missing Methods

**Severity**: Medium

**Description**: When a custom model is missing a required method (`fit`, `predict`, etc.), the error message is unclear and doesn't indicate which method is missing.

**Symptoms**:
- Generic Python exception during `TRAIN MODEL` or `PREDICT()`
- Error message: "AttributeError" without specifying missing method

**Workaround**:
- ✅ Always inherit from `ClassificationModel` or `RegressionModel` base classes
- ✅ Test your model with pytest before deploying to IRIS
- ✅ Review base class interface in `shared/models/base.py`

**Status**: Enhanced error messages planned for GA.

**Tracking**: Validation and error handling improvements

---

#### BUG-004: Model State Serialization Issues with Complex Objects

**Severity**: Low

**Description**: Models that contain complex nested objects (e.g., custom transformers, third-party models) may fail to serialize correctly during `TRAIN MODEL`.

**Symptoms**:
- `TRAIN MODEL` completes but model state is not saved
- `PREDICT()` fails because model is not fitted
- Error: "PickleError" or "Serialization failed"

**Workaround**:
- ✅ Override `_get_model_state()` and `_set_model_state()` methods
- ✅ Manually serialize complex objects using joblib or pickle
- ✅ Review `EnsembleFraudDetector` for example of complex state management

**Example**:
```python
def _get_model_state(self):
    """Custom serialization for complex model"""
    return {
        'model': joblib.dumps(self.model),  # Use joblib for complex objects
        'preprocessor': pickle.dumps(self.preprocessor),
        'metadata': self.metadata  # Simple objects can be direct
    }

def _set_model_state(self, state):
    """Custom deserialization"""
    self.model = joblib.loads(state['model'])
    self.preprocessor = pickle.loads(state['preprocessor'])
    self.metadata = state['metadata']
```

**Status**: Documentation will be improved with serialization best practices.

**Tracking**: State management enhancements

---

### SQL Integration Issues

#### BUG-005: JSON USING Clause Validation

**Severity**: Low

**Description**: Invalid JSON in the `USING` clause may produce unclear error messages instead of JSON validation errors.

**Symptoms**:
- SQL syntax error with unclear message
- JSON parsing fails silently

**Workaround**:
- ✅ Validate JSON syntax before using in SQL (use a JSON validator)
- ✅ Use single quotes for JSON in SQL: `USING '{"param": "value"}'`
- ✅ Escape double quotes if using double-quoted JSON

**Status**: Better JSON validation and error messages planned for GA.

**Example**:
```sql
-- ✅ CORRECT: Single quotes around JSON
TRAIN MODEL my_model
USING '{"model_name": "MyClassifier", "user_params": {"param1": 1}}'

-- ❌ INCORRECT: Unescaped double quotes
TRAIN MODEL my_model
USING {"model_name": "MyClassifier"}  -- Will fail
```

**Tracking**: SQL parameter validation improvements

---

#### BUG-006: PREDICT() Performance with Large Result Sets

**Severity**: Low

**Description**: `SELECT ... PREDICT()` on very large tables (>1M rows) may be slower than expected.

**Impact**: Batch predictions on large datasets may take several minutes.

**Workaround**:
- ✅ Use `WHERE` clauses to limit prediction scope
- ✅ Batch predictions in smaller chunks (e.g., 10K-100K rows at a time)
- ✅ Consider materialized views for frequently-used predictions
- ✅ Use SQL `TOP` or `LIMIT` for testing

**Status**: Performance optimization ongoing. Feedback welcome on specific performance requirements.

**Example**:
```sql
-- ✅ Good: Batch predictions
SELECT TOP 10000 id,
       PREDICT(MyModel) as prediction
FROM LargeTable
WHERE prediction_date = CURRENT_DATE

-- ⚠️ Slow: Full table prediction
SELECT id, PREDICT(MyModel) as prediction
FROM LargeTable  -- 5M rows
```

**Tracking**: Query performance optimization

---

### Demo-Specific Issues

#### BUG-007: Sales Forecasting Demo Data Generation Takes Time

**Severity**: Low

**Description**: The sales forecasting demo's data generation script can take 2-3 minutes to generate realistic multi-year timeseries data.

**Impact**: First run of sales demo is slower than other demos.

**Workaround**:
- ✅ This is expected behavior - realistic timeseries data generation takes time
- ✅ Data is cached after first generation
- ✅ Reduce data volume in config for faster testing (see demo README)

**Status**: Acceptable for demo purposes. Pre-generated data may be provided for GA.

---

#### BUG-008: DNA Similarity Demo Requires Additional Dependencies

**Severity**: Low

**Description**: DNA similarity demo requires `biopython` which may not be installed by default.

**Symptoms**:
- Import error: "No module named 'Bio'"
- Demo fails to run

**Workaround**:
```bash
# Install biopython
pip install biopython

# Or use demo-specific requirements
pip install -r demos/dna_similarity/requirements.txt
```

**Status**: Will be documented more clearly in installation guide.

**Tracking**: Demo dependency documentation

---

## Workarounds

### General Troubleshooting Workflow

When you encounter an issue:

1. **Check this document** - Is it a known issue?
2. **Check [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)** - Common solutions
3. **Check [`EAP_FAQ.md`](EAP_FAQ.md)** - Frequently asked questions
4. **Search JIRA** (if you have access) - Existing bug reports
5. **Contact support** - Email thomas.dyar@intersystems.com

### Quick Workaround Reference

| Issue | Quick Fix |
|-------|-----------|
| Terminal restart needed | `docker restart iml-custom-models-iris` |
| Permission denied (Linux) | `sudo chown -R 51773:51773 ./data` |
| Module not found | Verify model file placement, check symlink |
| JSON syntax error | Use single quotes in SQL: `USING '{...}'` |
| Slow predictions | Add `WHERE` clause, batch in smaller chunks |
| Model not fitted | Check serialization, override state methods |
| Import error | Install missing dependencies with pip |

---

## Fixed in Future Releases

### Planned for GA (2026.1)

Based on current EAP feedback and internal roadmap:

- ✅ Enhanced error messages for missing model methods
- ✅ Better JSON USING clause validation
- ✅ Improved installation scripts (symlink automation)
- ✅ Hot-reload for model changes (under investigation)
- ✅ Expanded production deployment documentation
- ✅ Multi-platform installation testing
- ✅ Performance optimizations for large predictions

### Under Consideration for Post-GA

Based on EAP feedback, we're considering:

- Timeseries model native support
- Model namespace/versioning
- Async training for large models
- Model monitoring dashboard integration
- Pre-built model templates library

**Your feedback will help prioritize these!**

---

## How to Report New Issues

### Before Reporting

Please confirm:
- ✅ Issue is not listed in this document
- ✅ Issue is not in [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)
- ✅ Issue is not in [`EAP_FAQ.md`](EAP_FAQ.md)
- ✅ You've tried basic troubleshooting (restart, check logs)

### Reporting Channels

**Email** (Recommended for bugs): thomas.dyar@intersystems.com

**GitHub Issues** (If enabled): Use bug report template

### Information to Include

**For Bug Reports**:
```
**Title**: Brief description (e.g., "Model fails to load on Windows")

**Description**:
What happened? What did you expect to happen?

**Environment**:
- OS: [e.g., macOS 14.1, Ubuntu 22.04, Windows 11]
- Python version: [e.g., 3.11.5]
- IRIS version: [e.g., 2025.2]
- Installation method: [Docker / Local]

**Steps to Reproduce**:
1. Step 1
2. Step 2
3. Step 3

**Error Messages**:
```
Paste full error messages here
```

**Screenshots** (if applicable):
[Attach screenshots]

**Workaround Found**: [If you found a workaround, share it!]
```

---

## Document Updates

This document will be updated during the EAP as new issues are discovered and workarounds are identified.

**Last Updated**: 2025-01-12
**Next Update**: As issues are reported during EAP

To get the latest version of this document:
```bash
git pull origin main
```

Or check the repository: https://github.com/intersystems-community/integratedml-custom-models

---

## Thank You

Thank you for reviewing this document and helping us identify issues during the EAP. Your patience with these limitations and your feedback will make Custom Models better for everyone!

**Questions?** Email thomas.dyar@intersystems.com

**— The InterSystems Data Platforms Product Team**
