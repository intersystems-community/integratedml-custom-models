# Feature Specification: Resolve Missing Model Base Classes & Fix Parameter Naming

**Feature Branch**: `004-figure-out-why`  
**Created**: 2026-04-07  
**Updated**: 2026-04-07 (added parameter naming and IRISModel interface tasks)  
**Status**: Draft  
**Input**: User description: "figure out why models are missing"; confirmed by DP-450070 (Tomo Okuyama EAP feedback)

## Background

Two root causes were identified through investigation of DP-450070 and internal Confluence QD docs:

**Root Cause 1 — Missing base classes**: The `shared/models/` directory does not exist on disk. Demo models (`EnsembleFraudDetector`, `DNASequenceClassifier`) import from `shared.models.ensemble` and `shared.models.classification` — these imports fail at runtime. `CustomCreditRiskClassifier` and `HybridForecastingModel` are referenced in documentation but have no implementation files at all.

**Root Cause 2 — Wrong parameter names in all EAP docs/SQL**: Our documentation and SQL examples use underscored parameter names (`isc_models_disabled`, `path_to_classifiers`, `path_to_regressors`) which IRIS silently ignores. The actual parameter names IRIS reads are concatenated (`iscmodelsdisabled`, `pathtoclassifiers`, `pathtoregressors`). This was confirmed by Sidd Gangwani on 2026-04-07 in DP-450070, with training logs showing:
```
Not creating an instance of isc_logistic_regression as isc_models_disabled is set to True
```
This caused EAP participants to believe `iscmodelsdisabled` was broken when it was working correctly.

**Root Cause 3 — No working IRISModel examples in the repo**: The actual IRIS pluggable models interface requires each `.py` file to define a class called `IRISModel` (not our base classes). Our EAP repo has no examples of this pattern. Users cannot write a working custom model by following our docs alone.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Developer Can Import Demo Models Without Error (Priority: P1)

A developer cloning the repo runs `python -c "from demos.fraud_detection.models.ensemble_fraud_detector import EnsembleFraudDetector"` and it succeeds without an ImportError. Currently this fails immediately.

**Why this priority**: Without importable models, no demo runs, no tests pass, and the project is non-functional. This is the blocking issue.

**Independent Test**: Run `python -c "from demos.fraud_detection.models.ensemble_fraud_detector import EnsembleFraudDetector; from demos.dna_similarity.models.dna_classifier import DNASequenceClassifier; print('OK')"` from the project root. If both import without error, this story is complete.

**Acceptance Scenarios**:

1. **Given** a clean checkout of the repository with dependencies installed, **When** a developer imports `EnsembleFraudDetector`, **Then** no `ImportError` or `ModuleNotFoundError` is raised
2. **Given** a clean checkout, **When** a developer imports `DNASequenceClassifier`, **Then** no `ImportError` is raised
3. **Given** the base classes exist, **When** a demo model class is instantiated with `**kwargs`, **Then** it initializes without error and `self.parameters` is populated from the kwargs

---

### User Story 2 - All Demo Unit Tests Pass (Priority: P2)

A developer runs `pytest -m "not integration" demos/` and all unit tests that were previously failing due to import errors now pass.

**Why this priority**: Restoring test coverage validates that the base class contracts are correct and that demo implementations work against them.

**Independent Test**: Run `pytest -m "not integration" demos/fraud_detection/tests/ demos/dna_similarity/tests/ -v` and observe zero import-related failures.

**Acceptance Scenarios**:

1. **Given** base classes are implemented, **When** `pytest -m "not integration" demos/fraud_detection/tests/` runs, **Then** all unit tests pass
2. **Given** base classes are implemented, **When** `pytest -m "not integration" demos/dna_similarity/tests/` runs, **Then** all tests pass
3. **Given** a test that instantiates a demo model, **When** it calls `fit(X, y)` with sample data, **Then** it completes without error

---

### User Story 3 - Credit Risk and Sales Forecasting Models Are Implemented (Priority: P3)

A developer can import `CustomCreditRiskClassifier` and `HybridForecastingModel` — the two models referenced in documentation that have no implementation files at all.

**Why this priority**: These are needed to make the credit risk and sales forecasting demos runnable, but the fraud and DNA demos are higher priority to unblock first.

**Independent Test**: Run `python -c "from demos.credit_risk.models.credit_risk_classifier import CustomCreditRiskClassifier; from demos.sales_forecasting.models.hybrid_forecasting_model import HybridForecastingModel; print('OK')"` without error.

**Acceptance Scenarios**:

1. **Given** the credit risk demo directory, **When** `CustomCreditRiskClassifier` is imported, **Then** no ImportError
2. **Given** the sales forecasting demo directory, **When** `HybridForecastingModel` is imported, **Then** no ImportError
3. **Given** both models, **When** unit tests in their respective `tests/` directories run, **Then** all pass

---

### User Story 4 - EAP Participant Can Suppress Built-in Models Using Correct Parameter Names (Priority: P1)

An EAP participant follows the Quick Guide or any SQL example in the repo and successfully suppresses ISC built-in models, seeing only their custom models in training runs.

**Why this priority**: This was the primary EAP bug report (DP-450070). The wrong parameter names in our docs caused IRIS to silently ignore the suppression flag, making users think the feature was broken.

**Independent Test**: All SQL examples in `demos/*/sql/` and `docs/QUICK_GUIDE_CUSTOM_MODELS.md` use `"iscmodelsdisabled"`, `"pathtoclassifiers"`, `"pathtoregressors"` (no underscores). Zero occurrences of the underscored variants remain in any non-archival file.

**Acceptance Scenarios**:

1. **Given** a SQL example from `demos/credit_risk/sql/create_model.sql`, **When** run against IRIS 2025.2+, **Then** built-in ISC models are suppressed when `"iscmodelsdisabled": 1` is set
2. **Given** any doc file in `docs/`, **When** a developer copies a USING block, **Then** the parameter names are correct and IRIS will honor them
3. **Given** a grep of the repo for `"isc_models_disabled"`, **Then** zero results are found in demo/doc/test files

---

### User Story 5 - EAP Participant Can Write a Working IRISModel File from Our Examples (Priority: P2)

An EAP participant reads our docs or repo examples and can write a Python file with an `IRISModel` class that IRIS will load and train.

**Why this priority**: Our current repo shows `shared.models.ClassificationModel` as the base to extend, but IRIS actually requires a specific `IRISModel` interface per `.py` file. Without working examples, EAP participants cannot write custom models.

**Independent Test**: The repo contains at least two working `IRISModel` example files in `examples/iris_models/` — one classifier and one regressor — with a README explaining the interface. Running `python -c "from examples.iris_models.example_classifier import IRISModel; m = IRISModel(); print(m.name)"` succeeds.

**Acceptance Scenarios**:

1. **Given** `examples/iris_models/example_classifier.py`, **When** imported, **Then** `IRISModel` class exists with `name` string attribute and `model` with `fit`/`predict`/`predict_proba`
2. **Given** `examples/iris_models/example_regressor.py`, **When** imported, **Then** `IRISModel` class exists with `name` and `model` with `fit`/`predict`
3. **Given** the example files, **When** a developer copies one as a starting point, **Then** IRIS can load and train it using `pathtoclassifiers`/`pathtoregressors`

---

### Edge Cases

- What happens when a demo model is instantiated with an empty `**kwargs`? Base class must handle gracefully, defaulting `self.parameters` to `{}`.
- What if `predict_proba()` is called on a model that doesn't implement it? Base class should raise `NotImplementedError` with a clear message.
- What if `fit()` is called before `predict()`? Models must validate trained state and raise a clear error if not trained.
- What if `_get_model_state()` / `_set_model_state()` encounter unpicklable objects? Must fail with an explicit error, not a silent corrupt state.
- What if a user passes `"isc_models_disabled"` (underscored)? IRIS silently ignores it — our docs must note this explicitly.

## Requirements *(mandatory)*

### Functional Requirements

#### Group A: Missing base classes (original scope)
- **FR-001**: The `shared/models/` package MUST exist with `__init__.py`, `base.py`, `classification.py`, `regression.py`, and `ensemble.py`
- **FR-002**: `IntegratedMLBaseModel` in `base.py` MUST accept `**kwargs` and expose them as `self.parameters`
- **FR-003**: `IntegratedMLBaseModel` MUST define abstract methods `fit(X, y)` and `predict(X)` that subclasses must implement
- **FR-004**: `IntegratedMLBaseModel` MUST provide `_get_model_state()` and `_set_model_state()` for serialization
- **FR-005**: `IntegratedMLBaseModel` MUST provide a `_engineer_features()` hook that subclasses can override for domain preprocessing
- **FR-006**: `ClassificationModel` MUST extend `IntegratedMLBaseModel` and add `predict_proba()` as an optional override
- **FR-007**: `EnsembleModel` MUST extend `IntegratedMLBaseModel` and support combining multiple sub-models
- **FR-008**: `RegressionModel` MUST extend `IntegratedMLBaseModel` for continuous value prediction
- **FR-009**: `demos/credit_risk/models/credit_risk_classifier.py` MUST exist with `CustomCreditRiskClassifier` extending `ClassificationModel`
- **FR-010**: `demos/sales_forecasting/models/hybrid_forecasting_model.py` MUST exist with `HybridForecastingModel` extending `RegressionModel`
- **FR-011**: All existing demo model imports MUST succeed after implementation (`EnsembleFraudDetector`, `DNASequenceClassifier`)

#### Group B: Parameter naming fix (added 2026-04-07)
- **FR-012**: All SQL `USING {}` blocks in `demos/*/sql/` MUST use `"iscmodelsdisabled"` (not `"isc_models_disabled"`)
- **FR-013**: All SQL `USING {}` blocks MUST use `"pathtoclassifiers"` (not `"path_to_classifiers"`)
- **FR-014**: All SQL `USING {}` blocks MUST use `"pathtoregressors"` (not `"path_to_regressors"`)
- **FR-015**: `docs/QUICK_GUIDE_CUSTOM_MODELS.md` and `docs/EAP_FAQ.md` MUST use the correct parameter names
- **FR-016**: Test files `tests/test_all_demos_e2e.py` and `tests/test_real_iris_integration.py` MUST use correct parameter names
- **FR-017**: `README.md` examples MUST use correct parameter names
- **FR-018**: `docs/EAP_KNOWN_ISSUES.md` MUST document that underscored names are silently ignored by IRIS

#### Group C: IRISModel interface examples (added 2026-04-07)
- **FR-019**: `examples/iris_models/` directory MUST exist with at least `example_classifier.py` and `example_regressor.py`
- **FR-020**: Each example file MUST define a class named `IRISModel` with a `name` string attribute and a `model` attribute that has `fit`, `predict`, `get_params`, `set_params` methods
- **FR-021**: `examples/iris_models/README.md` MUST explain the IRISModel interface, the difference from `shared.models` base classes, and how to deploy using `pathtoclassifiers`/`pathtoregressors`
- **FR-022**: Example classifier MUST include `predict_proba()` on its inner `model` object
- **FR-023**: Example files MUST be minimal and readable — under 80 lines each, no external deps beyond scikit-learn

### Key Entities

- **IntegratedMLBaseModel**: Core abstract base. Attributes: `parameters` (dict from kwargs), `_is_trained` (bool). Methods: `fit`, `predict`, `_engineer_features`, `_get_model_state`, `_set_model_state`
- **ClassificationModel**: Extends base for binary/multi-class. Adds: `predict_proba()`, label encoding helpers
- **RegressionModel**: Extends base for continuous outputs. Adds: numeric target validation
- **EnsembleModel**: Extends base for multi-model voting. Adds: sub-model registry, voting/averaging logic
- **CustomCreditRiskClassifier**: Demo model for loan default prediction, extends ClassificationModel
- **HybridForecastingModel**: Demo model for time-series, extends RegressionModel (Prophet + LightGBM)
- **IRISModel (examples)**: The actual IRIS pluggable interface — a standalone class per `.py` file with `name` + `model` attributes. Distinct from our `shared.models` hierarchy.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 4 demo model classes import without error from a clean checkout with dependencies installed
- **SC-002**: `pytest -m "not integration" demos/` completes with 0 import-related failures
- **SC-003**: All base class abstract methods are covered by at least one unit test each
- **SC-004**: A developer can go from `git clone` to a working demo model import in under 5 minutes following the README
- **SC-005**: `grep -r '"isc_models_disabled"' demos/ docs/ tests/ README.md` returns 0 results
- **SC-006**: `grep -r '"path_to_classifiers"\|"path_to_regressors"' demos/ docs/ tests/ README.md` returns 0 results
- **SC-007**: `examples/iris_models/` contains at least 2 working example files importable with `python -c "from examples.iris_models.example_classifier import IRISModel"`

## Assumptions

- Base classes use scikit-learn's `BaseEstimator` / `ClassifierMixin` conventions (`fit`/`predict` API)
- Serialization uses `pickle`/`joblib` (already used in `dna_classifier.py`)
- `HybridForecastingModel` will be a minimal stub that satisfies imports; full Prophet+LightGBM implementation is out of scope for this spec
- `CustomCreditRiskClassifier` will be a functional classifier using the existing preprocessing in `scripts/data_preprocessing.py`
- Parameter naming fix (FR-012 through FR-017) has already been applied as of 2026-04-07 commit on branch `004-figure-out-why`
- IRISModel examples are standalone and do NOT import from `shared.models` — they show the raw IRIS interface
