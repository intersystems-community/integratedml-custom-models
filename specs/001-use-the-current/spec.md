# Feature Specification: IntegratedML Custom Models Platform

**Feature Branch**: `001-use-the-current`
**Created**: 2025-10-10
**Status**: Draft
**Input**: User description: "use the current codebase to generate a cohesive specification for this project"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Deploy Custom Credit Risk Model (Priority: P1)

A data scientist needs to deploy a custom credit risk assessment model that includes domain-specific feature engineering (debt-to-income ratios, risk scoring) directly into the production IRIS database without writing integration code or exporting data.

**Why this priority**: This is the fundamental value proposition of the platform - enabling custom Python models with complex preprocessing to execute within SQL queries. Without this capability, users must export data, run external predictions, and import results.

**Independent Test**: Can be fully tested by creating a custom credit risk classifier with feature engineering enabled, training it via SQL CREATE MODEL command, and executing PREDICT queries that return risk assessments in under 50ms.

**Acceptance Scenarios**:

1. **Given** a dataset of 10,000 credit applications in an IRIS table, **When** a data scientist creates a model using SQL with custom parameters `{"enable_debt_ratio": 1, "enable_risk_scoring": 1}`, **Then** the model trains successfully and stores all engineered features and preprocessing pipelines

2. **Given** a trained credit risk model, **When** new credit applications are queried with `SELECT customer_id, PREDICT(CreditRiskModel)`, **Then** predictions return within 50ms and include risk probabilities

3. **Given** a custom model with 15+ engineered features from 8 base features, **When** the model is validated using VALIDATE MODEL SQL command, **Then** accuracy metrics and feature importance scores are returned

---

### User Story 2 - Deploy Ensemble Fraud Detection System (Priority: P2)

A fraud analyst needs to deploy an ensemble of multiple detection models (neural network, rule-based, behavioral analysis) that vote together and provide confidence scores for real-time transaction monitoring.

**Why this priority**: Demonstrates advanced multi-model capability and shows how the platform handles complex model architectures beyond simple classifiers. Critical for production fraud detection use cases.

**Independent Test**: Can be fully tested by deploying an ensemble with 3+ sub-models, processing 25,000 test transactions, and verifying that weighted voting produces consistent fraud flags with confidence thresholds.

**Acceptance Scenarios**:

1. **Given** transaction data with amount, merchant, location features, **When** an ensemble fraud detector is created with neural, rules, and anomaly sub-models, **Then** all sub-models train successfully and store their individual states

2. **Given** a trained ensemble model, **When** 1000 concurrent prediction requests are made, **Then** the system maintains sub-50ms latency and weighted voting produces consistent scores

3. **Given** fraudulent and legitimate transactions, **When** predictions are evaluated, **Then** the ensemble achieves better precision than any individual sub-model

---

### User Story 3 - Deploy Time-Series Sales Forecasting (Priority: P3)

A business analyst needs to forecast sales across multiple store locations using a hybrid model combining Prophet for seasonality detection and LightGBM for feature-based predictions.

**Why this priority**: Demonstrates regression capability and complex model composition. Shows the platform can handle time-series forecasting in addition to classification tasks.

**Independent Test**: Can be fully tested by training on 365 days of sales data across 5 stores and producing 30-day forecasts with accuracy measured by MAPE (Mean Absolute Percentage Error).

**Acceptance Scenarios**:

1. **Given** historical sales data with date, store, and sales amount, **When** a hybrid forecasting model is trained, **Then** Prophet detects seasonal patterns and LightGBM learns store-specific features

2. **Given** a trained forecasting model, **When** predictions are requested for future dates, **Then** forecasts include seasonality adjustments and achieve <30% MAPE on validation data

3. **Given** multiple store locations, **When** forecasts are generated, **Then** predictions account for store-specific patterns and holiday effects

---

### User Story 4 - Deploy DNA Sequence Similarity Classifier (Priority: P4)

A bioinformatics researcher needs to classify DNA sequences by similarity using custom distance metrics (GC content, k-mer matching, motif search) not available in standard ML libraries.

**Why this priority**: Demonstrates extreme customization capability with domain-specific algorithms. Shows the platform can handle specialized scientific computing use cases.

**Independent Test**: Can be fully tested by processing 5,000 DNA sequences with custom similarity calculations and achieving classification accuracy on sequence family prediction.

**Acceptance Scenarios**:

1. **Given** DNA sequences as text strings, **When** a custom similarity analyzer is trained, **Then** GC content ratios and k-mer frequencies are calculated for each sequence

2. **Given** sequences with known families, **When** predictions are made using K-NN with custom Hamming distance, **Then** similar sequences are correctly grouped by genetic similarity

3. **Given** new unknown sequences, **When** predictions are requested, **Then** the model returns predicted family classifications with similarity scores

---

### Edge Cases

- What happens when a model receives input data with missing features during prediction?
  - System fills with defaults or raises clear validation error based on preprocessing configuration

- What happens when training data volume is insufficient (<100 records)?
  - Model should train but log warnings about potential overfitting; benchmarks may not meet accuracy targets

- How does the system handle model state when IRIS restarts?
  - Models must persist fully via serialization; predictions should resume without retraining after restart

- What happens when prediction latency exceeds 50ms threshold?
  - System logs performance warnings; users can optimize by disabling expensive feature engineering options

- How does the system handle incompatible parameter updates?
  - Parameter validation during model creation returns clear error messages about invalid configurations

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST allow custom Python models to be deployed via SQL `CREATE MODEL` statements with JSON parameter configuration

- **FR-002**: System MUST support scikit-learn compatible models inheriting from base classes (ClassificationModel, RegressionModel, EnsembleModel)

- **FR-003**: Models MUST execute fit and predict operations entirely within IRIS database without requiring data export

- **FR-004**: System MUST accept model parameters via JSON USING clause syntax (IRIS 2025.2+ format)

- **FR-005**: Models MUST support custom feature engineering pipelines that execute consistently during training and prediction

- **FR-006**: System MUST serialize and persist trained model state (parameters, preprocessing pipelines, learned weights) across database sessions

- **FR-007**: System MUST provide VALIDATE MODEL SQL command that returns accuracy metrics and model diagnostics

- **FR-008**: System MUST support ensemble models that combine multiple sub-models with configurable voting strategies

- **FR-009**: Models MUST support parameter introspection via get_params() and set_params() for scikit-learn compatibility

- **FR-010**: System MUST handle both classification (binary/multi-class) and regression prediction tasks

- **FR-011**: System MUST preserve feature names and data types through preprocessing transformations

- **FR-012**: System MUST provide clear error messages for invalid parameters, missing features, or incompatible data types

- **FR-013**: Models MUST support custom preprocessing including imputation, scaling, encoding, and domain-specific transformations

- **FR-014**: System MUST allow models to define and use internal state (preprocessors, encoders, scalers) that persists with the model

- **FR-015**: System MUST support probability predictions (predict_proba) for classification tasks

### Key Entities

- **Custom Model**: A Python class implementing fit/predict interface with custom preprocessing logic, parameters, and internal state. Deployed to IRIS and invoked via SQL. Attributes include model class name, parameters dictionary, feature engineering configuration, trained state (scikit-learn pipeline, weights, preprocessors).

- **Training Dataset**: Table in IRIS database containing features (X) and target variable (y). Used to fit custom models via `CREATE MODEL...FROM table_name`. Attributes include table name, feature columns, target column, row count.

- **Model Parameters**: JSON configuration passed from SQL to Python model controlling behavior. Attributes include parameter name, value, type, validation rules. Examples: enable_debt_ratio (boolean), decision_threshold (float 0-1), ensemble_weights (array).

- **Prediction Request**: SQL query using PREDICT() function to generate predictions from trained model. Attributes include model name, input features (from SELECT), output format (class labels or probabilities).

- **Model State**: Serialized representation of trained model including learned weights, preprocessing artifacts, feature metadata. Attributes include pickle data, version, creation timestamp, fit status.

- **Base Model Classes**: Abstract classes (IntegratedMLBaseModel, ClassificationModel, RegressionModel, EnsembleModel) providing common interfaces. Attributes include parameter handling, state serialization methods, validation hooks.

- **Feature Engineering Pipeline**: Sequence of transformations applied to raw data before model training/prediction. Attributes include transformation steps, fitted encoders/scalers, feature names mapping.

- **Ensemble Sub-Model**: Individual model within an ensemble, with its own architecture and weight. Attributes include model instance, voting weight, confidence threshold.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Data scientists can deploy a custom model with feature engineering to production in under 5 minutes from model code to SQL integration

- **SC-002**: Prediction latency remains under 50ms (p95) for single-record queries across all demo models (credit risk, fraud, forecasting, DNA)

- **SC-003**: Models support training datasets from 1,000 to 100,000 records without degradation in training completion time proportional to dataset size

- **SC-004**: Model state persists correctly across IRIS database restarts with 100% prediction consistency (same inputs produce identical outputs before and after restart)

- **SC-005**: Custom preprocessing pipelines with 10+ feature transformations execute correctly during both training and prediction phases without feature name mismatches

- **SC-006**: Ensemble models combining 3+ sub-models produce weighted predictions with accuracy improvement of at least 5% over best individual sub-model

- **SC-007**: All four demo applications (credit risk, fraud detection, sales forecasting, DNA similarity) achieve documented accuracy benchmarks and latency targets

- **SC-008**: Parameter validation catches 100% of invalid configurations (wrong types, out-of-range values) before model training begins

- **SC-009**: Model serialization successfully stores and restores all internal state (pipelines, encoders, learned parameters) with byte-for-byte reproduction of predictions

- **SC-010**: Users can query model performance metrics (training time, prediction latency, accuracy) through VALIDATE MODEL SQL command

- **SC-011**: System handles missing features in prediction data with either imputation or clear error messages, preventing silent failures

- **SC-012**: Code formatting, linting, and test suite pass on all model code with 100% test coverage on core base classes

## Assumptions

- **Environment**: IRIS 2025.2+ with IntegratedML installed and Python 3.8+ runtime configured
- **Data Scale**: Demo datasets contain 1,000-100,000 records; production scale testing beyond this scope
- **Performance Baseline**: 50ms latency target assumes single-record predictions on standard database server hardware
- **Model Complexity**: Custom models may include arbitrary Python logic but must remain compatible with scikit-learn estimator interface
- **Default Behavior**: Missing feature values are imputed using median (numeric) or mode (categorical) strategies unless model overrides
- **Testing Framework**: pytest with IRIS database integration for unit and integration tests
- **Deployment Model**: Docker Compose for local development; production deployment patterns out of scope
- **Feature Engineering**: Models perform transformations in Python; no reliance on IRIS-specific SQL functions for preprocessing
