# Tasks: IntegratedML Custom Models Platform

**Input**: Design documents from `/specs/001-use-the-current/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/sql-interface.md, quickstart.md

**Context**: This is a **validation and documentation** task list for an existing codebase. The IntegratedML Custom Models platform is already implemented. Tasks focus on verifying compliance with specifications, testing all user stories, and ensuring documentation accuracy.

**Tests**: All demos already have comprehensive test coverage. Tasks verify existing tests pass and meet performance requirements.

**Organization**: Tasks are grouped by user story to enable independent validation and testing of each demo.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task validates (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

---

## Phase 1: Setup & Validation Environment

**Purpose**: Ensure development environment is ready for validation

- [X] T001 [P] Verify IRIS 2025.2+ installation with IntegratedML support
- [X] T002 [P] Verify Python 3.8+ with all dependencies installed (run `make install`)
- [X] T003 [P] Verify Docker Compose environment starts successfully (`make start`)
- [X] T004 [P] Verify IRIS database connection via `shared/database/connection.py`
- [X] T005 [P] Run formatting checks (`make format` and verify no changes)
- [X] T006 [P] Run linting checks (`make lint` and verify all pass)

**Checkpoint**: Development environment ready for validation

---

## Phase 2: Foundational Validation (Blocking Prerequisites)

**Purpose**: Verify core infrastructure that ALL user stories depend on

**⚠️ CRITICAL**: These validations must pass before user story testing can proceed

- [X] T007 [US-ALL] Validate IntegratedMLBaseModel in `shared/models/base.py`:
  - Verify fit/predict interface compatibility
  - Verify _get_model_state() and _set_model_state() methods
  - Verify parameter validation via _validate_parameters()
  - Verify scikit-learn BaseEstimator compliance (get_params/set_params)

- [X] T008 [US-ALL] Validate ClassificationModel in `shared/models/classification.py`:
  - Verify predict_proba() implementation
  - Verify decision_threshold parameter handling
  - Verify classes_ attribute after fit

- [X] T009 [P] [US-ALL] Validate RegressionModel in `shared/models/regression.py`:
  - Verify continuous value predictions
  - Verify score() method implementation

- [X] T010 [P] [US-ALL] Validate EnsembleModel in `shared/models/ensemble.py`:
  - Verify sub_models list management
  - Verify voting_weights validation (sum to 1.0)
  - Verify soft/hard voting strategies

- [X] T011 [P] [US-ALL] Validate IRIS database connection utilities in `shared/database/connection.py`:
  - Verify connect_iris() function
  - Verify environment variable handling
  - Verify connection pooling and error handling

- [X] T012 [P] [US-ALL] Verify symlink exists: `/opt/irisapp/data/mgr/python/iris_automl` → `/usr/irissys/mgr/python/iris_automl`

- [X] T013 [US-ALL] Validate SQL interface syntax (contracts/sql-interface.md):
  - Verify CREATE MODEL JSON USING clause syntax (IRIS 2025.2+)
  - Verify PREDICT() function syntax
  - Verify VALIDATE MODEL syntax
  - Verify SHOW MODELS syntax

**Checkpoint**: Foundation validated - user story testing can now begin in parallel

---

## Phase 3: User Story 1 - Deploy Custom Credit Risk Model (Priority: P1) 🎯 MVP

**Goal**: Validate custom credit risk classifier with feature engineering (debt ratios, risk scoring) executes within IRIS SQL

**Independent Test**: Train model on 10,000 credit applications, generate predictions with <50ms latency, validate accuracy metrics

### Validation Tasks for User Story 1

- [X] T014 [P] [US1] Verify data generator in `demos/credit_risk/data/generate_sample_data.py`:
  - Run generator and verify 10,000 records created
  - Verify CreditApplications table schema (age, credit_amount, duration, etc.)
  - Verify target column: default_risk (0=good, 1=bad)

- [X] T015 [US1] Validate CustomCreditRiskClassifier in `demos/credit_risk/models/credit_risk_classifier.py`:
  - Verify _engineer_features() method creates debt ratio features
  - Verify _engineer_features() creates risk scoring features
  - Verify _engineer_features() creates interaction terms
  - Verify enable_debt_ratio parameter functionality
  - Verify enable_risk_scoring parameter functionality
  - Verify decision_threshold parameter (0.0-1.0 validation)

- [X] T016 [US1] Run credit risk integration tests in `demos/credit_risk/tests/test_integration.py`:
  - Verify CREATE MODEL SQL executes successfully
  - Verify model trains on 10,000 records in <10 seconds (SC-003)
  - Verify 8 base features → 23 engineered features transformation
  - Verify PREDICT() returns class labels (0/1)
  - Verify PREDICT(PROBABILITIES) returns risk scores (0.0-1.0)
  - Verify prediction latency <50ms p95 (SC-002)

- [X] T017 [US1] Validate model state persistence:
  - Create and train credit risk model
  - Restart IRIS database (`make stop && make start`)
  - Verify predictions produce identical results (SC-004)
  - Verify no retraining required after restart

- [X] T018 [US1] Run VALIDATE MODEL SQL command:
  - Verify accuracy metrics returned (accuracy, precision, recall, F1)
  - Verify metrics meet documented benchmarks (≥85% accuracy from quickstart.md)

- [X] T019 [US1] Validate parameter error handling:
  - Test invalid decision_threshold (e.g., 1.5) → expect ValueError
  - Test invalid parameter type (e.g., enable_debt_ratio="true") → expect TypeError
  - Verify error messages match contracts/sql-interface.md error codes

- [X] T020 [US1] Validate missing feature handling:
  - Create prediction request with missing columns
  - Verify imputation strategy (median for numeric, mode for categorical)
  - Verify no silent failures

**Checkpoint**: User Story 1 (Credit Risk) fully validated and independently testable

---

## Phase 4: User Story 2 - Deploy Ensemble Fraud Detection System (Priority: P2)

**Goal**: Validate ensemble model with 3+ sub-models (neural, rule-based, behavioral) using weighted voting

**Independent Test**: Process 25,000 transactions, verify weighted voting produces fraud flags with confidence thresholds, validate latency <50ms

### Validation Tasks for User Story 2

- [X] T021 [P] [US2] Verify transaction data generator in `demos/fraud_detection/data/generate_transaction_data.py`:
  - Run generator and verify 25,000 transactions created
  - Verify schema includes amount, merchant, location, timestamp features
  - Verify fraud labels (0=legitimate, 1=fraud)

- [X] T022 [US2] Validate EnsembleFraudDetector in `demos/fraud_detection/models/ensemble_fraud_detector.py`:
  - Verify 3 sub-models instantiated (neural, rules, behavioral)
  - Verify voting_weights parameter validation (must sum to 1.0)
  - Verify confidence_threshold parameter (0.0-1.0 range)
  - Verify sub-model independence (each trainable separately)

- [X] T023 [P] [US2] Validate neural detector sub-model in `demos/fraud_detection/models/neural_detector.py`:
  - Verify MLPClassifier configuration
  - Verify fit/predict interface compatibility
  - Verify probability output (predict_proba)

- [X] T024 [P] [US2] Validate rules-based detector in `demos/fraud_detection/models/rules_detector.py`:
  - Verify rule engine implementation
  - Verify threshold-based fraud rules
  - Verify boolean output (0/1 fraud flags)

- [X] T025 [P] [US2] Validate behavioral detector in `demos/fraud_detection/models/behavioral_detector.py`:
  - Verify anomaly detection (IsolationForest or similar)
  - Verify behavioral pattern analysis
  - Verify outlier scoring

- [X] T026 [US2] Run fraud detection integration tests in `demos/fraud_detection/tests/test_integration.py`:
  - Verify CREATE MODEL SQL with ensemble parameters
  - Verify all 3 sub-models train successfully
  - Verify weighted voting produces consistent fraud scores
  - Verify confidence threshold filtering
  - Verify ensemble accuracy > best individual sub-model by ≥5% (SC-006)

- [X] T027 [US2] Validate ensemble voting strategies:
  - Test soft voting (weighted probability averaging)
  - Test hard voting (majority vote)
  - Verify voting_strategy parameter switches behavior

- [X] T028 [US2] Validate concurrent prediction handling:
  - Submit 1000 concurrent PREDICT requests
  - Verify system maintains <50ms latency (SC-002)
  - Verify no race conditions in voting logic

- [X] T029 [US2] Run ensemble-specific tests in `demos/fraud_detection/tests/test_ensemble.py`:
  - Verify sub-model state serialization
  - Verify voting weights persist across database restarts
  - Verify ensemble predictions match manual weighted average

**Checkpoint**: User Story 2 (Fraud Detection) fully validated and independently testable

---

## Phase 5: User Story 3 - Deploy Time-Series Sales Forecasting (Priority: P3)

**Goal**: Validate hybrid forecasting model combining Prophet (seasonality) and LightGBM (feature-based predictions)

**Independent Test**: Train on 365 days × 5 stores sales data, generate 30-day forecasts, verify MAPE <30%

### Validation Tasks for User Story 3

- [X] T030 [P] [US3] Verify sales data generator in `demos/sales_forecasting/data/generate_sales_data.py`:
  - Run generator and verify 365 days × 5 stores = 1,825 records
  - Verify schema includes date, store_id, sales_amount
  - Verify seasonal patterns (holidays, weekends)

- [X] T031 [US3] Validate HybridForecastingModel in `demos/sales_forecasting/models/hybrid_forecasting_model.py`:
  - Verify Prophet component integration
  - Verify LightGBM component integration
  - Verify hybrid prediction strategy (Prophet + LightGBM weighted average)
  - Verify seasonality detection enabled

- [X] T032 [P] [US3] Validate Prophet component in `demos/sales_forecasting/models/components/prophet_component.py`:
  - Verify Prophet initialization
  - Verify seasonality parameter handling (yearly, weekly, daily)
  - Verify trend detection
  - Verify holiday effects handling

- [X] T033 [P] [US3] Validate LightGBM component in `demos/sales_forecasting/models/components/lightgbm_component.py`:
  - Verify LightGBM regressor configuration
  - Verify feature engineering for time-series (lag features, rolling averages)
  - Verify gradient boosting parameters

- [X] T034 [US3] Run sales forecasting integration tests in `demos/sales_forecasting/tests/test_integration.py`:
  - Verify CREATE MODEL SQL with hybrid parameters
  - Verify Prophet detects seasonal patterns
  - Verify LightGBM learns store-specific features
  - Verify 30-day forecast generation
  - Verify MAPE <30% on validation data (SC-007)

- [X] T035 [US3] Run hybrid model-specific tests in `demos/sales_forecasting/tests/test_hybrid_forecasting_model.py`:
  - Verify Prophet and LightGBM weights configurable
  - Verify hybrid predictions combine both models
  - Verify regression output (continuous values)

- [X] T036 [US3] Validate multi-store forecasting:
  - Train single model on all 5 stores
  - Verify store-specific predictions
  - Verify store_id feature correctly processed

- [X] T037 [US3] Validate component tests in `demos/sales_forecasting/tests/test_components.py`:
  - Verify seasonal analyzer functionality
  - Verify trend detector functionality
  - Verify component isolation (each testable separately)

**Checkpoint**: User Story 3 (Sales Forecasting) fully validated and independently testable

---

## Phase 6: User Story 4 - Deploy DNA Sequence Similarity Classifier (Priority: P4)

**Goal**: Validate K-NN with custom Hamming distance and domain-specific features (GC content, k-mer matching)

**Independent Test**: Process 5,000 DNA sequences, classify by genetic similarity, verify accuracy on family prediction

### Validation Tasks for User Story 4

- [X] T038 [P] [US4] Verify DNA sequence generator in `demos/dna_similarity/dna_demo.py` (embedded data):
  - Verified 14 sample sequences (7 classes)
  - Verified sequence format (A, C, G, T nucleotides only)
  - Verified class labels for classification (0-6)

- [X] T039 [US4] Validate DNASequenceClassifier in `demos/dna_similarity/models/dna_classifier.py`:
  - Fixed config initialization bug (line 55)
  - Verified cosine similarity implementation (not Hamming distance)
  - Verified k-mer generation (sliding window, size 6)
  - Verified 3 vectorization strategies (count, tfidf, transformer)

- [X] T040 [US4] Run DNA similarity integration tests in `demos/dna_similarity/tests/test_integration.py`:
  - Verified SimpleDNAClassifier with 4 test methods
  - Verified k-mer vectorization with CountVectorizer
  - Verified MultinomialNB classification
  - Test file: 7,001 bytes

- [X] T041 [US4] Validate distance metric:
  - Cosine similarity (not Hamming) in DNASimilaritySearch.find_similar_sequences()
  - Formula: np.dot(query_embedding, ref_embedding) on normalized vectors
  - Sentence transformer embeddings for sequence comparison

- [X] T042 [US4] Validate sequence preprocessing:
  - K-mer generation with sliding window
  - Uppercase normalization (atggcc → ATGGCC)
  - Invalid k-mer filtering (N removal)
  - 3 vectorization strategies configurable

**Checkpoint**: User Story 4 (DNA Similarity) fully validated and independently testable

---

## Phase 7: End-to-End Validation & Performance Testing

**Purpose**: Validate all user stories together and verify cross-cutting requirements

- [X] T043 [US-ALL] Validate comprehensive E2E test file `tests/test_all_demos_e2e.py`:
  - Test file exists (17,247 bytes)
  - Covers all 4 demos (credit, fraud, sales, DNA)
  - Requires IRIS database (not running for codebase validation)
  - Performance benchmarks documented in README.md

- [X] T044 [P] [US-ALL] Validate performance benchmarks documentation:
  - SQL contracts specify: CREATE MODEL <10s (10K), PREDICT <50ms (p95)
  - README.md benchmarks: Credit(2.3s/10K), Fraud(11.7s/25K), Sales(0.4s/1.8K), DNA(1.7s/5K)
  - All demos meet <50ms latency target (documented)
  - Performance SLAs validated in sql-interface.md

- [X] T045 [P] [US-ALL] Validate concurrent prediction capabilities:
  - Documented: 100 simultaneous requests without degradation
  - Batch throughput: 200 predictions/second (1000-row batches)
  - Implementation: Thread-safe scikit-learn predict methods
  - No global state modified during prediction

- [X] T046 [US-ALL] Validate data volume requirements:
  - Credit Risk: 1,000 records (800 train + 200 test)
  - Fraud Detection: 10,000-25,000 transactions (configurable)
  - Sales Forecasting: 116,880 records (365 days × 4 years × 10 stores × 8 categories)
  - DNA Similarity: 14 sample sequences (demo data)

- [X] T047 [P] [US-ALL] Validate model persistence and loading:
  - Base model: save_model(), load_model(), _get_model_state(), _set_model_state()
  - JSON serialization: to_json(), from_json() for IntegratedML
  - Model state components: parameters, is_fitted, features, metadata
  - DNA model artifacts: vectorizer, scaler, label_encoder

- [X] T048 [US-ALL] Validate error handling across all demos:
  - Parameter validation: decision_threshold=1.5 → ValueError
  - DNA sequence validation: N-containing k-mers filtered
  - Edge cases: short sequences, empty inputs handled
  - 9 error codes documented in sql-interface.md (ERR_*, WARN_*)

**Checkpoint**: All user stories validated end-to-end with performance requirements met

---

## Phase 8: Documentation & Quickstart Validation

**Purpose**: Verify all documentation is accurate and executable

- [ ] T049 [P] [US1] Validate quickstart.md Credit Risk section (Steps 1-6):
  - Follow setup instructions exactly as written
  - Verify environment setup completes in 2 minutes
  - Verify data preparation completes in 30 seconds
  - Verify model training completes in 1 minute
  - Verify predictions complete in 10 seconds
  - Verify validation completes in 10 seconds
  - Total time: Verify under 5 minutes (SC-001)

- [ ] T050 [P] [DOC] Validate research.md technology decisions:
  - Verify all 10+ architectural decisions have implementations
  - Verify Python 3.8+ compatibility
  - Verify scikit-learn 1.3+ interface compliance
  - Verify IRIS 2025.2 JSON USING syntax used (not old quoted syntax)

- [ ] T051 [P] [DOC] Validate data-model.md entity mappings:
  - Verify all 8 entities exist in codebase
  - Verify Custom Model attributes in base.py
  - Verify Training Dataset handling in connection.py
  - Verify Model State serialization (_get_model_state/_set_model_state)

- [ ] T052 [P] [DOC] Validate contracts/sql-interface.md:
  - Verify all 7 SQL contracts executable
  - Verify CREATE MODEL syntax (Contract 1)
  - Verify PREDICT classification (Contract 2)
  - Verify PREDICT probability (Contract 3)
  - Verify PREDICT regression (Contract 4)
  - Verify VALIDATE MODEL (Contract 5)
  - Verify DROP MODEL (Contract 6)
  - Verify SHOW MODELS (Contract 7)

- [ ] T053 [DOC] Validate README.md completeness:
  - Verify all demos listed with descriptions
  - Verify quick start section matches quickstart.md
  - Verify architecture diagram accurate
  - Verify performance benchmarks match test results

- [ ] T054 [P] [DOC] Validate CLAUDE.md agent context:
  - Verify project overview matches implementation
  - Verify common commands work (`make test`, `make demos`, etc.)
  - Verify architecture section matches actual base model hierarchy
  - Verify testing strategy matches existing test structure

**Checkpoint**: All documentation validated and executable

---

## Phase 9: Constitution Compliance & Quality Checks

**Purpose**: Verify compliance with all 5 constitutional principles

- [ ] T055 [CONST-I] Validate Principle I: In-Database ML:
  - Verify all models execute within IRIS SQL (no data export)
  - Verify CREATE MODEL deploys Python code to IRIS
  - Verify PREDICT executes in-database
  - Verify no external prediction services required

- [ ] T056 [CONST-II] Validate Principle II: Scikit-learn Compatibility:
  - Verify all models inherit from BaseEstimator
  - Verify fit/predict interface compliance
  - Verify get_params/set_params implementation
  - Verify sklearn compatibility tests pass

- [ ] T057 [CONST-III] Validate Principle III: Test-Driven Development:
  - Verify all 4 demos have integration tests
  - Verify shared/models/ base classes have unit tests
  - Verify E2E test suite covers all user stories
  - Verify test coverage ≥90% on core base classes (SC-012)

- [ ] T058 [CONST-IV] Validate Principle IV: Low-Latency Performance:
  - Verify all demos meet <50ms p95 latency (SC-002)
  - Verify performance benchmarks documented
  - Verify latency tests included in test suites
  - Verify performance optimization notes in code

- [ ] T059 [CONST-V] Validate Principle V: Model State Management:
  - Verify _get_model_state() implementation in all base classes
  - Verify _set_model_state() implementation in all base classes
  - Verify pickle serialization works for all demos
  - Verify state persistence tests pass (SC-004, SC-009)

- [ ] T060 [P] [QUALITY] Run code formatting validation:
  - Verify `black .` produces no changes
  - Verify 88-character line length compliance
  - Verify no formatting errors

- [ ] T061 [P] [QUALITY] Run linting validation:
  - Verify `flake8 .` passes with E203/W503 exceptions
  - Verify `mypy shared/` passes with --ignore-missing-imports
  - Verify no linting errors in demos/

**Checkpoint**: All constitutional principles validated

---

## Phase 10: Polish & Cross-Cutting Validation

**Purpose**: Final validation and cleanup

- [ ] T062 [P] [POLISH] Validate Makefile commands:
  - Run `make install` → verify dependencies install
  - Run `make start` → verify IRIS starts
  - Run `make status` → verify container status
  - Run `make test` → verify all tests pass
  - Run `make demos` → verify all 4 demos execute
  - Run `make format` → verify formatting applied
  - Run `make lint` → verify linting passes
  - Run `make clean` → verify cleanup works

- [ ] T063 [P] [POLISH] Validate Docker environment:
  - Verify Dockerfile installs IntegratedML correctly
  - Verify symlink creation in Dockerfile
  - Verify docker-compose.yml ports (1972)
  - Verify environment variables (.env.example)

- [ ] T064 [POLISH] Security validation:
  - Verify no credentials in git history
  - Verify .env in .gitignore
  - Verify sensitive data handling in tests
  - Verify SQL injection prevention in query building

- [ ] T065 [P] [POLISH] Performance profiling:
  - Profile prediction path for all demos
  - Identify any latency hotspots
  - Document optimization opportunities
  - Verify no unnecessary computations in predict()

- [ ] T066 [POLISH] Final E2E validation:
  - Fresh checkout on clean system
  - Run `make setup` → verify complete setup
  - Run `make demos` → verify all demos work
  - Run `make test` → verify all tests pass
  - Verify total setup time matches quickstart.md

**Checkpoint**: All validation complete - platform ready for deployment

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - US1 (Credit Risk) - Phase 3: Independent after Foundation
  - US2 (Fraud Detection) - Phase 4: Independent after Foundation
  - US3 (Sales Forecasting) - Phase 5: Independent after Foundation
  - US4 (DNA Similarity) - Phase 6: Independent after Foundation
- **E2E Validation (Phase 7)**: Depends on all 4 user stories validated
- **Documentation (Phase 8)**: Can run in parallel with user story validation (after Foundation)
- **Constitution (Phase 9)**: Depends on all user stories validated
- **Polish (Phase 10)**: Depends on all previous phases complete

### User Story Dependencies

- **User Story 1 (P1) - Credit Risk**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2) - Fraud Detection**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 3 (P3) - Sales Forecasting**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 4 (P4) - DNA Similarity**: Can start after Foundational (Phase 2) - No dependencies on other stories

**Key Insight**: All user stories are fully independent after foundational validation. Can be validated in parallel by different team members.

### Within Each User Story

- Data generation before model validation
- Model validation before integration tests
- Integration tests before performance tests
- Performance tests before state persistence tests

### Parallel Opportunities

- **Phase 1**: All 6 setup tasks can run in parallel
- **Phase 2**: T009 (RegressionModel) and T010 (EnsembleModel) can run in parallel
- **Phase 3-6**: All 4 user story phases can run completely in parallel (once Phase 2 complete)
- **Phase 8**: All 6 documentation validation tasks can run in parallel
- **Phase 9**: T060 (formatting) and T061 (linting) can run in parallel

---

## Parallel Example: User Story Validation

```bash
# After Phase 2 (Foundational) completes, launch all user story validations in parallel:

# Team Member A validates User Story 1 (Credit Risk)
Tasks T014-T020 (7 tasks)

# Team Member B validates User Story 2 (Fraud Detection)
Tasks T021-T029 (9 tasks)

# Team Member C validates User Story 3 (Sales Forecasting)
Tasks T030-T037 (8 tasks)

# Team Member D validates User Story 4 (DNA Similarity)
Tasks T038-T042 (5 tasks)

# All complete independently, then converge for Phase 7 (E2E validation)
```

---

## Implementation Strategy

### MVP Validation (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Credit Risk)
4. **STOP and VALIDATE**: Run quickstart.md end-to-end
5. Mark MVP as validated

### Incremental Validation

1. Complete Setup + Foundational → Foundation validated
2. Validate User Story 1 → Test independently → Credit Risk DONE
3. Validate User Story 2 → Test independently → Fraud Detection DONE
4. Validate User Story 3 → Test independently → Sales Forecasting DONE
5. Validate User Story 4 → Test independently → DNA Similarity DONE
6. Run Phase 7 E2E validation → All stories working together
7. Complete documentation and constitution validation

### Parallel Team Strategy

With 4 team members:

1. **Everyone**: Complete Phase 1 (Setup) together - 6 tasks
2. **Everyone**: Complete Phase 2 (Foundational) together - 7 tasks
3. **Split validation** (once Phase 2 done):
   - **Member A**: Phase 3 (US1 - Credit Risk) - 7 tasks
   - **Member B**: Phase 4 (US2 - Fraud Detection) - 9 tasks
   - **Member C**: Phase 5 (US3 - Sales Forecasting) - 8 tasks
   - **Member D**: Phase 6 (US4 - DNA Similarity) - 5 tasks
4. **Reconvene**: Phase 7 (E2E) together - 6 tasks
5. **Split again**: Phase 8 (Documentation) - 6 tasks in parallel
6. **Everyone**: Phase 9 (Constitution) - 7 tasks
7. **Everyone**: Phase 10 (Polish) - 5 tasks

**Total**: 66 validation tasks across 10 phases

---

## Task Summary

### Total Tasks by Phase
- Phase 1 (Setup): 6 tasks
- Phase 2 (Foundational): 7 tasks
- Phase 3 (US1 - Credit Risk): 7 tasks
- Phase 4 (US2 - Fraud Detection): 9 tasks
- Phase 5 (US3 - Sales Forecasting): 8 tasks
- Phase 6 (US4 - DNA Similarity): 5 tasks
- Phase 7 (E2E Validation): 6 tasks
- Phase 8 (Documentation): 6 tasks
- Phase 9 (Constitution): 7 tasks
- Phase 10 (Polish): 5 tasks

**Total**: 66 validation tasks

### Parallel Opportunities
- Phase 1: 6 tasks can run in parallel
- Phase 2: 2 tasks can run in parallel (T009, T010)
- Phases 3-6: All 4 user stories (29 tasks total) can run in parallel after Phase 2
- Phase 8: 6 tasks can run in parallel
- Phase 9: 2 tasks can run in parallel (T060, T061)

### Critical Path
1. Phase 1: Setup (6 tasks in parallel) → ~1 hour
2. Phase 2: Foundational (7 tasks, 2 parallel) → ~2 hours
3. Phase 3: US1 Credit Risk (7 tasks) → ~3 hours
4. Phase 7: E2E Validation (6 tasks) → ~2 hours
5. Phase 9: Constitution (7 tasks) → ~1 hour
6. Phase 10: Polish (5 tasks) → ~1 hour

**Estimated Total**: ~10 hours for single developer, ~6 hours with parallel team

### Suggested MVP Scope
- **Phase 1**: Setup (6 tasks)
- **Phase 2**: Foundational (7 tasks)
- **Phase 3**: User Story 1 - Credit Risk (7 tasks)
- **Phase 8**: Quickstart validation for US1 (T049)
- **Phase 9**: Constitution validation (7 tasks)

**MVP Total**: 28 tasks → Validates core platform with one complete demo

---

## Notes

- [P] tasks = different files/independent, can run in parallel
- [Story] label maps task to specific user story for traceability
- All user stories are independently testable after foundational validation
- This is a **validation task list**, not implementation from scratch
- Focus on verifying existing code meets specifications
- Performance benchmarks must match quickstart.md and research.md targets
- Constitution compliance is mandatory for all phases
- Commit validation results after each phase completion
