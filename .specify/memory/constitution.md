# IntegratedML Custom Models Constitution

<!--
  ============================================================================
  SYNC IMPACT REPORT - Constitution v1.0.0
  ============================================================================

  Version Change: INITIAL → 1.0.0

  Rationale: Initial constitution creation for IntegratedML Custom Models
  project. This establishes the foundational governance and principles for
  developing custom ML models that execute within InterSystems IRIS SQL.

  Principles Defined:
  - I. In-Database ML (NEW)
  - II. Scikit-learn Compatibility (NEW)
  - III. Test-Driven Development (NEW)
  - IV. Low-Latency Performance (NEW)
  - V. Model State Management (NEW)

  Templates Requiring Updates:
  ✅ plan-template.md - Constitution Check section references this file
  ✅ spec-template.md - Requirements align with model development principles
  ✅ tasks-template.md - Task categorization supports testing and performance gates

  Follow-up TODOs: None - all critical fields populated
  ============================================================================
-->

## Core Principles

### I. In-Database ML

All machine learning models MUST execute within InterSystems IRIS SQL without
requiring data export or external processing pipelines.

**Rationale**: The core value proposition of IntegratedML Custom Models is
eliminating data movement. Models process data where it lives, enabling
real-time predictions via `PREDICT()` directly in SQL queries. This principle
is non-negotiable and defines the project's purpose.

**Requirements**:
- Models MUST be deployable via SQL `CREATE MODEL` statements
- Training MUST execute in-database using `FROM table` syntax
- Predictions MUST return directly to SQL result sets
- NO data export to external systems for model execution
- Parameters MUST be passed via JSON `USING` clause (IRIS 2025.2 syntax)

### II. Scikit-learn Compatibility

All custom models MUST implement the scikit-learn estimator interface
(`fit`, `predict`, `get_params`, `set_params`).

**Rationale**: IntegratedML requires scikit-learn compatible models to ensure
proper integration with the IRIS ML engine. This interface provides consistent
model lifecycle management, parameter handling, and serialization.

**Requirements**:
- Models MUST inherit from `IntegratedMLBaseModel` (shared/models/base.py)
- `fit(X, y, **params)` method MUST accept training data and IntegratedML params
- `predict(X)` method MUST return predictions matching IntegratedML expectations
- Models MUST support parameter serialization via `get_params()`/`set_params()`
- Custom preprocessing MUST occur within model methods, not externally

### III. Test-Driven Development (NON-NEGOTIABLE)

All features and models MUST have tests covering functional correctness,
integration with IRIS, and performance benchmarks.

**Rationale**: Given the critical nature of ML predictions in production
workflows (credit risk, fraud detection, etc.), comprehensive testing is
mandatory. Performance benchmarks ensure latency requirements are met.

**Requirements**:
- Unit tests MUST verify model logic (fit/predict behavior)
- Integration tests MUST verify SQL integration (CREATE MODEL, PREDICT queries)
- Performance benchmarks MUST measure training time and prediction latency
- Test data generators MUST produce realistic volumes (>1000 records minimum)
- Tests MUST be runnable via `make test` or `pytest demos/*/tests/`
- E2E test (`tests/test_all_demos_e2e.py`) MUST pass before releases

### IV. Low-Latency Performance

Model predictions MUST complete within 50ms (p95) to support real-time
applications.

**Rationale**: IntegratedML Custom Models target production use cases like
fraud detection and credit risk assessment that require immediate responses.
Sub-50ms latency ensures models can be used in interactive applications and
high-throughput batch processing.

**Requirements**:
- Prediction latency MUST be <50ms p95 for single-record predictions
- Training time SHOULD be documented in demo test results
- Performance benchmarks MUST be included in integration tests
- Models MUST avoid unnecessary computation in predict path
- Feature engineering MUST be optimized for repeated prediction calls

### V. Model State Management

Models MUST implement proper serialization to persist across database sessions
and deployments.

**Rationale**: IRIS stores trained models for reuse across queries and server
restarts. Proper state management ensures models remain available and
consistent without retraining.

**Requirements**:
- Models MUST implement `_get_model_state()` for serialization
- Models MUST implement `_set_model_state()` for deserialization
- Model state MUST include all trained parameters and preprocessing artifacts
- Serialization MUST be compatible with IRIS persistence mechanisms
- Models MUST handle version compatibility for state loaded from older versions

## Technical Standards

### Python Environment

- **Python Version**: 3.8+ (compatible with IRIS 2025.2 Python runtime)
- **Dependencies**: Managed via pyproject.toml with uv or pip
- **Code Style**: Black formatting (line length 88)
- **Type Hints**: Recommended but not required (mypy for shared/ modules)
- **Linting**: flake8 with E203/W503 exceptions

### IRIS Integration

- **IRIS Version**: 2025.2+ (required for JSON USING clause syntax)
- **IntegratedML Installation**: Via `intersystems-iris-automl` from InterSystems registry
- **Connection**: Environment variables (.env) for host/port/namespace/credentials
- **Deployment**: Docker Compose for reproducible IRIS environment
- **Symlink**: iris_automl symlink MUST be created in Python path

### Model Architecture

- **Base Classes**: Extend IntegratedMLBaseModel, ClassificationModel, RegressionModel, or EnsembleModel
- **Project Structure**: demos/{domain}/models/ for domain-specific implementations
- **Shared Utilities**: shared/database/ for IRIS connections, shared/utils/ for helpers
- **Documentation**: Each demo MUST have README explaining model architecture

## Development Workflow

### Feature Development Process

1. **Specification**: Create feature spec following spec-template.md
2. **Planning**: Generate implementation plan using plan-template.md
3. **Testing**: Write tests FIRST (TDD - fail, then implement)
4. **Implementation**: Build model following architecture patterns
5. **Integration**: Validate SQL integration with IRIS
6. **Benchmarking**: Measure and document performance
7. **Documentation**: Update demo README with results

### Quality Gates

- All tests MUST pass (`make test`)
- Code MUST be formatted (`make format`)
- Linting MUST pass (`make lint`)
- E2E test MUST complete successfully
- Performance benchmarks MUST meet latency requirements (<50ms p95)
- Demo results MUST be documented in README tables

### Code Review Requirements

- Changes MUST include tests (unit + integration)
- Performance impact MUST be documented for model changes
- SQL syntax MUST use IRIS 2025.2 JSON USING clause format
- Breaking changes MUST be called out explicitly
- Commit messages MUST follow conventional commits format

## Governance

This constitution supersedes all other development practices and guides all
technical decisions for IntegratedML Custom Models.

### Amendment Process

1. Propose amendment with clear rationale and scope of impact
2. Document breaking changes to existing principles
3. Update affected templates (plan, spec, tasks) for consistency
4. Increment constitution version per semantic versioning:
   - MAJOR: Backward-incompatible principle removal/redefinition
   - MINOR: New principle added or materially expanded guidance
   - PATCH: Clarifications, wording fixes, non-semantic refinements
5. Obtain approval from project maintainers
6. Execute migration plan if existing code affected

### Compliance Review

- All PRs MUST verify alignment with principles
- Constitution violations MUST be justified in plan.md Complexity Tracking table
- Unjustified complexity additions WILL be rejected
- Performance regressions below constitutional thresholds WILL be rejected
- Tests bypassing TDD process WILL be rejected

### Runtime Guidance

For AI development assistants: Consult CLAUDE.md for project-specific command
references, common workflows, and architectural patterns. The constitution
establishes WHAT to build; CLAUDE.md explains HOW to build it efficiently.

**Version**: 1.0.0 | **Ratified**: 2025-10-10 | **Last Amended**: 2025-10-10
