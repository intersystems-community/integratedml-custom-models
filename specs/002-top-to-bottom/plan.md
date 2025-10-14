# Implementation Plan: Documentation and Code Hygiene Review

**Branch**: `002-top-to-bottom` | **Date**: 2025-10-10 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-top-to-bottom/spec.md`

**Note**: This template is filled in by the `/plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

This feature reorganizes project documentation to improve developer onboarding and maintains code hygiene through comprehensive .gitignore patterns. The primary requirement is to establish a clear documentation hierarchy that enables new developers to locate setup instructions within 10 minutes and prevents accidental commits of temporary files or IDE configurations. The technical approach involves auditing existing documentation for duplication, consolidating content with cross-references, adding table of contents to major documents, and ensuring .gitignore covers all common Python/IDE/OS artifacts.

## Technical Context

**Language/Version**: Markdown (GitHub-flavored), Git 2.x+, Bash scripting for automation
**Primary Dependencies**: None (documentation work; existing tools: grep, find, tree, markdown linters optional)
**Storage**: Git version control for all documentation files (.md); no database required
**Testing**: Manual validation via documentation audit script, link checking (optional CI), developer usability testing
**Target Platform**: Cross-platform (macOS, Linux, Windows) - documentation readable in any markdown viewer
**Project Type**: Documentation reorganization - applies to existing IntegratedML Custom Models project structure
**Performance Goals**: Developer onboarding <10 minutes; documentation search/navigation <2 minutes per task
**Constraints**: Must preserve existing content; zero breaking changes to code; backward compatibility for all links
**Scale/Scope**: ~50 markdown files across project; 4 demo directories; docs/ reorganization; .gitignore comprehensive coverage

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: In-Database ML ✅ NOT APPLICABLE
- **Status**: N/A - This feature does not add or modify ML models
- **Justification**: Documentation reorganization does not affect SQL integration, data movement, or model execution
- **Verification**: No changes to shared/models/ or demos/*/models/; no SQL contracts modified

### Principle II: Scikit-learn Compatibility ✅ NOT APPLICABLE
- **Status**: N/A - No model interface changes
- **Justification**: Documentation work does not modify IntegratedMLBaseModel, fit/predict methods, or parameter handling
- **Verification**: No Python model code affected; shared/models/base.py unchanged

### Principle III: Test-Driven Development ✅ PASSES
- **Status**: PASSES - Documentation changes testable via audits and developer usability tests
- **Requirements Met**:
  - Documentation audit script validates structure (SC-002: zero duplicates)
  - Link validation checks for broken references (SC-005: zero broken links)
  - Developer usability test measures onboarding time (SC-001: <10 minutes)
  - .gitignore validation via git status after development tasks (SC-003: zero untracked artifacts)
- **Verification**: Create documentation_audit.sh script; run usability test with 3 developers; validate gitignore with common workflows

### Principle IV: Low-Latency Performance ✅ NOT APPLICABLE
- **Status**: N/A - No runtime performance impact
- **Justification**: Documentation changes do not affect model prediction latency, training time, or feature engineering
- **Verification**: No changes to predict() methods; no model state modifications

### Principle V: Model State Management ✅ NOT APPLICABLE
- **Status**: N/A - No serialization changes
- **Justification**: Documentation does not modify _get_model_state(), _set_model_state(), or persistence mechanisms
- **Verification**: No changes to model state methods or pickle compatibility

### Technical Standards Compliance ✅ PASSES

**Python Environment**: N/A - No Python code changes
**IRIS Integration**: N/A - No IRIS configuration changes
**Model Architecture**: N/A - No model hierarchy changes

**Quality Gates for Documentation**:
- ✅ Markdown formatting consistent (SC-012: markdown linter with zero errors)
- ✅ Table of contents added to long documents (SC-004: all docs >100 lines)
- ✅ Cross-references valid (SC-005: zero broken links)
- ✅ Demo READMEs focused (SC-007: <500 lines, demo-specific)

### Constitution Summary

**Overall Status**: ✅ **PASSES**

- 3 of 5 core principles NOT APPLICABLE (documentation work only)
- 2 of 5 core principles PASSES (TDD via audits/tests; quality gates defined)
- No constitution violations
- No complexity justifications needed

This feature enhances project maintainability without affecting ML functionality. All changes are additive (reorganization, consolidation, improved navigation) with zero breaking changes to existing code or model behavior.

## Project Structure

### Documentation (this feature)

```
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

**Current Structure** (documentation targets):
```
/Users/tdyar/ws/pluggable_iml/
├── README.md                          # Top-level entry point (FR-001)
├── CLAUDE.md                          # Project-specific AI guidance
├── LICENSE
├── .gitignore                         # Code hygiene target (FR-004, FR-013)
├── .env.example
├── requirements.txt
├── pyproject.toml
├── Makefile
├── docker-compose.yml
├── docs/                              # Centralized documentation (FR-002)
│   ├── architecture.md                # Architecture decisions (FR-008)
│   ├── api_reference.md               # Central API docs (FR-009)
│   ├── deployment.md
│   ├── DOCKER_SETUP.md
│   ├── QUICK_GUIDE_CUSTOM_MODELS.md
│   ├── user_guide.md
│   ├── architecture/                  # Architecture diagrams (FR-008)
│   ├── api/                           # API details
│   └── tutorials/                     # Step-by-step guides
│       ├── tutorial_01_credit_risk.md
│       ├── tutorial_02_fraud_detection.md
│       ├── tutorial_03_sales_forecasting.md
│       └── tutorial_04_custom_models.md
├── demos/                             # Demo-specific docs (FR-003)
│   ├── credit_risk/
│   │   ├── README.md                  # Demo-specific (FR-003, SC-007)
│   │   ├── TECHNICAL_DOCUMENTATION.md
│   │   ├── models/
│   │   ├── data/
│   │   └── tests/
│   ├── fraud_detection/
│   │   ├── README.md
│   │   ├── docs/
│   │   │   ├── ARCHITECTURE.md
│   │   │   └── PERFORMANCE_BENCHMARKS.md
│   │   ├── models/
│   │   ├── data/
│   │   └── tests/
│   ├── sales_forecasting/
│   │   ├── README.md
│   │   ├── models/
│   │   ├── data/
│   │   └── tests/
│   └── dna_similarity/
│       ├── README.md
│       ├── models/
│       ├── data/
│       └── tests/
├── shared/                            # Shared library code (no doc changes)
│   ├── models/
│   ├── database/
│   ├── utils/
│   └── testing/
├── specs/                             # Feature specifications (FR-010)
│   ├── 001-use-the-current/
│   └── 002-top-to-bottom/            # This feature
├── tests/                             # E2E tests
└── .github/                           # GitHub templates
    ├── ISSUE_TEMPLATE/
    ├── workflows/
    ├── pull_request_template.md
    └── REPOSITORY_SETUP.md
```

**Structure Decision**:

This is a documentation reorganization project applied to an existing IntegratedML Custom Models codebase. The structure follows "Option 1: Single project" pattern with specialized demo directories. Documentation targets include:

1. **Top-level** (SC-006): Reduce to <15 essential files by organizing content into subdirectories
2. **docs/** (FR-002): Centralize cross-cutting documentation with logical subdirectories
3. **demos/*** (FR-003): Each demo has focused README.md (<500 lines) with cross-references to central docs
4. **specs/** (FR-010): Feature specifications separate from implementation documentation

No source code modifications required - this feature only reorganizes markdown files and updates .gitignore.

## Complexity Tracking

*No complexity justifications required - all constitution checks passed.*

**Constitution Summary**: 3 of 5 principles N/A (documentation work only), 2 of 5 principles PASS. No violations detected.

This feature requires no architectural complexity justifications. All changes are standard documentation reorganization activities with established best practices (markdown hierarchy, gitignore patterns, cross-referencing).
