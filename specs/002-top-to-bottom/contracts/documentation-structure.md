# Documentation Structure Contract

**Feature**: Documentation and Code Hygiene Review
**Date**: 2025-10-10
**Version**: 1.0

## Overview

This contract defines the structural requirements for all documentation files in the IntegratedML Custom Models project. It specifies directory organization, file naming conventions, content requirements, and validation rules to ensure consistency and discoverability.

---

## Contract 1: Top-Level README

**Purpose**: Serve as the primary entry point for all project documentation.

**File Path**: `README.md` (repository root)

**Required Sections**:
1. **Project Title and Description** (H1)
2. **Table of Contents** (with jump links to all major sections)
3. **Overview** - 2-3 paragraph project summary
4. **Quick Start** - Minimal steps to run first demo (<5 minutes)
5. **Architecture Overview** - Base model hierarchy diagram + brief explanation
6. **Documentation Map** - Links to docs/, demos/, specs/ with purpose statements
7. **Performance Benchmarks** - Table with demo results (training time, accuracy, latency)
8. **Contributing** - Link to CONTRIBUTING.md or inline guidelines
9. **License** - Link to LICENSE file

**Preconditions**:
- Repository contains at least one demo in demos/
- docs/ directory exists with architecture.md and api_reference.md

**Postconditions**:
- New developers can locate setup instructions within 2 minutes (SC-001 component)
- All major documentation categories linked from TOC
- Performance benchmarks visible without navigating to individual demos

**Validation**:
```bash
# Check required sections exist
grep -q "^# " README.md  # H1 title
grep -q "## Table of Contents" README.md
grep -q "## Overview" README.md
grep -q "## Quick Start" README.md
grep -q "## Architecture" README.md
grep -q "## Documentation" README.md
```

---

## Contract 2: Demo README

**Purpose**: Provide demo-specific setup and usage instructions without duplicating project-wide documentation.

**File Path**: `demos/{demo_name}/README.md`

**Required Sections**:
1. **Demo Title** (H1) - e.g., "Credit Risk Classification Demo"
2. **Overview** - 1 paragraph describing demo purpose
3. **Quick Start** - 3-5 steps to run the demo
4. **Architecture** - Diagram + explanation of demo-specific model architecture
5. **Performance Benchmarks** - Training time, accuracy, latency for this demo
6. **Usage Examples** - 2-3 code snippets showing SQL integration
7. **Troubleshooting** - Common issues and solutions
8. **See Also** - Links to docs/api_reference.md, docs/architecture.md, extended technical docs

**Constraints**:
- **Maximum Length**: 500 lines (SC-007)
- **No Duplication**: MUST link to central API docs instead of duplicating method signatures (FR-005, FR-015)
- **Demo-Specific Only**: MUST NOT include project-wide setup instructions (those belong in top-level README)

**Preconditions**:
- Demo directory contains models/, data/, tests/ subdirectories
- Demo has at least one test file in tests/ subdirectory

**Postconditions**:
- Developers can run demo without reading project-wide documentation
- Extended technical content (if needed) extracted to demos/{demo_name}/docs/

**Validation**:
```bash
# Check length limit
LINES=$(wc -l < demos/credit_risk/README.md)
[ $LINES -le 500 ] || echo "ERROR: Demo README exceeds 500 lines"

# Check for API duplication (should link, not duplicate)
grep -q "docs/api_reference.md" demos/credit_risk/README.md || \
  echo "WARNING: Missing API reference link"
```

---

## Contract 3: Centralized API Reference

**Purpose**: Provide single authoritative source for all public API documentation.

**File Path**: `docs/api_reference.md`

**Required Sections**:
1. **Title** (H1) - "API Reference"
2. **Table of Contents** (required - document >100 lines)
3. **Base Model Classes** (H2)
   - IntegratedMLBaseModel (H3)
   - ClassificationModel (H3)
   - RegressionModel (H3)
   - EnsembleModel (H3)
4. **Public Methods** (H3 per method)
   - Method signature
   - Parameters (name, type, description)
   - Return value (type, description)
   - Usage example (code snippet)
   - Related methods (cross-references)

**Method Documentation Template**:
```markdown
### `fit(X, y, **params)`

**Purpose**: Train the model on provided dataset.

**Parameters**:
- `X` (pandas.DataFrame): Feature matrix with training data
- `y` (pandas.Series): Target variable values
- `**params` (dict): Additional IntegratedML parameters from USING clause

**Returns**: `self` (for method chaining)

**Raises**:
- `ValueError`: If X and y shapes mismatch
- `AttributeError`: If required preprocessing artifacts missing

**Example**:
\```python
model = CustomCreditRiskClassifier(enable_debt_ratio=True)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
\```

**See Also**: [predict()](#predict), [get_params()](#get_params)
```

**Preconditions**:
- All public methods in shared/models/ have docstrings
- Method signatures match scikit-learn estimator interface (fit, predict, get_params, set_params)

**Postconditions**:
- 100% coverage of public methods (SC-009)
- Demo READMEs link to this file instead of duplicating signatures (SC-002 zero duplication)

**Validation**:
```bash
# Extract all public methods from shared/models/ base classes
grep -r "def [a-z_]*(" shared/models/*.py | grep -v "def _" > methods.txt

# Check each method documented in API reference
while read method; do
  grep -q "$method" docs/api_reference.md || echo "Missing: $method"
done < methods.txt
```

---

## Contract 4: Architecture Documentation

**Purpose**: Explain design decisions and architectural patterns.

**File Path**: `docs/architecture.md`

**Required Sections**:
1. **Title** (H1) - "Architecture"
2. **Table of Contents** (required - document >100 lines)
3. **Base Model Hierarchy** (H2) - ASCII diagram + explanation
4. **Feature Engineering Pipeline** (H2) - How preprocessing works
5. **Model State Management** (H2) - Serialization and persistence
6. **IRIS Integration** (H2) - How models execute in SQL
7. **Directory Structure** (H2) - Explanation of demos/, shared/, tests/ organization

**Required Diagrams** (SC-008: at least 3 visual representations):
1. Base model class hierarchy
2. Directory structure tree
3. Data flow (training → prediction)

**Preconditions**:
- Base model classes exist in shared/models/
- At least one demo demonstrates architecture patterns

**Postconditions**:
- Senior developers understand design patterns without reading code (SC-001 architecture component)
- All architectural decisions documented with rationale

**Validation**:
```bash
# Check for required diagrams (ASCII art or code blocks)
grep -c "```" docs/architecture.md  # Should be ≥3 for diagrams
```

---

## Contract 5: .gitignore Patterns

**Purpose**: Prevent accidental commit of temporary files and IDE configurations.

**File Path**: `.gitignore` (repository root)

**Required Pattern Categories**:
1. **Python Artifacts** - `__pycache__/`, `*.pyc`, `.pytest_cache/`
2. **IDE Configurations** - `.vscode/`, `.idea/`, `*.swp`
3. **OS-Specific Files** - `.DS_Store`, `Thumbs.db`
4. **IntegratedML Artifacts** - `iris.log`, `iris.pid`, `docker/volumes/iris_data/`
5. **Model Artifacts** - `*.pkl`, `*.joblib`, `checkpoints/`

**Pattern Organization**:
```gitignore
# Category Comment
pattern1
pattern2

# Next Category Comment
pattern3
```

**Preconditions**:
- Development workflows include pytest, Docker, multiple IDEs (VS Code, PyCharm)

**Postconditions**:
- git status after pytest shows zero untracked __pycache__/ (SC-003)
- git status after IDE usage shows zero .vscode/ or .idea/ (SC-003)
- Covers at least 5 tools/environments (SC-011)

**Validation**:
```bash
# Run development tasks and check git status
pytest demos/credit_risk/tests/
docker-compose up -d
git status --short | grep -E "(__pycache__|.pytest_cache|.DS_Store)" && \
  echo "ERROR: Artifacts not ignored"
```

---

## Contract 6: Documentation Contribution Guide

**Purpose**: Define where developers should add new documentation.

**File Path**: `docs/CONTRIBUTING_DOCS.md`

**Required Content**:
1. **Decision Flowchart** - Where to add documentation based on content type
2. **Examples** - Concrete examples for each documentation category
3. **Cross-Reference Guidelines** - How to link between documents
4. **TOC Guidelines** - When to add table of contents (>100 lines)
5. **Link Format** - Relative links for docs, absolute paths for code

**Decision Tree**:
```
Is it a feature specification? → specs/###-feature/spec.md
Is it project-wide implementation? → docs/{architecture,api,deployment}/
Is it demo-specific? → demos/*/README.md
Is it repository maintenance? → .github/REPOSITORY_SETUP.md
Is it AI agent guidance? → CLAUDE.md
```

**Preconditions**:
- Project has established documentation structure (docs/, demos/, specs/)

**Postconditions**:
- Developers know where to add documentation without asking (FR-014)
- Documentation duplication prevented through clear guidelines

**Validation**: Manual review during code review

---

## Contract 7: Link Validation

**Purpose**: Ensure all documentation cross-references are valid.

**Validation Frequency**: Before merge to main branch

**Validation Script** (optional automation):
```bash
#!/bin/bash
# documentation_audit.sh

# Find all markdown files
find . -name "*.md" | while read file; do
  echo "Checking $file..."

  # Extract markdown links [text](path)
  grep -oE '\[.*\]\([^)]+\)' "$file" | while read link; do
    path=$(echo "$link" | sed -E 's/.*\(([^)]+)\)/\1/')

    # Skip external URLs
    [[ "$path" =~ ^https?:// ]] && continue

    # Skip anchors within same file
    [[ "$path" =~ ^# ]] && continue

    # Resolve relative path from file location
    dir=$(dirname "$file")
    target="$dir/$path"

    # Check if target exists
    [ -f "$target" ] || echo "  BROKEN: $link"
  done
done
```

**Success Criteria**: Zero broken links (SC-005)

**Error Response**: Pull request blocked until broken links fixed

---

## Performance Requirements

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Locate setup instructions | <2 minutes | Developer usability test |
| Find API reference | <1 minute | Navigation from demo README |
| Complete onboarding | <10 minutes | Full setup without help (SC-001) |
| Documentation search | <2 minutes | Find specific topic using TOC/grep |

---

## Versioning and Compatibility

**Documentation Version**: 1.0

**Backward Compatibility**:
- All existing links preserved or redirected
- No breaking changes to file paths
- Deprecated documents marked with migration guide

**Breaking Change Policy**:
- File moves REQUIRE updating all cross-references
- Directory restructuring REQUIRES deprecation notice period
- Major TOC changes REQUIRE communication in release notes

---

## Summary

All documentation structure contracts defined for:
- ✅ Top-level README (FR-001)
- ✅ Demo READMEs (FR-003, SC-007)
- ✅ Centralized API reference (FR-009, SC-009)
- ✅ Architecture documentation (FR-008, SC-008)
- ✅ .gitignore patterns (FR-004, FR-013, SC-003, SC-011)
- ✅ Contribution guidelines (FR-014)
- ✅ Link validation (SC-005)

Ready for quickstart.md generation (Phase 1 continued).
