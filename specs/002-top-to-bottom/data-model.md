# Data Model: Documentation and Code Hygiene Review

**Feature**: Documentation and Code Hygiene Review
**Branch**: 002-top-to-bottom
**Date**: 2025-10-10
**Status**: Design Complete

## Overview

This document defines the key entities involved in documentation organization and code hygiene management. These are conceptual entities representing documentation structure and git ignore patterns, not database tables or API objects.

---

## Entity Definitions

### 1. Documentation File

**Description**: A markdown (.md) file containing user-facing information about the project, organized within the three-tier documentation hierarchy (specs/, docs/, demos/).

**Attributes**:
- `file_path` (string): Absolute or relative path to .md file (e.g., "docs/architecture.md", "demos/credit_risk/README.md")
- `primary_topic` (enum): Main subject (setup | architecture | api | tutorial | demo | specification)
- `target_audience` (list[enum]): Intended readers (new_developer | senior_developer | stakeholder | contributor)
- `last_updated` (datetime): Last modification timestamp from git history
- `word_count` (integer): Approximate document size in words
- `line_count` (integer): Total lines (used for TOC requirement threshold >100 lines)
- `has_toc` (boolean): Whether document includes table of contents
- `cross_references` (list[string]): Links to other documentation files or code locations

**Relationships**:
- Belongs to: Documentation Directory
- Contains: Documentation Sections
- Links to: Other Documentation Files (cross-references)
- References: Code Files (via file paths with line numbers)

**Validation Rules**:
- `file_path` MUST end with `.md` extension
- `line_count` > 100 REQUIRES `has_toc` = true (FR-011)
- `primary_topic` = "api" IMPLIES file_path includes "docs/api_reference.md" (FR-009 centralized API docs)
- `primary_topic` = "demo" IMPLIES `line_count` ≤ 500 (SC-007 demo README length limit)
- `cross_references` MUST use relative paths for markdown links, absolute paths for code references (Decision 4)

**State Transitions**:
1. **Created**: New documentation file added to repository
2. **Draft**: Content being written; may not meet validation rules yet
3. **Complete**: All validation rules pass; ready for review
4. **Published**: Merged to main branch; available to users
5. **Deprecated**: Marked for removal or consolidation with migration guide

**Examples**:
```python
{
  "file_path": "docs/api_reference.md",
  "primary_topic": "api",
  "target_audience": ["new_developer", "senior_developer"],
  "last_updated": "2025-10-10T14:30:00Z",
  "word_count": 3500,
  "line_count": 285,
  "has_toc": true,
  "cross_references": [
    "shared/models/base.py:50-120",
    "shared/models/classification.py:30-80",
    "docs/architecture.md#base-model-hierarchy"
  ]
}
```

---

### 2. Documentation Directory

**Description**: A folder organizing related documentation files within the project structure (docs/, demos/*, specs/*).

**Attributes**:
- `directory_path` (string): Path relative to repository root (e.g., "docs/architecture", "demos/credit_risk")
- `purpose_statement` (string): Brief description of directory contents (e.g., "Architecture diagrams and design decisions")
- `has_index` (boolean): Whether directory contains README.md or index.md
- `subdirectory_count` (integer): Number of nested directories
- `file_count` (integer): Number of .md files directly in this directory

**Relationships**:
- Contains: Documentation Files
- Contains: Subdirectories (nested Documentation Directories)
- Part of: Parent Documentation Directory

**Validation Rules**:
- `directory_path` in ["docs/", "demos/*/", "specs/*/"] REQUIRES `has_index` = true for discoverability
- `directory_path` = "demos/*" REQUIRES file_count ≥ 1 (minimum: README.md per FR-003)
- `directory_path` = "docs/" RECOMMENDS subdirectories ["architecture/", "api/", "tutorials/"] per FR-002

**Directory Categories**:
- **Specification Directories** (`specs/###-feature/`): Feature specs, plans, research, data models, contracts
- **Implementation Directories** (`docs/`): Cross-cutting documentation (architecture, API, deployment)
- **Example Directories** (`demos/*/`): Demo-specific setup, usage, benchmarks

**Examples**:
```python
{
  "directory_path": "docs/architecture",
  "purpose_statement": "Architecture diagrams, base model hierarchy, design patterns",
  "has_index": false,  # No index.md; architecture.md in parent docs/
  "subdirectory_count": 0,
  "file_count": 0  # Content in docs/architecture.md, not subdirectory
}

{
  "directory_path": "demos/credit_risk",
  "purpose_statement": "Credit risk classification demo with custom feature engineering",
  "has_index": true,  # README.md present
  "subdirectory_count": 4,  # models/, data/, tests/, docs/
  "file_count": 2  # README.md + TECHNICAL_DOCUMENTATION.md
}
```

---

### 3. Code Artifact

**Description**: A temporary or generated file that should not be version controlled, identified by .gitignore patterns.

**Attributes**:
- `file_pattern` (string): Glob pattern matching artifact files (e.g., "*.pyc", "__pycache__/", ".DS_Store")
- `source_tool` (enum): Tool/environment generating artifact (pytest | vscode | pycharm | macos | docker | iris)
- `gitignore_rule` (string): Exact line in .gitignore preventing commit
- `category` (enum): Artifact type (python | ide | os | docker | database | temp)
- `cleanup_frequency` (enum): How often removed (never_committed | on_build | on_clean | manual)

**Relationships**:
- Matched by: .gitignore Rules
- Generated by: Development Tools/IDEs/Test Frameworks

**Validation Rules**:
- `source_tool` = pytest REQUIRES `file_pattern` in ["__pycache__/", ".pytest_cache/", "*.pyc"] per FR-004
- `source_tool` = vscode REQUIRES `file_pattern` = ".vscode/" per FR-004
- `source_tool` = pycharm REQUIRES `file_pattern` = ".idea/" per FR-004
- `source_tool` = macos REQUIRES `file_pattern` in [".DS_Store", "._*"] per FR-004
- `source_tool` = iris REQUIRES `file_pattern` in ["iris.log", "iris.pid", "docker/volumes/iris_data/"] per FR-013

**Artifact Categories**:
```
Python Artifacts (category=python):
  - __pycache__/      # Bytecode cache
  - *.pyc, *.pyo      # Compiled Python files
  - *.egg-info/       # Package metadata
  - .pytest_cache/    # Test framework cache

IDE Configurations (category=ide):
  - .vscode/          # VS Code settings
  - .idea/            # PyCharm/IntelliJ settings
  - *.swp, *.swo      # Vim swap files

OS-Specific (category=os):
  - .DS_Store         # macOS directory metadata
  - Thumbs.db         # Windows thumbnail cache

IntegratedML (category=database):
  - iris.log          # IRIS database logs
  - iris.pid          # IRIS process ID
  - docker/volumes/   # Docker persistent data
```

**Examples**:
```python
{
  "file_pattern": "__pycache__/",
  "source_tool": "pytest",
  "gitignore_rule": "__pycache__/",
  "category": "python",
  "cleanup_frequency": "never_committed"
}

{
  "file_pattern": ".DS_Store",
  "source_tool": "macos",
  "gitignore_rule": ".DS_Store",
  "category": "os",
  "cleanup_frequency": "manual"
}
```

---

### 4. Cross-Reference

**Description**: A link from one documentation file to another or from documentation to code, enabling navigation and preventing duplication.

**Attributes**:
- `source_document` (string): Path to markdown file containing the link
- `target_location` (string): Destination (markdown file path or code file with line numbers)
- `link_text` (string): Human-readable description shown to user
- `link_type` (enum): Purpose (navigation | related_reading | code_reference | api_documentation)
- `validity_status` (enum): Link health (valid | broken | redirects | not_checked)
- `last_validated` (datetime): When link was last checked

**Relationships**:
- Originates from: Documentation File (source)
- Points to: Documentation File or Code File (target)

**Validation Rules**:
- `link_type` = "navigation" IMPLIES target is markdown file in same or parent directory (relative link)
- `link_type` = "code_reference" IMPLIES target includes absolute path and optional line numbers (e.g., "shared/models/base.py:50")
- `link_type` = "api_documentation" IMPLIES target is "docs/api_reference.md#section"
- `validity_status` = "broken" TRIGGERS fix required before merge (SC-005 zero broken links)

**Link Formats**:
```markdown
# Navigation (relative markdown link)
[Architecture Overview](../docs/architecture.md)

# Code Reference (absolute path with line numbers)
See `shared/models/base.py:50-120` for base model implementation.

# API Documentation (anchor link)
Refer to [ClassificationModel.fit()](../docs/api_reference.md#classificationmodelfit) for usage.
```

**Examples**:
```python
{
  "source_document": "demos/credit_risk/README.md",
  "target_location": "docs/api_reference.md#classificationmodel",
  "link_text": "ClassificationModel API",
  "link_type": "api_documentation",
  "validity_status": "valid",
  "last_validated": "2025-10-10T14:30:00Z"
}

{
  "source_document": "docs/architecture.md",
  "target_location": "shared/models/base.py:50-120",
  "link_text": "IntegratedMLBaseModel implementation",
  "link_type": "code_reference",
  "validity_status": "valid",
  "last_validated": "2025-10-10T14:30:00Z"
}
```

---

### 5. Documentation Section

**Description**: A major heading within a documentation file, used for table of contents generation and navigation.

**Attributes**:
- `parent_file` (string): Path to markdown file containing this section
- `heading_level` (integer): Depth (1=H1, 2=H2, 3=H3, etc.)
- `heading_text` (string): Section title
- `anchor_id` (string): URL fragment for linking (e.g., "#overview")
- `content_type` (enum): Section purpose (overview | instructions | reference | examples | troubleshooting)
- `word_count` (integer): Approximate words in this section
- `subsection_count` (integer): Number of child sections (one level deeper)

**Relationships**:
- Part of: Documentation File
- Contains: Subsections (child Documentation Sections)
- Linked from: Table of Contents (if parent file >100 lines)

**Validation Rules**:
- `heading_level` MUST be in range [1, 6] (H1-H6 markdown limits)
- `heading_level` sequence MUST NOT skip levels (H1 → H2 → H3, not H1 → H3)
- `parent_file` with line_count > 100 REQUIRES table of contents linking to all heading_level=2 sections (FR-011)
- `anchor_id` MUST be lowercase with hyphens (e.g., "api-reference" for "API Reference")

**TOC Generation Logic**:
```markdown
## Table of Contents

- [Overview](#overview)               # heading_level=2
- [Setup](#setup)                     # heading_level=2
  - [Prerequisites](#prerequisites)   # heading_level=3 (optional in TOC)
- [API Reference](#api-reference)     # heading_level=2
```

**Examples**:
```python
{
  "parent_file": "docs/api_reference.md",
  "heading_level": 2,
  "heading_text": "ClassificationModel",
  "anchor_id": "#classificationmodel",
  "content_type": "reference",
  "word_count": 450,
  "subsection_count": 4  # fit(), predict(), get_params(), set_params()
}

{
  "parent_file": "demos/credit_risk/README.md",
  "heading_level": 2,
  "heading_text": "Quick Start",
  "anchor_id": "#quick-start",
  "content_type": "instructions",
  "word_count": 180,
  "subsection_count": 3  # Install, Configure, Run
}
```

---

## Entity Relationships Diagram

```
┌─────────────────────┐
│ Documentation       │
│ Directory           │
│ (docs/, demos/*)    │
└──────────┬──────────┘
           │ contains
           ▼
┌─────────────────────┐         ┌────────────────────┐
│ Documentation File  │────────>│ Documentation      │
│ (*.md)              │ part of │ Section (H1-H6)    │
└──────────┬──────────┘         └────────────────────┘
           │ links
           ▼
┌─────────────────────┐
│ Cross-Reference     │
│ (markdown links)    │
└──────────┬──────────┘
           │ points to
           ▼
      ┌─────────┐
      │ Code    │
      │ Files   │
      └─────────┘

┌─────────────────────┐         ┌────────────────────┐
│ .gitignore Rules    │────────>│ Code Artifact      │
│ (patterns)          │ matches │ (temp files)       │
└─────────────────────┘         └────────────────────┘
```

---

## Data Flow Sequences

### Sequence 1: Adding New Documentation

1. Developer creates new markdown file in appropriate directory (docs/, demos/*, specs/*)
2. System validates file meets requirements (TOC if >100 lines, cross-references use correct format)
3. Developer links new file from relevant locations (README, parent directory index)
4. Code review verifies all links valid, no duplication, correct directory placement
5. Merge to main branch; documentation published

### Sequence 2: Preventing Accidental Commits

1. Developer runs tests, generates __pycache__/ artifacts
2. Git status checks .gitignore patterns
3. Matching artifacts excluded from untracked files list
4. Only intentional changes appear in git status
5. Pre-commit validation (if enabled) confirms no unwanted artifacts staged

### Sequence 3: Consolidating Duplicate Documentation

1. Documentation audit script finds duplicate content (same API signatures in demo README and docs/api_reference.md)
2. Maintainer removes duplicate from demo README
3. Maintainer adds cross-reference link to central docs/api_reference.md
4. Link validation confirms reference points to correct section
5. SC-002 success criteria met (zero duplicates)

---

## Summary

The data model supports:
- ✅ Documentation organization into three-tier hierarchy (specs/, docs/, demos/)
- ✅ Code hygiene through .gitignore pattern management
- ✅ Cross-referencing to prevent duplication
- ✅ Table of contents generation for long documents
- ✅ Link validation to ensure zero broken references

All entities validated against functional requirements (FR-001 through FR-015) and success criteria (SC-001 through SC-012).

Ready for contracts/ generation (Phase 1 continued).
