# Research: Documentation and Code Hygiene Review

**Feature**: Documentation and Code Hygiene Review
**Branch**: 002-top-to-bottom
**Date**: 2025-10-10

## Overview

This document captures research decisions for reorganizing project documentation and improving code hygiene. All decisions are based on industry best practices for Python project documentation structure and git ignore patterns.

---

## Decision 1: Documentation Directory Structure

**Question**: How should we organize documentation across docs/, demos/, and specs/ directories to minimize duplication and maximize discoverability?

**Decision**: Three-tier documentation hierarchy

```
docs/              # Cross-cutting implementation documentation
├── architecture/  # Design decisions, patterns, base model hierarchy
├── api/           # Centralized API reference for shared/models/
└── tutorials/     # Step-by-step how-to guides

demos/*/           # Demo-specific documentation
└── README.md      # Demo setup, usage, performance benchmarks (max 500 lines)

specs/             # Feature specifications (what/why, not how)
└── ###-feature/   # Feature spec, plan, data models, contracts
```

**Rationale**:
- Separation of concerns: specs/ = requirements, docs/ = implementation, demos/ = examples
- Avoids duplication: demos/ link to docs/api_reference.md rather than duplicating method signatures
- Scalability: New demos add one README; new features add one spec directory
- Clear ownership: docs/ = project-wide, demos/* = demo-specific, specs/* = feature-specific

**Alternatives Considered**:
1. **Single docs/ for everything** - Rejected: Creates confusion between specs and implementation docs
2. **Wikis or external tools** - Rejected: Requirement to keep all docs in git
3. **Per-module README files** - Rejected: 50+ READMEs hard to navigate; prefer centralized docs/

**References**:
- Python Packaging Guide: https://packaging.python.org/en/latest/guides/writing-documentation/
- Divio Documentation System: https://documentation.divio.com/

---

## Decision 2: Table of Contents Strategy

**Question**: Should table of contents be generated automatically or maintained manually? What threshold triggers TOC requirement?

**Decision**: Manual TOC at top of documents >100 lines using markdown internal links

**Example Format**:
```markdown
# Document Title

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
```

**Rationale**:
- Manual control: Allows curating navigation focus (hide minor sections)
- GitHub compatibility: Works in all markdown viewers without plugins
- 100-line threshold: Short docs don't need TOC; longer docs improve with jump links
- Maintenance burden low: TOC updates happen when adding/removing major sections

**Alternatives Considered**:
1. **Automated TOC generator** - Rejected: Adds build step; not required per spec assumptions
2. **No TOC, rely on outline view** - Rejected: Not all markdown viewers have outline view
3. **50-line threshold** - Rejected: Too aggressive; many documents 50-100 lines don't need TOC

**References**:
- GitHub Markdown Guide: https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#section-links

---

## Decision 3: .gitignore Pattern Organization

**Question**: How should .gitignore patterns be organized for clarity and maintainability?

**Decision**: Group patterns by category with comments

```gitignore
# Python artifacts
__pycache__/
*.py[cod]
*.egg-info/

# IDE configurations
.vscode/
.idea/
*.swp

# OS-specific
.DS_Store
Thumbs.db

# IntegratedML specific
iris.log
iris.pid
docker/volumes/iris_data/
```

**Rationale**:
- Readability: Comments explain why each section exists
- Maintainability: New patterns added to correct category
- Completeness: Covers Python, IDEs (VS Code, PyCharm, Vim), OS (macOS, Windows), project-specific (IRIS)
- Standard order: Language → IDE → OS → Project follows common practice

**Alternatives Considered**:
1. **Alphabetical ordering** - Rejected: Harder to find related patterns
2. **Global .gitignore_global** - Rejected: Requires per-developer setup; team-shared .gitignore better
3. **Separate .dockerignore** - Accepted for Docker context; .gitignore for version control

**References**:
- GitHub gitignore templates: https://github.com/github/gitignore
- Python gitignore best practices: https://www.toptal.com/developers/gitignore/api/python

---

## Decision 4: Cross-Reference Link Format

**Question**: What link format should be used for cross-references between documentation files and to code?

**Decision**: Use relative markdown links for docs, absolute paths with line numbers for code

**Documentation links** (relative paths):
```markdown
See [API Reference](../docs/api_reference.md#classificationmodel) for method details.
```

**Code references** (absolute paths with line numbers):
```markdown
The base model is defined in `shared/models/base.py:50-120`.
```

**Rationale**:
- Relative doc links: Portable across file moves; work in local editors and GitHub
- Line number ranges: Helps locate exact code sections; optional (can be omitted if fragile)
- Absolute code paths: Clear from any documentation location
- Markdown native: No special tools required for link validation

**Alternatives Considered**:
1. **Absolute URLs to GitHub** - Rejected: Breaks in forks; requires internet for local viewing
2. **Generated link registry** - Rejected: Adds complexity per spec assumptions
3. **No line numbers** - Accepted as optional; include when stable, omit if code changes frequently

**References**:
- Markdown link syntax: https://www.markdownguide.org/basic-syntax/#links

---

## Decision 5: Documentation Contribution Guidelines

**Question**: How should developers know where to add new documentation (README vs. docs/ vs. specs/)?

**Decision**: Create docs/CONTRIBUTING_DOCS.md with decision flowchart

**Flowchart logic**:
```
Is it a feature specification? → specs/###-feature/spec.md
Is it project-wide? → docs/{architecture,api,tutorials,deployment}/
Is it demo-specific? → demos/*/README.md
Is it repository setup? → .github/REPOSITORY_SETUP.md
Is it AI agent guidance? → CLAUDE.md
```

**Rationale**:
- Clear decision tree: Eliminates ambiguity about document placement
- Self-service: Developers can determine correct location without asking
- Prevents duplication: Encourages linking to existing docs rather than copy-pasting
- Part of FR-014: Documentation contribution guide requirement

**Alternatives Considered**:
1. **No guideline, rely on code review** - Rejected: Reactive rather than proactive
2. **Embedded in top-level README** - Rejected: Clutters entry point; dedicated guide better
3. **Wiki page** - Rejected: Assumes external wiki availability per spec assumptions

**References**:
- Write the Docs: https://www.writethedocs.org/guide/writing/beginners-guide-to-docs/

---

## Decision 6: Demo README Length Limit

**Question**: How should we enforce the 500-line limit for demo READMEs while preserving necessary technical content?

**Decision**: Extract detailed technical content to demo-specific subdirectories; README focuses on quickstart

**Demo README Structure** (<500 lines):
```markdown
# Demo Name

## Quick Start (3-step setup)
## Architecture Overview (diagram + 1 paragraph)
## Performance Benchmarks (results table)
## Usage Examples (2-3 code snippets)
## Troubleshooting (common issues)
## See Also (links to docs/ and demo/docs/)
```

**Extended Content** (if needed):
- `demos/*/docs/ARCHITECTURE.md` - Detailed architecture
- `demos/*/docs/PERFORMANCE_BENCHMARKS.md` - Extended benchmarks
- `demos/*/TECHNICAL_DOCUMENTATION.md` - Deep dive technical details

**Rationale**:
- Quickstart focus: New users get running in <10 minutes per SC-001
- Depth available: Technical details available for deep dive without cluttering README
- Link-based navigation: README links to extended docs for exploration
- Consistent structure: All demo READMEs follow same format

**Alternatives Considered**:
1. **No line limit, long READMEs** - Rejected: Violates SC-007; overwhelming for new users
2. **1000-line limit** - Rejected: Too permissive; encourages duplication
3. **External documentation site** - Rejected: Assumes external hosting per spec assumptions

**References**:
- README best practices: https://github.com/matiassingers/awesome-readme

---

## Decision 7: Markdown Linting Rules

**Question**: Which markdown linting rules should be enforced to ensure consistent formatting?

**Decision**: Use markdownlint with relaxed rules for line length and list markers

**Enabled Rules**:
- Heading hierarchy (H1 → H2 → H3, no skipping)
- Consistent list markers (- for unordered, 1. for ordered)
- Code fence syntax (triple backticks with language identifier)
- No trailing whitespace
- Blank lines around headings and lists

**Disabled Rules**:
- MD013 (line length): Allow long lines for code snippets and links
- MD007 (list indentation): Allow flexible indentation for nested lists
- MD033 (inline HTML): Allow for complex tables or diagrams

**Rationale**:
- Consistency: Ensures all docs follow same formatting conventions per FR-012
- Readability: Enforces structure that improves navigation
- Flexibility: Relaxes rules that conflict with technical writing needs
- Optional enforcement: Per spec assumptions, linting is optional (manual validation acceptable)

**Alternatives Considered**:
1. **No linting, manual review only** - Accepted per spec; linting optional for automation
2. **Strict markdownlint default rules** - Rejected: Too rigid for technical documentation
3. **Custom linter** - Rejected: Adds complexity; standard tools sufficient

**References**:
- markdownlint rules: https://github.com/DavidAnson/markdownlint/blob/main/doc/Rules.md

---

## Decision 8: Link Validation Strategy

**Question**: How should broken links be detected and prevented?

**Decision**: Manual link validation during code review + optional CI check

**Manual Process**:
1. Reviewer clicks all links in changed documentation
2. Internal links (markdown files, code files) validated by checking file existence
3. External links (GitHub, documentation sites) validated by clicking

**Optional CI Automation** (if team chooses to implement):
```bash
# Check internal markdown links
find docs/ demos/ -name "*.md" -exec markdown-link-check {} \;

# Check code file references
grep -r "\.py:[0-9]" docs/ demos/ | while read ref; do
  # Extract file path and verify existence
done
```

**Rationale**:
- Pragmatic: Manual validation sufficient per spec assumptions (no automated link checking required)
- Scalable: CI automation available if team wants it, but not mandatory
- Code review gate: Broken links caught before merge
- Zero broken links goal: SC-005 measured via manual audit

**Alternatives Considered**:
1. **Required CI link checking** - Rejected: Spec assumes manual validation acceptable
2. **No validation, rely on user bug reports** - Rejected: Violates SC-005 goal
3. **Pre-commit hook** - Considered but optional; not in initial implementation

**References**:
- markdown-link-check: https://github.com/tcort/markdown-link-check

---

## Summary

All research decisions documented with clear rationale. No NEEDS CLARIFICATION items remain. Key themes:

1. **Three-tier hierarchy**: specs/ (what/why) → docs/ (how) → demos/ (examples)
2. **Manual-first approach**: Automation optional, manual processes documented
3. **Standard tools**: Markdown, git, bash - no specialized documentation generators
4. **Clear guidelines**: Decision flowcharts and contribution guides prevent confusion
5. **Link-based navigation**: Cross-references instead of duplication

Ready for Phase 1: Design artifacts (data-model.md, contracts/, quickstart.md).
