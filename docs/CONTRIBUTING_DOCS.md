# Contributing to Documentation

This guide helps you determine where to add new documentation in the IntegratedML Custom Models project.

## Table of Contents

- [Documentation Organization](#documentation-organization)
- [Decision Flowchart](#decision-flowchart)
- [Documentation Standards](#documentation-standards)
- [File Naming Conventions](#file-naming-conventions)
- [Markdown Guidelines](#markdown-guidelines)
- [Review Process](#review-process)

## Documentation Organization

Our documentation follows a three-tier structure:

### 📚 Core Documentation (`docs/`)
**Purpose**: Cross-cutting technical documentation for the entire project

**When to use**:
- API reference documentation
- Architecture and design patterns
- Deployment guides
- User guides and tutorials
- Project-wide setup instructions

**Examples**:
- `docs/api_reference.md` - Complete API documentation
- `docs/architecture.md` - System design and patterns
- `docs/deployment.md` - Production deployment strategies
- `docs/user_guide.md` - End-to-end usage instructions

### 🎯 Demo Applications (`demos/`)
**Purpose**: Demo-specific setup and usage instructions

**When to use**:
- Demo-specific installation steps
- Demo-specific data requirements
- Demo-specific configuration
- Demo-specific usage examples

**Examples**:
- `demos/credit_risk/README.md` - Credit risk demo setup
- `demos/fraud_detection/README.md` - Fraud detection demo setup
- `demos/sales_forecasting/README.md` - Sales forecasting demo setup
- `demos/dna_similarity/README.md` - DNA similarity demo setup

### 📋 Feature Specifications (`specs/`)
**Purpose**: Design documents and implementation plans for new features

**When to use**:
- Feature specifications with user stories
- Implementation plans with architecture decisions
- Task breakdowns and validation results
- Research and design decisions

**Examples**:
- `specs/001-feature-name/spec.md` - Feature specification
- `specs/001-feature-name/plan.md` - Implementation plan
- `specs/001-feature-name/tasks.md` - Task breakdown
- `specs/001-feature-name/research.md` - Research decisions

## Decision Flowchart

```
┌─────────────────────────────────────────────────────────┐
│  What type of documentation do you need to add/update?  │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
   ┌─────────┐ ┌─────────┐ ┌──────────┐
   │  API    │ │ Feature │ │  Demo    │
   │  Docs?  │ │ Design? │ │ Specific?│
   └────┬────┘ └────┬────┘ └────┬─────┘
        │           │           │
        │           │           │
        ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌──────────┐
   │ Is it   │ │ Is it a │ │ Which    │
   │ for a   │ │ new     │ │ demo?    │
   │ specific│ │ feature │ │          │
   │ class?  │ │ spec?   │ │          │
   └────┬────┘ └────┬────┘ └────┬─────┘
        │           │           │
    YES │ NO        │ YES       │
        │           │           │
        ▼           ▼           ▼
┌───────────────────────────────────────────────────┐
│                  DESTINATION                       │
├───────────────────────────────────────────────────┤
│                                                    │
│  API Documentation                                 │
│  ├─ Specific class → docs/api_reference.md        │
│  ├─ Base classes → docs/api_reference.md          │
│  └─ Utilities → docs/api_reference.md             │
│                                                    │
│  Architecture & Design                             │
│  ├─ System design → docs/architecture.md          │
│  ├─ Integration patterns → docs/architecture.md   │
│  ├─ Data flow → docs/architecture.md              │
│  └─ Design patterns → docs/architecture.md        │
│                                                    │
│  Deployment & Operations                           │
│  ├─ Docker setup → docs/deployment.md             │
│  ├─ Cloud deployment → docs/deployment.md         │
│  ├─ Configuration → docs/deployment.md            │
│  └─ Performance tuning → docs/deployment.md       │
│                                                    │
│  User Guides & Tutorials                           │
│  ├─ Getting started → docs/user_guide.md          │
│  ├─ Step-by-step guides → docs/user_guide.md      │
│  ├─ Common workflows → docs/user_guide.md         │
│  └─ Troubleshooting → docs/user_guide.md          │
│                                                    │
│  Demo-Specific Documentation                       │
│  ├─ Credit risk → demos/credit_risk/README.md     │
│  ├─ Fraud detection → demos/fraud_detection/...   │
│  ├─ Sales forecasting → demos/sales_forecasting/..│
│  └─ DNA similarity → demos/dna_similarity/...     │
│                                                    │
│  Feature Specifications                            │
│  ├─ New feature spec → specs/NNN-name/spec.md     │
│  ├─ Implementation plan → specs/NNN-name/plan.md  │
│  ├─ Task breakdown → specs/NNN-name/tasks.md      │
│  └─ Research → specs/NNN-name/research.md         │
│                                                    │
│  Project-Level Documentation                       │
│  ├─ Main entry point → README.md                  │
│  ├─ Contributing → CONTRIBUTING.md                │
│  ├─ Doc contributions → docs/CONTRIBUTING_DOCS.md │
│  └─ License → LICENSE                             │
└───────────────────────────────────────────────────┘
```

## Quick Decision Guide

### Is it about a specific model class or base class?
→ **Add to** `docs/api_reference.md`

### Is it about system architecture, design patterns, or integration?
→ **Add to** `docs/architecture.md`

### Is it about Docker, cloud deployment, or production configuration?
→ **Add to** `docs/deployment.md`

### Is it a step-by-step guide for end users?
→ **Add to** `docs/user_guide.md`

### Is it specific to ONE demo (setup, data, config, usage)?
→ **Add to** `demos/[demo-name]/README.md`

### Is it a new feature specification or implementation plan?
→ **Create new** `specs/NNN-feature-name/` directory with:
- `spec.md` - Feature specification with user stories
- `plan.md` - Implementation plan with architecture decisions
- `tasks.md` - Task breakdown with dependencies
- `research.md` - Research and design decisions

### Is it project-level information (contributing, license, main README)?
→ **Add to** root-level `.md` files

## Documentation Standards

### File Requirements

1. **Table of Contents**: All files >100 lines must include a table of contents
   - Use automated TOC generation: `bash scripts/generate_toc.sh [file]`
   - Place TOC after title and before first section
   - Use GitHub-flavored markdown anchor links

2. **Cross-References**: Prefer linking over duplication
   - ✅ Good: "See [API Reference](../api_reference.md) for complete documentation"
   - ❌ Bad: Copy-pasting API documentation into demo READMEs

3. **Demo README Size**: Keep demo READMEs under 500 lines
   - Focus on demo-specific setup and usage
   - Link to main docs for project-wide information

4. **Code Examples**: Include working code snippets
   - Use proper syntax highlighting (```python, ```sql, ```bash)
   - Ensure examples are tested and functional
   - Include expected output where helpful

### Structure Guidelines

1. **Clear Headings**: Use descriptive heading hierarchy
   - H1 (`#`) for document title only
   - H2 (`##`) for major sections
   - H3 (`###`) for subsections
   - H4+ for detailed breakdowns

2. **Visual Aids**: Include diagrams where helpful
   - ASCII diagrams for architecture
   - Code flow diagrams for complex logic
   - Tables for comparison matrices

3. **Examples First**: Lead with practical examples
   - Show working code before explaining theory
   - Include realistic use cases
   - Provide copy-paste ready snippets

## File Naming Conventions

### Core Documentation (`docs/`)
- Use snake_case for multi-word files: `api_reference.md`
- Use UPPERCASE for special files: `QUICK_GUIDE_CUSTOM_MODELS.md`
- Be descriptive: `deployment.md`, `architecture.md`, `user_guide.md`

### Demo READMEs (`demos/*/`)
- Always use `README.md` for main demo documentation
- Additional files use descriptive names: `TROUBLESHOOTING.md`, `ADVANCED_USAGE.md`

### Feature Specs (`specs/NNN-feature-name/`)
- Directory naming: `NNN-short-kebab-case-name/` (e.g., `002-top-to-bottom/`)
- Standard files: `spec.md`, `plan.md`, `tasks.md`, `research.md`
- Additional files: descriptive names matching content

## Markdown Guidelines

### Required Elements

1. **Front Matter**: Start with clear title
   ```markdown
   # Document Title

   Brief description of what this document covers.
   ```

2. **Table of Contents**: For files >100 lines
   ```markdown
   ## Table of Contents

   - [Section 1](#section-1)
   - [Section 2](#section-2)
   ```

3. **Code Blocks**: Always specify language
   ```markdown
   ```python
   # Python code here
   ```

   ```sql
   -- SQL code here
   ```
   ```

4. **Links**: Use descriptive link text
   - ✅ Good: `See [API Reference](docs/api_reference.md) for details`
   - ❌ Bad: `See [here](docs/api_reference.md) for details`

### Formatting Standards

1. **Lists**: Use `-` for unordered lists, `1.` for ordered
2. **Emphasis**: `**bold**` for important terms, `*italic*` for emphasis
3. **Code**: Use backticks for inline code: `model.fit(X, y)`
4. **Tables**: Use GitHub-flavored markdown tables with alignment
5. **Line Length**: No hard limit, but keep lines readable (aim for <120 chars)

### Emojis in Headers

When using emojis in section headers, be aware of GitHub anchor behavior:
- GitHub strips emojis from anchor links
- Example: `## 🎯 Core Features` becomes `#core-features` (not `#-core-features`)
- Test TOC links to ensure they work correctly

## Review Process

### Before Submitting

1. **Run TOC Generation**: Update table of contents
   ```bash
   bash scripts/generate_toc.sh docs/your_file.md > /tmp/toc.md
   # Review and add to your file
   ```

2. **Check Links**: Verify all internal links work
   ```bash
   # Manual verification in markdown preview
   ```

3. **Validate Examples**: Test all code snippets
   ```bash
   # Run code examples to ensure they work
   python -c "your example code"
   ```

4. **Check File Size**: Ensure demo READMEs are <500 lines
   ```bash
   wc -l demos/*/README.md
   ```

5. **Review Formatting**: Check markdown rendering
   - Preview in VS Code or GitHub
   - Verify code blocks have language tags
   - Ensure tables render correctly

### Pull Request Checklist

- [ ] TOC added/updated for files >100 lines
- [ ] Cross-references used instead of duplication
- [ ] Code examples tested and working
- [ ] Links verified and functional
- [ ] File placed in correct directory per flowchart
- [ ] Formatting follows markdown guidelines
- [ ] Demo READMEs under 500 lines (if applicable)

## Common Scenarios

### Adding Documentation for a New Model Class

1. **Add API documentation** to `docs/api_reference.md`:
   - Class signature with parameters
   - Method descriptions with examples
   - Return types and exceptions

2. **Update architecture docs** (if new pattern):
   - Add to class hierarchy diagram
   - Document design patterns used
   - Explain integration approach

3. **Create demo** (if demonstrating usage):
   - Add demo-specific README with setup
   - Link to main API docs for reference
   - Include working code examples

### Adding a New Demo

1. **Create demo directory**: `demos/new-demo/`
2. **Add demo README**: `demos/new-demo/README.md`
   - Demo-specific setup and data requirements
   - Link to main docs for project-wide setup
   - Include usage examples specific to this demo
3. **Update main README**: Add demo to list with description
4. **Update Documentation Map**: Add demo to README.md documentation section

### Documenting a New Feature

1. **Create spec directory**: `specs/NNN-feature-name/`
2. **Add specification**: `specs/NNN-feature-name/spec.md`
3. **Add implementation plan**: `specs/NNN-feature-name/plan.md`
4. **Add task breakdown**: `specs/NNN-feature-name/tasks.md`
5. **Document in main docs** (after implementation):
   - Update API reference if new classes
   - Update architecture if new patterns
   - Update user guide if user-facing

### Updating Existing Documentation

1. **Find the file** using the decision flowchart above
2. **Make changes** following markdown guidelines
3. **Update TOC** if structure changed
4. **Test examples** if code changed
5. **Verify links** if file moved or renamed

## Questions?

If you're unsure where documentation belongs:

1. **Check the flowchart** above
2. **Review existing documentation** in similar areas
3. **Ask in pull request** - reviewers can help guide placement
4. **When in doubt**: Default to creating a new file in `docs/` and link from README.md

## Examples

### Example 1: Documenting a Bug Fix
**Question**: Where do I document this bug fix in the credit risk classifier?

**Answer**:
- Update API documentation in `docs/api_reference.md` if behavior changed
- Update demo README if usage changed: `demos/credit_risk/README.md`
- Add to git commit message with details
- No new documentation file needed for simple bug fixes

### Example 2: Adding a New Deployment Strategy
**Question**: We now support Kubernetes deployment. Where does this go?

**Answer**:
- Add comprehensive guide to `docs/deployment.md`
- Add new section for Kubernetes configuration
- Update TOC in deployment.md
- Link from main README.md if it's a major deployment option

### Example 3: Creating a Tutorial
**Question**: I wrote a step-by-step tutorial for beginners. Where should it go?

**Answer**:
- Add to `docs/user_guide.md` if it's a general tutorial
- Add to demo README if it's demo-specific: `demos/[demo-name]/README.md`
- Create new file `docs/tutorials/your-tutorial.md` if it's substantial (>500 lines)
- Link from main README.md documentation map

## Additional Resources

- [Main README](../README.md) - Project overview and documentation map
- [Architecture Docs](architecture.md) - System design and patterns
- [API Reference](api_reference.md) - Complete API documentation
- [Contributing Guide](../CONTRIBUTING.md) - General contribution guidelines
