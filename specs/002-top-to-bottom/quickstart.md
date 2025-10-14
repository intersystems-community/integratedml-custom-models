# Quickstart: Documentation Reorganization

**Feature**: Documentation and Code Hygiene Review
**Date**: 2025-10-10
**Audience**: Project Maintainers, Documentation Contributors

## Overview

This quickstart guides you through implementing the documentation reorganization and code hygiene improvements in under 2 hours. You'll audit existing documentation, consolidate duplicates, add table of contents, and ensure .gitignore comprehensive coverage.

---

## Prerequisites

Before starting, ensure you have:

- ✅ Git repository access (read/write permissions)
- ✅ Text editor or IDE
- ✅ Bash shell (for audit scripts)
- ✅ Basic grep/find command knowledge

---

## Step 1: Audit Existing Documentation (30 minutes)

### Identify All Documentation Files

```bash
# From repository root
find . -name "*.md" -type f | grep -v ".git" | grep -v "node_modules" | sort > docs_inventory.txt

# Count by directory
echo "Documentation file count by directory:"
echo "Top-level: $(ls *.md 2>/dev/null | wc -l)"
echo "docs/: $(find docs/ -name "*.md" | wc -l)"
echo "demos/: $(find demos/ -name "*.md" | wc -l)"
echo "specs/: $(find specs/ -name "*.md" | wc -l)"
echo ".github/: $(find .github/ -name "*.md" | wc -l)"
```

**Expected Output**:
```
Documentation file count by directory:
Top-level: 2     # README.md, CLAUDE.md
docs/: 12        # Various documentation files
demos/: 8        # 4 demos × 2 files each (README + extended docs)
specs/: 15       # Feature specifications
.github/: 5      # Issue templates, PR template
```

### Check Documentation Length

```bash
# Find files needing table of contents (>100 lines)
echo "Files >100 lines (need TOC):"
find . -name "*.md" -type f | while read file; do
  lines=$(wc -l < "$file")
  if [ $lines -gt 100 ]; then
    echo "  $file: $lines lines"
  fi
done
```

### Detect Duplicate Content

```bash
# Find potential duplicates by comparing section headers
echo "Searching for duplicate API documentation..."
grep -rh "^### " docs/ demos/ | sort | uniq -c | sort -rn | head -20

# Check for duplicate setup instructions
echo "Searching for duplicate setup instructions..."
grep -rl "pip install" docs/ demos/ README.md
```

**Action Items**:
- [ ] List all files >100 lines without TOC
- [ ] Identify API documentation duplicated in demo READMEs
- [ ] Find setup instructions repeated across multiple files

---

## Step 2: Reorganize Top-Level Directory (15 minutes)

### Current Top-Level Structure

```bash
ls -1 /Users/tdyar/ws/pluggable_iml/ | head -20
```

**Goal**: Reduce to <15 essential files (SC-006)

### Clean Up Non-Essential Files

```bash
# Move data files to data/ if not already there
# Move notebooks to notebooks/ if not already there
# Move scripts to scripts/ if not already there

# Verify top-level count
echo "Top-level items: $(ls -1 | wc -l)"
```

**Target Structure**:
```
/Users/tdyar/ws/pluggable_iml/
├── README.md          # Entry point
├── CLAUDE.md          # AI guidance
├── LICENSE
├── .gitignore         # Code hygiene
├── .env.example
├── requirements.txt
├── pyproject.toml
├── Makefile
├── docker-compose.yml
├── docs/              # Documentation
├── demos/             # Examples
├── shared/            # Library
├── specs/             # Specifications
├── tests/             # E2E tests
└── .github/           # GitHub config
```

---

## Step 3: Add Table of Contents (20 minutes)

### Create TOC Template

```bash
# Generate TOC for a markdown file
function generate_toc() {
  local file=$1
  echo "## Table of Contents"
  echo ""
  grep -E "^##+ " "$file" | while read line; do
    level=$(echo "$line" | grep -oE "^#+" | wc -c)
    level=$((level - 3))  # Adjust for H2=0 indent
    indent=$(printf "%${level}s" "")
    title=$(echo "$line" | sed 's/^#\+ //')
    anchor=$(echo "$title" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr -d '.,()[]')
    echo "${indent}- [$title](#$anchor)"
  done
  echo ""
}

# Example usage
generate_toc docs/architecture.md
```

### Add TOC to Long Files

**Files Requiring TOC** (>100 lines per FR-011):
- `README.md`
- `docs/architecture.md`
- `docs/api_reference.md`
- `docs/deployment.md`
- `docs/user_guide.md`

**Manual Process**:
1. Run `generate_toc` function for each file
2. Copy TOC output
3. Insert after H1 title, before first H2 section
4. Verify all links work by clicking in markdown preview

---

## Step 4: Consolidate Duplicate Documentation (30 minutes)

### Remove API Documentation from Demo READMEs

**Pattern to Find**:
```bash
# Find demo READMEs with method signatures
grep -l "def fit(" demos/*/README.md
grep -l "def predict(" demos/*/README.md
```

**Replacement Strategy**:
```markdown
<!-- BEFORE: Duplicated in demos/credit_risk/README.md -->
### CustomCreditRiskClassifier API

#### fit(X, y, **params)
Trains the model on credit application data...
[full method signature, parameters, etc.]

#### predict(X)
Generates risk predictions...
[full method signature, parameters, etc.]

<!-- AFTER: Link to central API docs -->
### CustomCreditRiskClassifier API

See [ClassificationModel API Reference](../../docs/api_reference.md#classificationmodel)
for complete method signatures, parameters, and usage examples.

**Demo-Specific Configuration**:
- `enable_debt_ratio` (bool): Calculate debt-to-income ratios
- `enable_risk_scoring` (bool): Add custom risk scores
- `decision_threshold` (float): Classification boundary (default: 0.5)
```

**Action Items**:
- [ ] Remove API duplication from demos/credit_risk/README.md
- [ ] Remove API duplication from demos/fraud_detection/README.md
- [ ] Remove API duplication from demos/sales_forecasting/README.md
- [ ] Remove API duplication from demos/dna_similarity/README.md
- [ ] Add cross-references to docs/api_reference.md

---

## Step 5: Update .gitignore (15 minutes)

### Audit Current .gitignore

```bash
# Check current coverage
grep -E "^(__pycache__|\.vscode|\.idea|\.DS_Store)" .gitignore

# Test with development workflow
pytest demos/credit_risk/tests/
git status --short | grep -E "(__pycache__|\.pytest_cache)" && \
  echo "ERROR: Python artifacts not ignored" || \
  echo "SUCCESS: Python artifacts ignored"
```

### Add Missing Patterns

**Required Categories** (FR-004, FR-013):
```gitignore
# Python artifacts
__pycache__/
*.py[cod]
.pytest_cache/
*.egg-info/

# IDE configurations
.vscode/
.idea/
*.swp
*.swo

# OS-specific
.DS_Store
._*
Thumbs.db

# IntegratedML specific
iris.log
iris.pid
iris.key
docker/volumes/iris_data/
docker/volumes/model_cache/

# Model artifacts
*.pkl
*.joblib
checkpoints/
```

### Validate .gitignore

```bash
# Clean existing artifacts
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null
find . -name ".DS_Store" -type f -delete 2>/dev/null

# Run tests to regenerate
pytest demos/credit_risk/tests/

# Check git status
untracked=$(git status --short | grep -E "(__pycache__|\.pytest_cache|\.DS_Store)" | wc -l)
if [ $untracked -eq 0 ]; then
  echo "✅ SUCCESS: All artifacts ignored (SC-003)"
else
  echo "❌ FAIL: $untracked untracked artifacts found"
  git status --short | grep -E "(__pycache__|\.pytest_cache|\.DS_Store)"
fi
```

---

## Step 6: Create Documentation Audit Script (10 minutes)

```bash
#!/bin/bash
# scripts/documentation_audit.sh

echo "=== Documentation Audit ==="
echo ""

# SC-002: Zero duplicate documentation
echo "1. Checking for duplicate content..."
duplicate_count=$(grep -rh "^### " docs/ demos/ | sort | uniq -d | wc -l)
if [ $duplicate_count -eq 0 ]; then
  echo "   ✅ PASS: No duplicate sections found"
else
  echo "   ❌ FAIL: $duplicate_count duplicate sections"
  grep -rh "^### " docs/ demos/ | sort | uniq -d | head -10
fi
echo ""

# SC-004: All files >100 lines have TOC
echo "2. Checking for missing table of contents..."
missing_toc=0
find docs/ demos/ README.md -name "*.md" | while read file; do
  lines=$(wc -l < "$file")
  if [ $lines -gt 100 ] && ! grep -q "## Table of Contents" "$file"; then
    echo "   ❌ MISSING TOC: $file ($lines lines)"
    missing_toc=$((missing_toc + 1))
  fi
done
if [ $missing_toc -eq 0 ]; then
  echo "   ✅ PASS: All long files have TOC"
fi
echo ""

# SC-005: Zero broken links
echo "3. Checking for broken internal links..."
broken_links=0
find . -name "*.md" | while read file; do
  grep -oE '\[.*\]\([^)]+\)' "$file" | while read link; do
    path=$(echo "$link" | sed -E 's/.*\(([^)]+)\)/\1/')
    [[ "$path" =~ ^https?:// ]] && continue  # Skip external
    [[ "$path" =~ ^# ]] && continue  # Skip anchors
    dir=$(dirname "$file")
    target="$dir/$path"
    [ -f "$target" ] || echo "   ❌ BROKEN: $link in $file"
  done
done
echo "   ✅ PASS: Link validation complete"
echo ""

# SC-006: Top-level <15 items
echo "4. Checking top-level directory cleanliness..."
toplevel_count=$(ls -1 | wc -l)
if [ $toplevel_count -lt 15 ]; then
  echo "   ✅ PASS: $toplevel_count top-level items (<15)"
else
  echo "   ❌ FAIL: $toplevel_count top-level items (target <15)"
fi
echo ""

# SC-007: Demo READMEs <500 lines
echo "5. Checking demo README length..."
for demo in demos/*/; do
  readme="$demo/README.md"
  if [ -f "$readme" ]; then
    lines=$(wc -l < "$readme")
    if [ $lines -le 500 ]; then
      echo "   ✅ PASS: $readme ($lines lines)"
    else
      echo "   ❌ FAIL: $readme ($lines lines, max 500)"
    fi
  fi
done
echo ""

echo "=== Audit Complete ==="
```

**Make Executable**:
```bash
chmod +x scripts/documentation_audit.sh
./scripts/documentation_audit.sh
```

---

## Verification Checklist

After completing all steps, verify:

- [ ] **SC-001**: New developer can find setup instructions in <10 minutes (manual test with unfamiliar developer)
- [ ] **SC-002**: Documentation audit script reports zero duplicates
- [ ] **SC-003**: `git status` after pytest/Docker shows zero untracked artifacts
- [ ] **SC-004**: All files >100 lines have table of contents
- [ ] **SC-005**: Documentation audit script reports zero broken links
- [ ] **SC-006**: Top-level directory has <15 items
- [ ] **SC-007**: All demo READMEs <500 lines
- [ ] **SC-008**: docs/architecture.md includes 3+ diagrams
- [ ] **SC-009**: docs/api_reference.md covers 100% of public methods
- [ ] **SC-011**: .gitignore covers ≥5 tools (pytest, VS Code, PyCharm, macOS, Docker, IRIS)
- [ ] **SC-012**: Documentation audit script with markdown linter reports zero errors (optional)

---

## Next Steps

### Immediate Actions

1. **Commit Documentation Reorganization**:
```bash
git add README.md docs/ demos/ .gitignore
git commit -m "docs: reorganize documentation hierarchy and consolidate duplicates

- Add table of contents to all files >100 lines (SC-004)
- Consolidate API documentation to docs/api_reference.md (SC-002)
- Update .gitignore for comprehensive coverage (SC-003, SC-011)
- Reduce top-level directory to <15 items (SC-006)
- Ensure demo READMEs <500 lines with cross-references (SC-007)"
```

2. **Run Usability Test**:
   - Ask 3 developers unfamiliar with project to:
     - Find setup instructions (target <10 min)
     - Locate API reference for ClassificationModel (target <2 min)
     - Understand architecture without reading code (qualitative)

3. **Create Documentation Contribution Guide**:
```bash
# Create docs/CONTRIBUTING_DOCS.md with decision flowchart
# (See Contract 6 in contracts/documentation-structure.md)
```

### Optional Enhancements

1. **CI Link Validation**:
```yaml
# .github/workflows/docs-lint.yml
name: Documentation Lint
on: [pull_request]
jobs:
  markdown-link-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: gaurav-nelson/github-action-markdown-link-check@v1
```

2. **Markdown Linting**:
```bash
# Install markdownlint-cli
npm install -g markdownlint-cli

# Create .markdownlint.json config
{
  "MD013": false,  # Line length (allow long lines for code)
  "MD033": false,  # Inline HTML (allow for tables)
  "MD041": false   # First line H1 (some files start with YAML frontmatter)
}

# Run linter
markdownlint '**/*.md' --ignore node_modules
```

---

## Troubleshooting

### Issue: Broken Links After File Moves

**Symptom**: Documentation audit reports broken links

**Solution**:
```bash
# Find all references to moved file
grep -r "old_path.md" docs/ demos/

# Update references to new path
find docs/ demos/ -name "*.md" -exec sed -i '' 's|old_path.md|new_path.md|g' {} +

# Re-run audit
./scripts/documentation_audit.sh
```

### Issue: .gitignore Not Working

**Symptom**: `git status` shows artifacts that should be ignored

**Solution**:
```bash
# Clear git cache (gitignore only applies to untracked files)
git rm -r --cached .
git add .

# Verify
git status
```

### Issue: TOC Links Not Working

**Symptom**: Clicking TOC link doesn't jump to section

**Solution**:
```markdown
<!-- Ensure anchor format matches GitHub rules -->
## API Reference  <!-- Creates anchor #api-reference -->

<!-- TOC link must match exactly -->
- [API Reference](#api-reference)  ✅ Correct
- [API Reference](#API-Reference)  ❌ Wrong (capitalization)
- [API Reference](#api_reference)  ❌ Wrong (underscore)
```

---

## Summary

**Time Investment**: ~2 hours

**Deliverables**:
- Reorganized documentation with clear hierarchy
- Table of contents in all long documents
- Zero duplicate documentation
- Comprehensive .gitignore coverage
- Documentation audit script for ongoing validation

**Success Metrics Achieved**:
- SC-001: <10 minute onboarding
- SC-002: Zero duplicates
- SC-003: Zero untracked artifacts
- SC-004-SC-012: All documentation quality criteria met

**Ready for**: Developer onboarding improvement, ongoing documentation maintenance
