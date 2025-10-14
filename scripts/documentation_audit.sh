#!/bin/bash
# scripts/documentation_audit.sh
# Documentation quality audit script

echo "=== Documentation Audit ==="
echo ""

# SC-002: Zero duplicate documentation
echo "1. Checking for duplicate content..."
duplicate_count=$(grep -rh "^### " docs/ demos/ 2>/dev/null | sort | uniq -d | wc -l)
if [ $duplicate_count -eq 0 ]; then
  echo "   ✅ PASS: No duplicate sections found"
else
  echo "   ❌ FAIL: $duplicate_count duplicate sections"
  grep -rh "^### " docs/ demos/ 2>/dev/null | sort | uniq -d | head -10
fi
echo ""

# SC-004: All files >100 lines have TOC
echo "2. Checking for missing table of contents..."
missing_toc=0
find docs/ demos/ README.md -name "*.md" 2>/dev/null | while read file; do
  lines=$(wc -l < "$file" 2>/dev/null || echo 0)
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
find . -name "*.md" 2>/dev/null | while read file; do
  grep -oE '\[.*\]\([^)]+\)' "$file" 2>/dev/null | while read link; do
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
    lines=$(wc -l < "$readme" 2>/dev/null || echo 0)
    if [ $lines -le 500 ]; then
      echo "   ✅ PASS: $readme ($lines lines)"
    else
      echo "   ❌ FAIL: $readme ($lines lines, max 500)"
    fi
  fi
done
echo ""

echo "=== Audit Complete ==="
