#!/bin/bash
# scripts/generate_toc.sh
# Generate table of contents for markdown files

generate_toc() {
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

# If called directly with filename argument
if [ $# -gt 0 ]; then
  generate_toc "$1"
fi
