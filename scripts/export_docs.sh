#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p docs/pdf

docs=(
  docs/project-proposal-requirements.md
  docs/design-spec.md
  docs/user-manual.md
  docs/test-cases.md
  docs/summary-report.md
  docs/ai-usage.md
)

for doc in "${docs[@]}"; do
  base="$(basename "$doc" .md)"
  pandoc "$doc" \
    --resource-path=.:docs \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V colorlinks=true \
    -o "docs/pdf/${base}.pdf"
done
