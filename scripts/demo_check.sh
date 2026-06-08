#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

fail=0

check_file() {
  local label="$1"
  local path="$2"
  if [[ -s "$path" ]]; then
    printf '[ok] %s: %s\n' "$label" "$path"
  else
    printf '[missing] %s: %s\n' "$label" "$path" >&2
    fail=1
  fi
}

check_config_value() {
  local key="$1"
  local expected="$2"
  if grep -Eq "^${key}[[:space:]]*=[[:space:]]*${expected}[[:space:]]*$" config.ini; then
    printf '[ok] config %s=%s\n' "$key" "$expected"
  else
    printf '[warn] config should use %s=%s for the recorded defense flow\n' "$key" "$expected" >&2
  fi
}

printf 'Checking defense/demo readiness...\n\n'

check_file "requirements PDF" "Project2026.pdf"
check_file "tutorial GIF" "docs/tutorial.gif"
check_file "tutorial MP4" "docs/demo.mp4"
check_file "screenshot CNN board" "docs/screenshots/01-cnn-board.png"
check_file "screenshot search preview" "docs/screenshots/02-search-preview.png"
check_file "screenshot Classical evaluator" "docs/screenshots/03-classical-evaluator.png"
check_file "screenshot NNUE panel" "docs/screenshots/04-nnue-panel.png"
check_file "screenshot MCTS tree" "docs/screenshots/05-mcts-tree.png"
check_file "final project report PDF" "docs/pdf/final-project-report.pdf"
check_file "design PDF" "docs/pdf/design-spec.pdf"
check_file "manual PDF" "docs/pdf/user-manual.pdf"
check_file "test-cases PDF" "docs/pdf/test-cases.pdf"
check_file "summary PDF" "docs/pdf/summary-report.pdf"
check_file "AI usage PDF" "docs/pdf/ai-usage.pdf"
check_file "NNUE runtime model" "models/nnue-halfkp-demo.bin"
check_file "CNN runtime model" "models/lc0-cnn-small-112p-4x32-policy4672-wdl3.bin"
check_file "BT4 runtime model" "models/lc0-bt4-tiny-96x4x4h.bin"

check_config_value "window.fullscreen" "false"
check_config_value "startup.arch" "cnn"

if command -v ffprobe >/dev/null 2>&1; then
  mp4_summary="$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,nb_frames,duration \
    -of csv=p=0 docs/demo.mp4 2>/dev/null || true)"
  gif_summary="$(ffprobe -v error -select_streams v:0 -count_frames \
    -show_entries stream=width,height,nb_read_frames,duration \
    -of csv=p=0 docs/tutorial.gif 2>/dev/null || true)"
  [[ -n "$mp4_summary" ]] && printf '[ok] MP4 stream: %s\n' "$mp4_summary"
  [[ -n "$gif_summary" ]] && printf '[ok] GIF stream: %s\n' "$gif_summary"
else
  printf '[warn] ffprobe not installed; skipped media metadata check\n' >&2
fi

printf '\nRunning automated tests...\n'
./scripts/test.sh

if [[ "$fail" -ne 0 ]]; then
  printf '\nDemo readiness check failed. Run ./scripts/import_models.sh and re-run this script.\n' >&2
  exit 1
fi

printf '\nDemo readiness check passed.\n'
