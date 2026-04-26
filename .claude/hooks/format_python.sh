#!/usr/bin/env bash
# PostToolUse hook: ruff format + auto-fix on Python edits inside lucid/ or tests/.
#
# Silent on success. Non-blocking: ruff failures don't fail the edit
# (Claude can react to remaining lint output in subsequent turns).

set -u

input=$(cat)
file_path=$(printf '%s' "$input" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null || true)

case "$file_path" in
  *.py) ;;
  *) exit 0 ;;
esac

case "$file_path" in
  */lucid/*|*/tests/*) ;;
  *) exit 0 ;;
esac

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}" || exit 0

uv run --quiet ruff format "$file_path" >/dev/null 2>&1 || true
uv run --quiet ruff check --fix "$file_path" >/dev/null 2>&1 || true

exit 0
