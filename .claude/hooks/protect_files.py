#!/usr/bin/env python3
"""PreToolUse hook: block edits to protected paths.

Blocks:
- `.env.local` and any other `.env*` (except `.env.example`) — secret files.
- `prompts/<name>/v<N>.md` — frozen prompt versions. CLAUDE.md rule:
  "Never edit a published prompt version in place." Use the
  `bump-prompt-version` skill to create v<N+1>.md instead.
- `calibration-runs/**` — recorded calibration artefacts are immutable
  provenance for κ/α numbers.

Exit 2 = block + surface stderr to the model. Exit 0 = allow.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0  # Don't block on malformed hook input.

    file_path = data.get("tool_input", {}).get("file_path", "")
    if not file_path:
        return 0

    name = Path(file_path).name

    # Secrets.
    if re.fullmatch(r"\.env(\..+)?", name) and name != ".env.example":
        print(
            f"BLOCKED: {file_path}\n"
            "Reason: contains secrets (ANTHROPIC_API_KEY etc.). "
            "If you really need to edit it, do so manually outside Claude.",
            file=sys.stderr,
        )
        return 2

    # Frozen prompt versions: prompts/<anything>/v<digits>.md
    if re.search(r"/prompts/[^/]+/v\d+\.md$", file_path):
        print(
            f"BLOCKED: {file_path}\n"
            "Reason: prompt versions are immutable once shipped. "
            "Use the bump-prompt-version skill to create the next version.",
            file=sys.stderr,
        )
        return 2

    # Recorded calibration runs.
    if "/calibration-runs/" in file_path:
        print(
            f"BLOCKED: {file_path}\n"
            "Reason: calibration-runs/ holds immutable provenance for κ/α numbers.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
