"""Prompt file loading + frontmatter validation.

Every module loads its prompt via :func:`load_prompt`. The loader parses
the YAML-subset frontmatter, extracts the body, and verifies that the
declared ``hash`` matches ``sha256`` of the body — a mismatch raises
loudly so audit provenance can never claim a prompt version that doesn't
match what ran.

Intentional limits:

- Frontmatter is a trivial key/value subset (``key: value`` per line, no
  nesting, double- or single-quoted strings stripped). Full PyYAML is not
  in the dep set and the format never needs more than this.
- Hash verification is on the **body only** (everything after the closing
  ``---\\n``). This lets the ``hash:`` field itself be bumped without a
  recursive-hash problem.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

__all__ = ["PromptFile", "load_prompt"]


ThinkingMode = Literal["disabled", "adaptive"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True, slots=True)
class PromptFile:
    """Parsed contents of ``prompts/module_<letter>/<version>.md``."""

    path: Path
    module: str
    version: str
    model: str
    thinking_mode: str
    effort: str
    citation: str
    purpose: str
    body: str
    body_hash: str


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)\Z", re.DOTALL)

_REQUIRED_KEYS = frozenset(
    {"version", "module", "model", "thinking_mode", "effort", "citation", "purpose", "hash"}
)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise ValueError("prompt file has no YAML frontmatter")
    fm_text, body = match.group(1), match.group(2)

    fm: dict[str, str] = {}
    for line in fm_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"malformed frontmatter line: {line!r}")
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        fm[key] = val
    return fm, body


def load_prompt(
    module: str,
    version: str,
    *,
    root: Path = Path("prompts"),
) -> PromptFile:
    """Load and validate ``prompts/module_<module>/<version>.md``.

    Raises :class:`ValueError` if any required frontmatter key is missing
    or if the declared ``hash`` does not equal ``sha256`` of the body.
    """
    path = root / f"module_{module.lower()}" / f"{version}.md"
    if not path.is_file():
        raise FileNotFoundError(f"prompt file not found: {path}")

    text = path.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(text)

    missing = _REQUIRED_KEYS - fm.keys()
    if missing:
        raise ValueError(f"{path}: frontmatter missing keys {sorted(missing)}")

    body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if fm["hash"] != body_hash:
        raise ValueError(
            f"{path}: frontmatter hash {fm['hash']!r} does not match "
            f"sha256(body) {body_hash!r} — bump `hash:` after editing body"
        )

    return PromptFile(
        path=path,
        module=fm["module"],
        version=fm["version"],
        model=fm["model"],
        thinking_mode=fm["thinking_mode"],
        effort=fm["effort"],
        citation=fm["citation"],
        purpose=fm["purpose"],
        body=body,
        body_hash=body_hash,
    )
