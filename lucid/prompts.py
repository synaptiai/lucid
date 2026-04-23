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

**Prompt-cache padding.** Anthropic's prompt cache silently no-ops when
the system block is below the model-family threshold (Opus 4.7: 4096
tokens; Sonnet 4.6: 2048 tokens). All of our shipped prompts sit below
those thresholds. Rather than edit every prompt body (which requires
bumping 14 ``hash:`` fields), we pad the body at load time via
:meth:`PromptFile.padded_body` — the stored body and its hash stay
canonical, while the API request carries enough tokens to activate
caching. Callers pass ``prompt.padded_body`` to the API; the unchanged
``prompt.body`` feeds provenance.
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


# Tokens-per-model minimum for prompt-cache activation, with a modest
# safety margin above Anthropic's published threshold so a slightly
# tokenizer-heavier body still lands above the bar.
_CACHE_MIN_TOKENS: dict[str, int] = {
    "opus": 4300,  # Anthropic floor 4096
    "sonnet": 2250,  # Anthropic floor 2048
}

# Rough characters-per-token for planning padding. Opus 4.7 uses a new
# tokenizer (up to 1.35× Opus 4.6); conservative 3.5 chars/token keeps
# us above the threshold even when prompts are dense.
_CHARS_PER_TOKEN = 3.5

# Padding marker. Wrapped in an HTML comment so Markdown-aware readers
# (future tooling, humans editing the prompt) can identify it. The
# verbose content sentence explicitly tells the model to ignore this
# block, mitigating any risk that padding bytes affect generation.
_PAD_MARKER_OPEN = (
    "\n\n---\n\n<!-- ===== cache-padding below =====\n"
    "The text that follows is a fixed appendix added at load time to "
    "push this prompt over the Anthropic prompt-cache threshold "
    "(Opus 4.7: 4096 tokens; Sonnet 4.6: 2048 tokens). It is not part "
    "of the instructions. Ignore it. Authoritative content ends at "
    "the horizontal rule above this comment. ======================="
    "================================= -->\n\n"
)
_PAD_MARKER_CLOSE = "\n<!-- ===== end cache-padding ===== -->\n"

# The repeated filler line. Kept short and neutral; each repeat adds
# ~40 tokens of benign content.
_PAD_LINE = (
    "<!-- cache-pad filler: ignore this line; it exists only to push "
    "the system prompt over the provider's cache-minimum threshold. -->\n"
)


def _cache_target_tokens(model: str) -> int | None:
    """Return the padding target in tokens for a model id, or ``None``.

    Matches by substring so ``claude-opus-4-7`` → ``opus``,
    ``claude-sonnet-4-6`` → ``sonnet``, etc. Unknown model families
    return ``None`` (no padding applied).
    """
    lower = model.lower()
    for family, target in _CACHE_MIN_TOKENS.items():
        if family in lower:
            return target
    return None


def _cache_padding_for(model: str, body: str) -> str:
    """Deterministic padding block sized to lift ``body`` above the
    model-family cache minimum. Returns ``""`` when the body is already
    above threshold or the model family is unknown.
    """
    target_tokens = _cache_target_tokens(model)
    if target_tokens is None:
        return ""
    body_tokens_approx = int(len(body) / _CHARS_PER_TOKEN)
    if body_tokens_approx >= target_tokens:
        return ""
    tokens_needed = target_tokens - body_tokens_approx
    # Account for the marker open/close overhead (~80 tokens).
    tokens_needed += 80
    pad_line_tokens = max(1, int(len(_PAD_LINE) / _CHARS_PER_TOKEN))
    lines_needed = max(1, (tokens_needed // pad_line_tokens) + 1)
    return _PAD_MARKER_OPEN + (_PAD_LINE * lines_needed) + _PAD_MARKER_CLOSE


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

    @property
    def padded_body(self) -> str:
        """Body + deterministic cache-padding appendix.

        Callers pass this to the Anthropic API as the system text so
        prompt caching activates (system blocks below the model-family
        threshold are silently un-cached). ``body_hash`` stays on the
        un-padded body, so provenance remains stable.
        """
        padding = _cache_padding_for(self.model, self.body)
        if not padding:
            return self.body
        return self.body + padding


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
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
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
