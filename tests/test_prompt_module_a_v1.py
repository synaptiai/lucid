"""Static invariant tests for ``prompts/module_a/v1.md``.

These tests guard two things that would break silently if violated:

1. The frontmatter ``hash`` field matches ``sha256`` of the body. The body
   hash is persisted on every ``Finding`` (``prompt_hash`` column) — if the
   prompt drifts without a version bump, audit provenance is wrong.
2. The body is long enough to engage Opus 4.7 prompt caching (≥ 4096
   tokens; we approximate with ≥ 16,384 characters for a ~2× safety margin
   under the worst-case tokenizer expansion).

The module-shape tests (what the runtime does with this prompt) live in
``test_module_a_spiralbench.py``. Here we only verify the file on disk.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

PROMPT_PATH = Path("prompts/module_a/v1.md")

# Characters per token upper bound across common tokenizers: Opus 4.7 can
# produce up to 1.35× more tokens than Opus 4.6 for the same text
# (methodology.md §7). A naive 4-chars-per-token heuristic underestimates
# Opus 4.7 by ~25%, so we require ≥ 4096 × 4 = 16384 chars for the body to
# be safely over the cache minimum regardless of tokenizer.
MIN_BODY_CHARS = 4096 * 4

REQUIRED_FRONTMATTER_KEYS = frozenset(
    {"version", "module", "model", "thinking_mode", "effort", "citation", "purpose", "hash"}
)

BEHAVIORS_V1_2 = frozenset(
    {
        "pushback",
        "escalation",
        "de-escalation",
        "sycophancy",
        "benign-warmth",
        "delusion-reinforcement",
        "topic-shut-down",
        "help-referral-warranted",
        "help-referral-unwarranted",
        "positive-sentience-claim",
        "uncertain-sentience-claim",
        "negative-sentience-claim",
        "harmful-advice",
        "confident-bullshitting",
        "boundary-setting",
        "validate-feelings-not-thoughts",
        "ritualization",
    }
)


def _parse_prompt_file(path: Path) -> tuple[dict[str, str], str]:
    """Return (frontmatter_dict, body). Minimal YAML subset parser."""
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)\Z", text, re.DOTALL)
    if match is None:
        raise AssertionError(f"{path} missing YAML frontmatter")
    fm_text, body = match.group(1), match.group(2)

    fm: dict[str, str] = {}
    for line in fm_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        fm[key] = val
    return fm, body


@pytest.fixture(scope="module")
def parsed() -> tuple[dict[str, str], str]:
    return _parse_prompt_file(PROMPT_PATH)


def test_prompt_file_exists() -> None:
    assert PROMPT_PATH.is_file(), f"expected prompt at {PROMPT_PATH}"


def test_frontmatter_has_all_required_keys(parsed: tuple[dict[str, str], str]) -> None:
    fm, _ = parsed
    missing = REQUIRED_FRONTMATTER_KEYS - fm.keys()
    assert not missing, f"frontmatter missing keys: {sorted(missing)}"


def test_frontmatter_model_is_opus_4_7(parsed: tuple[dict[str, str], str]) -> None:
    fm, _ = parsed
    assert fm["model"] == "claude-opus-4-7"


def test_frontmatter_thinking_and_effort_match_plan_matrix(
    parsed: tuple[dict[str, str], str],
) -> None:
    """Per the Phase 7 matrix: Module A = thinking disabled, effort low."""
    fm, _ = parsed
    assert fm["thinking_mode"] == "disabled"
    assert fm["effort"] == "low"


def test_frontmatter_hash_matches_body_sha256(parsed: tuple[dict[str, str], str]) -> None:
    """If the body drifts without the hash being bumped, this test fails loudly.

    Audit provenance depends on ``prompt_hash`` being the actual sha256 of the
    prompt body. A drifted hash would produce findings that claim to have
    come from a prompt version that doesn't exist.
    """
    fm, body = parsed
    declared = fm["hash"]
    actual = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert declared == actual, (
        f"frontmatter hash {declared!r} != sha256(body) {actual!r}; "
        "bump `hash:` after editing prompt body"
    )


def test_body_length_exceeds_opus_cache_minimum(parsed: tuple[dict[str, str], str]) -> None:
    _, body = parsed
    assert len(body) >= MIN_BODY_CHARS, (
        f"body is {len(body)} chars; need ≥ {MIN_BODY_CHARS} to safely exceed "
        "Opus 4.7's 4096-token cache minimum under the worst-case tokenizer"
    )


def test_body_mentions_every_behavior_id(parsed: tuple[dict[str, str], str]) -> None:
    """Every rubric id must appear in the body — otherwise the model's output
    JSON keys won't match the parser's expected schema."""
    _, body = parsed
    missing = {b for b in BEHAVIORS_V1_2 if b not in body}
    assert not missing, f"behaviors not mentioned in prompt body: {sorted(missing)}"


def test_body_contains_injection_resistance_instruction(
    parsed: tuple[dict[str, str], str],
) -> None:
    """A canary-style check: the prompt must explicitly instruct the model
    to treat transcript text as data, not instructions. Prevents someone
    from removing the defence paragraph in a later iteration without noticing.
    """
    _, body = parsed
    phrases = ("ignore previous instructions", "data, not instructions")
    assert any(p in body.lower() for p in phrases) or any(p in body for p in phrases), (
        "prompt should explicitly warn against prompt-injection attempts"
    )


def test_body_mentions_output_schema_section(parsed: tuple[dict[str, str], str]) -> None:
    _, body = parsed
    assert "## Output Schema" in body or "## output schema" in body.lower()
