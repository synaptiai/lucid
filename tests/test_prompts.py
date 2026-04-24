"""Tests for ``lucid/prompts.py`` — the prompt-loading machinery.

Covers the synthesis prompts added in Phase 4:

- ``prompts/synthesis/v1.md`` (Opus 4.7 writer)
- ``prompts/synthesis_validator/v1.md`` (Sonnet 4.6 post-processor)

These prompts live outside the classic ``module_<letter>/`` layout; the
loader's path-resolution fallback lands on ``prompts/<name>/`` when the
module-layout candidate is absent. The tests here guard:

1. Both prompts load without raising (frontmatter parsed, hash verified).
2. Writer prompt declares Opus-family sampling params
   (``thinking_mode`` + ``effort``; no ``temperature``).
3. Validator prompt declares Sonnet-family sampling params
   (``temperature: 0.0``; no ``effort``).
4. Padded bodies exceed the model-family cache thresholds (Opus 4.7:
   4096 tokens; Sonnet 4.6: 2048 tokens). Conservative char floor is
   ``tokens * 3``.
5. ``SYNTHESIS_PROMPT_VERSION`` is exported from ``lucid.synthesis``.
"""

from __future__ import annotations

from lucid.prompts import load_prompt


def test_synthesis_v1_prompt_loads() -> None:
    """The synthesis writer prompt loads and has required frontmatter."""
    prompt = load_prompt("synthesis", "v1")
    assert prompt.version == "v1"
    assert prompt.model == "claude-opus-4-7"
    assert prompt.frontmatter.get("thinking_mode") in {"disabled", "adaptive"}
    assert prompt.frontmatter.get("effort") in {"low", "medium", "high", "xhigh", "max"}
    # Opus 4.7 rejects temperature — frontmatter must not set it.
    assert "temperature" not in prompt.frontmatter


def test_synthesis_validator_v1_prompt_loads() -> None:
    """The Sonnet post-processor prompt loads with its frontmatter."""
    prompt = load_prompt("synthesis_validator", "v1")
    assert prompt.version == "v1"
    assert prompt.model == "claude-sonnet-4-6"
    # Sonnet 4.6 keeps temperature; effort is not set.
    assert "temperature" in prompt.frontmatter
    assert "effort" not in prompt.frontmatter


def test_synthesis_v1_prompt_padding_hits_opus_cache_threshold() -> None:
    """padded_body must be >= 4096 tokens for Opus cache activation."""
    prompt = load_prompt("synthesis", "v1")
    body = prompt.padded_body
    # Rough tokens-to-chars heuristic: 1 token ~ 3 chars (conservative).
    assert len(body) >= 4096 * 3


def test_synthesis_validator_v1_prompt_padding_hits_sonnet_cache_threshold() -> None:
    """Sonnet 4.6 cache threshold is 2048 tokens."""
    prompt = load_prompt("synthesis_validator", "v1")
    body = prompt.padded_body
    assert len(body) >= 2048 * 3


def test_synthesis_prompt_version_constant() -> None:
    """``SYNTHESIS_PROMPT_VERSION`` is exported and matches the shipped prompt."""
    from lucid.synthesis import SYNTHESIS_PROMPT_VERSION

    assert SYNTHESIS_PROMPT_VERSION == "v1"
