"""Sonnet 4.6 post-processor: structure Opus markdown into validated blocks.

Runs after the :class:`~lucid.synthesis.session.SynthesisSession`
completes. Reads each populated :class:`~lucid.schemas.ReportSection`
(``insufficient_evidence=False``, ``markdown`` non-empty), calls Sonnet
4.6 via ``client.messages.parse()`` using
:class:`~lucid.schemas.SynthesisSectionOutput` as the ``output_format``,
and returns the section enriched with ``blocks`` + ``citation_confidence``.

Declined sections are skipped — no post-processing needed.

Failures (Sonnet unreachable, parse failure, timeout) degrade
gracefully: the original section is returned unchanged and a warning is
logged. The section stays renderable from ``markdown`` alone.

SDK shape (anthropic 0.96.0): ``AsyncMessages.parse(...)`` accepts
``output_format=<pydantic model>`` and returns a
:class:`anthropic.types.ParsedMessage`. The parsed model lives on
``.parsed_output`` (a property that walks content blocks looking for a
``ParsedTextBlock`` with ``parsed_output`` set). When the property
returns ``None`` we fall back to extracting text blocks and validating
the JSON directly.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import orjson

from lucid.prompts import load_prompt
from lucid.schemas import ReportSection, SynthesisSectionOutput

if TYPE_CHECKING:  # pragma: no cover
    from anthropic import AsyncAnthropic

_LOGGER = logging.getLogger(__name__)

SONNET_MODEL = "claude-sonnet-4-6"
SONNET_MAX_TOKENS = 4000  # generous; sections are well under 500 words


def _extract_parsed(response: Any) -> SynthesisSectionOutput | None:
    """Extract the :class:`SynthesisSectionOutput` from a ParsedMessage response.

    anthropic 0.96.0 exposes the parsed value as ``response.parsed_output``
    (a property that walks content blocks for the parsed JSON). If that
    property returns ``None``, fall back to reconstructing from content-block
    text via :meth:`SynthesisSectionOutput.model_validate_json`.

    Returns ``None`` if both paths fail; the caller logs a warning and
    returns the section unchanged.
    """
    # Primary path: SDK's own parse machinery.
    candidate = getattr(response, "parsed_output", None)
    if isinstance(candidate, SynthesisSectionOutput):
        return candidate

    # Genuine fallback: reconstruct from text content blocks.
    try:
        text = "".join(
            block.text
            for block in getattr(response, "content", []) or []
            if hasattr(block, "text") and isinstance(block.text, str)
        )
        if not text:
            return None
        return SynthesisSectionOutput.model_validate_json(text)
    except (ValueError, AttributeError) as err:
        _LOGGER.warning("Content-block JSON reconstruction failed: %s", err)
        return None


async def post_process_section(
    *,
    async_client: AsyncAnthropic,
    section: ReportSection,
    valid_finding_ids: list[str],
    valid_turn_ids: list[str],
    tool_call_counts: dict[str, int] | None = None,
) -> ReportSection:
    """Enrich one :class:`ReportSection` with Sonnet-structured blocks.

    Declined sections (``insufficient_evidence=True``) are returned
    unchanged without an SDK call. Sonnet failures return the section
    unchanged with a warning log. Success returns a ``model_copy`` with
    ``blocks`` + ``citation_confidence`` set.

    Parameters
    ----------
    valid_finding_ids, valid_turn_ids
        Sonnet drops citations to ids not in these lists per the
        ``synthesis_validator/v1.md`` contract.
    tool_call_counts
        Advisory — Sonnet extracts ``aggregate_claim`` /
        ``aggregate_support`` but does not cross-reference them.
        Cross-checking is the Python aggregate-claim validator's job.
    """
    if section.insufficient_evidence:
        # Declined sections have no prose to structure — skip the SDK
        # round-trip entirely.
        return section

    try:
        prompt = load_prompt("synthesis_validator", "v1")
    except Exception as err:  # any load failure logs and skips
        _LOGGER.warning(
            "Sonnet post-process skipped for section %s: prompt load failed: %s",
            section.section_id,
            err,
        )
        return section

    user_payload = {
        "section_id": section.section_id,
        "raw_markdown": section.markdown,
        "valid_finding_ids": valid_finding_ids,
        "valid_turn_ids": valid_turn_ids,
        "tool_call_counts": tool_call_counts or {},
    }
    user_content = orjson.dumps(user_payload).decode()

    temperature = float(prompt.frontmatter.get("temperature", "0.0"))

    try:
        response = await async_client.messages.parse(
            model=SONNET_MODEL,
            max_tokens=SONNET_MAX_TOKENS,
            temperature=temperature,
            system=prompt.padded_body,
            messages=[{"role": "user", "content": user_content}],
            output_format=SynthesisSectionOutput,
        )
    except Exception as err:  # SDK + network errors swallowed for degradation
        _LOGGER.warning(
            "Sonnet post-process failed for section %s: %s",
            section.section_id,
            err,
        )
        return section

    structured = _extract_parsed(response)
    if structured is None:
        _LOGGER.warning(
            "Sonnet post-process for section %s: could not extract parsed JSON",
            section.section_id,
        )
        return section

    return section.model_copy(
        update={
            "blocks": list(structured.blocks),
            "citation_confidence": structured.citation_confidence,
        }
    )


async def post_process_sections(
    *,
    async_client: AsyncAnthropic,
    sections: list[ReportSection],
    valid_finding_ids: list[str],
    valid_turn_ids: list[str],
    tool_call_counts: dict[str, int] | None = None,
) -> list[ReportSection]:
    """Batch post-process — runs Sonnet per section concurrently.

    Per-section failures are already isolated inside
    :func:`post_process_section` (returns the section unchanged on any
    error), so :func:`asyncio.gather` with the default
    ``return_exceptions=False`` is safe — no exception propagates.
    ``asyncio.gather`` preserves input order in the result list.
    Declined sections pass through unchanged.
    """
    if not sections:
        return []
    return list(
        await asyncio.gather(
            *(
                post_process_section(
                    async_client=async_client,
                    section=section,
                    valid_finding_ids=valid_finding_ids,
                    valid_turn_ids=valid_turn_ids,
                    tool_call_counts=tool_call_counts,
                )
                for section in sections
            )
        )
    )


__all__ = ["post_process_section", "post_process_sections"]
