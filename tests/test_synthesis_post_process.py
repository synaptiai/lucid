"""Tests for the Sonnet post-processor."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from lucid.schemas import ReportSection, SynthesisBlock, SynthesisSectionOutput
from lucid.synthesis.post_process import post_process_section, post_process_sections


def _section(
    *,
    sid: str = "exec_summary",
    declined: bool = False,
    markdown: str = "Prose [F:f1].",
    citations: list[str] | None = None,
) -> ReportSection:
    return ReportSection(
        audit_run_id="run-pp",
        section_id=sid,
        markdown="" if declined else markdown,
        cited_finding_ids=[] if declined else (citations or ["f1"]),
        cited_turn_ids=[],
        insufficient_evidence=declined,
        decline_reason="thin evidence" if declined else None,
        created_at=datetime.now(tz=UTC),
    )


def _mock_sonnet_response(structured: SynthesisSectionOutput) -> MagicMock:
    """Fake anthropic ParsedMessage with `.parsed_output` set.

    ``parsed_output`` is a property on the real ParsedMessage class;
    ``MagicMock`` happily accepts attribute assignment so we don't need
    to build a real instance.
    """
    response = MagicMock()
    response.parsed_output = structured
    return response


@pytest.mark.asyncio
async def test_declined_section_skipped_without_sdk_call() -> None:
    """Sections with insufficient_evidence=True are returned unchanged, no SDK call."""
    async_client = MagicMock()
    async_client.messages.parse = AsyncMock(side_effect=AssertionError("should not be called"))
    section = _section(declined=True)
    result = await post_process_section(
        async_client=async_client,
        section=section,
        valid_finding_ids=["f1"],
        valid_turn_ids=[],
    )
    assert result is section  # exactly the same object
    async_client.messages.parse.assert_not_called()


@pytest.mark.asyncio
async def test_populated_section_gets_enriched_with_blocks() -> None:
    """Sonnet returns SynthesisSectionOutput; section gains blocks + confidence."""
    structured = SynthesisSectionOutput(
        section_id="exec_summary",
        blocks=[SynthesisBlock(text="Prose.", citations=["f1"])],
        raw_markdown="Prose [F:f1].",
        citation_confidence=0.88,
        cited_finding_ids=["f1"],
        cited_turn_ids=[],
    )
    async_client = MagicMock()
    async_client.messages.parse = AsyncMock(return_value=_mock_sonnet_response(structured))
    section = _section()
    result = await post_process_section(
        async_client=async_client,
        section=section,
        valid_finding_ids=["f1"],
        valid_turn_ids=[],
    )
    assert len(result.blocks) == 1
    assert result.blocks[0].text == "Prose."
    assert result.citation_confidence == 0.88
    # Other fields preserved.
    assert result.section_id == section.section_id
    assert result.markdown == section.markdown
    async_client.messages.parse.assert_awaited_once()


@pytest.mark.asyncio
async def test_sonnet_exception_returns_section_unchanged() -> None:
    """SDK raising any exception -> section returned unchanged + warning logged."""
    async_client = MagicMock()
    async_client.messages.parse = AsyncMock(side_effect=RuntimeError("Sonnet down"))
    section = _section()
    result = await post_process_section(
        async_client=async_client,
        section=section,
        valid_finding_ids=["f1"],
        valid_turn_ids=[],
    )
    # Unchanged.
    assert result.blocks == []
    assert result.citation_confidence is None
    assert result.markdown == section.markdown


@pytest.mark.asyncio
async def test_sonnet_unparseable_response_returns_section_unchanged() -> None:
    """parsed_output=None AND content-extraction fails -> graceful fallback."""
    async_client = MagicMock()
    bad_response = MagicMock()
    bad_response.parsed_output = None
    bad_response.content = []  # empty — nothing to extract
    async_client.messages.parse = AsyncMock(return_value=bad_response)
    section = _section()
    result = await post_process_section(
        async_client=async_client,
        section=section,
        valid_finding_ids=["f1"],
        valid_turn_ids=[],
    )
    assert result.blocks == []
    assert result.citation_confidence is None


@pytest.mark.asyncio
async def test_batch_post_process_isolates_failures() -> None:
    """post_process_sections: one section's failure doesn't affect siblings."""
    structured = SynthesisSectionOutput(
        section_id="exec_summary",
        blocks=[SynthesisBlock(text="Prose.", citations=["f1"])],
        raw_markdown="Prose [F:f1].",
        citation_confidence=0.9,
        cited_finding_ids=["f1"],
        cited_turn_ids=[],
    )
    # First call succeeds, second raises.
    async_client = MagicMock()
    async_client.messages.parse = AsyncMock(
        side_effect=[
            _mock_sonnet_response(structured),
            RuntimeError("rate limit"),
        ]
    )
    section_a = _section(sid="exec_summary")
    section_b = _section(sid="top_3_actions", markdown="Another [F:f2]", citations=["f2"])
    results = await post_process_sections(
        async_client=async_client,
        sections=[section_a, section_b],
        valid_finding_ids=["f1", "f2"],
        valid_turn_ids=[],
    )
    assert len(results) == 2
    # First enriched.
    assert len(results[0].blocks) == 1
    assert results[0].citation_confidence == 0.9
    # Second unchanged.
    assert results[1].blocks == []
    assert results[1].citation_confidence is None


@pytest.mark.asyncio
async def test_empty_sections_list_returns_empty() -> None:
    async_client = MagicMock()
    async_client.messages.parse = AsyncMock(side_effect=AssertionError("should not be called"))
    results = await post_process_sections(
        async_client=async_client,
        sections=[],
        valid_finding_ids=[],
        valid_turn_ids=[],
    )
    assert results == []
