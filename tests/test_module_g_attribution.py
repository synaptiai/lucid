"""Tests for Module G (deterministic time/model attribution).

No LLM involved. Covers:

- Claude Code conversations (``model`` already set) → declared, confidence 1.0.
- Claude.ai conversations (``model`` None) → inferred from ``updated_at`` via
  ``ANTHROPIC_DEFAULT_MODEL_TIMELINE`` at confidence 0.6.
- Boundary dates (release day should resolve to the newly released model).
- Pre-timeline dates (before the earliest entry) → ``confidence=0.3`` and
  explanation flags uncertainty rather than silently picking a default.
- Idempotency: two runs produce the same finding ids and turn_ids_hash.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from lucid.modules.base import ModuleCorpus
from lucid.modules.module_g_attribution import (
    ANTHROPIC_DEFAULT_MODEL_TIMELINE,
    PROMPT_VERSION,
    ModuleGAttribution,
)
from lucid.schemas import Conversation, Finding, ModuleName, Source


def _conversation(
    conv_id: str,
    updated_at: datetime,
    *,
    model: str | None = None,
    source: Source = Source.CLAUDE_AI,
) -> Conversation:
    return Conversation(
        id=conv_id,
        source=source,
        source_path="/tmp/export",
        created_at=updated_at,
        updated_at=updated_at,
        model=model,
        turn_count=1,
    )


def _corpus(*convs: Conversation) -> ModuleCorpus:
    return ModuleCorpus(
        conversations={c.id: c for c in convs},
        turns_by_conversation={c.id: [] for c in convs},
        audit_run_id="run-abc",
    )


async def test_claude_code_conversation_uses_declared_model() -> None:
    conv = _conversation(
        "c1",
        datetime(2026, 3, 1, tzinfo=UTC),
        model="claude-opus-4-6",
        source=Source.CLAUDE_CODE,
    )
    results = await ModuleGAttribution().run(_corpus(conv))

    assert len(results) == 1
    finding = results[0]
    assert isinstance(finding, Finding)
    assert finding.module == ModuleName.G_ATTRIBUTION
    assert finding.behavior == "model=claude-opus-4-6"
    assert finding.confidence == 1.0
    assert finding.metadata["source"] == "declared"
    assert finding.metadata["year_month"] == "2026-03"


async def test_claude_ai_inferred_from_timeline() -> None:
    """2026-03-01: Sonnet 4.6 is the default (released 2026-02-17)."""
    conv = _conversation("c1", datetime(2026, 3, 1, tzinfo=UTC))
    results = await ModuleGAttribution().run(_corpus(conv))
    finding = results[0]
    assert isinstance(finding, Finding)
    assert finding.behavior == "model=claude-sonnet-4-6"
    assert finding.confidence == pytest.approx(0.6)
    assert finding.metadata["source"] == "inferred"


async def test_release_day_boundary_picks_new_model() -> None:
    """Same calendar day as a release → newly released model."""
    conv = _conversation("c1", datetime(2026, 2, 17, tzinfo=UTC))
    results = await ModuleGAttribution().run(_corpus(conv))
    finding = results[0]
    assert isinstance(finding, Finding)
    assert finding.behavior == "model=claude-sonnet-4-6"


async def test_day_before_release_picks_prior_model() -> None:
    conv = _conversation("c1", datetime(2026, 2, 16, tzinfo=UTC))
    results = await ModuleGAttribution().run(_corpus(conv))
    finding = results[0]
    assert isinstance(finding, Finding)
    # Before Sonnet 4.6 release (2026-02-17), default is Sonnet 4.5.
    assert finding.behavior == "model=claude-sonnet-4-5"


async def test_pre_timeline_date_marked_unknown() -> None:
    conv = _conversation("c1", datetime(2023, 1, 1, tzinfo=UTC))
    results = await ModuleGAttribution().run(_corpus(conv))
    finding = results[0]
    assert isinstance(finding, Finding)
    assert finding.behavior == "model=unknown"
    assert finding.confidence == pytest.approx(0.3)
    assert finding.metadata["source"] == "inferred"


async def test_multiple_conversations_each_get_a_finding() -> None:
    convs = [
        _conversation("c1", datetime(2026, 1, 15, tzinfo=UTC)),
        _conversation("c2", datetime(2026, 3, 1, tzinfo=UTC)),
        _conversation("c3", datetime(2025, 10, 1, tzinfo=UTC)),
    ]
    results = await ModuleGAttribution().run(_corpus(*convs))
    assert len(results) == 3
    assert {r.conversation_id for r in results if isinstance(r, Finding)} == {"c1", "c2", "c3"}


async def test_idempotent_ids_across_runs() -> None:
    conv = _conversation("c1", datetime(2026, 3, 1, tzinfo=UTC))
    corpus = _corpus(conv)

    first = await ModuleGAttribution().run(corpus)
    second = await ModuleGAttribution().run(corpus)

    assert isinstance(first[0], Finding)
    assert isinstance(second[0], Finding)
    assert first[0].id == second[0].id
    assert first[0].turn_ids_hash == second[0].turn_ids_hash


async def test_prompt_version_populates_finding_provenance() -> None:
    conv = _conversation("c1", datetime(2026, 3, 1, tzinfo=UTC))
    results = await ModuleGAttribution().run(_corpus(conv))
    finding = results[0]
    assert isinstance(finding, Finding)
    assert finding.prompt_version == PROMPT_VERSION
    assert finding.prompt_hash  # non-empty
    assert finding.detected_by == ["deterministic-attribution"]


def test_timeline_is_chronologically_ordered() -> None:
    """Binary-search correctness depends on monotonic date ordering."""
    dates = [entry[0] for entry in ANTHROPIC_DEFAULT_MODEL_TIMELINE]
    assert dates == sorted(dates), "ANTHROPIC_DEFAULT_MODEL_TIMELINE must be sorted by date"
