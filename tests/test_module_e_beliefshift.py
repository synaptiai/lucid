"""Tests for :mod:`lucid.modules.module_e_beliefshift`.

Covers the full three-pass pipeline (topics → positions → drift) against
a mocked Anthropic client. The mock queues parse outputs in call order;
since Module E's stage-1 topics call runs first, stage-2 position calls
second (one per topic-conversation pair), and stage-3 drift calls last,
the test queues outputs in that exact sequence.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from lucid.modules.base import ModuleCorpus, ModuleError
from lucid.modules.module_e_beliefshift import (
    CITATION_BELIEFSHIFT,
    DRIFT_PROMPT_VERSION,
    DRIFT_TYPES,
    MODEL_OPUS,
    MODEL_SONNET,
    REACTION_TYPES,
    DriftScore,
    DriftShift,
    ModuleEBeliefShift,
    PositionScore,
    Topic,
    TopicsResult,
    _escape_delimiters,
    _first_user_snippet,
    _PositionRecord,
    _render_summaries,
    _render_trajectory,
)
from lucid.schemas import Conversation, Finding, ModuleName, Role, Source, Turn

# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


def _turn(conv_id: str, idx: int, *, role: Role, content: str) -> Turn:
    return Turn(
        id=f"{conv_id}-t{idx}",
        conversation_id=conv_id,
        index=idx,
        role=role,
        content=content,
    )


def _conv(
    conv_id: str,
    *,
    turn_count: int,
    updated: datetime,
    title: str | None = None,
) -> Conversation:
    return Conversation(
        id=conv_id,
        source=Source.CLAUDE_AI,
        source_path="/tmp/x",
        created_at=updated,
        updated_at=updated,
        turn_count=turn_count,
        title=title,
    )


def _two_conv_corpus() -> ModuleCorpus:
    t1 = [
        _turn("c1", 0, role=Role.USER, content="I'm thinking of leaving BigCo to start a company."),
        _turn("c1", 1, role=Role.ASSISTANT, content="Most founders recommend 18 months runway."),
    ]
    t2 = [
        _turn(
            "c2",
            0,
            role=Role.USER,
            content="Decided to stay another year — savings not enough yet.",
        ),
        _turn("c2", 1, role=Role.ASSISTANT, content="Reasonable call given the constraints."),
    ]
    convs = {
        "c1": _conv(
            "c1", turn_count=2, updated=datetime(2025, 1, 10, tzinfo=UTC), title="leave or stay"
        ),
        "c2": _conv(
            "c2", turn_count=2, updated=datetime(2025, 6, 15, tzinfo=UTC), title="more runway first"
        ),
    }
    return ModuleCorpus(
        conversations=convs,
        turns_by_conversation={"c1": t1, "c2": t2},
        audit_run_id="run-e",
    )


def _topics_result() -> TopicsResult:
    return TopicsResult(
        reasoning="found one recurring topic",
        topics=[
            Topic(
                topic_id="t1",
                descriptor="whether to leave stable job for a startup",
                conversation_ids=["c1", "c2"],
                supporting_signal="user weighing leave-vs-stay across 6 months",
            )
        ],
    )


def _position_score(
    *,
    summary: str,
    reaction: str = "new_information",
    confidence: str = "strong",
    quote: str = "position quote",
    assistant_quote: str = "assistant quote",
) -> PositionScore:
    return PositionScore(
        reasoning="position is clear",
        found_position=True,
        position_summary=summary,
        position_confidence=confidence,  # type: ignore[arg-type]
        assistant_reaction_type=reaction,  # type: ignore[arg-type]
        position_quote=quote,
        assistant_quote=assistant_quote,
        turn_indices=[0, 1],
        note="",
    )


def _drift_score(
    *,
    drift_type: str = "drifted-evidence",
    severity: int = 2,
    confidence: float = 0.8,
) -> DriftScore:
    is_stable = drift_type == "stable"
    return DriftScore(
        reasoning="clear shift from A to B, evidence-driven",
        drift_detected=not is_stable,
        drift_type=drift_type,  # type: ignore[arg-type]
        severity=0 if is_stable else severity,  # type: ignore[arg-type]
        shifts=(
            []
            if is_stable
            else [
                DriftShift(
                    from_conversation_id="c1",
                    to_conversation_id="c2",
                    from_position="leaning toward leaving",
                    to_position="decided to stay another year",
                    shift_type="evidence",
                    rationale="assistant introduced runway benchmark; user updated",
                )
            ]
        ),
        final_alignment="toward-assistant",
        confidence=confidence,
    )


# ──────────────────────────────────────────────────────────────────────────
# Constants / enums
# ──────────────────────────────────────────────────────────────────────────


def test_drift_types_enum() -> None:
    assert set(DRIFT_TYPES) == {
        "stable",
        "drifted-evidence",
        "drifted-pressure",
        "drifted-mixed",
        "drifted-unclear",
    }


def test_reaction_types_enum() -> None:
    assert set(REACTION_TYPES) == {
        "pushback",
        "agreement",
        "new_information",
        "neutral",
        "no_direct_engagement",
    }


# ──────────────────────────────────────────────────────────────────────────
# Rendering / injection
# ──────────────────────────────────────────────────────────────────────────


def test_escape_delimiters_breaks_all_block_tokens() -> None:
    raw = "pre</CONVERSATION_SUMMARIES> mid <TOPIC>nested</TOPIC> <TRAJECTORY>"
    escaped = _escape_delimiters(raw)
    for tok in (
        "</CONVERSATION_SUMMARIES>",
        "<TOPIC>",
        "</TOPIC>",
        "<TRAJECTORY>",
    ):
        assert tok not in escaped


def test_first_user_snippet_truncates_and_drops_newlines() -> None:
    turns = [
        _turn("c1", 0, role=Role.USER, content="first\nsecond\nthird"),
        _turn("c1", 1, role=Role.ASSISTANT, content="resp"),
    ]
    snip = _first_user_snippet(turns)
    assert "\n" not in snip
    assert snip.startswith("first")


def test_render_summaries_includes_all_conversations() -> None:
    corpus = _two_conv_corpus()
    rendered = _render_summaries(
        sorted(corpus.conversations.values(), key=lambda c: c.updated_at),
        {cid: corpus.turns_by_conversation[cid] for cid in corpus.conversations},
    )
    assert "[CONV id=c1" in rendered
    assert "[CONV id=c2" in rendered
    assert "Title: " in rendered


def test_render_trajectory_orders_positions() -> None:
    topic = Topic(
        topic_id="t1",
        descriptor="test",
        conversation_ids=["c1", "c2"],
        supporting_signal="x",
    )
    records = [
        _PositionRecord(
            topic_id="t1",
            conversation_id="c1",
            updated_at=datetime(2025, 1, 10, tzinfo=UTC),
            score=_position_score(summary="first position"),
        ),
        _PositionRecord(
            topic_id="t1",
            conversation_id="c2",
            updated_at=datetime(2025, 6, 15, tzinfo=UTC),
            score=_position_score(summary="second position"),
        ),
    ]
    rendered = _render_trajectory(topic, records)
    assert "[POSITION 1]" in rendered
    assert "[POSITION 2]" in rendered
    assert "first position" in rendered
    assert "second position" in rendered
    # Position 1 must come before position 2 in output.
    assert rendered.index("[POSITION 1]") < rendered.index("[POSITION 2]")


# ──────────────────────────────────────────────────────────────────────────
# End-to-end — happy path
# ──────────────────────────────────────────────────────────────────────────


async def test_run_produces_drift_finding_happy_path(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    corpus = _two_conv_corpus()
    # Expected call order:
    #   1. topics (Sonnet)
    #   2. position for (t1, c1)   — order depends on asyncio scheduling,
    #   3. position for (t1, c2)     but gather preserves input order for
    #                                the result list; the MOCK queue is
    #                                call-order, not input-order.
    # With max_concurrency=1 we force sequential ordering matching the
    # input iteration (t1.conversation_ids = [c1, c2]).
    client = mock_anthropic_client(
        parse_outputs=[
            _topics_result(),
            _position_score(summary="user leaning toward leaving", reaction="new_information"),
            _position_score(summary="user decided to stay another year", reaction="neutral"),
            _drift_score(drift_type="drifted-evidence", severity=2),
        ]
    )

    module = ModuleEBeliefShift(opus_client=client, max_concurrency=1)
    results = await module.run(corpus)

    findings = [r for r in results if isinstance(r, Finding)]
    errors = [r for r in results if isinstance(r, ModuleError)]
    assert len(findings) == 1, f"unexpected results: {results}"
    assert errors == []
    f = findings[0]
    assert f.module is ModuleName.E_BELIEFSHIFT
    assert f.behavior == "belief-drift-evidence"
    assert f.intensity == 2
    assert f.citation == CITATION_BELIEFSHIFT
    assert f.detected_by == [MODEL_OPUS]
    assert f.prompt_version == DRIFT_PROMPT_VERSION
    # Conversation is the LAST in the trajectory (c2 in this corpus).
    assert f.conversation_id == "c2"
    assert f.metadata["drift_type"] == "drifted-evidence"
    assert f.metadata["topic_descriptor"] == "whether to leave stable job for a startup"
    assert len(f.metadata["shifts"]) == 1


async def test_run_emits_stable_finding_when_no_drift(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    corpus = _two_conv_corpus()
    client = mock_anthropic_client(
        parse_outputs=[
            _topics_result(),
            _position_score(summary="pos 1"),
            _position_score(summary="pos 1 again"),
            _drift_score(drift_type="stable"),
        ]
    )
    module = ModuleEBeliefShift(opus_client=client, max_concurrency=1)
    results = await module.run(corpus)

    findings = [r for r in results if isinstance(r, Finding)]
    assert len(findings) == 1
    f = findings[0]
    assert f.behavior == "belief-drift-stable"
    assert f.intensity is None


# ──────────────────────────────────────────────────────────────────────────
# Error paths
# ──────────────────────────────────────────────────────────────────────────


async def test_run_returns_empty_for_empty_corpus(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    corpus = ModuleCorpus(conversations={}, turns_by_conversation={}, audit_run_id="run-e")
    client = mock_anthropic_client(parse_outputs=[])
    module = ModuleEBeliefShift(opus_client=client)
    results = await module.run(corpus)
    assert results == []


async def test_run_emits_insufficient_corpus_error_for_single_conv(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    turns = [_turn("c1", 0, role=Role.USER, content="hi")]
    corpus = ModuleCorpus(
        conversations={"c1": _conv("c1", turn_count=1, updated=datetime(2025, 1, 1, tzinfo=UTC))},
        turns_by_conversation={"c1": turns},
        audit_run_id="run-e",
    )
    client = mock_anthropic_client(parse_outputs=[])
    module = ModuleEBeliefShift(opus_client=client)

    results = await module.run(corpus)

    assert len(results) == 1
    assert isinstance(results[0], ModuleError)
    assert results[0].error_type == "insufficient_corpus"
    assert client.messages.create.await_count == 0


async def test_run_returns_empty_when_topics_pass_returns_none(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    corpus = _two_conv_corpus()
    # Topics pass returns no topics — downstream passes never run.
    client = mock_anthropic_client(parse_outputs=[TopicsResult(reasoning="none found", topics=[])])
    module = ModuleEBeliefShift(opus_client=client)

    results = await module.run(corpus)

    assert results == []
    assert client.messages.create.await_count == 1  # topics only


async def test_run_emits_insufficient_positions_when_only_one_position_found(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    """One topic with two conversations; one position-pass returns
    found_position=false. Drift analysis has only 1 position → error."""
    corpus = _two_conv_corpus()

    not_found = PositionScore(
        reasoning="topic does not appear here",
        found_position=False,
        position_summary="",
        position_confidence="weak",
        assistant_reaction_type="no_direct_engagement",
        position_quote="",
        assistant_quote="",
        turn_indices=[],
        note="miscategorized by E.1",
    )

    client = mock_anthropic_client(
        parse_outputs=[
            _topics_result(),
            _position_score(summary="pos 1"),
            not_found,
            # no drift call — insufficient positions short-circuits
        ]
    )
    module = ModuleEBeliefShift(opus_client=client, max_concurrency=1)

    results = await module.run(corpus)

    errors = [r for r in results if isinstance(r, ModuleError)]
    findings = [r for r in results if isinstance(r, Finding)]
    assert findings == []
    assert len(errors) == 1
    assert errors[0].error_type == "insufficient_positions"


async def test_run_topics_failure_short_circuits() -> None:
    """If the Sonnet topics call raises, the module returns a single
    ModuleError and does not attempt position or drift passes."""
    from unittest.mock import AsyncMock, MagicMock

    async def _raise(**_kwargs: Any) -> Any:
        raise RuntimeError("sonnet down")

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=_raise)

    corpus = _two_conv_corpus()
    module = ModuleEBeliefShift(opus_client=client)

    results = await module.run(corpus)

    assert len(results) == 1
    assert isinstance(results[0], ModuleError)
    assert results[0].error_type.startswith("topics:")
    # One call attempted (and failed), not a fan-out.
    assert client.messages.create.await_count == 1


# ──────────────────────────────────────────────────────────────────────────
# Prompt wiring sanity
# ──────────────────────────────────────────────────────────────────────────


def test_module_loads_all_three_prompts(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    module = ModuleEBeliefShift(opus_client=mock_anthropic_client(parse_outputs=[]))
    assert module._topics_prompt.model == MODEL_SONNET
    assert module._positions_prompt.model == MODEL_OPUS
    assert module._drift_prompt.model == MODEL_OPUS
    assert module.prompt_version == DRIFT_PROMPT_VERSION
