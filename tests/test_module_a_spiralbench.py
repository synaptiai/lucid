"""Tests for :mod:`lucid.modules.module_a_spiralbench`.

Mocks the Anthropic client via the ``mock_anthropic_client`` factory in
``conftest.py``. Verifies the chunking logic, incident → Finding
conversion (including same-behavior-same-turn roll-up), error isolation
(one bad window does not crash the module), and the cache-control
payload shape sent on the system prompt.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast

from anthropic.types import Usage

from lucid.modules.base import ModuleCorpus, ModuleError
from lucid.modules.module_a_spiralbench import (
    BEHAVIORS,
    CHUNK_SIZE,
    MODEL,
    BehaviorIncident,
    ModuleASpiralBench,
    SpiralBenchIncidents,
    SpiralBenchScore,
    _iter_windows,
    _render_window,
    _score_to_findings,
)
from lucid.prompts import load_prompt
from lucid.schemas import Conversation, Finding, ModuleName, Role, Source, Turn

# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


def _turn(conv_id: str, idx: int, *, role: Role = Role.USER, content: str = "") -> Turn:
    return Turn(
        id=f"{conv_id}-t{idx}",
        conversation_id=conv_id,
        index=idx,
        role=role,
        content=content or f"turn {idx}",
    )


def _alternating_turns(conv_id: str, n: int) -> list[Turn]:
    return [
        _turn(
            conv_id,
            i,
            role=Role.USER if i % 2 == 0 else Role.ASSISTANT,
            content=f"{'U' if i % 2 == 0 else 'A'} says something at turn {i}",
        )
        for i in range(n)
    ]


def _corpus(conv_ids: list[str], turns_by_conv: dict[str, list[Turn]]) -> ModuleCorpus:
    convs = {
        cid: Conversation(
            id=cid,
            source=Source.CLAUDE_AI,
            source_path="/tmp/x",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 2, 1, tzinfo=UTC),
            turn_count=len(turns_by_conv[cid]),
        )
        for cid in conv_ids
    }
    return ModuleCorpus(
        conversations=convs,
        turns_by_conversation=turns_by_conv,
        audit_run_id="run-test",
    )


def _empty_score() -> SpiralBenchScore:
    return SpiralBenchScore(
        reasoning="nothing notable",
        incidents=SpiralBenchIncidents(),
    )


def _score_with(**behavior_incidents: list[BehaviorIncident]) -> SpiralBenchScore:
    return SpiralBenchScore(
        reasoning="seeded",
        incidents=SpiralBenchIncidents.model_validate(behavior_incidents),
    )


# ──────────────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────────────


def test_iter_windows_splits_at_chunk_size() -> None:
    turns = _alternating_turns("c1", 25)
    windows = list(_iter_windows("c1", turns))
    assert len(windows) == 3
    assert [len(w.turns) for w in windows] == [10, 10, 5]
    assert [w.start_index for w in windows] == [0, 10, 20]


def test_iter_windows_skips_windows_with_no_assistant_turn() -> None:
    turns = [_turn("c1", i, role=Role.USER) for i in range(5)]
    assert list(_iter_windows("c1", turns)) == []


def test_iter_windows_empty_turns_yields_nothing() -> None:
    assert list(_iter_windows("c1", [])) == []


def test_render_window_tags_roles_and_indices() -> None:
    turns = _alternating_turns("c1", 4)
    window = next(iter(_iter_windows("c1", turns)))
    rendered = _render_window(window)
    assert "<TRANSCRIPT_BLOCK>" in rendered
    assert "[USER t=0]" in rendered
    assert "[ASSISTANT t=1]" in rendered
    assert "</TRANSCRIPT_BLOCK>" in rendered


# ──────────────────────────────────────────────────────────────────────────
# Findings construction
# ──────────────────────────────────────────────────────────────────────────


def test_score_to_findings_attributes_to_correct_assistant_turn() -> None:
    turns = _alternating_turns("c1", 6)
    windows = list(_iter_windows("c1", turns))
    window = windows[0]
    # Assistant turns in this 6-turn window are at positions 1, 3, 5.
    # The model reports a sycophancy incident at turn_index=1 (the second
    # assistant turn — window position 3).
    score = _score_with(
        sycophancy=[BehaviorIncident(snippet="quote A", intensity=2, turn_index=1)],
    )
    findings = _score_to_findings(
        score,
        window,
        audit_run_id="run-1",
        prompt_hash="abc",
        detected_at=datetime(2026, 4, 22, tzinfo=UTC),
    )
    assert len(findings) == 1
    f = findings[0]
    assert f.behavior == "sycophancy"
    assert f.intensity == 2
    assert f.conversation_id == "c1"
    assert f.turn_ids == ["c1-t3"]  # 2nd assistant turn has absolute index 3
    assert f.quote_assistant == "quote A"


def test_score_to_findings_rolls_up_same_behavior_same_turn() -> None:
    turns = _alternating_turns("c1", 4)
    window = next(iter(_iter_windows("c1", turns)))
    # Two sycophancy incidents at the SAME assistant turn; rubric allows it.
    # The finding must roll up to intensity=max and extra snippets go to
    # evidence_quotes.
    score = _score_with(
        sycophancy=[
            BehaviorIncident(snippet="mild one", intensity=1, turn_index=0),
            BehaviorIncident(snippet="bad one", intensity=3, turn_index=0),
        ],
    )
    findings = _score_to_findings(
        score,
        window,
        audit_run_id="run-1",
        prompt_hash="abc",
        detected_at=datetime(2026, 4, 22, tzinfo=UTC),
    )
    assert len(findings) == 1
    f = findings[0]
    assert f.intensity == 3
    assert f.quote_assistant == "bad one"
    assert f.evidence_quotes == ["mild one"]
    assert "2 incident(s)" in f.explanation


def test_score_to_findings_keeps_separate_findings_for_different_turns() -> None:
    turns = _alternating_turns("c1", 6)
    window = next(iter(_iter_windows("c1", turns)))
    score = _score_with(
        pushback=[
            BehaviorIncident(snippet="at turn 1", intensity=2, turn_index=0),
            BehaviorIncident(snippet="at turn 3", intensity=1, turn_index=1),
        ],
    )
    findings = _score_to_findings(
        score,
        window,
        audit_run_id="run-1",
        prompt_hash="abc",
        detected_at=datetime(2026, 4, 22, tzinfo=UTC),
    )
    assert {f.turn_ids[0] for f in findings} == {"c1-t1", "c1-t3"}


def test_score_to_findings_ignores_out_of_range_turn_index() -> None:
    """If the model hallucinates an invalid turn_index, drop the incident
    but keep scoring — one bad index must not lose the rest."""
    turns = _alternating_turns("c1", 4)
    window = next(iter(_iter_windows("c1", turns)))
    score = _score_with(
        pushback=[
            BehaviorIncident(snippet="real", intensity=2, turn_index=0),
            BehaviorIncident(snippet="bogus", intensity=3, turn_index=99),
        ],
    )
    findings = _score_to_findings(
        score,
        window,
        audit_run_id="run-1",
        prompt_hash="abc",
        detected_at=datetime(2026, 4, 22, tzinfo=UTC),
    )
    assert len(findings) == 1
    assert findings[0].quote_assistant == "real"


# ──────────────────────────────────────────────────────────────────────────
# End-to-end via mocked client
# ──────────────────────────────────────────────────────────────────────────


async def test_run_produces_findings_and_uses_cache_control(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    turns = _alternating_turns("c1", 6)
    corpus = _corpus(["c1"], {"c1": turns})

    score = _score_with(
        sycophancy=[BehaviorIncident(snippet="flattery", intensity=2, turn_index=0)],
    )
    client = mock_anthropic_client(parse_outputs=[score])

    module = ModuleASpiralBench(client=client)
    results = await module.run(corpus)

    # Exactly one Finding, correctly attributed.
    findings = [r for r in results if isinstance(r, Finding)]
    assert len(findings) == 1
    assert findings[0].module == ModuleName.A_SPIRALBENCH
    assert findings[0].behavior == "sycophancy"
    assert findings[0].detected_by == [MODEL]

    # Verify that the mock client received the cache_control marker.
    call = client.messages.parse.await_args
    system = call.kwargs["system"]
    assert isinstance(system, list)
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert call.kwargs["model"] == MODEL
    assert call.kwargs["output_format"] is SpiralBenchScore


async def test_run_chunks_long_conversations_into_multiple_calls(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    turns = _alternating_turns("c1", 25)  # 3 windows of size 10,10,5
    corpus = _corpus(["c1"], {"c1": turns})

    client = mock_anthropic_client(
        parse_outputs=[_empty_score(), _empty_score(), _empty_score()],
    )
    module = ModuleASpiralBench(client=client)
    await module.run(corpus)
    assert client.messages.parse.await_count == 3


async def test_run_skips_conversations_with_no_assistant_turns(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    turns = [_turn("c1", i, role=Role.USER) for i in range(3)]
    corpus = _corpus(["c1"], {"c1": turns})

    client = mock_anthropic_client(parse_outputs=[])
    module = ModuleASpiralBench(client=client)
    results = await module.run(corpus)
    assert results == []
    assert client.messages.parse.await_count == 0


async def test_run_isolates_per_window_errors(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    """First window's parse raises; second window's parse succeeds; the
    module should produce a ModuleError for the first and a Finding for
    the second — no raising out of run()."""
    turns_a = _alternating_turns("c1", 4)
    turns_b = _alternating_turns("c2", 4)
    corpus = _corpus(["c1", "c2"], {"c1": turns_a, "c2": turns_b})

    class BoomError(Exception):
        pass

    good_score = _score_with(
        pushback=[BehaviorIncident(snippet="corrected them", intensity=1, turn_index=0)],
    )

    # Build a custom mock: raise for the first call, succeed for the second.
    from unittest.mock import AsyncMock, MagicMock

    responses = [BoomError("boom"), good_score]

    async def _parse(**_kwargs: Any) -> Any:
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        resp = MagicMock()
        resp.parsed_output = item
        resp.usage = Usage(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        return resp

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.parse = AsyncMock(side_effect=_parse)

    module = ModuleASpiralBench(client=client)
    results = await module.run(corpus)

    errors = [r for r in results if isinstance(r, ModuleError)]
    findings = [r for r in results if isinstance(r, Finding)]
    assert len(errors) == 1
    assert errors[0].error_type == "BoomError"
    assert len(findings) == 1


async def test_run_returns_empty_for_empty_corpus(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    corpus = _corpus([], {})
    client = mock_anthropic_client(parse_outputs=[])
    module = ModuleASpiralBench(client=client)
    assert await module.run(corpus) == []


# ──────────────────────────────────────────────────────────────────────────
# Prompt wiring sanity
# ──────────────────────────────────────────────────────────────────────────


def test_module_loads_v1_prompt_and_propagates_hash_to_findings(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    client = mock_anthropic_client(parse_outputs=[])
    module = ModuleASpiralBench(client=client)
    expected = load_prompt("a", "v1")
    assert module._prompt.body_hash == expected.body_hash
    assert module._prompt.model == MODEL


def test_behaviors_constant_matches_rubric_cardinality() -> None:
    assert len(BEHAVIORS) == 17
    # All behavior ids appear as aliases/names on the incidents model.
    aliases = {
        (info.alias or name)
        for name, info in SpiralBenchIncidents.model_fields.items()
    }
    assert set(BEHAVIORS) == aliases


def test_chunk_size_matches_plan() -> None:
    """Plan §6: 10-turn chunking. If we change it, acknowledge by updating
    the test (and regression-testing cache hit rates)."""
    assert CHUNK_SIZE == 10


# silence unused-import lint
_ = cast
