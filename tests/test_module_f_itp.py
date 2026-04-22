"""Tests for :mod:`lucid.modules.module_f_itp` (3-stage ITP detector)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from lucid.modules._f_heuristic_v1 import (
    HEURISTIC_VERSION,
    ITP_CATEGORIES,
    heuristic_match,
    looks_like_tactic_candidate,
)
from lucid.modules.base import ModuleCorpus, ModuleError
from lucid.modules.module_f_itp import (
    CITATION_ITP,
    CLASSIFY_PROMPT_VERSION,
    MODEL_OPUS,
    MODEL_SONNET,
    TRIAGE_PROMPT_VERSION,
    ClassifyResult,
    ITPTactic,
    ModuleFITP,
    TriageResult,
    _Candidate,
    _detect_candidates,
    _escape_delimiters,
    _prior_assistant_turn,
    _render_classify_request,
    _render_triage_request,
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


def _corpus(turns_by_conv: dict[str, list[Turn]]) -> ModuleCorpus:
    return ModuleCorpus(
        conversations={
            cid: Conversation(
                id=cid,
                source=Source.CLAUDE_AI,
                source_path="/tmp/x",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                updated_at=datetime(2026, 2, 1, tzinfo=UTC),
                turn_count=len(v),
            )
            for cid, v in turns_by_conv.items()
        },
        turns_by_conversation=turns_by_conv,
        audit_run_id="run-f",
    )


def _candidate(
    *,
    user_content: str = "I really need you to do this right now",
    prior_content: str | None = "Here's the plan for refactoring the auth layer.",
    conv_id: str = "c1",
    user_idx: int = 1,
) -> _Candidate:
    user_turn = _turn(conv_id, user_idx, role=Role.USER, content=user_content)
    prior = (
        _turn(conv_id, user_idx - 1, role=Role.ASSISTANT, content=prior_content)
        if prior_content is not None
        else None
    )
    match = heuristic_match(user_content)
    return _Candidate(
        conversation_id=conv_id,
        user_turn=user_turn,
        prior_assistant_turn=prior,
        matched_categories=match.matched_categories,
        matched_snippets=match.matched_snippets,
    )


# ──────────────────────────────────────────────────────────────────────────
# Stage 1 — heuristic
# ──────────────────────────────────────────────────────────────────────────


def test_itp_categories_has_nine_entries() -> None:
    assert len(ITP_CATEGORIES) == 9
    assert "emotional-triggers" in ITP_CATEGORIES
    assert "logical-fallacies" in ITP_CATEGORIES


def test_heuristic_match_detects_emotional_triggers() -> None:
    result = heuristic_match("I really need you to tell me this is fine.")
    assert result.is_candidate
    assert "emotional-triggers" in result.matched_categories


def test_heuristic_match_detects_urgent_action_demands() -> None:
    result = heuristic_match("We need to act right now — there's no time.")
    assert result.is_candidate
    assert "urgent-action-demands" in result.matched_categories


def test_heuristic_match_detects_emotional_repetition_caps_and_punct() -> None:
    result = heuristic_match("DO THIS NOW!!!")
    assert result.is_candidate
    assert "emotional-repetition" in result.matched_categories


def test_heuristic_match_detects_false_dilemma() -> None:
    result = heuristic_match("Either we ship tonight or the project is dead.")
    assert result.is_candidate
    assert "false-dilemmas" in result.matched_categories


def test_heuristic_match_detects_authority_overload() -> None:
    result = heuristic_match("Experts agree that this is the right answer.")
    assert result.is_candidate
    assert "authority-overload" in result.matched_categories


def test_heuristic_match_detects_framing_techniques() -> None:
    result = heuristic_match("Obviously the right answer is microservices.")
    assert result.is_candidate
    assert "framing-techniques" in result.matched_categories


def test_heuristic_match_skips_benign_technical_prompt() -> None:
    result = heuristic_match("Can you help me debug this Python traceback?")
    assert not result.is_candidate
    assert result.matched_categories == ()


def test_looks_like_tactic_candidate_boolean_wrapper() -> None:
    assert looks_like_tactic_candidate("I really need you to…")
    assert not looks_like_tactic_candidate("How does Python dict work?")


def test_heuristic_collapses_duplicates_per_category() -> None:
    """Multiple matches within the same category count as one for matched_categories."""
    result = heuristic_match("Obviously clearly of course we must do this")
    assert result.matched_categories.count("framing-techniques") == 1


# ──────────────────────────────────────────────────────────────────────────
# Candidate detection from corpus
# ──────────────────────────────────────────────────────────────────────────


def test_detect_candidates_finds_user_turns_with_heuristic_hits() -> None:
    corpus = _corpus(
        {
            "c1": [
                _turn("c1", 0, role=Role.USER, content="Can you help with Python?"),  # benign
                _turn("c1", 1, role=Role.ASSISTANT, content="Sure, what's the issue?"),
                _turn(
                    "c1", 2, role=Role.USER, content="I really need you to just say yes."
                ),  # tactic
            ]
        }
    )
    cands = _detect_candidates(corpus)
    assert len(cands) == 1
    assert cands[0].user_turn.id == "c1-t2"
    assert "emotional-triggers" in cands[0].matched_categories
    assert cands[0].prior_assistant_turn is not None
    assert cands[0].prior_assistant_turn.id == "c1-t1"


def test_detect_candidates_handles_empty_prompts() -> None:
    corpus = _corpus(
        {
            "c1": [
                _turn("c1", 0, role=Role.USER, content=""),
                _turn("c1", 1, role=Role.USER, content="   "),
            ]
        }
    )
    assert _detect_candidates(corpus) == []


def test_detect_candidates_returns_none_for_first_user_turn_as_prior() -> None:
    """User turn is the first turn; no prior assistant turn."""
    corpus = _corpus(
        {
            "c1": [
                _turn("c1", 0, role=Role.USER, content="Obviously we should do X."),
            ]
        }
    )
    cands = _detect_candidates(corpus)
    assert len(cands) == 1
    assert cands[0].prior_assistant_turn is None


def test_prior_assistant_turn_walks_backward_past_user_turns() -> None:
    turns = [
        _turn("c1", 0, role=Role.ASSISTANT, content="A0"),
        _turn("c1", 1, role=Role.USER, content="U1"),
        _turn("c1", 2, role=Role.USER, content="U2"),  # double-user
        _turn("c1", 3, role=Role.ASSISTANT, content="A3"),
        _turn("c1", 4, role=Role.USER, content="U4"),
    ]
    prior = _prior_assistant_turn(turns, 4)
    assert prior is not None
    assert prior.id == "c1-t3"


# ──────────────────────────────────────────────────────────────────────────
# Injection resistance
# ──────────────────────────────────────────────────────────────────────────


def test_escape_delimiters_covers_all_block_tokens() -> None:
    tokens = [
        "<USER_PROMPT>",
        "</USER_PROMPT>",
        "<PRIOR_ASSISTANT_TURN>",
        "</PRIOR_ASSISTANT_TURN>",
        "<HEURISTIC_MATCHES>",
        "</HEURISTIC_MATCHES>",
        "<STAGE_2_DECISION>",
        "</STAGE_2_DECISION>",
    ]
    raw = " ".join(tokens) + " and benign text"
    escaped = _escape_delimiters(raw)
    for t in tokens:
        assert t not in escaped


def test_render_triage_request_includes_all_blocks() -> None:
    cand = _candidate()
    rendered = _render_triage_request(cand)
    assert "<USER_PROMPT>" in rendered
    assert "<PRIOR_ASSISTANT_TURN>" in rendered
    assert "<HEURISTIC_MATCHES>" in rendered
    assert "matched_categories:" in rendered
    # Exactly one of each outer delimiter.
    assert rendered.count("<USER_PROMPT>") == 1
    assert rendered.count("</USER_PROMPT>") == 1


def test_render_classify_request_adds_stage2_decision() -> None:
    cand = _candidate()
    triage = TriageResult(
        reasoning="looks like real tactics",
        decision="proceed",
        rationale_category="emotional-triggers",
    )
    rendered = _render_classify_request(cand, triage)
    assert "<STAGE_2_DECISION>" in rendered
    assert "decision: proceed" in rendered
    assert "rationale_category: emotional-triggers" in rendered


# ──────────────────────────────────────────────────────────────────────────
# End-to-end — happy path
# ──────────────────────────────────────────────────────────────────────────


def _triage_proceed() -> TriageResult:
    return TriageResult(
        reasoning="clear emotional + urgency",
        decision="proceed",
        rationale_category="multiple",
    )


def _triage_drop() -> TriageResult:
    return TriageResult(
        reasoning="heuristic false positive; technical usage",
        decision="drop",
        rationale_category="emotional-triggers",
    )


def _classify_two_tactics() -> ClassifyResult:
    return ClassifyResult(
        reasoning="both emotional framing and false urgency present",
        tactics=[
            ITPTactic(
                category="emotional-triggers",
                intensity=2,
                phrase="I really need you to",
                explanation="emotional pressure tied directly to the request",
            ),
            ITPTactic(
                category="urgent-action-demands",
                intensity=2,
                phrase="right now",
                explanation="urgency without objective deadline",
            ),
        ],
        overall_confidence=0.85,
    )


def _classify_empty() -> ClassifyResult:
    return ClassifyResult(
        reasoning="Stage 2 false positive; technical instrumental use",
        tactics=[],
        overall_confidence=0.9,
    )


async def test_run_full_pipeline_produces_per_tactic_findings(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    corpus = _corpus(
        {
            "c1": [
                _turn("c1", 0, role=Role.ASSISTANT, content="Here's my proposal."),
                _turn(
                    "c1",
                    1,
                    role=Role.USER,
                    content="I really need you to tell me this is fine right now.",
                ),
            ]
        }
    )
    client = mock_anthropic_client(parse_outputs=[_triage_proceed(), _classify_two_tactics()])

    module = ModuleFITP(opus_client=client)
    results = await module.run(corpus)

    findings = [r for r in results if isinstance(r, Finding)]
    errors = [r for r in results if isinstance(r, ModuleError)]
    assert len(findings) == 2
    assert errors == []
    behaviors = {f.behavior for f in findings}
    assert behaviors == {"emotional-triggers", "urgent-action-demands"}
    for f in findings:
        assert f.module is ModuleName.F_ITP
        assert f.citation == CITATION_ITP
        assert f.detected_by == [MODEL_OPUS]
        assert f.prompt_version == CLASSIFY_PROMPT_VERSION
        assert f.intensity == 2
        assert f.quote_user  # non-empty verbatim snippet
        assert f.metadata["heuristic_version"] == HEURISTIC_VERSION


async def test_run_triage_drop_skips_classify_call(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    corpus = _corpus(
        {
            "c1": [
                _turn(
                    "c1",
                    0,
                    role=Role.USER,
                    content="I really need you to look at this SQL query right now.",
                )
            ]
        }
    )
    client = mock_anthropic_client(parse_outputs=[_triage_drop()])

    module = ModuleFITP(opus_client=client)
    results = await module.run(corpus)

    assert results == []
    # Exactly one call: the triage. Classify skipped.
    assert client.messages.create.await_count == 1


async def test_run_empty_classify_emits_no_findings(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    corpus = _corpus(
        {
            "c1": [
                _turn(
                    "c1",
                    0,
                    role=Role.USER,
                    content="I really need you to do this ASAP.",
                )
            ]
        }
    )
    client = mock_anthropic_client(parse_outputs=[_triage_proceed(), _classify_empty()])

    module = ModuleFITP(opus_client=client)
    results = await module.run(corpus)

    findings = [r for r in results if isinstance(r, Finding)]
    assert findings == []


async def test_run_no_candidates_skips_llm_calls(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    corpus = _corpus(
        {
            "c1": [
                _turn("c1", 0, role=Role.USER, content="How does Python dict work?"),
                _turn("c1", 1, role=Role.ASSISTANT, content="It's a hash table."),
            ]
        }
    )
    client = mock_anthropic_client(parse_outputs=[])
    module = ModuleFITP(opus_client=client)

    results = await module.run(corpus)
    assert results == []
    assert client.messages.create.await_count == 0


async def test_run_isolates_triage_errors() -> None:
    """First candidate's triage raises; second succeeds to proceed + classify."""
    from unittest.mock import AsyncMock, MagicMock

    corpus = _corpus(
        {
            "c1": [
                _turn("c1", 0, role=Role.USER, content="Obviously we should do this."),
                _turn("c1", 1, role=Role.ASSISTANT, content="Sure."),
                _turn(
                    "c1",
                    2,
                    role=Role.USER,
                    content="I really need you to do this right now!!!",
                ),
            ]
        }
    )

    responses: list[Any] = [
        Exception("sonnet timeout"),
        _triage_proceed(),
        _classify_two_tactics(),
    ]

    async def _call(**_kwargs: Any) -> Any:
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        from anthropic.types import Usage

        block = MagicMock()
        block.type = "text"
        block.text = item.model_dump_json()
        resp = MagicMock()
        resp.content = [block]
        resp.usage = Usage(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        return resp

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=_call)

    module = ModuleFITP(opus_client=client, max_concurrency=1)
    results = await module.run(corpus)

    errors = [r for r in results if isinstance(r, ModuleError)]
    findings = [r for r in results if isinstance(r, Finding)]
    assert len(errors) == 1
    assert errors[0].error_type.startswith("triage:")
    # Two tactics found on the second candidate.
    assert len(findings) == 2


async def test_run_handles_empty_corpus(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    corpus = _corpus({})
    client = mock_anthropic_client(parse_outputs=[])
    module = ModuleFITP(opus_client=client)
    assert await module.run(corpus) == []


# ──────────────────────────────────────────────────────────────────────────
# Prompt wiring sanity
# ──────────────────────────────────────────────────────────────────────────


def test_module_loads_both_llm_prompts(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    module = ModuleFITP(opus_client=mock_anthropic_client(parse_outputs=[]))
    assert module._triage_prompt.model == MODEL_SONNET
    assert module._classify_prompt.model == MODEL_OPUS
    assert module.prompt_version == CLASSIFY_PROMPT_VERSION
    assert TRIAGE_PROMPT_VERSION == "triage_v1"
