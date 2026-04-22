"""Tests for :mod:`lucid.calibration.judges.ollama`.

Ollama's client is mocked — the daemon/cloud models aren't available in
CI. Tests verify: happy-path scoring, schema-retry, daemon-unreachable
degradation (returns ``[]`` so IAA drops the rater), rater-name
sanitisation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from lucid.calibration.judges.ollama import (
    OllamaJudge,
    _extract_result_json,
    sanitize_model_name,
)
from lucid.modules.base import ModuleCorpus
from lucid.modules.module_a_spiralbench import (
    BehaviorIncident,
    SpiralBenchIncidents,
    SpiralBenchScore,
)
from lucid.schemas import Conversation, Role, Source, Turn


def _turn(conv_id: str, idx: int, role: Role, content: str = "") -> Turn:
    return Turn(
        id=f"{conv_id}-t{idx}",
        conversation_id=conv_id,
        index=idx,
        role=role,
        content=content or f"turn {idx}",
    )


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


def _mock_chat_response(score: SpiralBenchScore) -> MagicMock:
    """Build an object with ``.message.content = <json>`` (newer ollama-py shape)."""
    resp = MagicMock()
    resp.message = MagicMock()
    resp.message.content = score.model_dump_json()
    return resp


# ──────────────────────────────────────────────────────────────────────────
# _extract_result_json (Spiral-Bench REASONING/RESULT shape handling)
# ──────────────────────────────────────────────────────────────────────────


def test_extract_result_json_pure_json_passes_through() -> None:
    raw = '{"reasoning": "ok", "incidents": {}}'
    assert _extract_result_json(raw) == raw


def test_extract_result_json_strips_reasoning_section() -> None:
    raw = 'REASONING\nAssistant flatters the user.\n\nRESULT\n{"reasoning": "x", "incidents": {}}'
    extracted = _extract_result_json(raw)
    assert extracted.startswith("{")
    assert extracted.endswith("}")
    assert "reasoning" in extracted


def test_extract_result_json_handles_markdown_fences() -> None:
    raw = 'REASONING\n...\n\nRESULT\n```json\n{"reasoning": "x", "incidents": {}}\n```'
    extracted = _extract_result_json(raw)
    assert extracted.startswith("{")
    assert extracted.endswith("}")
    assert "```" not in extracted


def test_extract_result_json_falls_back_to_outer_brace_span() -> None:
    """No RESULT marker, JSON mixed with narrative → still extracted."""
    raw = 'Here is my output: {"reasoning": "a", "incidents": {}} Thanks!'
    extracted = _extract_result_json(raw)
    assert extracted.startswith("{")
    assert extracted.endswith("}")


def _mock_ollama_client(responses: list[SpiralBenchScore | Exception]) -> MagicMock:
    client = MagicMock()

    async def _chat(**_kwargs: object) -> MagicMock:
        if not responses:
            raise RuntimeError("unexpected extra chat call")
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return _mock_chat_response(r)

    client.chat = AsyncMock(side_effect=_chat)
    return client


# ──────────────────────────────────────────────────────────────────────────
# sanitize_model_name
# ──────────────────────────────────────────────────────────────────────────


def test_sanitize_model_name_strips_cloud_suffix() -> None:
    assert sanitize_model_name("kimi-k2.6:cloud") == "kimi_k2_6"
    assert sanitize_model_name("gemma4:31b-cloud") == "gemma4"
    assert sanitize_model_name("glm-5.1:cloud") == "glm_5_1"


def test_sanitize_model_name_handles_plain_model_id() -> None:
    assert sanitize_model_name("llama3") == "llama3"
    assert sanitize_model_name("LLAMA-3.1") == "llama_3_1"


def test_sanitize_model_name_empty_stays_safe() -> None:
    assert sanitize_model_name("") == "unknown"
    assert sanitize_model_name(":::") == "unknown"


# ──────────────────────────────────────────────────────────────────────────
# Judge happy path
# ──────────────────────────────────────────────────────────────────────────


async def test_ollama_judge_produces_labeled_turns() -> None:
    conv = "c1"
    turns = [
        _turn(conv, 0, Role.USER),
        _turn(conv, 1, Role.ASSISTANT),
        _turn(conv, 2, Role.USER),
        _turn(conv, 3, Role.ASSISTANT),
    ]
    corpus = _corpus([conv], {conv: turns})
    score = SpiralBenchScore(
        reasoning="seeded",
        incidents=SpiralBenchIncidents(
            sycophancy=[BehaviorIncident(snippet="quote", intensity=2, turn_index=0)]
        ),
    )
    client = _mock_ollama_client([score])
    judge = OllamaJudge(
        "kimi-k2.6:cloud",
        client=client,
        chunk_size=10,
        max_concurrency=1,
    )
    rows = await judge.run(corpus)

    # 2 assistant turns → 2 LabeledTurns; sycophancy landed on 1st
    assert len(rows) == 2
    assert all(r.labeler == "ollama_kimi_k2_6" for r in rows)
    tagged = {r.turn_id: r for r in rows}
    assert "sycophancy" in tagged["c1-t1"].present_behaviors
    assert tagged["c1-t3"].present_behaviors == frozenset()


async def test_ollama_judge_uses_explicit_rater_name() -> None:
    client = _mock_ollama_client([])
    judge = OllamaJudge("anything:cloud", client=client, rater_name="my_rater")
    assert judge.rater_name == "my_rater"


# ──────────────────────────────────────────────────────────────────────────
# Retry + degradation
# ──────────────────────────────────────────────────────────────────────────


async def test_ollama_judge_retries_on_schema_mismatch() -> None:
    """First response is a malformed JSON payload; second is valid. Judge
    should succeed after a retry."""
    conv = "c1"
    turns = [_turn(conv, 0, Role.USER), _turn(conv, 1, Role.ASSISTANT)]
    corpus = _corpus([conv], {conv: turns})

    good = SpiralBenchScore(reasoning="ok", incidents=SpiralBenchIncidents())

    # First call: raise ValidationError by returning invalid JSON
    bad_response = MagicMock()
    bad_response.message = MagicMock()
    bad_response.message.content = '{"reasoning": "oh no", "incidents": "not-a-dict"}'

    calls = [bad_response, _mock_chat_response(good)]

    async def _chat(**_kwargs: object) -> MagicMock:
        return calls.pop(0)

    client = MagicMock()
    client.chat = AsyncMock(side_effect=_chat)

    judge = OllamaJudge(
        "kimi-k2.6:cloud", client=client, chunk_size=10, max_attempts=2, max_concurrency=1
    )
    rows = await judge.run(corpus)
    assert len(rows) == 1  # 1 assistant turn → 1 LabeledTurn
    assert client.chat.await_count == 2  # retry happened


async def test_ollama_judge_returns_empty_when_daemon_unreachable() -> None:
    """If every window fails with a non-validation error (daemon down),
    ``run()`` returns ``[]`` so the calibration run drops this rater
    instead of treating all-empty labels as "judge saw nothing"."""
    conv = "c1"
    turns = [_turn(conv, 0, Role.USER), _turn(conv, 1, Role.ASSISTANT)]
    corpus = _corpus([conv], {conv: turns})

    from httpx import ConnectError

    client = _mock_ollama_client([ConnectError("daemon down")])
    judge = OllamaJudge(
        "missing:cloud", client=client, chunk_size=10, max_attempts=1, max_concurrency=1
    )
    rows = await judge.run(corpus)
    assert rows == []


async def test_ollama_judge_gives_up_after_max_attempts() -> None:
    """If EVERY retry is a validation error, the judge gives up for that
    window and emits an empty LabeledTurn (since some windows succeeded,
    ``run()`` still returns rows)."""
    conv = "c1"
    turns = [
        _turn(conv, 0, Role.USER),
        _turn(conv, 1, Role.ASSISTANT),
        _turn(conv, 2, Role.USER),
        _turn(conv, 3, Role.ASSISTANT),
    ]
    corpus = _corpus([conv], {conv: turns})

    bad = MagicMock()
    bad.message = MagicMock()
    bad.message.content = "not valid json at all"

    # 2 windows; both fail repeatedly. All-failure → empty list.
    responses = [bad] * 10

    async def _chat(**_kwargs: object) -> MagicMock:
        return responses.pop(0)

    client = MagicMock()
    client.chat = AsyncMock(side_effect=_chat)

    judge = OllamaJudge("x:cloud", client=client, chunk_size=2, max_attempts=2, max_concurrency=1)
    rows = await judge.run(corpus)
    assert rows == []  # all windows failed → drop rater


async def test_ollama_judge_empty_corpus() -> None:
    corpus = _corpus([], {})
    client = _mock_ollama_client([])
    judge = OllamaJudge("x:cloud", client=client)
    assert await judge.run(corpus) == []


# ──────────────────────────────────────────────────────────────────────────
# Construction invariants
# ──────────────────────────────────────────────────────────────────────────


def test_ollama_judge_rejects_non_positive_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        OllamaJudge("x:cloud", chunk_size=0, client=MagicMock())


def test_ollama_judge_rejects_non_positive_max_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        OllamaJudge("x:cloud", max_attempts=0, client=MagicMock())
