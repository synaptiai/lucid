"""Contract tests for :mod:`lucid.modules.base`.

Covers the types every module relies on (``ModuleCorpus``, ``ModuleError``,
``ModuleResult``) and the concurrency helper's semaphore cap + order
preservation + exception propagation semantics.
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime

import pytest

from lucid.modules.base import (
    ModuleCorpus,
    ModuleError,
    ModuleResult,
    run_with_bounded_concurrency,
)
from lucid.schemas import (
    Conversation,
    Finding,
    ModuleName,
    Role,
    Source,
    Turn,
)

# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _make_conversation(conv_id: str = "c1") -> Conversation:
    return Conversation(
        id=conv_id,
        source=Source.CLAUDE_AI,
        source_path="/tmp/export",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 2, 1, tzinfo=UTC),
        turn_count=2,
    )


def _make_turn(conv_id: str, idx: int) -> Turn:
    return Turn(
        id=f"{conv_id}-t{idx}",
        conversation_id=conv_id,
        index=idx,
        role=Role.USER if idx % 2 == 0 else Role.ASSISTANT,
        content=f"turn {idx}",
    )


def _make_finding() -> Finding:
    return Finding(
        id="f1",
        audit_run_id="run1",
        conversation_id="c1",
        turn_ids=["c1-t0"],
        turn_ids_hash="hash",
        module=ModuleName.A_SPIRALBENCH,
        behavior="off-ramp-missed",
        intensity=2,
        confidence=0.8,
        explanation="test",
        citation="test",
        detected_by=["claude-opus-4-7"],
        detected_at=datetime(2026, 4, 22, tzinfo=UTC),
        prompt_version="v1",
        prompt_hash="abc",
    )


# ──────────────────────────────────────────────────────────────────────────
# ModuleCorpus
# ──────────────────────────────────────────────────────────────────────────


def test_module_corpus_is_frozen() -> None:
    corpus = ModuleCorpus(
        conversations={"c1": _make_conversation("c1")},
        turns_by_conversation={"c1": [_make_turn("c1", 0)]},
        audit_run_id="run1",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        corpus.audit_run_id = "mutated"  # type: ignore[misc]


def test_module_corpus_exposes_ordered_turns() -> None:
    turns = [_make_turn("c1", i) for i in range(5)]
    corpus = ModuleCorpus(
        conversations={"c1": _make_conversation("c1")},
        turns_by_conversation={"c1": turns},
        audit_run_id="run1",
    )
    assert [t.index for t in corpus.turns_by_conversation["c1"]] == [0, 1, 2, 3, 4]


# ──────────────────────────────────────────────────────────────────────────
# ModuleError
# ──────────────────────────────────────────────────────────────────────────


def test_module_error_captures_structured_fields() -> None:
    err = ModuleError(
        module=ModuleName.A_SPIRALBENCH,
        conversation_id="c1",
        error_type="parse_error",
        message="model returned invalid JSON",
    )
    assert err.module == ModuleName.A_SPIRALBENCH
    assert err.conversation_id == "c1"
    assert err.error_type == "parse_error"


def test_module_error_allows_conversation_none_for_cross_corpus() -> None:
    err = ModuleError(
        module=ModuleName.H_MEMORY,
        conversation_id=None,
        error_type="embedding_error",
        message="voyage api unreachable",
    )
    assert err.conversation_id is None


def test_module_error_is_frozen() -> None:
    err = ModuleError(
        module=ModuleName.A_SPIRALBENCH,
        conversation_id="c1",
        error_type="x",
        message="y",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        err.message = "mutated"  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────
# ModuleResult union
# ──────────────────────────────────────────────────────────────────────────


def test_module_result_accepts_finding() -> None:
    result: ModuleResult = _make_finding()
    assert isinstance(result, Finding)


def test_module_result_accepts_module_error() -> None:
    err = ModuleError(
        module=ModuleName.A_SPIRALBENCH,
        conversation_id="c1",
        error_type="x",
        message="y",
    )
    result: ModuleResult = err
    assert isinstance(result, ModuleError)


# ──────────────────────────────────────────────────────────────────────────
# run_with_bounded_concurrency
# ──────────────────────────────────────────────────────────────────────────


async def test_concurrency_respects_cap() -> None:
    """Never more than `max_concurrency` coroutines in-flight at once."""
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def task(i: int) -> int:
        nonlocal in_flight, peak
        async with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        async with lock:
            in_flight -= 1
        return i

    results = await run_with_bounded_concurrency(
        [task(i) for i in range(20)],
        max_concurrency=3,
    )
    assert results == list(range(20))
    assert peak <= 3, f"peak concurrency was {peak}, expected ≤ 3"


async def test_concurrency_preserves_submission_order() -> None:
    """Results come back in input order even when tasks finish out-of-order."""

    async def task(i: int) -> int:
        # Later tasks finish first.
        await asyncio.sleep(0.01 * (10 - i))
        return i

    results = await run_with_bounded_concurrency(
        [task(i) for i in range(10)],
        max_concurrency=5,
    )
    assert results == list(range(10))


async def test_concurrency_empty_input() -> None:
    results = await run_with_bounded_concurrency([], max_concurrency=5)
    assert results == []


async def test_concurrency_propagates_exceptions() -> None:
    """One coroutine raising propagates — helper does not swallow."""

    async def good() -> int:
        return 1

    async def bad() -> int:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await run_with_bounded_concurrency([good(), bad(), good()], max_concurrency=2)


async def test_concurrency_rejects_non_positive_cap() -> None:
    async def noop() -> int:
        return 0

    coro = noop()
    try:
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            await run_with_bounded_concurrency([coro], max_concurrency=0)
    finally:
        coro.close()
