"""Tests for the deterministic scoring loop in lucid.run."""
from __future__ import annotations

from typing import Any

import pytest

from lucid.run import _run_scoring_loop
from lucid.schemas import ModuleName


@pytest.mark.asyncio
async def test_run_scoring_loop_invokes_every_enabled_module_in_order(monkeypatch):
    """The loop calls invoke_module_for_run once per enabled module in
    the order supplied, passing the full conversation id list each
    time, and returns the per-module result dicts."""
    call_log: list[tuple[ModuleName, tuple[str, ...]]] = []

    async def _spy(*, module: ModuleName, conversation_ids: list[str], **_) -> dict[str, Any]:
        call_log.append((module, tuple(conversation_ids)))
        return {"module": module.value, "status": "completed", "findings_stored": 0}

    monkeypatch.setattr("lucid.run.invoke_module_for_run", _spy)

    progress: list[tuple[str, str]] = []

    def _log(level: str, message: str) -> None:
        progress.append((level, message))

    results = await _run_scoring_loop(
        store=None,  # the spy ignores it
        audit_run_id="run-scoring-1",
        enabled_modules=[
            ModuleName.A_SPIRALBENCH,
            ModuleName.B_SHARMA,
            ModuleName.G_ATTRIBUTION,
        ],
        sampled_ids=["c1", "c2"],
        progress_log=_log,
        anthropic_client=None,
        embedding_provider=None,
        allow_module_d=False,
        per_module_usd={},
        debited_modules=set(),
        spend_tracker={"accrued_usd": 0.0, "budget_usd": 10.0},
    )

    assert call_log == [
        (ModuleName.A_SPIRALBENCH, ("c1", "c2")),
        (ModuleName.B_SHARMA, ("c1", "c2")),
        (ModuleName.G_ATTRIBUTION, ("c1", "c2")),
    ]
    assert len(results) == 3
    assert [r["module"] for r in results] == ["A", "B", "G"]
    # Each module invocation is preceded by a "Running module X" progress line.
    running_lines = [m for level, m in progress if "Running module" in m]
    assert running_lines == ["Running module A", "Running module B", "Running module G"]


@pytest.mark.asyncio
async def test_run_scoring_loop_empty_enabled_modules_returns_empty(monkeypatch):
    """Empty enabled_modules list = no invocations, empty result."""
    called = False

    async def _spy(**_) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr("lucid.run.invoke_module_for_run", _spy)

    results = await _run_scoring_loop(
        store=None,
        audit_run_id="run-scoring-2",
        enabled_modules=[],
        sampled_ids=["c1"],
        progress_log=lambda *_: None,
        anthropic_client=None,
        embedding_provider=None,
        allow_module_d=False,
        per_module_usd={},
        debited_modules=set(),
        spend_tracker={"accrued_usd": 0.0, "budget_usd": 10.0},
    )

    assert results == []
    assert called is False


@pytest.mark.asyncio
async def test_run_scoring_loop_propagates_exception(monkeypatch):
    """Per-module transport-level exceptions propagate (the loop does not
    swallow them — upstream decides audit-run status)."""
    async def _spy(**_) -> dict[str, Any]:
        raise RuntimeError("anthropic down")

    monkeypatch.setattr("lucid.run.invoke_module_for_run", _spy)

    with pytest.raises(RuntimeError, match="anthropic down"):
        await _run_scoring_loop(
            store=None,
            audit_run_id="run-scoring-3",
            enabled_modules=[ModuleName.A_SPIRALBENCH],
            sampled_ids=["c1"],
            progress_log=lambda *_: None,
            anthropic_client=None,
            embedding_provider=None,
            allow_module_d=False,
            per_module_usd={},
            debited_modules=set(),
            spend_tracker={"accrued_usd": 0.0, "budget_usd": 10.0},
        )
