"""Tests for the status-aggregation logic in _execute_scoring.

The audit aggregator must treat documented intentional gates
(``skipped``, ``no_embedding_provider``, ``no_client``) as
successful outcomes — flipping to ``partial`` was user-facing wrong:
an audit without ``VOYAGE_API_KEY`` would exit non-zero even though
every module ran its correct path. See :data:`lucid.run._NON_FAILURE_STATUSES`.
"""

from __future__ import annotations

from typing import Any

import pytest

from lucid.run import _execute_scoring
from lucid.schemas import ModuleName


def _patch_loop_with(monkeypatch, results: list[dict[str, Any]]) -> None:
    """Replace ``_run_scoring_loop`` with a coroutine that returns
    ``results`` verbatim. The aggregation logic under test reads the
    list unchanged, so we only need a fake producer."""

    async def _fake(**_: Any) -> list[dict[str, Any]]:
        return results

    monkeypatch.setattr("lucid.run._run_scoring_loop", _fake)


@pytest.mark.asyncio
async def test_all_completed_produces_completed_status(monkeypatch):
    results = [
        {"module": "A", "status": "completed"},
        {"module": "B", "status": "completed"},
        {"module": "G", "status": "completed"},
    ]
    _patch_loop_with(monkeypatch, results)
    status, reason = await _execute_scoring(
        store=None,
        registry=None,
        audit_run_id="run-1",
        enabled_modules=[
            ModuleName.A_SPIRALBENCH,
            ModuleName.B_SHARMA,
            ModuleName.G_ATTRIBUTION,
        ],
        sampled_ids=["c1"],
        progress_log=lambda *_: None,
        anthropic_client=None,
        embedding_provider=None,
        allow_module_d=False,
        per_module_usd={},
        debited_modules=set(),
        spend_tracker={"accrued_usd": 0.0, "budget_usd": 10.0},
    )
    assert status == "completed"
    assert "all modules" in reason.lower()


@pytest.mark.asyncio
async def test_skipped_module_d_still_counts_as_completed(monkeypatch):
    """Module D opted out (--no-include-module-d) must NOT flip to partial."""
    results = [
        {"module": "A", "status": "completed"},
        {"module": "D", "status": "skipped"},  # D opted out
        {"module": "G", "status": "completed"},
    ]
    _patch_loop_with(monkeypatch, results)
    status, reason = await _execute_scoring(
        store=None,
        registry=None,
        audit_run_id="run-2",
        enabled_modules=[
            ModuleName.A_SPIRALBENCH,
            ModuleName.D_PERSPECTIVE,
            ModuleName.G_ATTRIBUTION,
        ],
        sampled_ids=["c1"],
        progress_log=lambda *_: None,
        anthropic_client=None,
        embedding_provider=None,
        allow_module_d=False,
        per_module_usd={},
        debited_modules=set(),
        spend_tracker={"accrued_usd": 0.0, "budget_usd": 10.0},
    )
    assert status == "completed", f"expected completed, got {status} (reason: {reason})"


@pytest.mark.asyncio
async def test_no_embedding_provider_counts_as_completed(monkeypatch):
    """Module H without VOYAGE_API_KEY must NOT flip the audit to partial.

    This is the common case for users who opt into the core audit
    without running Module H — exit 0 is correct.
    """
    results = [
        {"module": "A", "status": "completed"},
        {"module": "H", "status": "no_embedding_provider"},
        {"module": "G", "status": "completed"},
    ]
    _patch_loop_with(monkeypatch, results)
    status, _reason = await _execute_scoring(
        store=None,
        registry=None,
        audit_run_id="run-3",
        enabled_modules=[
            ModuleName.A_SPIRALBENCH,
            ModuleName.H_MEMORY,
            ModuleName.G_ATTRIBUTION,
        ],
        sampled_ids=["c1"],
        progress_log=lambda *_: None,
        anthropic_client=None,
        embedding_provider=None,
        allow_module_d=False,
        per_module_usd={},
        debited_modules=set(),
        spend_tracker={"accrued_usd": 0.0, "budget_usd": 10.0},
    )
    assert status == "completed"


@pytest.mark.asyncio
async def test_no_client_counts_as_completed(monkeypatch):
    """No ANTHROPIC_API_KEY on an LLM module — stays completed (not partial)."""
    results = [
        {"module": "A", "status": "no_client"},
        {"module": "G", "status": "completed"},
    ]
    _patch_loop_with(monkeypatch, results)
    status, _reason = await _execute_scoring(
        store=None,
        registry=None,
        audit_run_id="run-4",
        enabled_modules=[ModuleName.A_SPIRALBENCH, ModuleName.G_ATTRIBUTION],
        sampled_ids=["c1"],
        progress_log=lambda *_: None,
        anthropic_client=None,
        embedding_provider=None,
        allow_module_d=False,
        per_module_usd={},
        debited_modules=set(),
        spend_tracker={"accrued_usd": 0.0, "budget_usd": 10.0},
    )
    assert status == "completed"


@pytest.mark.asyncio
async def test_real_failure_flips_to_partial(monkeypatch):
    """A genuine 'failed' status (module crashed, not gated) flips to partial."""
    results = [
        {"module": "A", "status": "completed"},
        {"module": "B", "status": "failed"},
        {"module": "G", "status": "completed"},
    ]
    _patch_loop_with(monkeypatch, results)
    status, reason = await _execute_scoring(
        store=None,
        registry=None,
        audit_run_id="run-5",
        enabled_modules=[
            ModuleName.A_SPIRALBENCH,
            ModuleName.B_SHARMA,
            ModuleName.G_ATTRIBUTION,
        ],
        sampled_ids=["c1"],
        progress_log=lambda *_: None,
        anthropic_client=None,
        embedding_provider=None,
        allow_module_d=False,
        per_module_usd={},
        debited_modules=set(),
        spend_tracker={"accrued_usd": 0.0, "budget_usd": 10.0},
    )
    assert status == "partial"
    assert "B=failed" in reason


@pytest.mark.asyncio
async def test_empty_enabled_modules_returns_vacuous_completed(monkeypatch):
    """No enabled modules → empty results → vacuous 'all(...)' is True → completed."""
    _patch_loop_with(monkeypatch, [])
    status, _reason = await _execute_scoring(
        store=None,
        registry=None,
        audit_run_id="run-6",
        enabled_modules=[],
        sampled_ids=[],
        progress_log=lambda *_: None,
        anthropic_client=None,
        embedding_provider=None,
        allow_module_d=False,
        per_module_usd={},
        debited_modules=set(),
        spend_tracker={"accrued_usd": 0.0, "budget_usd": 10.0},
    )
    assert status == "completed"
