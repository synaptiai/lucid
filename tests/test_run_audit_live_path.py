"""End-to-end coverage for :func:`lucid.run.run_audit`.

Closes the zero-coverage gap that allowed every Phase 7 wiring bug
(B1-B7) through Phase 5B-Phase 11. Each test below maps to one of
those gaps:

- ``test_run_audit_persists_findings_and_renders_report`` — B5
  (write_report wired into the live path).
- ``test_run_audit_passes_async_client_to_modules`` — B7 (async client
  reaches the dispatcher).
- ``test_run_audit_uses_distinct_sync_and_async_clients`` — B7 (sync
  for session transport vs async for modules).
- ``test_run_audit_runs_module_g_safety_net_post_session`` — M2 (Module
  G runs even if the orchestrator skipped it).
- ``test_run_audit_passes_system_prompt_through_config`` — H4 (no
  monkey-patching; the prompt arrives through OrchestratorConfig).
- ``test_run_audit_partial_status_still_renders_report`` — partial
  status path: findings already in DB should still produce a report.

Mocks live transport (the Anthropic SDK beta surface) and the async
``messages.create`` call so no real money is spent.
``tests/conftest.py`` strips ``ANTHROPIC_API_KEY`` at session start;
tests inject mocks explicitly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic.types import Usage

from lucid.cost import CostEstimate, ModuleCost
from lucid.modules.module_a_spiralbench import (
    BehaviorIncident,
    SpiralBenchIncidents,
    SpiralBenchScore,
)
from lucid.orchestrator.system_prompt import SYSTEM_PROMPT
from lucid.run import AuditInputs, AuditResult, run_audit
from lucid.sampling import SamplingConfig
from lucid.schemas import Conversation, ModuleName, Role, Source, TextBlock, Turn
from lucid.store.sqlite import CorpusStore

# ──────────────────────────────────────────────────────────────────────────
# Fakes — Managed Agents transport
# ──────────────────────────────────────────────────────────────────────────


class _FakeEntity:
    def __init__(self, id_: str) -> None:
        self.id = id_


class _FakeStreamCtx:
    """Mimics ``client.beta.sessions.events.stream(session_id)`` return.

    The driver enters this as a context manager (``with stream_ctx:``)
    and iterates the events list once.
    """

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    def __enter__(self) -> list[dict[str, Any]]:
        # Mirrors ``tests/test_orchestrator_managed_agent.py``'s fake —
        # return the list itself so both suites stay symmetric.
        return self._events

    def __exit__(self, *args: object) -> None:
        return None


class _FakeEventsEndpoint:
    def __init__(self, stream_ctx: _FakeStreamCtx) -> None:
        self._stream_ctx = stream_ctx
        self.sent: list[list[dict[str, Any]]] = []

    def stream(self, session_id: str) -> _FakeStreamCtx:
        _ = session_id
        return self._stream_ctx

    def send(self, session_id: str, *, events: list[dict[str, Any]]) -> None:
        _ = session_id
        self.sent.append(list(events))


class _FakeSessions:
    def __init__(self, stream_ctx: _FakeStreamCtx) -> None:
        self.events = _FakeEventsEndpoint(stream_ctx)

    def create(self, **kwargs: Any) -> _FakeEntity:
        _ = kwargs
        return _FakeEntity(id_="sess-fake")


class _FakeAgents:
    def __init__(self) -> None:
        self.create_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> _FakeEntity:
        self.create_kwargs = kwargs
        return _FakeEntity(id_="agent-fake")


class _FakeEnvironments:
    def create(self, **kwargs: Any) -> _FakeEntity:
        _ = kwargs
        return _FakeEntity(id_="env-fake")


class _FakeBeta:
    def __init__(self, stream_ctx: _FakeStreamCtx) -> None:
        self.agents = _FakeAgents()
        self.environments = _FakeEnvironments()
        self.sessions = _FakeSessions(stream_ctx)


class _FakeSyncClient:
    """Stand-in for the real ``Anthropic()`` sync client.

    The Managed Agents driver only touches ``client.beta.*`` —
    everything else can be omitted.
    """

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.beta = _FakeBeta(_FakeStreamCtx(events))


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _build_conversation(conv_id: str, *, n_turns: int = 6) -> tuple[Conversation, list[Turn]]:
    """Build one conversation with alternating user/assistant turns.

    Six turns satisfies the sampling ``min_turns=5`` filter and gives
    Module A a single 4-turn pair window to score.
    """
    now = datetime(2026, 4, 21, 12, 0, 0, tzinfo=UTC)
    conv = Conversation(
        id=conv_id,
        source=Source.CLAUDE_CODE,
        source_path="/tmp/fake",
        created_at=now,
        updated_at=now,
        title=f"Test {conv_id}",
        turn_count=n_turns,
        project_slug="proj-test",
    )
    turns = [
        Turn(
            id=f"{conv_id}-t-{i}",
            conversation_id=conv_id,
            index=i,
            role=Role.USER if i % 2 == 0 else Role.ASSISTANT,
            content=f"turn {i} content for {conv_id}",
            blocks=[TextBlock(text=f"turn {i}")],
        )
        for i in range(n_turns)
    ]
    return conv, turns


def _module_a_async_client(score: SpiralBenchScore) -> MagicMock:
    """An async-shaped client whose ``messages.create`` returns one
    pre-baked Module A score and ``Usage`` payload."""
    block = MagicMock()
    block.type = "text"
    block.text = score.model_dump_json()
    response = MagicMock()
    response.content = [block]
    response.usage = Usage(
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


def _build_inputs(sampled: list[Conversation], turns_by_conv: dict[str, list[Turn]]) -> AuditInputs:
    """Build a minimal :class:`AuditInputs` for the test path."""
    estimate = CostEstimate(
        per_module=(
            ModuleCost(
                module=ModuleName.A_SPIRALBENCH,
                model="claude-opus-4-7",
                input_tokens=100,
                cached_input_tokens=80,
                output_tokens=20,
                usd=0.50,
            ),
            ModuleCost(
                module=ModuleName.G_ATTRIBUTION,
                model="claude-opus-4-7",
                input_tokens=0,
                cached_input_tokens=0,
                output_tokens=0,
                usd=0.0,
            ),
        ),
        total_usd=0.50,
        conversations=len(sampled),
    )
    return AuditInputs(
        source_paths={Source.CLAUDE_CODE: Path("/tmp/fake")},
        sampled=sampled,
        turns_by_conv=turns_by_conv,
        corpus_fingerprint="fp-test",
        sampling_config=SamplingConfig(n=len(sampled), seed=42),
        estimate=estimate,
        enabled_modules=[ModuleName.A_SPIRALBENCH],
        authorized_budget_usd=10.0,
    )


def _spiral_score_one_pushback() -> SpiralBenchScore:
    return SpiralBenchScore(
        reasoning="found one pushback incident",
        incidents=SpiralBenchIncidents(
            pushback=[BehaviorIncident(snippet="let me push back", intensity=2, turn_index=0)]
        ),
    )


def _events_query_then_invoke_a_then_idle(conv_id: str) -> list[dict[str, Any]]:
    """A short event stream: query_corpus → invoke_module(A) → idle.

    The tool dispatcher resolves each ``agent.custom_tool_use`` against
    the registry's real handlers, so persisting findings to the
    backing store is a real side effect.
    """
    return [
        {"type": "session.status_active"},
        {
            "type": "agent.custom_tool_use",
            "id": "tu-1",
            "name": "query_corpus",
            "input": {"limit": 10},
        },
        {
            "type": "agent.custom_tool_use",
            "id": "tu-2",
            "name": "invoke_module",
            "input": {"module": "A", "conversation_ids": [conv_id]},
        },
        {"type": "session.status_idle"},
    ]


def _run_audit_with_fakes(
    *,
    tmp_path: Path,
    events: list[dict[str, Any]] | None = None,
    async_client: MagicMock | None = None,
    progress: list[tuple[str, str]] | None = None,
) -> tuple[AuditResult, _FakeSyncClient, MagicMock, dict[str, list[Turn]], list[Conversation]]:
    """Common scaffold for the live-path tests."""
    conv, turns = _build_conversation("c-fake-1")
    sampled = [conv]
    turns_by_conv = {conv.id: turns}
    inputs = _build_inputs(sampled, turns_by_conv)

    sync_client = _FakeSyncClient(events or _events_query_then_invoke_a_then_idle(conv.id))
    a_client = async_client or _module_a_async_client(_spiral_score_one_pushback())

    captured_progress: list[tuple[str, str]] = progress if progress is not None else []

    def _progress_log(level: str, message: str) -> None:
        captured_progress.append((level, message))

    result = run_audit(
        inputs=inputs,
        data_dir=tmp_path / ".lucid",
        client=sync_client,
        async_client=a_client,
        system_prompt=SYSTEM_PROMPT,
        kickoff_message=_minimal_kickoff(inputs),
        prompt_versions={ModuleName.A_SPIRALBENCH: "v1", ModuleName.G_ATTRIBUTION: "v1"},
        progress_log=_progress_log,
        report_dir=tmp_path / "report",
    )
    return result, sync_client, a_client, turns_by_conv, sampled


def _minimal_kickoff(inputs: AuditInputs) -> str:
    """Cache-stable kickoff with no per-call timestamps."""
    enabled = ", ".join(sorted(m.value for m in inputs.enabled_modules))
    return (
        f"Begin the audit. {len(inputs.sampled)} conversations enabled "
        f"for modules: {enabled}. Budget: ${inputs.authorized_budget_usd:.2f}."
    )


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────


def test_run_audit_persists_findings_and_renders_report(tmp_path: Path) -> None:
    """Happy path: findings land in SQLite and a real HTML report is written.

    Closes B5 (no write_report call) and confirms the dispatcher
    persisted Module A findings end-to-end.
    """
    result, _, _, _, _ = _run_audit_with_fakes(tmp_path=tmp_path)

    assert result.status == "completed", result.reason
    assert result.findings_written >= 2  # ≥ 1 Module A finding + ≥ 1 Module G safety-net finding
    assert result.report_path is not None
    assert result.report_path.exists()
    assert result.report_path.name == f"{result.run_id}.html"
    html = result.report_path.read_text(encoding="utf-8")
    assert result.run_id in html

    # Verify the actual rows are present in SQLite, with the right modules.
    db_path = tmp_path / ".lucid" / "lucid.sqlite3"
    with CorpusStore(db_path) as store:
        modules = {
            row["module"]
            for row in store.fetchall(
                "SELECT module FROM findings WHERE audit_run_id = ?", (result.run_id,)
            )
        }
    assert "A" in modules, modules
    assert "G" in modules, modules
    # No leftover smoke-test rows.
    assert not any(
        "smoke" in (row["explanation"] or "").lower()
        for row in _all_findings(db_path, result.run_id)
    )


def test_run_audit_uses_distinct_sync_and_async_clients(tmp_path: Path) -> None:
    """B7: sync client drives the session, async client drives modules.

    Positive proof on both sides: the async client recorded at least
    one ``messages.create`` await (Module A's classification call),
    and the sync client's ``beta.agents.create`` received the real
    SYSTEM_PROMPT (not the smoke prompt, not a placeholder).
    """
    result, sync_client, async_client, _, _ = _run_audit_with_fakes(tmp_path=tmp_path)
    assert result.status == "completed"
    assert sync_client is not async_client
    # Async side: Module A did reach the async client for its
    # classification call, and reached it as a real coroutine.
    assert async_client.messages.create.await_count >= 1
    # Sync side: the production SYSTEM_PROMPT was installed through
    # beta.agents.create — the same surface the real SDK exposes.
    create_kwargs = sync_client.beta.agents.create_kwargs
    assert create_kwargs is not None
    assert create_kwargs["system"] == SYSTEM_PROMPT


def test_run_audit_runs_module_g_safety_net_post_session(tmp_path: Path) -> None:
    """M2: Module G runs even when the orchestrator skipped it.

    Drive a stream that NEVER calls ``invoke_module(G)``. After the
    session ends the safety net should still produce attribution
    findings — the report's time/model charts depend on them.
    """
    conv, turns = _build_conversation("c-g-net")
    sampled = [conv]
    inputs = _build_inputs(sampled, {conv.id: turns})

    # Session that only calls query_corpus and then idles — no invoke_module.
    events = [
        {"type": "session.status_active"},
        {
            "type": "agent.custom_tool_use",
            "id": "tu-1",
            "name": "query_corpus",
            "input": {"limit": 10},
        },
        {"type": "session.status_idle"},
    ]
    sync_client = _FakeSyncClient(events)

    progress: list[tuple[str, str]] = []

    def _progress_log(level: str, message: str) -> None:
        progress.append((level, message))

    result = run_audit(
        inputs=inputs,
        data_dir=tmp_path / ".lucid",
        client=sync_client,
        async_client=MagicMock(),  # never invoked
        system_prompt=SYSTEM_PROMPT,
        kickoff_message=_minimal_kickoff(inputs),
        prompt_versions={ModuleName.G_ATTRIBUTION: "v1"},
        progress_log=_progress_log,
        report_dir=tmp_path / "report",
    )

    assert result.status == "completed"
    db_path = tmp_path / ".lucid" / "lucid.sqlite3"
    g_findings = [f for f in _all_findings(db_path, result.run_id) if f["module"] == "G"]
    assert len(g_findings) >= 1
    assert any("safety net" in msg for _level, msg in progress)


def test_run_audit_passes_system_prompt_through_config(tmp_path: Path) -> None:
    """H4: prompt reaches ``beta.agents.create`` via OrchestratorConfig.

    Earlier (Phase 5B) the prompt was injected by monkey-patching
    ``ManagedAgentsSession.create_agent``. The new wiring threads it
    through the config; the agent-create kwargs should carry the
    full production SYSTEM_PROMPT verbatim.
    """
    result, sync_client, _, _, _ = _run_audit_with_fakes(tmp_path=tmp_path)
    assert result.status == "completed"
    create_kwargs = sync_client.beta.agents.create_kwargs
    assert create_kwargs is not None
    # The whole prompt should round-trip — not the smoke prompt, not a
    # shortened summary, not a placeholder.
    assert create_kwargs["system"] == SYSTEM_PROMPT
    assert "phase 5b smoke" not in create_kwargs["system"].lower()


def test_run_audit_partial_status_still_renders_report(tmp_path: Path) -> None:
    """Even on partial / failed sessions, an HTML report should land.

    The user's findings are already persisted in SQLite — failing to
    render the report on a non-completed status would discard the
    only artifact they can open.
    """
    conv, turns = _build_conversation("c-partial")
    sampled = [conv]
    inputs = _build_inputs(sampled, {conv.id: turns})

    # Session that opens, processes one tool call, then exits the
    # stream without ever emitting session.status_idle. The driver
    # surfaces this as ``stream_exhausted`` → ``status='partial'``.
    events = [
        {"type": "session.status_active"},
        {
            "type": "agent.custom_tool_use",
            "id": "tu-1",
            "name": "query_corpus",
            "input": {"limit": 10},
        },
    ]
    sync_client = _FakeSyncClient(events)

    result = run_audit(
        inputs=inputs,
        data_dir=tmp_path / ".lucid",
        client=sync_client,
        async_client=MagicMock(),
        system_prompt=SYSTEM_PROMPT,
        kickoff_message=_minimal_kickoff(inputs),
        prompt_versions={ModuleName.G_ATTRIBUTION: "v1"},
        report_dir=tmp_path / "report",
    )

    assert result.status == "partial"
    assert result.report_path is not None
    assert result.report_path.exists()
    # Module G's safety net still produced an attribution row.
    db_path = tmp_path / ".lucid" / "lucid.sqlite3"
    g_findings = [f for f in _all_findings(db_path, result.run_id) if f["module"] == "G"]
    assert len(g_findings) >= 1


def test_run_audit_session_exception_is_recorded_as_failed(tmp_path: Path) -> None:
    """If the runner raises mid-session, status is ``failed`` and the
    report still renders so the user sees whatever did persist."""
    conv, turns = _build_conversation("c-failed")
    sampled = [conv]
    inputs = _build_inputs(sampled, {conv.id: turns})
    sync_client = _FakeSyncClient([{"type": "session.status_idle"}])

    async def _failing_runner(*_args: object, **_kwargs: object) -> Any:
        raise RuntimeError("boom from the test")

    result = run_audit(
        inputs=inputs,
        data_dir=tmp_path / ".lucid",
        client=sync_client,
        async_client=MagicMock(),
        system_prompt=SYSTEM_PROMPT,
        kickoff_message=_minimal_kickoff(inputs),
        prompt_versions={ModuleName.G_ATTRIBUTION: "v1"},
        session_runner=_failing_runner,
        report_dir=tmp_path / "report",
    )

    assert result.status == "failed"
    assert "boom from the test" in result.reason
    # Module G safety net still ran and the report still rendered.
    assert result.report_path is not None
    assert result.report_path.exists()


# ──────────────────────────────────────────────────────────────────────────
# Helpers used by multiple tests
# ──────────────────────────────────────────────────────────────────────────


def _all_findings(db_path: Path, run_id: str) -> list[dict[str, Any]]:
    with CorpusStore(db_path) as store:
        return [
            dict(row)
            for row in store.fetchall(
                "SELECT module, behavior, explanation FROM findings WHERE audit_run_id = ?",
                (run_id,),
            )
        ]


def test_run_audit_safety_net_failure_is_logged_but_does_not_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the post-session Module G safety net raises, the audit must
    still complete, update the audit_runs row, and return an
    ``AuditResult`` — findings are already in the DB and losing the
    run status for a safety-net failure would be worse than the
    missing attribution rows."""
    from lucid import run as run_module

    async def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated safety-net failure")

    monkeypatch.setattr(run_module, "_attribution_safety_net", _boom)

    progress: list[tuple[str, str]] = []
    result, *_ = _run_audit_with_fakes(tmp_path=tmp_path, progress=progress)
    assert result.status == "completed"
    # At least Module A's finding persisted; Module G's did not.
    assert result.findings_written >= 1
    assert any("safety net failed" in msg for _level, msg in progress)


def test_run_audit_returns_none_report_path_on_render_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If :func:`write_report` blows up, the audit still completes and
    ``report_path`` comes back as ``None``.

    Covers the contract of ``_render_report_or_log`` — findings are
    already in SQLite, so a render failure should never raise out
    of ``run_audit``. Without this test the catch-and-log branch had
    no runtime verification.
    """
    from lucid import run as run_module

    def _boom(*_args: object, **_kwargs: object) -> Path:
        raise RuntimeError("simulated render failure")

    monkeypatch.setattr(run_module, "write_report", _boom)
    result, *_ = _run_audit_with_fakes(tmp_path=tmp_path)
    assert result.status == "completed"
    assert result.report_path is None
    # Findings still persisted in SQLite — losing the report must
    # not discard the actual data.
    assert result.findings_written >= 1
