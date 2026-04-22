"""ManagedAgentsSession tests against a fake Anthropic client.

The fake mimics just enough of the SDK surface for the event loop:
`client.beta.agents.create`, `client.beta.environments.create`,
`client.beta.sessions.create`, and `client.beta.sessions.events.stream`.

The stream can be primed with a list of events; the driver consumes them
in order, dispatches custom-tool calls through the registry, and stops
on `session.status_idle`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from lucid.orchestrator.managed_agent import (
    MANAGED_AGENTS_BETA_HEADER,
    ManagedAgentsSession,
    OrchestratorConfig,
)
from lucid.orchestrator.tools import build_tool_registry
from lucid.store import initialize_db
from lucid.store.sqlite import CorpusStore

# ----- fakes ----------------------------------------------------------


class _FakeEntity:
    def __init__(self, id_: str) -> None:
        self.id = id_


class _FakeStreamCtx:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events
        self.closed = False

    def __enter__(self) -> list[dict[str, Any]]:
        return self._events

    def __exit__(self, *args: object) -> None:
        self.closed = True


class _FakeEventsEndpoint:
    def __init__(self, stream: _FakeStreamCtx) -> None:
        self._stream = stream
        self.sent_events: list[list[dict[str, Any]]] = []

    def stream(self, session_id: str) -> _FakeStreamCtx:
        _ = session_id
        return self._stream

    def send(self, session_id: str, *, events: list[dict[str, Any]]) -> None:
        _ = session_id
        self.sent_events.append(list(events))


class _FakeSessions:
    def __init__(self, stream: _FakeStreamCtx) -> None:
        self.events = _FakeEventsEndpoint(stream)

    def create(self, **kwargs: Any) -> _FakeEntity:
        _ = kwargs
        return _FakeEntity(id_="sess-fake")


class _FakeAgents:
    def __init__(self) -> None:
        self.last_create_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> _FakeEntity:
        self.last_create_kwargs = kwargs
        return _FakeEntity(id_="agent-fake")


class _FakeEnvironments:
    def create(self, **kwargs: Any) -> _FakeEntity:
        _ = kwargs
        return _FakeEntity(id_="env-fake")


class _FakeBeta:
    def __init__(self, stream: _FakeStreamCtx) -> None:
        self.agents = _FakeAgents()
        self.environments = _FakeEnvironments()
        self.sessions = _FakeSessions(stream)


class _FakeClient:
    def __init__(self, stream: _FakeStreamCtx) -> None:
        self.beta = _FakeBeta(stream)


# ----- helpers --------------------------------------------------------


def _seed_store(tmp_path: Path) -> tuple[CorpusStore, str]:
    from datetime import UTC, datetime

    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    store = CorpusStore(db)
    store.connect()
    store.connect().execute(
        """
        INSERT INTO audit_runs (
            id, sources_json, source_paths_json, started_at, corpus_stats_json,
            token_usage_json, sampling_config_json, status, corpus_fingerprint,
            prompt_versions_json, schema_version, skipped_modules_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "run-42",
            '["claude-code"]',
            '{"claude-code": "/tmp"}',
            datetime(2026, 4, 21, tzinfo=UTC).isoformat(),
            "{}",
            "{}",
            "{}",
            "running",
            "fp",
            "{}",
            1,
            "[]",
        ),
    )
    return store, "run-42"


# ----- tests ----------------------------------------------------------


async def test_session_completes_on_status_idle(tmp_path: Path) -> None:
    events = [
        {"type": "session.status_active"},
        {"type": "session.status_idle"},
    ]
    stream = _FakeStreamCtx(events)
    client = _FakeClient(stream)

    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    session = ManagedAgentsSession(
        client=client, registry=registry, config=OrchestratorConfig(run_id=run_id)
    )

    outcome = await session.run(kickoff_message="Start audit")
    assert outcome.completed
    assert outcome.reason == "session.status_idle"
    assert outcome.handles.agent_id == "agent-fake"
    assert outcome.handles.session_id == "sess-fake"
    # Kickoff message sent immediately after opening the stream.
    assert client.beta.sessions.events.sent_events
    assert client.beta.sessions.events.sent_events[0][0]["type"] == "user.message"
    store.close()


async def test_session_dispatches_custom_tool_call(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = [
        {"type": "session.status_active"},
        {
            "type": "agent.custom_tool_use",
            "name": "query_corpus",
            "input": {},
            "tool_use_id": "tu-xyz",
        },
        {"type": "session.status_idle"},
    ]
    stream = _FakeStreamCtx(events)
    client = _FakeClient(stream)

    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    session = ManagedAgentsSession(
        client=client, registry=registry, config=OrchestratorConfig(run_id=run_id)
    )

    outcome = await session.run(kickoff_message="Begin")
    assert outcome.tool_calls == 1
    # Sent events: kickoff user.message, then user.custom_tool_result for the dispatched call.
    types_sent = [evt["type"] for batch in client.beta.sessions.events.sent_events for evt in batch]
    assert "user.custom_tool_result" in types_sent
    store.close()


async def test_session_create_agent_includes_all_custom_tools(tmp_path: Path) -> None:
    stream = _FakeStreamCtx([{"type": "session.status_idle"}])
    client = _FakeClient(stream)
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    session = ManagedAgentsSession(
        client=client, registry=registry, config=OrchestratorConfig(run_id=run_id)
    )

    await session.run(kickoff_message="ok")
    kwargs = client.beta.agents.last_create_kwargs
    assert kwargs is not None
    tool_names = {t["name"] for t in kwargs["tools"]}
    assert {
        "query_corpus",
        "get_conversation",
        "get_turn_window",
        "invoke_module",
        "store_finding",
        "get_findings",
        "log_progress",
        "estimate_remaining_cost",
    }.issubset(tool_names)
    assert all(t["type"] == "custom" for t in kwargs["tools"])
    # beta.agents.create takes `system` as a plain string (not messages-API blocks).
    assert isinstance(kwargs["system"], str)
    assert "Lucid" in kwargs["system"]
    store.close()


def test_beta_header_constant_matches_methodology() -> None:
    # Locked in methodology.md §1 — change here should be paired with a doc update.
    assert MANAGED_AGENTS_BETA_HEADER == "managed-agents-2026-04-01"


# ──────────────────────────────────────────────────────────────────────────
# Heartbeat watchdog (M4)
# ──────────────────────────────────────────────────────────────────────────


class _SlowAsyncStream:
    """An async iterator that yields one event quickly, then sleeps long.

    The stream has no ``__enter__``, so :func:`_iter_stream` routes it
    through the async-iter branch — which means the long ``await
    asyncio.sleep`` inside ``__anext__`` is a real cancellation point.
    """

    def __init__(self, fast_event: dict[str, Any], slow_seconds: float) -> None:
        self._fast = fast_event
        self._slow = slow_seconds
        self._yielded = False

    def __aiter__(self) -> _SlowAsyncStream:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if not self._yielded:
            self._yielded = True
            return self._fast
        # Block on a real awaitable so the watchdog's cancellation
        # actually fires here. The sleep duration must comfortably
        # exceed the test's expected watchdog latency.
        await asyncio.sleep(self._slow)
        raise StopAsyncIteration


class _SlowEventsEndpoint:
    def __init__(self, stream: _SlowAsyncStream) -> None:
        self._stream = stream
        self.sent: list[list[dict[str, Any]]] = []

    def stream(self, session_id: str) -> _SlowAsyncStream:
        _ = session_id
        return self._stream

    def send(self, session_id: str, *, events: list[dict[str, Any]]) -> None:
        _ = session_id
        self.sent.append(list(events))


class _SlowSessions:
    def __init__(self, stream: _SlowAsyncStream) -> None:
        self.events = _SlowEventsEndpoint(stream)

    def create(self, **kwargs: Any) -> _FakeEntity:
        _ = kwargs
        return _FakeEntity(id_="sess-slow")


class _SlowBeta:
    def __init__(self, stream: _SlowAsyncStream) -> None:
        self.agents = _FakeAgents()
        self.environments = _FakeEnvironments()
        self.sessions = _SlowSessions(stream)


class _SlowClient:
    def __init__(self, stream: _SlowAsyncStream) -> None:
        self.beta = _SlowBeta(stream)


async def test_session_marks_partial_when_heartbeat_stalls(tmp_path: Path) -> None:
    """An async stream that sleeps after one event triggers the
    watchdog. The session must surface as partial with a
    stall-shaped reason and at least one diagnostics entry.

    ``slow_seconds`` is comfortably larger than ``stall_seconds`` so
    the cancellation lands inside the ``await asyncio.sleep`` rather
    than after StopAsyncIteration — even on a slow CI machine where
    scheduling jitter delays the watchdog tick.
    """
    stream = _SlowAsyncStream({"type": "session.status_active"}, slow_seconds=5.0)
    client = _SlowClient(stream)

    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    config = OrchestratorConfig(
        run_id=run_id,
        heartbeat_stall_seconds=0.05,
        heartbeat_check_interval_seconds=0.05,
    )
    session = ManagedAgentsSession(client=client, registry=registry, config=config)

    outcome = await session.run(kickoff_message="kickoff")

    assert outcome.completed is False
    assert outcome.reason.startswith("stalled"), outcome.reason
    assert any("stalled" in d for d in outcome.diagnostics)
    # First event made it through; the second never did because the
    # watchdog interrupted the wait.
    assert outcome.events_received == 1
    store.close()


async def test_session_does_not_stall_when_watchdog_disabled(tmp_path: Path) -> None:
    """``heartbeat_stall_seconds=0`` disables the watchdog entirely.

    The same slow stream that triggers a stall in the test above must
    now be allowed to run to its natural completion (here, the
    ``StopAsyncIteration`` after the slow sleep elapses).
    """
    stream = _SlowAsyncStream({"type": "session.status_active"}, slow_seconds=0.05)
    client = _SlowClient(stream)

    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    config = OrchestratorConfig(
        run_id=run_id,
        heartbeat_stall_seconds=0,  # disabled
        heartbeat_check_interval_seconds=0.01,
    )
    session = ManagedAgentsSession(client=client, registry=registry, config=config)

    outcome = await session.run(kickoff_message="kickoff")

    assert outcome.completed is False  # natural exit (stream_exhausted), not stalled
    assert outcome.reason == "stream_exhausted"
    assert outcome.events_received == 1
    # No stall diagnostics with the watchdog off.
    assert not any("stalled" in d for d in outcome.diagnostics)
    store.close()


async def test_session_external_cancellation_is_not_swallowed(tmp_path: Path) -> None:
    """A cancellation from outside (parent task cancels run()) must
    propagate as ``CancelledError``, not be converted to a partial
    outcome — that would mask the caller's intent."""
    # A stream that blocks indefinitely so the test can cancel mid-run.
    stream = _SlowAsyncStream({"type": "session.status_active"}, slow_seconds=10.0)
    client = _SlowClient(stream)

    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    config = OrchestratorConfig(
        run_id=run_id,
        heartbeat_stall_seconds=0,  # disable watchdog so only EXTERNAL cancel fires
        heartbeat_check_interval_seconds=0,
    )
    session = ManagedAgentsSession(client=client, registry=registry, config=config)

    task = asyncio.create_task(session.run(kickoff_message="kickoff"))
    # Give the task a moment to start the stream and yield once.
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    store.close()


async def test_session_completes_normally_with_watchdog_running(tmp_path: Path) -> None:
    """Watchdog enabled but stream finishes promptly — the watchdog
    must be cancelled cleanly without affecting the outcome.

    A ``check_interval`` smaller than the test runtime guarantees
    the watchdog ticks at least once during the test, exercising the
    "stalled?" branch (which evaluates False) before the finally
    block cancels it.
    """
    events = [
        {"type": "session.status_active"},
        {"type": "session.status_idle"},
    ]
    stream = _FakeStreamCtx(events)
    client = _FakeClient(stream)
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    config = OrchestratorConfig(
        run_id=run_id,
        heartbeat_stall_seconds=10.0,  # enabled but will not fire
        heartbeat_check_interval_seconds=0.05,
    )
    session = ManagedAgentsSession(client=client, registry=registry, config=config)

    outcome = await session.run(kickoff_message="kickoff")

    assert outcome.completed is True
    assert outcome.reason == "session.status_idle"
    assert outcome.diagnostics == []
    store.close()


async def test_session_watchdog_disabled_when_check_interval_zero(tmp_path: Path) -> None:
    """``heartbeat_check_interval_seconds=0`` disables the watchdog
    independently of ``heartbeat_stall_seconds``.

    Without an explicit test for this branch, a future refactor that
    drops the ``check_interval <= 0`` guard would silently make a
    misconfigured (non-zero stall, zero interval) pair behave like a
    disabled watchdog without surfacing the bug.
    """
    stream = _SlowAsyncStream({"type": "session.status_active"}, slow_seconds=0.05)
    client = _SlowClient(stream)
    store, run_id = _seed_store(tmp_path)
    registry = build_tool_registry(store=store, audit_run_id=run_id)
    config = OrchestratorConfig(
        run_id=run_id,
        heartbeat_stall_seconds=10.0,  # would fire if interval were positive
        heartbeat_check_interval_seconds=0,
    )
    session = ManagedAgentsSession(client=client, registry=registry, config=config)

    outcome = await session.run(kickoff_message="kickoff")

    assert outcome.completed is False
    assert outcome.reason == "stream_exhausted"
    assert not any("stalled" in d for d in outcome.diagnostics)
    store.close()


# Keep pytest import from ruff-warning
_ = pytest
