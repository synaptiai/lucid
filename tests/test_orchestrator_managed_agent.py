"""ManagedAgentsSession tests against a fake Anthropic client.

The fake mimics just enough of the SDK surface for the event loop:
`client.beta.agents.create`, `client.beta.environments.create`,
`client.beta.sessions.create`, and `client.beta.sessions.events.stream`.

The stream can be primed with a list of events; the driver consumes them
in order, dispatches custom-tool calls through the registry, and stops
on `session.status_idle`.
"""

from __future__ import annotations

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


# Keep pytest import from ruff-warning
_ = pytest
