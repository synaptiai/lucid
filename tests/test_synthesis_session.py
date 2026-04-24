"""SynthesisSession tests against a fake Anthropic client.

Mirrors the pattern from the deleted ``tests/test_orchestrator_managed_agent.py``.
The fake mimics just enough of the SDK surface for the event loop:
``client.beta.agents.{create,list,archive}``, ``client.beta.environments.create``,
``client.beta.sessions.{create,delete}``, and ``client.beta.sessions.events.{stream,send}``.

The stream can be primed with a list of events; the driver consumes them
in order, dispatches custom-tool calls through the registry, and stops
on ``session.finished``.

Continuation-nudge tests intentionally absent — the feature was dropped
from ``SynthesisSession`` (see module docstring).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from lucid.store import initialize_db
from lucid.store.sqlite import CorpusStore
from lucid.synthesis.session import (
    SynthesisConfig,
    SynthesisSession,
)
from lucid.synthesis.tools import build_synthesis_registry

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
        self.deleted_ids: list[str] = []

    def create(self, **kwargs: Any) -> _FakeEntity:
        _ = kwargs
        return _FakeEntity(id_="sess-fake")

    def delete(self, session_id: str, **_kwargs: Any) -> Any:
        self.deleted_ids.append(session_id)
        return MagicMock(id=session_id, deleted=True)


class _FakeNamedAgent:
    """Agent entity with both id and name, for list() responses."""

    def __init__(self, id_: str, name: str) -> None:
        self.id = id_
        self.name = name


class _FakeAgents:
    """Fake of ``client.beta.agents`` covering create / list / archive."""

    def __init__(self, prepopulated: list[_FakeNamedAgent] | None = None) -> None:
        self.last_create_kwargs: dict[str, Any] | None = None
        self._agents: list[_FakeNamedAgent] = list(prepopulated or [])
        self.archived_ids: list[str] = []

    def create(self, **kwargs: Any) -> _FakeNamedAgent:
        self.last_create_kwargs = kwargs
        created = _FakeNamedAgent(id_="agent-fake", name=str(kwargs.get("name", "agent-fake")))
        self._agents.append(created)
        return created

    def list(self, *, include_archived: bool = False, **_kwargs: Any) -> list[_FakeNamedAgent]:
        _ = include_archived
        return list(self._agents)

    def archive(self, agent_id: str, **_kwargs: Any) -> Any:
        self.archived_ids.append(agent_id)
        self._agents = [a for a in self._agents if a.id != agent_id]
        return MagicMock(id=agent_id, archived=True)


class _FakeEnvironments:
    def __init__(self) -> None:
        self.deleted_ids: list[str] = []

    def create(self, **kwargs: Any) -> _FakeEntity:
        _ = kwargs
        return _FakeEntity(id_="env-fake")

    def delete(self, environment_id: str, **_kwargs: Any) -> Any:
        self.deleted_ids.append(environment_id)
        return MagicMock(id=environment_id, deleted=True)


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
            "run-synth",
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
    return store, "run-synth"


def _make_session(client: _FakeClient, store: CorpusStore, run_id: str) -> SynthesisSession:
    registry = build_synthesis_registry(
        store=store,
        audit_run_id=run_id,
        progress_log=lambda *_: None,
    )
    config = SynthesisConfig(
        run_id=run_id,
        prompt_version="v1",
        system_prompt="You are the Lucid synthesis agent.",
    )
    return SynthesisSession(client=client, registry=registry, config=config)


# ----- tests ----------------------------------------------------------


async def test_synthesis_session_runs_to_completion_with_fake_stream(tmp_path: Path) -> None:
    """``session.finished`` terminates the event loop cleanly."""
    events: list[dict[str, Any]] = [
        {"type": "session.status_active"},
        {"type": "session.finished"},
    ]
    stream = _FakeStreamCtx(events)
    client = _FakeClient(stream)

    store, run_id = _seed_store(tmp_path)
    try:
        session = _make_session(client, store, run_id)
        outcome = await session.run(kickoff_message="Start synthesis")
        assert outcome.completed
        assert outcome.reason == "session.finished"
        assert outcome.handles.agent_id == "agent-fake"
        assert outcome.handles.session_id == "sess-fake"
        # Kickoff sent immediately after opening the stream.
        assert client.beta.sessions.events.sent_events
        assert client.beta.sessions.events.sent_events[0][0]["type"] == "user.message"
    finally:
        store.close()


async def test_synthesis_session_dispatches_custom_tool_calls(tmp_path: Path) -> None:
    """An ``agent.custom_tool_use`` event triggers dispatch + tool_result reply."""
    events: list[dict[str, Any]] = [
        {"type": "session.status_active"},
        {
            "type": "agent.custom_tool_use",
            "name": "log_progress",
            "input": {"stage": "planning", "message": "hi"},
            "id": "tu-1",
        },
        {"type": "session.finished"},
    ]
    stream = _FakeStreamCtx(events)
    client = _FakeClient(stream)

    store, run_id = _seed_store(tmp_path)
    try:
        session = _make_session(client, store, run_id)
        outcome = await session.run(kickoff_message="Begin")
        assert outcome.tool_calls == 1
        # Sent events: kickoff user.message + user.custom_tool_result reply.
        types_sent = [
            evt["type"] for batch in client.beta.sessions.events.sent_events for evt in batch
        ]
        assert "user.custom_tool_result" in types_sent
        # The tool_use correlation id is echoed back as custom_tool_use_id.
        result_batch = [
            batch
            for batch in client.beta.sessions.events.sent_events
            if batch[0]["type"] == "user.custom_tool_result"
        ]
        assert result_batch
        assert result_batch[0][0]["custom_tool_use_id"] == "tu-1"
    finally:
        store.close()


async def test_synthesis_session_counts_write_report_section(tmp_path: Path) -> None:
    """A successful ``write_report_section`` increments ``sections_written``.

    The real handler (Task 5.1) validates cited_finding_ids against the
    findings table, so we send a declined section — it persists without
    requiring a real finding row.
    """
    events: list[dict[str, Any]] = [
        {"type": "session.status_active"},
        {
            "type": "agent.custom_tool_use",
            "name": "write_report_section",
            "input": {
                "section_id": "exec_summary",
                "insufficient_evidence": True,
                "decline_reason": "No qualifying findings in the sample.",
            },
            "id": "tu-write",
        },
        {"type": "session.finished"},
    ]
    stream = _FakeStreamCtx(events)
    client = _FakeClient(stream)

    store, run_id = _seed_store(tmp_path)
    try:
        session = _make_session(client, store, run_id)
        outcome = await session.run(kickoff_message="Write the report")
        assert outcome.tool_calls == 1
        # insufficient_evidence=True counts as declined, not written.
        assert outcome.sections_written == 0
        assert outcome.sections_declined == 1
    finally:
        store.close()


async def test_synthesis_session_teardown_deletes_env_and_session(tmp_path: Path) -> None:
    """The per-run environment and session are deleted in the finally block."""
    stream = _FakeStreamCtx([{"type": "session.finished"}])
    client = _FakeClient(stream)

    store, run_id = _seed_store(tmp_path)
    try:
        session = _make_session(client, store, run_id)
        outcome = await session.run(kickoff_message="kickoff")
        assert outcome.handles.environment_id in client.beta.environments.deleted_ids
        assert outcome.handles.session_id in client.beta.sessions.deleted_ids
    finally:
        store.close()


async def test_synthesis_session_continues_past_status_idle(tmp_path: Path) -> None:
    """``session.status_idle`` is transient — the loop must NOT break on it.

    Live run run-0cc74fc5cdd2 (2026-04-24) showed that breaking on
    ``session.status_idle`` after the first agent turn's
    ``span.model_request_end`` cuts the session short, producing 0 tool
    calls beyond the first. This test guards against the regression by
    simulating an event stream that interleaves two tool_uses around an
    idle transition: a buggy loop breaks at the first idle and sees only
    1 tool call; the fix sees 2.
    """
    events: list[dict[str, Any]] = [
        {"type": "session.status_running"},
        {
            "type": "agent.custom_tool_use",
            "name": "log_progress",
            "input": {"stage": "planning", "message": "turn one"},
            "id": "tu-1",
        },
        {"type": "span.model_request_end"},
        # Transient idle BETWEEN agent turns — pre-fix code broke here.
        {"type": "session.status_idle"},
        {"type": "session.status_running"},
        {
            "type": "agent.custom_tool_use",
            "name": "log_progress",
            "input": {"stage": "writing", "message": "turn two"},
            "id": "tu-2",
        },
        {"type": "span.model_request_end"},
        {"type": "session.finished"},
    ]
    stream = _FakeStreamCtx(events)
    client = _FakeClient(stream)

    store, run_id = _seed_store(tmp_path)
    try:
        session = _make_session(client, store, run_id)
        outcome = await session.run(kickoff_message="multi-turn synthesis")
        # Fix: BOTH tool calls dispatched, terminates on session.finished.
        # Buggy: only 1 tool call dispatched, terminates on session.status_idle.
        assert outcome.tool_calls == 2
        assert outcome.completed
        assert outcome.reason == "session.finished"
        # Both tool_use ids were echoed back in user.custom_tool_result replies.
        result_batches = [
            batch
            for batch in client.beta.sessions.events.sent_events
            if batch[0]["type"] == "user.custom_tool_result"
        ]
        echoed_ids = [batch[0]["custom_tool_use_id"] for batch in result_batches]
        assert echoed_ids == ["tu-1", "tu-2"]
    finally:
        store.close()


class _BlockingStreamCtx:
    """Sync stream that yields one event then blocks forever in ``next()``.

    Simulates the live SDK behaviour observed in run-d049b5ac04df: after
    the agent emits its final text message the HTTP stream goes silent,
    and the C-level ``next()`` call blocks waiting for bytes that never
    arrive. The pre-fix ``_iter_stream`` would pin the event loop here
    and the stall watchdog never got scheduled.
    """

    def __init__(self, first_event: dict[str, Any], block_seconds: float = 30.0) -> None:
        self._first_event = first_event
        self._block_seconds = block_seconds
        self._emitted = False
        # ``_release`` is set in ``__exit__`` so the blocked thread can
        # unwind promptly when the session tears down — otherwise
        # pytest waits for the full ``block_seconds`` at teardown.
        self._release = threading.Event()
        self.closed = False

    def __enter__(self) -> _BlockingStreamCtx:
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True
        self._release.set()

    def __iter__(self) -> _BlockingStreamCtx:
        return self

    def __next__(self) -> dict[str, Any]:
        if not self._emitted:
            self._emitted = True
            return self._first_event
        # Real blocking wait — this is the whole point of the test.
        # Uses ``threading.Event.wait`` (which blocks the worker thread
        # without yielding to the event loop, mimicking C-land HTTP
        # reads) rather than ``time.sleep`` so teardown can release the
        # thread instead of making pytest wait the full window.
        # If the fix regresses, this will also pin the event loop and
        # the watchdog won't fire.
        self._release.wait(self._block_seconds)
        raise StopIteration


async def test_synthesis_session_stall_watchdog_fires_under_sync_iterator(
    tmp_path: Path,
) -> None:
    """The stall watchdog must fire even when the SDK returns a sync
    iterator that blocks between events.

    Regression guard for live run run-d049b5ac04df (2026-04-24) where
    the session hung 3+ minutes after Opus emitted a final text message
    and went silent. Root cause: ``_iter_stream`` used a sync
    ``for event in stream`` loop whose ``next()`` call blocked in
    C-land; ``await asyncio.sleep(0)`` only yields BETWEEN events so
    the watchdog task was never scheduled.

    Simulates: sync stream emits one event, then blocks forever.
    Expected: stall watchdog fires within ~stall_seconds and run()
    returns with outcome.completed=False + reason containing
    ``stalled``. Bounded wall-clock verifies the fix (the pre-fix path
    would never return within the assertion window).
    """
    stream = _BlockingStreamCtx(first_event={"type": "session.status_running"})
    client = _FakeClient(stream)  # type: ignore[arg-type]

    store, run_id = _seed_store(tmp_path)
    try:
        registry = build_synthesis_registry(
            store=store,
            audit_run_id=run_id,
            progress_log=lambda *_: None,
        )
        # Tight heartbeat so the test completes in ~1-2s.
        config = SynthesisConfig(
            run_id=run_id,
            prompt_version="v1",
            system_prompt="You are the Lucid synthesis agent.",
            heartbeat_stall_seconds=1.0,
            heartbeat_check_interval_seconds=0.2,
        )
        session = SynthesisSession(client=client, registry=registry, config=config)

        start = time.monotonic()
        outcome = await session.run(kickoff_message="begin")
        elapsed = time.monotonic() - start

        # The watchdog must have fired; the session is partial, not complete.
        assert outcome.completed is False
        assert "stalled" in outcome.reason
        # Bounded wall time: heartbeat 1s + check interval 0.2s + slack.
        # If the fix regresses and next() pins the loop, we'd sit here
        # for block_seconds (30s) before anything could unwind.
        assert elapsed < 5.0, f"run() took {elapsed:.2f}s — watchdog likely did not fire"
    finally:
        store.close()
