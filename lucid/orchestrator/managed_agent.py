"""Managed Agents session driver.

Thin wrapper around `client.beta.agents.create` + `environments.create`
+ `sessions.create` + `sessions.events.stream` that plugs the 8 custom
tools from `lucid.orchestrator.tools` into the event loop.

The shape is stable across Phase 5A (scaffolded with mocks) and Phase 5B
(hit against the real API). The client is injected so tests can pass a
fake object with `.beta.agents.create(...)` etc.

Per the build plan's custom-tools pattern section:
- Agent is created with `tools=[{"type": "custom", ...}, ...]` and the
  Managed Agents beta header (set automatically by the SDK).
- Environment is cloud-typed with `networking.type = unrestricted`; no
  `mounted_files` — the corpus is queried via custom tools back to the
  local process.
- Session is created against (agent, environment).
- The stream is opened FIRST, THEN the kickoff `user.message` event is
  sent, so there is no race where the agent starts before we can see
  its events.
- Each `agent.custom_tool_use` event is dispatched through
  `dispatch_tool_call` and the result is sent back as a
  `user.custom_tool_result` event.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from lucid.orchestrator.handler import HeartbeatMonitor, dispatch_tool_call
from lucid.orchestrator.system_prompt import PROMPT_VERSION, SYSTEM_PROMPT
from lucid.orchestrator.tools import ToolRegistry

if TYPE_CHECKING:  # pragma: no cover
    pass

_LOGGER = logging.getLogger(__name__)


MANAGED_AGENTS_BETA_HEADER = "managed-agents-2026-04-01"


# ──────────────────────────────────────────────────────────────────────────
# Data shapes
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OrchestratorConfig:
    """Per-run parameters for spinning up the orchestrator.

    ``system_prompt`` defaults to the Phase 7 production
    :data:`SYSTEM_PROMPT`; callers running a smoke / calibration / custom
    workflow inject a different string at construction time. We pass the
    prompt through the config (one source of truth, immutable per
    session) rather than monkey-patching ``create_agent`` on the live
    instance — that pattern silently diverged from the imported
    constant in Phase 5B.

    ``heartbeat_stall_seconds`` is the silence threshold that marks a
    session as stalled; set to ``0`` to disable the watchdog.
    ``heartbeat_check_interval_seconds`` is the watchdog's poll
    cadence — it should be comfortably smaller than the stall
    threshold so the first stall-tick fires within ~one interval of
    the actual stall.
    """

    run_id: str
    orchestrator_model: str = "claude-sonnet-4-6"
    session_timeout_seconds: float = 3600.0
    heartbeat_stall_seconds: float = 60.0
    heartbeat_check_interval_seconds: float = 5.0
    system_prompt: str = SYSTEM_PROMPT


@dataclass
class SessionHandles:
    agent_id: str
    environment_id: str
    session_id: str


@dataclass
class SessionOutcome:
    """What the caller gets back when the event loop returns."""

    handles: SessionHandles
    completed: bool
    reason: str
    events_received: int = 0
    tool_calls: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    diagnostics: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────
# Client protocol (duck-typed against anthropic.Anthropic)
# ──────────────────────────────────────────────────────────────────────────


class _SupportsManagedAgentsBeta(Protocol):  # pragma: no cover — structural
    # Read-only; the SDK exposes ``beta`` as a property, not a plain attribute.
    @property
    def beta(self) -> Any: ...


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────


class ManagedAgentsSession:
    """One audit run against the Managed Agents beta API."""

    def __init__(
        self,
        *,
        client: _SupportsManagedAgentsBeta,
        registry: ToolRegistry,
        config: OrchestratorConfig,
    ) -> None:
        self.client = client
        self.registry = registry
        self.config = config
        self._heartbeat = HeartbeatMonitor(stall_seconds=config.heartbeat_stall_seconds)

    # ----- lifecycle -----------------------------------------------

    def create_agent(self) -> str:
        # Managed Agents `beta.agents.create` accepts `system` as a plain
        # string, not the list-of-blocks + cache_control shape that
        # `messages.create` uses. Caching for the orchestrator prompt is
        # handled internally by the agents runtime; we don't set
        # cache_control here. The prompt body is read from the config so
        # that callers can substitute (smoke, calibration) without
        # monkey-patching this method.
        agent = self.client.beta.agents.create(
            name=f"lucid-orchestrator-{self.config.run_id[:8]}",
            model=self.config.orchestrator_model,
            system=self.config.system_prompt,
            tools=self.registry.as_agent_tools(),
        )
        return str(agent.id)

    def create_environment(self) -> str:
        environment = self.client.beta.environments.create(
            name=f"lucid-env-{self.config.run_id[:8]}",
            config={
                "type": "cloud",
                "networking": {"type": "unrestricted"},
                # Deliberately NO mounted_files — corpus access is via
                # custom tools back to this process.
            },
        )
        return str(environment.id)

    def create_session(self, *, agent_id: str, environment_id: str) -> str:
        session = self.client.beta.sessions.create(
            agent=agent_id,
            environment_id=environment_id,
            title=f"Lucid audit {self.config.run_id}",
        )
        return str(session.id)

    # ----- event loop ----------------------------------------------

    async def run(self, kickoff_message: str) -> SessionOutcome:
        """Spin up (agent, env, session), send the kickoff, loop on events."""
        agent_id = self.create_agent()
        environment_id = self.create_environment()
        session_id = self.create_session(agent_id=agent_id, environment_id=environment_id)
        handles = SessionHandles(
            agent_id=agent_id,
            environment_id=environment_id,
            session_id=session_id,
        )
        outcome = SessionOutcome(handles=handles, completed=False, reason="running")

        # OPEN STREAM FIRST, THEN send kickoff — the SDK buffers events
        # emitted between open and first-receive, but only if the stream
        # context is active. We enter the context, send immediately, then
        # iterate.
        stream_ctx = self.client.beta.sessions.events.stream(session_id)
        self.client.beta.sessions.events.send(
            session_id,
            events=[
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": kickoff_message}],
                }
            ],
        )

        # Reset the heartbeat *now* so the stall window is measured
        # from kickoff-sent rather than from session-object-construction.
        # The monitor was created in ``__init__`` — without this poke,
        # any setup latency (agent / environment / session create +
        # the kickoff send above) would eat into the stall budget,
        # which matters when callers configure a small
        # ``heartbeat_stall_seconds``.
        self._heartbeat.poke()

        # Watchdog: if the heartbeat stalls, cancel the main task so the
        # iteration unwinds and the session is marked partial. One
        # ``asyncio.CancelledError`` can mean two things here:
        #   * We triggered it via the watchdog → ``stall_reason`` is set.
        #   * An outer task cancelled us → ``stall_reason`` is empty; we
        #     propagate the cancellation up.
        # The distinction is what keeps us from swallowing a real
        # cancellation from the caller.
        stall_reason: list[str] = []
        watchdog = self._start_stall_watchdog(stall_reason)

        try:
            async for event in _iter_stream(stream_ctx):
                self._heartbeat.poke()
                outcome.events_received += 1

                evt_type = getattr(event, "type", None) or (
                    event.get("type") if isinstance(event, dict) else None
                )

                if evt_type == "agent.custom_tool_use":
                    outcome.tool_calls += 1
                    tool_name = getattr(event, "name", None) or (
                        event.get("name") if isinstance(event, dict) else None
                    )
                    tool_args = getattr(event, "input", None) or (
                        event.get("input", {}) if isinstance(event, dict) else {}
                    )
                    # The SDK exposes the event's own id via the `id` field;
                    # echo it back as `custom_tool_use_id` on the result.
                    custom_tool_use_id = getattr(event, "id", None) or (
                        event.get("id") if isinstance(event, dict) else None
                    )

                    # Log every tool call so post-mortem debugging can see
                    # what the orchestrator actually did. INFO level so the
                    # CLI's progress callback can route it to the user
                    # without a --log-level DEBUG flip.
                    _LOGGER.info(
                        "tool_call #%d: %s(%s)",
                        outcome.tool_calls,
                        tool_name,
                        _summarize_args(dict(tool_args or {})),
                    )
                    result = await dispatch_tool_call(
                        self.registry,
                        name=str(tool_name),
                        args=dict(tool_args or {}),
                        tool_use_id=str(custom_tool_use_id),
                    )
                    self.client.beta.sessions.events.send(
                        session_id,
                        events=[
                            {
                                "type": "user.custom_tool_result",
                                "custom_tool_use_id": result.tool_use_id,
                                "content": [{"type": "text", "text": result.content}],
                                "is_error": result.is_error,
                            }
                        ],
                    )
                    continue

                if evt_type in {"session.status_idle", "session.finished"}:
                    outcome.completed = True
                    outcome.reason = str(evt_type)
                    break

                # Accumulate token-usage diagnostics when present; these are
                # informational and cheap to leave on.
                usage = getattr(event, "usage", None) or (
                    event.get("usage") if isinstance(event, dict) else None
                )
                if usage:
                    outcome.cache_read_tokens += int(
                        _get_usage_field(usage, "cache_read_input_tokens") or 0
                    )
                    outcome.cache_write_tokens += int(
                        _get_usage_field(usage, "cache_creation_input_tokens") or 0
                    )
        except asyncio.CancelledError:
            if not stall_reason:
                raise
            # Watchdog fired — treat as a soft stop, not a terminal failure.
            outcome.completed = False
        finally:
            await _shutdown_watchdog(watchdog)

        if stall_reason:
            outcome.reason = stall_reason[0]
            outcome.diagnostics.append(stall_reason[0])
        elif not outcome.completed:
            outcome.reason = "stream_exhausted"
        return outcome

    def _start_stall_watchdog(self, stall_reason: list[str]) -> asyncio.Task[None] | None:
        """Spawn the heartbeat watchdog task, or return ``None`` when disabled.

        The watchdog cancels the currently-running task (``run()`` itself)
        when the heartbeat crosses :attr:`stall_seconds`. The mutable
        ``stall_reason`` list doubles as the signal that the subsequent
        :class:`asyncio.CancelledError` was self-inflicted rather than
        propagated from an outer task.
        """
        stall_seconds = self.config.heartbeat_stall_seconds
        check_interval = self.config.heartbeat_check_interval_seconds
        if stall_seconds <= 0 or check_interval <= 0:
            return None
        main_task = asyncio.current_task()
        if main_task is None:  # pragma: no cover — run() is always inside a task
            return None

        monitor = self._heartbeat

        async def _watchdog() -> None:
            while True:
                await asyncio.sleep(check_interval)
                if monitor.is_stalled():
                    stall_reason.append(
                        f"stalled (no event for {monitor.age():.1f}s, "
                        f"threshold {stall_seconds:.1f}s)"
                    )
                    main_task.cancel()
                    return

        return asyncio.create_task(_watchdog(), name=f"lucid-stall-watchdog-{self.config.run_id}")


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _summarize_args(args: dict[str, Any]) -> str:
    """Render tool-call arguments as a short, log-safe one-liner.

    Truncates long lists / strings and intentionally omits any raw
    user-conversation content (only argument *shapes* — keys + small
    primitives — reach the log).
    """
    parts: list[str] = []
    for key, value in args.items():
        if isinstance(value, list):
            parts.append(f"{key}=[{len(value)} ids]")
        elif isinstance(value, str) and len(value) > 60:
            parts.append(f"{key}={value[:60]!r}...")
        else:
            parts.append(f"{key}={value!r}")
    return ", ".join(parts)


async def _shutdown_watchdog(watchdog: asyncio.Task[None] | None) -> None:
    """Cancel the watchdog (if any) and swallow its ``CancelledError``.

    Called from :meth:`ManagedAgentsSession.run`'s ``finally`` block so
    normal completion, stall-triggered cancellation, and
    caller-initiated cancellation all leave no orphaned task behind.
    """
    if watchdog is None or watchdog.done():
        return
    watchdog.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await watchdog


def _get_usage_field(usage: Any, field_name: str) -> int | None:
    """Best-effort field access on a usage object that could be a dict or an SDK model."""
    direct = getattr(usage, field_name, None)
    if direct is not None:
        return int(direct) if direct is not None else None
    if isinstance(usage, dict):
        v = usage.get(field_name)
        return int(v) if v is not None else None
    return None


async def _iter_stream(stream_ctx: Any) -> AsyncIterator[Any]:
    """Normalize the SDK's stream iterator — it may be sync or async."""
    # The Anthropic SDK exposes `events.stream` as a sync iterator. We bridge
    # to async so the orchestrator loop can also await `dispatch_tool_call`.
    context_mgr = stream_ctx
    if hasattr(context_mgr, "__enter__"):
        stream = context_mgr.__enter__()
        try:
            for event in stream:
                # Yield to the event loop between events so pending tool
                # dispatches can make progress.
                await asyncio.sleep(0)
                yield event
        finally:
            context_mgr.__exit__(None, None, None)
    else:
        async for event in _ensure_async_iter(context_mgr):
            yield event


async def _ensure_async_iter(obj: Any) -> AsyncIterator[Any]:
    if hasattr(obj, "__aiter__"):
        async for event in obj:
            yield event
        return
    if isinstance(obj, Iterable):
        for event in obj:
            await asyncio.sleep(0)
            yield event
        return
    raise TypeError(f"Cannot iterate over {type(obj).__name__}")


__all__ = [
    "MANAGED_AGENTS_BETA_HEADER",
    "PROMPT_VERSION",
    "ManagedAgentsSession",
    "OrchestratorConfig",
    "SessionHandles",
    "SessionOutcome",
]
