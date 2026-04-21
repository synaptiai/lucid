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
    """Per-run parameters for spinning up the orchestrator."""

    run_id: str
    orchestrator_model: str = "claude-sonnet-4-6"
    session_timeout_seconds: float = 3600.0
    heartbeat_stall_seconds: float = 60.0


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
    beta: Any


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
        self._heartbeat = HeartbeatMonitor(
            stall_seconds=config.heartbeat_stall_seconds
        )

    # ----- lifecycle -----------------------------------------------

    def create_agent(self) -> str:
        agent = self.client.beta.agents.create(
            name=f"lucid-orchestrator-{self.config.run_id[:8]}",
            model=self.config.orchestrator_model,
            system=_system_prompt_with_cache_control(),
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
        session_id = self.create_session(
            agent_id=agent_id, environment_id=environment_id
        )
        handles = SessionHandles(
            agent_id=agent_id,
            environment_id=environment_id,
            session_id=session_id,
        )
        outcome = SessionOutcome(handles=handles, completed=False, reason="running")

        # OPEN STREAM FIRST, then send the kickoff to avoid the race the
        # docs warn about.
        stream_ctx = self.client.beta.sessions.events.stream(session_id)

        async for event in _iter_stream(stream_ctx):
            self._heartbeat.poke()
            outcome.events_received += 1

            if outcome.events_received == 1:
                # First event tells us the stream is open; send the kickoff.
                self.client.beta.sessions.events.send(
                    session_id,
                    events=[
                        {
                            "type": "user.message",
                            "content": [
                                {"type": "text", "text": kickoff_message}
                            ],
                        }
                    ],
                )

            evt_type = getattr(event, "type", None) or (
                event.get("type") if isinstance(event, dict) else None
            )

            if evt_type == "agent.custom_tool_use":
                outcome.tool_calls += 1
                tool_name = (
                    getattr(event, "name", None)
                    or (event.get("name") if isinstance(event, dict) else None)
                )
                tool_args = (
                    getattr(event, "input", None)
                    or (event.get("input", {}) if isinstance(event, dict) else {})
                )
                tool_use_id = (
                    getattr(event, "tool_use_id", None)
                    or (event.get("tool_use_id") if isinstance(event, dict) else None)
                )

                result = await dispatch_tool_call(
                    self.registry,
                    name=str(tool_name),
                    args=dict(tool_args or {}),
                    tool_use_id=str(tool_use_id),
                )
                self.client.beta.sessions.events.send(
                    session_id,
                    events=[
                        {
                            "type": "user.custom_tool_result",
                            "tool_use_id": result.tool_use_id,
                            "content": result.content,
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

        if not outcome.completed:
            outcome.reason = "stream_exhausted"
        return outcome


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _get_usage_field(usage: Any, field_name: str) -> int | None:
    """Best-effort field access on a usage object that could be a dict or an SDK model."""
    direct = getattr(usage, field_name, None)
    if direct is not None:
        return int(direct) if direct is not None else None
    if isinstance(usage, dict):
        v = usage.get(field_name)
        return int(v) if v is not None else None
    return None


def _system_prompt_with_cache_control() -> list[dict[str, Any]]:
    """Return the system prompt as a single content block with cache_control.

    The 1-hour TTL matches the plan: the orchestrator prompt is stable
    across every session in a multi-module audit.
    """
    return [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]


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
