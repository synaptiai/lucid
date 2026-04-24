"""Synthesis Managed Agents session driver.

Runs *after* the deterministic scoring phase to produce narrative
report sections. Thin async wrapper around the Managed Agents beta
API (agents.create + environments.create + sessions.create +
sessions.events.stream + sessions.events.send) wired into the
synthesis tool registry from lucid.synthesis.tools.

See docs/plans/2026-04-24-synthesis-agent-refactor.md Task 3.5.

Historical note: the orchestrator session (deleted in Phase 2) lived
at lucid/orchestrator/managed_agent.py. This file is structurally
similar but simpler — no continuation_nudges (rejected by the state
machine), no PROMPT_VERSION import (passed via config), no legacy
event-state handling.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import orjson

from lucid.orchestrator.tools import ToolRegistry
from lucid.synthesis.handler import HeartbeatMonitor, dispatch_tool_call
from lucid.synthesis.lifecycle import (
    get_or_create_synthesis_agent,
    prune_stale_synthesis_agents,
)

if TYPE_CHECKING:  # pragma: no cover
    pass

_LOGGER = logging.getLogger(__name__)


MANAGED_AGENTS_BETA_HEADER = "managed-agents-2026-04-01"


# ──────────────────────────────────────────────────────────────────────────
# Config / outcome shapes
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SynthesisConfig:
    """Per-run parameters for the synthesis session.

    ``prompt_version`` keys the Managed Agents lifecycle — the agent is
    named ``lucid-synthesis-v<prompt_version>`` and reused across runs
    at the same version. Callers pass the literal explicitly (Phase 4.2
    introduces ``SYNTHESIS_PROMPT_VERSION`` at the prompt-loading
    boundary; until then, whoever constructs the config owns the value).

    ``heartbeat_stall_seconds`` is the silence threshold that marks a
    session as stalled; set to ``0`` to disable the watchdog.
    ``heartbeat_check_interval_seconds`` is the watchdog's poll cadence.

    ``keep_ephemeral=True`` skips session + environment teardown for
    debugging; the long-lived agent is never deleted here either way.
    """

    run_id: str
    prompt_version: str
    system_prompt: str
    model: str = "claude-opus-4-7"
    session_timeout_seconds: float = 3600.0
    heartbeat_stall_seconds: float = 60.0
    heartbeat_check_interval_seconds: float = 5.0
    keep_ephemeral: bool = False


@dataclass
class SynthesisHandles:
    agent_id: str
    environment_id: str
    session_id: str


@dataclass
class SynthesisOutcome:
    """What the caller gets back when the event loop returns."""

    handles: SynthesisHandles
    completed: bool
    reason: str
    events_received: int = 0
    tool_calls: int = 0
    sections_written: int = 0
    sections_declined: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    diagnostics: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────
# Client protocol (duck-typed against anthropic.Anthropic)
# ──────────────────────────────────────────────────────────────────────────


class _SupportsManagedAgentsBeta(Protocol):  # pragma: no cover — structural
    @property
    def beta(self) -> Any: ...


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────


class SynthesisSession:
    """One synthesis run against the Managed Agents beta API."""

    def __init__(
        self,
        *,
        client: _SupportsManagedAgentsBeta,
        registry: ToolRegistry,
        config: SynthesisConfig,
    ) -> None:
        self.client = client
        self.registry = registry
        self.config = config
        self._heartbeat = HeartbeatMonitor(stall_seconds=config.heartbeat_stall_seconds)

    # ----- lifecycle -----------------------------------------------

    def get_or_create_agent(self) -> str:
        """Find the synthesis agent for this prompt version or create it."""
        return get_or_create_synthesis_agent(
            self.client,
            model=self.config.model,
            system_prompt=self.config.system_prompt,
            tools=self.registry.as_agent_tools(),
            prompt_version=self.config.prompt_version,
        )

    def create_environment(self) -> str:
        environment = self.client.beta.environments.create(
            name=f"lucid-synth-env-{self.config.run_id[:8]}",
            config={
                "type": "cloud",
                "networking": {"type": "unrestricted"},
                # Deliberately NO mounted_files — corpus access is via
                # read-only custom tools back to this process.
            },
        )
        return str(environment.id)

    def create_session(self, *, agent_id: str, environment_id: str) -> str:
        session = self.client.beta.sessions.create(
            agent=agent_id,
            environment_id=environment_id,
            title=f"Lucid synthesis {self.config.run_id}",
        )
        return str(session.id)

    # ----- event loop ----------------------------------------------

    async def run(self, kickoff_message: str) -> SynthesisOutcome:
        """Spin up (agent, env, session), send kickoff, loop on events."""
        agent_id = self.get_or_create_agent()
        # Archive stale synthesis agents + legacy orchestrator agents on
        # every run so the console stays pruned. Failures are logged and
        # never block the audit.
        try:
            prune_stale_synthesis_agents(
                self.client, current_prompt_version=self.config.prompt_version
            )
        except Exception as err:
            _LOGGER.warning("Stale-agent prune failed: %s", err)

        environment_id = self.create_environment()
        session_id = self.create_session(agent_id=agent_id, environment_id=environment_id)
        handles = SynthesisHandles(
            agent_id=agent_id,
            environment_id=environment_id,
            session_id=session_id,
        )
        outcome = SynthesisOutcome(handles=handles, completed=False, reason="running")

        # OPEN STREAM FIRST, THEN send kickoff — see CLAUDE.md §Managed
        # Agents conventions for the race rationale. The orchestrator
        # session learned this the hard way in Phase 5B testing.
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

        # Reset the heartbeat from kickoff-sent, not from session-object-
        # construction, so setup latency doesn't eat the stall budget.
        self._heartbeat.poke()

        # Watchdog: if the heartbeat stalls, cancel the main task so
        # iteration unwinds and the session is marked partial. A
        # ``CancelledError`` with ``stall_reason`` set was self-inflicted;
        # otherwise it came from an outer task and we propagate.
        stall_reason: list[str] = []
        watchdog = self._start_stall_watchdog(stall_reason)

        try:
            async for event in _iter_stream(stream_ctx):
                self._heartbeat.poke()
                outcome.events_received += 1

                evt_type = getattr(event, "type", None) or (
                    event.get("type") if isinstance(event, dict) else None
                )

                # Diagnostic: log every event type seen at INFO so post-mortems can
                # tell the difference between "session idled because agent emitted
                # text" vs "stream exhausted" vs "session.finished cleanly". Costs
                # one log line per event; outweighed by the debugging signal when
                # a session underdelivers (e.g. 2 tool_calls + 0 sections).
                _LOGGER.info("synthesis event #%d: %s", outcome.events_received, evt_type)

                # When the agent emits text (agent.text / agent.message / similar),
                # capture the first 80 chars as a preview — agent-emitted text
                # between tool calls is the classic "informational session"
                # anti-pattern. Enough to diagnose without dumping corpus content.
                if evt_type and (
                    "text" in str(evt_type).lower() or "message" in str(evt_type).lower()
                ):
                    text_preview = ""
                    # Try common shapes: event.text, event.content[0].text, event.delta.text
                    for attr in ("text", "delta"):
                        val = getattr(event, attr, None)
                        if isinstance(val, str) and val:
                            text_preview = val[:80]
                            break
                        inner_text = getattr(val, "text", None) if val is not None else None
                        if isinstance(inner_text, str) and inner_text:
                            text_preview = inner_text[:80]
                            break
                    # Also try .content as a list of blocks
                    content = getattr(event, "content", None)
                    if not text_preview and isinstance(content, list) and content:
                        first_block = content[0]
                        block_text = getattr(first_block, "text", None)
                        if isinstance(block_text, str) and block_text:
                            text_preview = block_text[:80]
                    if text_preview:
                        _LOGGER.warning(
                            "synthesis agent emitted text (preview, 80 chars): %r",
                            text_preview,
                        )
                    else:
                        _LOGGER.warning(
                            "synthesis agent emitted %s event (no text captured)", evt_type
                        )

                if evt_type == "agent.custom_tool_use":
                    outcome.tool_calls += 1
                    tool_name = getattr(event, "name", None) or (
                        event.get("name") if isinstance(event, dict) else None
                    )
                    tool_args = getattr(event, "input", None) or (
                        event.get("input", {}) if isinstance(event, dict) else {}
                    )
                    custom_tool_use_id = getattr(event, "id", None) or (
                        event.get("id") if isinstance(event, dict) else None
                    )

                    _LOGGER.info(
                        "synthesis tool_call #%d: %s(%s)",
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
                    # Count successful write_report_section outcomes for
                    # downstream telemetry. A dropped parse here must
                    # never abort the run — swallow and continue.
                    if str(tool_name) == "write_report_section":
                        try:
                            result_payload = orjson.loads(result.content)
                            if isinstance(result_payload, dict) and result_payload.get("ok"):
                                if result_payload.get("insufficient_evidence"):
                                    outcome.sections_declined += 1
                                else:
                                    outcome.sections_written += 1
                        except Exception:
                            pass
                    _LOGGER.info(
                        "sending user.custom_tool_result for %s (tool_use_id=%s)",
                        tool_name,
                        result.tool_use_id,
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

                # `session.status_idle` is a TRANSIENT state — it fires between
                # every agent turn (after model_request_end, while the server
                # waits for our user.custom_tool_result or the next user.message).
                # Breaking on it cuts the session after one agent turn. The real
                # terminal event is `session.finished`. Genuine hangs (agent idle
                # with nothing to respond to) are caught by the stall watchdog
                # at heartbeat_stall_seconds (default 60s).
                #
                # See live run run-0cc74fc5cdd2 (2026-04-24) for the diagnostic
                # event trace that identified this.
                if evt_type == "session.finished":
                    outcome.completed = True
                    outcome.reason = str(evt_type)
                    break

                # Accumulate token-usage diagnostics when present.
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
            # Watchdog fired — treat as soft stop, not terminal failure.
            outcome.completed = False
        finally:
            await _shutdown_watchdog(watchdog)
            self._teardown_ephemeral(handles, outcome)

        if stall_reason:
            outcome.reason = stall_reason[0]
            outcome.diagnostics.append(stall_reason[0])
        elif not outcome.completed:
            outcome.reason = "stream_exhausted"
        _LOGGER.info(
            "synthesis outcome: completed=%s reason=%s tool_calls=%d events=%d",
            outcome.completed,
            outcome.reason,
            outcome.tool_calls,
            outcome.events_received,
        )
        return outcome

    def _teardown_ephemeral(self, handles: SynthesisHandles, outcome: SynthesisOutcome) -> None:
        """Delete the per-run session + environment unless keep_ephemeral.

        The agent itself is long-lived and shared across runs of the
        same ``prompt_version``; never deleted here.
        """
        if self.config.keep_ephemeral:
            _LOGGER.info(
                "keep_ephemeral=True; leaving session %s and environment %s in place",
                handles.session_id,
                handles.environment_id,
            )
            return

        cleanups: tuple[tuple[str, Callable[[], Any]], ...] = (
            ("session", lambda: self.client.beta.sessions.delete(handles.session_id)),
            (
                "environment",
                lambda: self.client.beta.environments.delete(handles.environment_id),
            ),
        )
        for label, op in cleanups:
            try:
                op()
            except Exception as err:
                _LOGGER.warning("Failed to delete %s: %s", label, err)
                outcome.diagnostics.append(f"{label}_cleanup_failed: {err}")

    def _start_stall_watchdog(self, stall_reason: list[str]) -> asyncio.Task[None] | None:
        """Spawn the heartbeat watchdog task, or return ``None`` if disabled.

        The watchdog cancels the running ``run()`` task when the
        heartbeat crosses ``stall_seconds``. ``stall_reason`` doubles as
        the signal that the subsequent ``CancelledError`` was
        self-inflicted rather than propagated from an outer task.
        """
        stall_seconds = self.config.heartbeat_stall_seconds
        check_interval = self.config.heartbeat_check_interval_seconds
        if stall_seconds <= 0 or check_interval <= 0:
            return None
        main_task = asyncio.current_task()
        if main_task is None:  # pragma: no cover — run() always inside a task
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

        return asyncio.create_task(_watchdog(), name=f"lucid-synth-watchdog-{self.config.run_id}")


# ──────────────────────────────────────────────────────────────────────────
# Helpers (lifted verbatim from the deleted managed_agent.py)
# ──────────────────────────────────────────────────────────────────────────


def _summarize_args(args: dict[str, Any]) -> str:
    """Render tool-call arguments as a short, log-safe one-liner.

    Only argument *shapes* (keys + small primitives) reach the log.
    Long strings are truncated; lists are summarized by length.
    """
    parts: list[str] = []
    for key, value in args.items():
        if isinstance(value, list):
            parts.append(f"{key}=[{len(value)} items]")
        elif isinstance(value, str) and len(value) > 60:
            parts.append(f"{key}={value[:60]!r}...")
        else:
            parts.append(f"{key}={value!r}")
    return ", ".join(parts)


async def _shutdown_watchdog(watchdog: asyncio.Task[None] | None) -> None:
    """Cancel the watchdog (if any) and swallow its ``CancelledError``."""
    if watchdog is None or watchdog.done():
        return
    watchdog.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await watchdog


def _get_usage_field(usage: Any, field_name: str) -> int | None:
    """Best-effort field access on a usage obj (dict or SDK model)."""
    direct = getattr(usage, field_name, None)
    if direct is not None:
        return int(direct)
    if isinstance(usage, dict):
        v = usage.get(field_name)
        return int(v) if v is not None else None
    return None


async def _iter_stream(stream_ctx: Any) -> AsyncIterator[Any]:
    """Normalize the SDK's stream iterator — sync context mgr or async."""
    context_mgr = stream_ctx
    if hasattr(context_mgr, "__enter__"):
        stream = context_mgr.__enter__()
        try:
            for event in stream:
                # Yield to the event loop so pending tool dispatches can
                # make progress between events.
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
    "SynthesisConfig",
    "SynthesisHandles",
    "SynthesisOutcome",
    "SynthesisSession",
]
