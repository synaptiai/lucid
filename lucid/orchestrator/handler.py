"""Event dispatcher for Managed Agents session events.

The dispatcher is deliberately sync-over-async: `dispatch_tool_call` is
an `async def` that the event loop awaits for each `agent.custom_tool_use`
event. A heartbeat watchdog is exposed separately so callers can drop
into checkpoint mode without blocking on `next(stream)`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from lucid.orchestrator.tools import ToolRegistry, to_tool_result_content

_LOGGER = logging.getLogger(__name__)


class ToolDispatchError(RuntimeError):
    """Raised when the orchestrator emitted a tool call the dispatcher can't serve."""


@dataclass
class ToolResult:
    tool_use_id: str
    content: str  # JSON-serialized result payload
    is_error: bool = False


async def dispatch_tool_call(
    registry: ToolRegistry,
    *,
    name: str,
    args: dict[str, Any],
    tool_use_id: str,
) -> ToolResult:
    """Run the handler bound to `name` and return a ToolResult event payload.

    Surfaces handler-level errors as `{"error": "...", "message": "..."}`
    results rather than raising, so a single bad arg from the orchestrator
    doesn't abort the entire session.
    """
    tool = registry.get(name)
    if tool is None:
        return ToolResult(
            tool_use_id=tool_use_id,
            content=to_tool_result_content(
                {"error": "unknown_tool", "message": f"No tool named {name!r}"}
            ),
            is_error=True,
        )

    try:
        payload = await tool.handler(args)
    except asyncio.CancelledError:  # pragma: no cover — propagated from outer cancel
        raise
    except Exception as err:
        _LOGGER.exception("tool %r raised during dispatch", name)
        return ToolResult(
            tool_use_id=tool_use_id,
            content=to_tool_result_content({"error": "handler_exception", "message": str(err)}),
            is_error=True,
        )

    is_error = isinstance(payload, dict) and "error" in payload
    return ToolResult(
        tool_use_id=tool_use_id,
        content=to_tool_result_content(payload),
        is_error=is_error,
    )


class HeartbeatMonitor:
    """Tracks time-since-last-event; callers poll `is_stalled()`.

    Separate object (not a closure) so tests can assert behaviour by
    manipulating the internal clock and so the CLI can read current age
    for progress rendering.
    """

    def __init__(self, *, stall_seconds: float = 60.0, clock: object | None = None) -> None:
        self._stall_seconds = stall_seconds
        self._clock = clock or time
        self._last_seen: float = self._clock.monotonic()  # type: ignore[attr-defined]

    def poke(self) -> None:
        """Mark that an event was just received."""
        self._last_seen = self._clock.monotonic()  # type: ignore[attr-defined]

    def age(self) -> float:
        """Seconds since the last `poke()`."""
        return float(self._clock.monotonic()) - self._last_seen  # type: ignore[attr-defined]

    def is_stalled(self) -> bool:
        return self.age() > self._stall_seconds


__all__ = [
    "HeartbeatMonitor",
    "ToolDispatchError",
    "ToolResult",
    "dispatch_tool_call",
]
