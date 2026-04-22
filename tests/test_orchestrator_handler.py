"""Handler tests: tool dispatch + heartbeat monitor."""

from __future__ import annotations

import json
from typing import Any

from lucid.orchestrator.handler import HeartbeatMonitor, dispatch_tool_call
from lucid.orchestrator.tools import CustomTool, ToolRegistry


def _registry_with(name: str, handler) -> ToolRegistry:  # type: ignore[no-untyped-def]
    reg = ToolRegistry()
    reg.register(
        CustomTool(
            name=name,
            description="test tool",
            input_schema={"type": "object"},
            handler=handler,
        )
    )
    return reg


# ----- dispatch_tool_call --------------------------------------------


async def test_dispatch_returns_handler_result() -> None:
    async def echo(args: dict[str, Any]) -> dict[str, Any]:
        return {"echoed": args}

    registry = _registry_with("echo", echo)
    result = await dispatch_tool_call(
        registry, name="echo", args={"hello": "world"}, tool_use_id="tu-1"
    )
    assert result.tool_use_id == "tu-1"
    assert not result.is_error
    assert json.loads(result.content) == {"echoed": {"hello": "world"}}


async def test_dispatch_unknown_tool_returns_error_result() -> None:
    registry = ToolRegistry()
    result = await dispatch_tool_call(registry, name="nope", args={}, tool_use_id="tu-2")
    assert result.is_error
    payload = json.loads(result.content)
    assert payload["error"] == "unknown_tool"


async def test_dispatch_handler_exception_surfaced_as_error_result() -> None:
    async def kaboom(args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")

    registry = _registry_with("kaboom", kaboom)
    result = await dispatch_tool_call(registry, name="kaboom", args={}, tool_use_id="tu-3")
    assert result.is_error
    payload = json.loads(result.content)
    assert payload["error"] == "handler_exception"
    assert payload["message"] == "boom"


async def test_dispatch_passes_through_handler_error_payload() -> None:
    async def complaining(args: dict[str, Any]) -> dict[str, Any]:
        return {"error": "not_found", "message": "no such thing"}

    registry = _registry_with("complaining", complaining)
    result = await dispatch_tool_call(registry, name="complaining", args={}, tool_use_id="tu-4")
    assert result.is_error
    payload = json.loads(result.content)
    assert payload["error"] == "not_found"


# ----- HeartbeatMonitor -----------------------------------------------


class _ManualClock:
    """Deterministic clock for heartbeat tests."""

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t


def test_heartbeat_not_stalled_immediately() -> None:
    clock = _ManualClock()
    mon = HeartbeatMonitor(stall_seconds=60.0, clock=clock)
    assert not mon.is_stalled()


def test_heartbeat_stalls_after_timeout() -> None:
    clock = _ManualClock()
    mon = HeartbeatMonitor(stall_seconds=5.0, clock=clock)
    clock.t = 10.0
    assert mon.is_stalled()


def test_heartbeat_poke_resets_age() -> None:
    clock = _ManualClock()
    mon = HeartbeatMonitor(stall_seconds=5.0, clock=clock)
    clock.t = 10.0
    mon.poke()
    assert not mon.is_stalled()


def test_heartbeat_monitor_exposes_stall_seconds_property() -> None:
    """Public read of the stall threshold — used by the watchdog's
    log line in :func:`ManagedAgentsSession._start_stall_watchdog`."""
    mon = HeartbeatMonitor(stall_seconds=42.5)
    assert mon.stall_seconds == 42.5
