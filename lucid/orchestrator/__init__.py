"""Managed Agents orchestrator package.

Public surface:
- `ToolRegistry` + `CustomTool` — declarative tool schemas matching the
  Managed Agents `{"type": "custom", "name": "...", ...}` shape.
- `build_tool_registry(store, cost_estimator, run_id)` — constructs the
  registry with bound handlers for a given audit run.
- `dispatch_tool_call(registry, name, args)` — run a single tool call.
- `SYSTEM_PROMPT` — routing prompt for the Sonnet 4.6 orchestrator
  (padded past the 2048-token Sonnet cache minimum; `cache_control`
  applied by the caller).
"""

from lucid.orchestrator.handler import (
    ToolDispatchError,
    dispatch_tool_call,
)
from lucid.orchestrator.system_prompt import SYSTEM_PROMPT
from lucid.orchestrator.tools import (
    CustomTool,
    ToolHandler,
    ToolRegistry,
    build_tool_registry,
)

__all__ = [
    "SYSTEM_PROMPT",
    "CustomTool",
    "ToolDispatchError",
    "ToolHandler",
    "ToolRegistry",
    "build_tool_registry",
    "dispatch_tool_call",
]
