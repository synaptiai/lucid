"""Managed Agents orchestrator package.

Legacy home for tool schemas, the custom-tool dispatcher, and the
agent-lifecycle helpers. The orchestrator session itself is gone — the
deterministic scoring loop in :mod:`lucid.run` invokes modules directly.
Phase 3 of the synthesis-agent refactor will move ``handler``,
``lifecycle``, and the read-only tool handlers into ``lucid/synthesis/``.

Public surface:
- :class:`ToolRegistry` + :class:`CustomTool` — declarative tool schemas
  matching the Managed Agents ``{"type": "custom", "name": "...", ...}``
  shape.
- :func:`build_tool_registry` — constructs the registry with bound
  handlers for a given audit run.
- :func:`dispatch_tool_call` — run a single tool call.
"""

from lucid.orchestrator.tools import (
    CustomTool,
    ToolHandler,
    ToolRegistry,
    build_tool_registry,
)

# TODO(phase-3.6): drop these re-exports once tools.py moves to
# lucid/synthesis/tools.py and all call sites import directly from
# lucid.synthesis.handler. Kept during the Phase 3 transition so
# external importers of `lucid.orchestrator` don't break mid-refactor.
from lucid.synthesis.handler import (
    ToolDispatchError,
    dispatch_tool_call,
)

__all__ = [
    "CustomTool",
    "ToolDispatchError",
    "ToolHandler",
    "ToolRegistry",
    "build_tool_registry",
    "dispatch_tool_call",
]
