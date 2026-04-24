"""Legacy home for tool schemas (``tools.py``).

The orchestrator session itself is gone — the deterministic scoring
loop in :mod:`lucid.run` invokes modules directly, and the synthesis
lifecycle + event handler + (soon) read-only tools live under
:mod:`lucid.synthesis`. Phase 3.4 of the synthesis-agent refactor will
move ``tools.py`` into ``lucid/synthesis/`` too, at which point this
package disappears.

Public surface:
- :class:`ToolRegistry` + :class:`CustomTool` — declarative tool schemas
  matching the Managed Agents ``{"type": "custom", "name": "...", ...}``
  shape.
- :func:`build_tool_registry` — constructs the registry with bound
  handlers for a given audit run.

The previous re-exports of :class:`HeartbeatMonitor`,
:class:`ToolDispatchError`, and :func:`dispatch_tool_call` from
:mod:`lucid.synthesis.handler` were dropped — they were never used via
this package root (verified by grep) and caused a circular import
between ``lucid.synthesis.__init__`` and ``lucid.orchestrator.tools``.
Import from :mod:`lucid.synthesis.handler` directly instead.
"""

from lucid.orchestrator.tools import (
    CustomTool,
    ToolHandler,
    ToolRegistry,
    build_tool_registry,
)

__all__ = [
    "CustomTool",
    "ToolHandler",
    "ToolRegistry",
    "build_tool_registry",
]
