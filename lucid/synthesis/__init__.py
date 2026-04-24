"""Synthesis session: Managed Agents narrative writer.

Runs *after* the deterministic scoring phase. Reads the findings table,
spot-reads the corpus via read-only custom tools, writes narrative
sections of the HTML report with inline ``[F:finding_id]`` citation
tokens. A Sonnet 4.6 post-processor structures the prose into
validated blocks; uncited or invalid prose is dropped at render time.

See docs/plans/2026-04-24-synthesis-agent-refactor.md for the design.
"""

from lucid.synthesis.handler import HeartbeatMonitor, dispatch_tool_call

__all__ = ["HeartbeatMonitor", "dispatch_tool_call"]
