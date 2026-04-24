"""Custom tools for the synthesis session.

Reuses the read-only handlers from :mod:`lucid.orchestrator.tools`
(``query_corpus``, ``get_conversation``, ``get_turn_window``,
``get_findings``, ``log_progress``). Adds a synthesis-specific
``write_report_section`` tool that persists agent-written prose into
the ``report_sections`` table (Task 5.1 will implement the real
handler; this task ships a STUB that returns ``{"ok": True}``
without DB side effects).

Does NOT include ``invoke_module``, ``store_finding``, or
``estimate_remaining_cost`` — the scoring loop owns module
invocation and spend tracking; synthesis is read-only w.r.t. findings.

See docs/plans/2026-04-24-synthesis-agent-refactor.md Phase 3.4.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lucid.orchestrator.tools import (
    CustomTool,
    ToolRegistry,
    make_get_conversation_tool,
    make_get_findings_tool,
    make_get_turn_window_tool,
    make_log_progress_tool,
    make_query_corpus_tool,
)
from lucid.store.sqlite import CorpusStore


def make_write_report_section_tool(
    store: CorpusStore,
    audit_run_id: str,
) -> CustomTool:
    """Stub tool: real implementation lands in Task 5.1.

    Accepts arbitrary args, returns ``{"ok": True, "stub": True}``
    without persisting. Keeps the synthesis session's tool loop alive
    end-to-end while the full validator (citation existence checks,
    aggregate-support validation) is built in Phase 5.
    """
    # Deliberately ignore store + audit_run_id for now — the stub has
    # no side effects. Task 5.1 wires them into real persistence.
    _ = store
    _ = audit_run_id

    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        # Deliberately a no-op for now. Task 5.1 replaces this with
        # validated persistence to report_sections.
        return {"ok": True, "stub": True, "args_received": list(args.keys())}

    return CustomTool(
        name="write_report_section",
        description=(
            "Persist one agent-written narrative section to the report. "
            "Accepts section_id (str), markdown (str) with inline "
            "[F:finding_id] / [T:turn_id] citation tokens, "
            "cited_finding_ids (list[str]), cited_turn_ids (list[str]), "
            "insufficient_evidence (bool, default false), and "
            "decline_reason (str, required when insufficient_evidence=true). "
            "Citations are validated against the findings/turns tables; "
            "unknown IDs are rejected. STUB in Phase 3.4 — replaced in "
            "Task 5.1 with validated persistence."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "section_id": {"type": "string", "minLength": 1, "maxLength": 64},
                "markdown": {"type": "string"},
                "cited_finding_ids": {"type": "array", "items": {"type": "string"}},
                "cited_turn_ids": {"type": "array", "items": {"type": "string"}},
                "insufficient_evidence": {"type": "boolean", "default": False},
                "decline_reason": {"type": ["string", "null"]},
            },
            "required": ["section_id"],
        },
        handler=_handler,
    )


def build_synthesis_registry(
    *,
    store: CorpusStore,
    audit_run_id: str,
    progress_log: Callable[[str, str], None],
) -> ToolRegistry:
    """Wire up the synthesis session's tool registry.

    5 read-only tools shared with the orchestrator/scoring registry
    (via module-level factories in :mod:`lucid.orchestrator.tools`)
    plus the new ``write_report_section`` (stub today; Task 5.1
    completes it).
    """
    registry = ToolRegistry()
    registry.register(make_query_corpus_tool(store))
    registry.register(make_get_conversation_tool(store))
    registry.register(make_get_turn_window_tool(store))
    registry.register(make_get_findings_tool(store, audit_run_id))
    registry.register(make_log_progress_tool(progress_log))
    registry.register(make_write_report_section_tool(store, audit_run_id))
    return registry


__all__ = [
    "build_synthesis_registry",
    "make_write_report_section_tool",
]
