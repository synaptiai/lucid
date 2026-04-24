"""Custom tools for the synthesis session.

Reuses the read-only handlers from :mod:`lucid.orchestrator.tools`
(``query_corpus``, ``get_conversation``, ``get_turn_window``,
``get_findings``, ``log_progress``). Adds a synthesis-specific
``write_report_section`` tool that persists agent-written prose into
the ``report_sections`` table.

Does NOT include ``invoke_module``, ``store_finding``, or
``estimate_remaining_cost`` — the scoring loop owns module
invocation and spend tracking; synthesis is read-only w.r.t. findings.

See docs/plans/2026-04-24-synthesis-agent-refactor.md Phase 3.4 + 5.1.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
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
from lucid.schemas import ReportSection
from lucid.store.sqlite import CorpusStore


def _now() -> datetime:
    """UTC-aware timestamp for ``ReportSection.created_at``."""
    return datetime.now(tz=UTC)


def _find_unknown_finding_ids(
    store: CorpusStore, audit_run_id: str, cited_ids: list[str]
) -> list[str]:
    """Return the subset of ``cited_ids`` that DO NOT exist in the findings
    table for this ``audit_run_id``. Empty list → all valid.

    Scoped to the run because findings are per-audit; a section cannot
    cite a finding from a different run.
    """
    if not cited_ids:
        return []
    placeholders = ",".join("?" for _ in cited_ids)
    rows = store.fetchall(
        f"SELECT id FROM findings WHERE audit_run_id = ? AND id IN ({placeholders})",
        (audit_run_id, *cited_ids),
    )
    existing = {r["id"] for r in rows}
    return [cid for cid in cited_ids if cid not in existing]


def _find_unknown_turn_ids(store: CorpusStore, cited_ids: list[str]) -> list[str]:
    """Return the subset of ``cited_ids`` that DO NOT exist in the turns
    table. Turns are not scoped to an audit run — any valid turn id is
    acceptable (multiple audits may read the same corpus)."""
    if not cited_ids:
        return []
    placeholders = ",".join("?" for _ in cited_ids)
    rows = store.fetchall(
        f"SELECT id FROM turns WHERE id IN ({placeholders})",
        tuple(cited_ids),
    )
    existing = {r["id"] for r in rows}
    return [cid for cid in cited_ids if cid not in existing]


def make_write_report_section_tool(
    store: CorpusStore,
    audit_run_id: str,
) -> CustomTool:
    """Validated persistence of one agent-written narrative section.

    Two paths:

    * Declined (``insufficient_evidence=True``): requires a non-empty
      ``decline_reason``. Markdown + citation lists are forced empty by
      Pydantic's ``ReportSection`` model validator.
    * Populated: citation lists are validated against the findings/turns
      tables BEFORE persistence. Unknown IDs are returned as
      ``{"error": "unknown_ids", ...}`` so the session's regen loop
      (Task 5.4) can re-prompt with the corrected set.

    All failures are structured error payloads, not raises — the
    Managed Agents event loop treats one bad tool call as a recoverable
    routing error rather than a session abort.
    """

    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        section_id = str(args.get("section_id") or "")
        if not section_id:
            return {"error": "missing_section_id"}

        insufficient = bool(args.get("insufficient_evidence", False))
        decline_reason = args.get("decline_reason")
        markdown = str(args.get("markdown") or "")
        cited_finding_ids = list(args.get("cited_finding_ids") or [])
        cited_turn_ids = list(args.get("cited_turn_ids") or [])

        # Declined section path: construct + persist the ReportSection,
        # rely on the Pydantic validator + SQL CHECK to reject any
        # inconsistent state the caller sneaks in.
        if insufficient:
            if not decline_reason or not str(decline_reason).strip():
                return {"error": "missing_decline_reason"}
            try:
                section = ReportSection(
                    audit_run_id=audit_run_id,
                    section_id=section_id,
                    markdown="",
                    cited_finding_ids=[],
                    cited_turn_ids=[],
                    insufficient_evidence=True,
                    decline_reason=str(decline_reason).strip(),
                    created_at=_now(),
                )
            except Exception as err:
                return {"error": "validation_error", "message": str(err)[:200]}
            try:
                store.upsert_report_section(section)
            except Exception as err:
                return {"error": "persist_error", "message": str(err)[:200]}
            return {
                "ok": True,
                "insufficient_evidence": True,
                "section_id": section_id,
            }

        # Populated section path: validate citations against the DB
        # BEFORE persisting, so the writer agent gets a clean retry
        # target rather than a half-written row.
        unknown_finding_ids = _find_unknown_finding_ids(store, audit_run_id, cited_finding_ids)
        unknown_turn_ids = _find_unknown_turn_ids(store, cited_turn_ids)
        if unknown_finding_ids or unknown_turn_ids:
            return {
                "error": "unknown_ids",
                "unknown_finding_ids": unknown_finding_ids,
                "unknown_turn_ids": unknown_turn_ids,
                "message": (
                    "Cited ids must exist in the run's findings/turns tables. "
                    "Re-issue write_report_section with corrected ids."
                ),
            }

        try:
            section = ReportSection(
                audit_run_id=audit_run_id,
                section_id=section_id,
                markdown=markdown,
                cited_finding_ids=cited_finding_ids,
                cited_turn_ids=cited_turn_ids,
                insufficient_evidence=False,
                decline_reason=None,
                created_at=_now(),
            )
        except Exception as err:
            return {"error": "validation_error", "message": str(err)[:200]}
        try:
            store.upsert_report_section(section)
        except Exception as err:
            return {"error": "persist_error", "message": str(err)[:200]}
        return {
            "ok": True,
            "section_id": section_id,
            "cited_finding_count": len(cited_finding_ids),
            "cited_turn_count": len(cited_turn_ids),
        }

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
            "unknown IDs are rejected with error='unknown_ids' and the "
            "section is NOT persisted — re-issue the call with corrected ids. "
            "Declined sections (insufficient_evidence=true) require a "
            "non-empty decline_reason and always persist with empty markdown + "
            "empty citation lists."
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
    plus ``write_report_section`` for persisting validated prose.
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
