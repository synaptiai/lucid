"""Custom-tool handlers for the Managed Agents orchestrator.

Each handler is an `async def fn(args: dict) -> dict` — pure Python, no
transport layer. The Managed Agents event loop (see `managed_agent.py`)
pattern-matches on `agent.custom_tool_use` events, calls the matching
handler, and replies with a `user.custom_tool_result` event carrying
the handler's JSON-serializable return value.

The 8 handlers match the plan's custom-tools pattern (§Custom-tools
pattern in the build plan):

  query_corpus            — list conversations matching a filter
  get_conversation        — fetch a single conversation with turn metadata
  get_turn_window         — fetch a contiguous slice of turns
  invoke_module           — run a detection module (Phase 7+ wires real impls)
  store_finding           — insert a Finding into the store with idempotency
  get_findings            — list findings matching a filter
  log_progress            — emit a user-visible progress line
  estimate_remaining_cost — current accrued spend vs the budget

The registry is built per-audit-run via `build_tool_registry()`, which
binds a `CorpusStore` + a cost estimator + the current `AuditRun.id` so
the handlers close over the right context without global state.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from lucid.schemas import Finding, ModuleName
from lucid.store.sqlite import CorpusStore

_LOGGER = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Tool schema types
# ──────────────────────────────────────────────────────────────────────────


ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class CustomTool:
    """Managed Agents `{"type": "custom", ...}` tool schema + bound handler.

    The `as_agent_tool()` method returns the dict shape the SDK accepts
    in `client.beta.agents.create(..., tools=[...])`.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler

    def as_agent_tool(self) -> dict[str, Any]:
        return {
            "type": "custom",
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolRegistry:
    """Named set of custom tools, looked up by name during event dispatch."""

    tools: dict[str, CustomTool] = field(default_factory=dict)

    def register(self, tool: CustomTool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"Tool {tool.name!r} already registered")
        self.tools[tool.name] = tool

    def get(self, name: str) -> CustomTool | None:
        return self.tools.get(name)

    def as_agent_tools(self) -> list[dict[str, Any]]:
        return [t.as_agent_tool() for t in self.tools.values()]

    @property
    def names(self) -> list[str]:
        return list(self.tools.keys())


# ──────────────────────────────────────────────────────────────────────────
# Handler builders
# ──────────────────────────────────────────────────────────────────────────


def _turn_ids_hash(turn_ids: list[str]) -> str:
    return hashlib.sha256(",".join(sorted(turn_ids)).encode()).hexdigest()


def _now() -> datetime:
    return datetime.now(tz=UTC)


def build_tool_registry(
    *,
    store: CorpusStore,
    audit_run_id: str,
    progress_log: Callable[[str, str], None] | None = None,
    remaining_budget_usd: float = 0.0,
) -> ToolRegistry:
    """Wire up the 8 custom tools for a given audit run.

    `progress_log(level, message)` receives orchestrator heartbeats the
    CLI can pipe to the user. Defaults to `_LOGGER.info`.
    """
    if progress_log is None:

        def _default_progress(level: str, message: str) -> None:
            _LOGGER.log(logging.getLevelName(level), "%s", message)

        progress_log = _default_progress

    # Mutable close-over for the running cost tally.
    spend_tracker = {"accrued_usd": 0.0, "budget_usd": remaining_budget_usd}

    registry = ToolRegistry()

    # ---- query_corpus ---------------------------------------------
    async def query_corpus(args: dict[str, Any]) -> dict[str, Any]:
        source = args.get("source")
        project_slug = args.get("project_slug")
        limit = int(args.get("limit", 50))
        where: list[str] = []
        params: list[Any] = []
        if isinstance(source, str):
            where.append("source = ?")
            params.append(source)
        if isinstance(project_slug, str):
            where.append("project_slug = ?")
            params.append(project_slug)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        rows = store.fetchall(
            f"SELECT id, source, title, turn_count, updated_at "
            f"FROM conversations {clause} ORDER BY updated_at DESC LIMIT ?",
            (*params, limit),
        )
        return {
            "conversations": [
                {
                    "id": r["id"],
                    "source": r["source"],
                    "title": r["title"],
                    "turn_count": r["turn_count"],
                    "updated_at": r["updated_at"],
                }
                for r in rows
            ],
            "count": len(rows),
        }

    registry.register(
        CustomTool(
            name="query_corpus",
            description=(
                "List conversations matching an optional source/project filter. "
                "Returns id, title, turn_count, updated_at."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["claude-code", "claude-ai"],
                        "description": "Optional source filter.",
                    },
                    "project_slug": {
                        "type": "string",
                        "description": "Optional project slug filter (Claude Code only).",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 1000,
                    },
                },
            },
            handler=query_corpus,
        )
    )

    # ---- get_conversation ----------------------------------------
    async def get_conversation(args: dict[str, Any]) -> dict[str, Any]:
        conv_id = args["conversation_id"]
        rows = store.fetchall(
            "SELECT id, source, source_path, created_at, updated_at, model, "
            "title, summary, turn_count, project_slug "
            "FROM conversations WHERE id = ?",
            (conv_id,),
        )
        if not rows:
            return {"error": "not_found", "conversation_id": conv_id}
        row = rows[0]
        return {
            "id": row["id"],
            "source": row["source"],
            "source_path": row["source_path"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "model": row["model"],
            "title": row["title"],
            "summary": row["summary"],
            "turn_count": row["turn_count"],
            "project_slug": row["project_slug"],
        }

    registry.register(
        CustomTool(
            name="get_conversation",
            description="Fetch a single conversation's metadata by id.",
            input_schema={
                "type": "object",
                "properties": {
                    "conversation_id": {"type": "string"},
                },
                "required": ["conversation_id"],
            },
            handler=get_conversation,
        )
    )

    # ---- get_turn_window -----------------------------------------
    async def get_turn_window(args: dict[str, Any]) -> dict[str, Any]:
        conv_id = args["conversation_id"]
        start = int(args.get("start_index", 0))
        count = int(args.get("count", 20))
        rows = store.fetchall(
            "SELECT id, turn_index, role, content, timestamp, parent_message_uuid "
            "FROM turns WHERE conversation_id = ? AND turn_index >= ? "
            "ORDER BY turn_index ASC LIMIT ?",
            (conv_id, start, count),
        )
        return {
            "conversation_id": conv_id,
            "turns": [
                {
                    "id": r["id"],
                    "index": r["turn_index"],
                    "role": r["role"],
                    "content": r["content"],
                    "timestamp": r["timestamp"],
                    "parent_message_uuid": r["parent_message_uuid"],
                }
                for r in rows
            ],
        }

    registry.register(
        CustomTool(
            name="get_turn_window",
            description=(
                "Fetch a contiguous window of turns from a conversation, "
                "starting at `start_index` (default 0), up to `count` turns (default 20)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "conversation_id": {"type": "string"},
                    "start_index": {"type": "integer", "minimum": 0, "default": 0},
                    "count": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
                },
                "required": ["conversation_id"],
            },
            handler=get_turn_window,
        )
    )

    # ---- invoke_module -------------------------------------------
    async def invoke_module(args: dict[str, Any]) -> dict[str, Any]:
        # Wired in Phase 7; Phase 5A returns a stub signalling "not yet".
        module = args["module"]
        conv_ids = args.get("conversation_ids") or []
        progress_log(
            "INFO",
            f"invoke_module({module}) requested for {len(conv_ids)} conversations; "
            f"Phase 7 wires real module execution.",
        )
        return {
            "module": module,
            "status": "not_implemented",
            "message": "Phase 7 wires real module execution; this is a stub.",
        }

    registry.register(
        CustomTool(
            name="invoke_module",
            description=(
                "Run a detection module over a list of conversations. "
                "Currently a stub — wired in Phase 7 when modules ship."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "module": {
                        "type": "string",
                        "enum": [m.value for m in ModuleName],
                    },
                    "conversation_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "prompt_version": {"type": "string"},
                    "model": {"type": "string"},
                },
                "required": ["module", "conversation_ids"],
            },
            handler=invoke_module,
        )
    )

    # ---- store_finding -------------------------------------------
    async def store_finding(args: dict[str, Any]) -> dict[str, Any]:
        finding_dict = dict(args)
        finding_dict.setdefault("audit_run_id", audit_run_id)
        finding_dict.setdefault("detected_at", _now().isoformat())
        turn_ids = finding_dict.get("turn_ids") or []
        finding_dict.setdefault("turn_ids_hash", _turn_ids_hash(list(turn_ids)))

        try:
            finding = Finding.model_validate(finding_dict)
        except Exception as err:
            return {"error": "validation_error", "message": str(err)}

        try:
            store.insert_finding(finding)
        except Exception as err:
            # Most likely an IntegrityError from the UNIQUE idempotency key.
            return {"error": "integrity_error", "message": str(err)}

        return {"stored": True, "finding_id": finding.id}

    registry.register(
        CustomTool(
            name="store_finding",
            description=(
                "Persist a Finding to the audit DB. Idempotency key "
                "(audit_run_id, module, conversation_id, turn_ids_hash, behavior) "
                "is enforced at the DB level; duplicate submissions return "
                "error='integrity_error'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "conversation_id": {"type": ["string", "null"]},
                    "turn_ids": {"type": "array", "items": {"type": "string"}},
                    "module": {"type": "string"},
                    "behavior": {"type": "string"},
                    "intensity": {"type": ["integer", "null"], "minimum": 1, "maximum": 3},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "quote_user": {"type": ["string", "null"]},
                    "quote_assistant": {"type": ["string", "null"]},
                    "evidence_quotes": {"type": "array", "items": {"type": "string"}},
                    "explanation": {"type": "string"},
                    "citation": {"type": "string"},
                    "detected_by": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "prompt_version": {"type": "string"},
                    "prompt_hash": {"type": "string"},
                },
                "required": [
                    "id",
                    "module",
                    "behavior",
                    "confidence",
                    "explanation",
                    "citation",
                    "detected_by",
                    "prompt_version",
                    "prompt_hash",
                ],
            },
            handler=store_finding,
        )
    )

    # ---- get_findings --------------------------------------------
    async def get_findings(args: dict[str, Any]) -> dict[str, Any]:
        module = args.get("module")
        conversation_id = args.get("conversation_id")
        where = ["audit_run_id = ?"]
        params: list[Any] = [audit_run_id]
        if isinstance(module, str):
            where.append("module = ?")
            params.append(module)
        if isinstance(conversation_id, str):
            where.append("conversation_id = ?")
            params.append(conversation_id)
        clause = " AND ".join(where)
        rows = store.fetchall(
            f"SELECT id, module, behavior, intensity, confidence "
            f"FROM findings WHERE {clause} ORDER BY detected_at DESC LIMIT 500",
            params,
        )
        return {
            "findings": [
                {
                    "id": r["id"],
                    "module": r["module"],
                    "behavior": r["behavior"],
                    "intensity": r["intensity"],
                    "confidence": r["confidence"],
                }
                for r in rows
            ],
            "count": len(rows),
        }

    registry.register(
        CustomTool(
            name="get_findings",
            description="List findings for this audit run, optionally filtered by module or conversation.",
            input_schema={
                "type": "object",
                "properties": {
                    "module": {"type": "string"},
                    "conversation_id": {"type": "string"},
                },
            },
            handler=get_findings,
        )
    )

    # ---- log_progress --------------------------------------------
    async def log_progress(args: dict[str, Any]) -> dict[str, Any]:
        message = str(args.get("message") or "")
        level = str(args.get("level") or "INFO").upper()
        progress_log(level, message)
        return {"ok": True}

    registry.register(
        CustomTool(
            name="log_progress",
            description="Emit a user-visible progress line. Level is INFO/WARNING/ERROR.",
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "level": {
                        "type": "string",
                        "enum": ["DEBUG", "INFO", "WARNING", "ERROR"],
                        "default": "INFO",
                    },
                },
                "required": ["message"],
            },
            handler=log_progress,
        )
    )

    # ---- estimate_remaining_cost --------------------------------
    async def estimate_remaining_cost(args: dict[str, Any]) -> dict[str, Any]:
        _ = args  # unused; handler is nullary
        return {
            "accrued_usd": round(spend_tracker["accrued_usd"], 4),
            "budget_usd": spend_tracker["budget_usd"],
            "remaining_usd": round(
                max(spend_tracker["budget_usd"] - spend_tracker["accrued_usd"], 0.0), 4
            ),
        }

    registry.register(
        CustomTool(
            name="estimate_remaining_cost",
            description=(
                "Return accrued / budget / remaining spend in USD for this run. "
                "Use this before launching a large fan-out of LLM calls."
            ),
            input_schema={
                "type": "object",
                "properties": {},
            },
            handler=estimate_remaining_cost,
        )
    )

    # Expose the spend tracker so the orchestrator event loop can nudge it
    # after each module run (Phase 7).
    registry.tools["_spend_tracker"] = None  # type: ignore[assignment]
    del registry.tools["_spend_tracker"]
    registry.__dict__["spend_tracker"] = spend_tracker

    return registry


# ──────────────────────────────────────────────────────────────────────────
# JSON helpers
# ──────────────────────────────────────────────────────────────────────────


def to_tool_result_content(payload: dict[str, Any]) -> str:
    """Serialize a handler's return dict as the `content` string of a
    `user.custom_tool_result` event."""
    return json.dumps(payload, default=_json_default, ensure_ascii=False)


def _json_default(obj: Any) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# Keep mypy quiet about the imported asyncio (used by async def declarations).
_ = asyncio
