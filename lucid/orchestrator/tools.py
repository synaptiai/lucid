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
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import orjson

from lucid.modules.base import ModuleCorpus, ModuleError, ModuleResult
from lucid.schemas import (
    ContentBlock,
    Conversation,
    Finding,
    ModuleName,
    Role,
    Source,
    Turn,
)
from lucid.store.sqlite import CorpusStore

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

    from lucid.cost import CostEstimate
    from lucid.modules.embeddings import EmbeddingProvider

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


# ──────────────────────────────────────────────────────────────────────────
# Module-run helpers (used by ``invoke_module``)
# ──────────────────────────────────────────────────────────────────────────


def _row_to_conversation(row: Any) -> Conversation:
    """Rehydrate a Conversation from a sqlite3.Row."""
    return Conversation(
        id=row["id"],
        source=Source(row["source"]),
        source_path=row["source_path"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        model=row["model"],
        title=row["title"],
        summary=row["summary"],
        turn_count=row["turn_count"],
        project_slug=row["project_slug"],
        metadata=orjson.loads(row["metadata_json"]) if row["metadata_json"] else {},
    )


def _row_to_turn(row: Any) -> Turn:
    """Rehydrate a Turn from a sqlite3.Row.

    ``blocks_json`` is stored as a JSON-serialized list of content blocks;
    Pydantic's discriminator handles tagged-union reconstruction via
    :class:`lucid.schemas.ContentBlock`.
    """
    from pydantic import TypeAdapter

    blocks_adapter: TypeAdapter[list[ContentBlock]] = TypeAdapter(list[ContentBlock])
    blocks_raw = orjson.loads(row["blocks_json"]) if row["blocks_json"] else []
    blocks = blocks_adapter.validate_python(blocks_raw)
    return Turn(
        id=row["id"],
        conversation_id=row["conversation_id"],
        index=row["turn_index"],
        role=Role(row["role"]),
        content=row["content"],
        blocks=blocks,
        timestamp=(datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else None),
        parent_message_uuid=row["parent_message_uuid"],
        token_count=row["token_count"],
    )


def _load_module_corpus(
    store: CorpusStore,
    audit_run_id: str,
    conversation_ids: Sequence[str],
) -> ModuleCorpus:
    """Hydrate a ``ModuleCorpus`` for the requested conversation_ids.

    Raises if any id in ``conversation_ids`` is not present in the
    store. Empty list produces an empty corpus — modules handle that
    gracefully.
    """
    if not conversation_ids:
        return ModuleCorpus(
            conversations={},
            turns_by_conversation={},
            audit_run_id=audit_run_id,
        )

    placeholders = ",".join("?" for _ in conversation_ids)
    conv_rows = store.fetchall(
        f"SELECT id, source, source_path, created_at, updated_at, model, "
        f"title, summary, turn_count, project_slug, metadata_json "
        f"FROM conversations WHERE id IN ({placeholders}) "
        f"ORDER BY updated_at ASC",
        tuple(conversation_ids),
    )
    conversations: dict[str, Conversation] = {
        row["id"]: _row_to_conversation(row) for row in conv_rows
    }

    turn_rows = store.fetchall(
        f"SELECT id, conversation_id, turn_index, role, content, blocks_json, "
        f"timestamp, parent_message_uuid, token_count "
        f"FROM turns WHERE conversation_id IN ({placeholders}) "
        f"ORDER BY conversation_id, turn_index ASC",
        tuple(conversation_ids),
    )
    turns_by_conversation: dict[str, list[Turn]] = {cid: [] for cid in conversations}
    for row in turn_rows:
        conv_id = row["conversation_id"]
        turns_by_conversation.setdefault(conv_id, []).append(_row_to_turn(row))

    return ModuleCorpus(
        conversations=conversations,
        turns_by_conversation=turns_by_conversation,
        audit_run_id=audit_run_id,
    )


def _load_findings(
    store: CorpusStore,
    audit_run_id: str,
    conversation_ids: Sequence[str],
) -> list[Finding]:
    """Rehydrate Findings for this audit run, optionally scoped to
    conversation_ids.

    Used by Module C (FindingsModule) to read existing A/B sycophancy
    events for second-pass classification.
    """
    params: list[Any] = [audit_run_id]
    clause = "audit_run_id = ?"
    if conversation_ids:
        placeholders = ",".join("?" for _ in conversation_ids)
        clause += f" AND conversation_id IN ({placeholders})"
        params.extend(conversation_ids)

    rows = store.fetchall(
        f"SELECT id, audit_run_id, conversation_id, turn_ids_json, turn_ids_hash, "
        f"module, behavior, intensity, confidence, confidence_alpha, "
        f"confidence_beta, quote_user, quote_assistant, evidence_quotes_json, "
        f"explanation, citation, detected_by_json, detected_at, prompt_version, "
        f"prompt_hash, metadata_json FROM findings WHERE {clause}",
        params,
    )
    findings: list[Finding] = []
    for row in rows:
        findings.append(
            Finding(
                id=row["id"],
                audit_run_id=row["audit_run_id"],
                conversation_id=row["conversation_id"],
                turn_ids=orjson.loads(row["turn_ids_json"]) if row["turn_ids_json"] else [],
                turn_ids_hash=row["turn_ids_hash"],
                module=ModuleName(row["module"]),
                behavior=row["behavior"],
                intensity=row["intensity"],
                confidence=row["confidence"],
                confidence_alpha=row["confidence_alpha"],
                confidence_beta=row["confidence_beta"],
                quote_user=row["quote_user"],
                quote_assistant=row["quote_assistant"],
                evidence_quotes=(
                    orjson.loads(row["evidence_quotes_json"]) if row["evidence_quotes_json"] else []
                ),
                explanation=row["explanation"],
                citation=row["citation"],
                detected_by=orjson.loads(row["detected_by_json"]),
                detected_at=datetime.fromisoformat(row["detected_at"]),
                prompt_version=row["prompt_version"],
                prompt_hash=row["prompt_hash"],
                metadata=orjson.loads(row["metadata_json"]) if row["metadata_json"] else {},
            )
        )
    return findings


async def _run_module(
    module_enum: ModuleName,
    *,
    corpus: ModuleCorpus,
    store: CorpusStore,
    audit_run_id: str,
    anthropic_client: AsyncAnthropic | None,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[ModuleResult]:
    """Instantiate and run the detection module for ``module_enum``.

    Deferred imports keep the registry's startup footprint small and
    avoid a hard dependency on all module files for tools that only
    need the store handlers (e.g. tests of ``query_corpus``).
    """
    if module_enum is ModuleName.A_SPIRALBENCH:
        assert anthropic_client is not None
        from lucid.modules.module_a_spiralbench import ModuleASpiralBench

        return await ModuleASpiralBench(client=anthropic_client).run(corpus)

    if module_enum is ModuleName.B_SHARMA:
        assert anthropic_client is not None
        from lucid.modules.module_b_sharma import ModuleBSharma

        return await ModuleBSharma(opus_client=anthropic_client).run(corpus)

    if module_enum is ModuleName.C_SYCEVAL:
        assert anthropic_client is not None
        from lucid.modules.module_c_syceval import ModuleCSycEval

        conversation_ids = list(corpus.conversations.keys())
        findings = _load_findings(store, audit_run_id, conversation_ids)
        return await ModuleCSycEval(client=anthropic_client).run(corpus, findings)

    if module_enum is ModuleName.D_PERSPECTIVE:
        assert anthropic_client is not None
        from lucid.modules.module_d_perspective import ModuleDPerspective

        return await ModuleDPerspective(client=anthropic_client).run(corpus)

    if module_enum is ModuleName.E_BELIEFSHIFT:
        assert anthropic_client is not None
        from lucid.modules.module_e_beliefshift import ModuleEBeliefShift

        return await ModuleEBeliefShift(opus_client=anthropic_client).run(corpus)

    if module_enum is ModuleName.F_ITP:
        assert anthropic_client is not None
        from lucid.modules.module_f_itp import ModuleFITP

        return await ModuleFITP(opus_client=anthropic_client).run(corpus)

    if module_enum is ModuleName.G_ATTRIBUTION:
        from lucid.modules.module_g_attribution import ModuleGAttribution

        return await ModuleGAttribution().run(corpus)

    if module_enum is ModuleName.H_MEMORY:
        assert anthropic_client is not None
        assert embedding_provider is not None
        from lucid.modules.module_h_memory import ModuleHMemory

        return await ModuleHMemory(
            opus_client=anthropic_client,
            store=store,
            embedding_provider=embedding_provider,
        ).run(corpus)

    raise ValueError(f"unsupported module: {module_enum}")


async def run_attribution_safety_net(
    *,
    store: CorpusStore,
    audit_run_id: str,
    conversation_ids: Sequence[str],
) -> tuple[int, int, list[dict[str, Any]]]:
    """Run Module G directly and persist its findings.

    Used by :func:`lucid.run.run_audit` after the orchestrator session
    ends to guarantee attribution findings exist even if Sonnet 4.6
    skipped ``invoke_module(G)`` under token pressure. Deterministic
    (no LLM call), idempotent — a second run surfaces every finding
    as an ``idempotent_skips`` collision against the UNIQUE key.

    Writes a :class:`~lucid.schemas.ModuleProgress` row to the
    ``module_progress`` table so resume tooling can see the safety
    net ran (or failed) without inspecting the findings table.

    Returns ``(stored, idempotent_skips, errors)`` matching the
    :func:`_persist_findings` tuple shape.
    """
    from lucid.modules.module_g_attribution import ModuleGAttribution

    if not conversation_ids:
        # Empty corpus — no work to do, no progress row needed.
        return 0, 0, []

    store.mark_module_started(audit_run_id, ModuleName.G_ATTRIBUTION)
    try:
        corpus = _load_module_corpus(store, audit_run_id, conversation_ids)
        results = await ModuleGAttribution().run(corpus)
    except Exception as err:
        store.mark_module_finished(
            audit_run_id,
            ModuleName.G_ATTRIBUTION,
            status="failed",
            error_message=str(err)[:500],
        )
        raise

    stored, idempotent, errors = _persist_findings(store, results)
    summary = (
        f"{len(errors)} per-conversation error(s); first: {errors[0]['error_type']}"
        if errors
        else None
    )
    store.mark_module_finished(
        audit_run_id,
        ModuleName.G_ATTRIBUTION,
        status="completed",
        error_message=summary,
    )
    return stored, idempotent, errors


def _persist_findings(
    store: CorpusStore,
    results: Sequence[ModuleResult],
) -> tuple[int, int, list[dict[str, Any]]]:
    """Write every ``Finding`` result to the store; return (stored,
    idempotent_collisions, module_errors).

    ``idempotent_collisions`` counts findings whose insertion raised a
    uniqueness error — that is the expected behaviour on resume / retry
    and is not a failure. Other insertion errors surface via the return
    value as individual ``module_errors`` entries so the orchestrator
    can see them without aborting the module pass.
    """
    stored = 0
    collisions = 0
    errors: list[dict[str, Any]] = []

    for result in results:
        if isinstance(result, ModuleError):
            errors.append(
                {
                    "conversation_id": result.conversation_id,
                    "error_type": result.error_type,
                    "message": result.message[:200],
                }
            )
            continue
        try:
            store.insert_finding(result)
            stored += 1
        except Exception as err:
            msg = str(err)
            if "UNIQUE constraint failed" in msg or "integrity" in msg.lower():
                collisions += 1
            else:
                errors.append(
                    {
                        "conversation_id": result.conversation_id,
                        "error_type": f"insert:{type(err).__name__}",
                        "message": msg[:200],
                    }
                )
    return stored, collisions, errors


async def invoke_module_for_run(
    *,
    module: ModuleName,
    conversation_ids: list[str],
    store: CorpusStore,
    audit_run_id: str,
    anthropic_client: AsyncAnthropic | None,
    embedding_provider: EmbeddingProvider | None,
    allow_module_d: bool,
    progress_log: Callable[[str, str], None],
    per_module_usd: dict[ModuleName, float],
    debited_modules: set[ModuleName],
    spend_tracker: dict[str, float],
) -> dict[str, Any]:
    """Run one detection module for a given audit run.

    Standalone version of the ``invoke_module`` custom-tool handler.
    Used directly by the deterministic scoring loop in
    :mod:`lucid.run` (Task 2.2) and — for now — also by the inline
    closure in :func:`build_tool_registry` so the tool-based execution
    path and the direct-call path share one implementation.

    Parameters carry what the closure used to close over: the store,
    the audit run id, the LLM client, the embedding provider, the
    Module D gate, the progress sink, the per-module cost-estimate
    table, the set of modules already debited this run, and the
    mutable spend tracker. Every side effect (``store.mark_module_*``,
    ``progress_log``, spend debits, finding persistence) happens here
    — callers should not replicate them.

    Returns a dict mirroring the custom-tool response shape. Successful
    runs produce ``{"status": "completed", "findings_stored": ...,
    "idempotent_skips": ..., "module_errors": ...}``; pre-flight gates
    short-circuit with ``status`` set to one of ``skipped`` /
    ``no_embedding_provider`` / ``no_client``; internal failures
    surface as ``{"error": "corpus_load_failed" | "module_run_failed"}``.
    The ``unknown_module`` parse error lives in the closure wrapper and
    never reaches this function (the ``module`` parameter is already
    an enum).
    """
    module_name = module.value

    # Module D is opt-in.
    if module is ModuleName.D_PERSPECTIVE and not allow_module_d:
        skip_message = "Module D is opt-in; start the audit with --include-module-d to enable it."
        progress_log(
            "INFO",
            f"invoke_module(D) called without --include-module-d; "
            f"skipping over {len(conversation_ids)} conversations.",
        )
        store.mark_module_finished(
            audit_run_id, module, status="skipped", error_message=skip_message
        )
        return {
            "module": module_name,
            "status": "skipped",
            "message": skip_message,
        }

    # Module H requires an embedding provider in addition to the client.
    if module is ModuleName.H_MEMORY and embedding_provider is None:
        skip_message = (
            "Module H requires an embedding provider (Voyage AI). "
            "Ensure VOYAGE_API_KEY is set or an EmbeddingProvider is "
            "injected into the tool registry."
        )
        progress_log(
            "WARNING",
            "invoke_module(H) called without embedding_provider; skipping. "
            "Module H requires Voyage or an equivalent EmbeddingProvider.",
        )
        store.mark_module_finished(
            audit_run_id, module, status="skipped", error_message=skip_message
        )
        return {
            "module": module_name,
            "status": "no_embedding_provider",
            "message": skip_message,
        }

    # LLM-backed modules need a client.
    llm_modules = {
        ModuleName.A_SPIRALBENCH,
        ModuleName.B_SHARMA,
        ModuleName.C_SYCEVAL,
        ModuleName.D_PERSPECTIVE,
        ModuleName.E_BELIEFSHIFT,
        ModuleName.F_ITP,
        ModuleName.H_MEMORY,
    }
    if module in llm_modules and anthropic_client is None:
        skip_message = (
            "anthropic_client not injected into tool registry; this "
            "happens on --dry-run and when ANTHROPIC_API_KEY is unset. "
            "Run a real audit with a valid key to execute this module."
        )
        progress_log(
            "WARNING",
            f"invoke_module({module_name}) called without anthropic_client; skipping module run.",
        )
        store.mark_module_finished(
            audit_run_id, module, status="skipped", error_message=skip_message
        )
        return {
            "module": module_name,
            "status": "no_client",
            "message": skip_message,
        }

    # Real run begins here. Mark started before the corpus load so
    # a corpus-load failure still leaves a ``failed`` row behind.
    store.mark_module_started(audit_run_id, module)

    try:
        corpus = _load_module_corpus(store, audit_run_id, conversation_ids)
    except Exception as err:
        err_message = str(err)[:500]
        store.mark_module_finished(audit_run_id, module, status="failed", error_message=err_message)
        return {
            "error": "corpus_load_failed",
            "module": module_name,
            "message": err_message,
        }

    progress_log(
        "INFO",
        f"invoke_module({module_name}) running over {len(corpus.conversations)} conversations.",
    )

    try:
        results = await _run_module(
            module,
            corpus=corpus,
            store=store,
            audit_run_id=audit_run_id,
            anthropic_client=anthropic_client,
            embedding_provider=embedding_provider,
        )
    except Exception as err:
        err_message = str(err)[:500]
        store.mark_module_finished(audit_run_id, module, status="failed", error_message=err_message)
        return {
            "error": "module_run_failed",
            "module": module_name,
            "message": err_message,
        }

    stored, idempotent, module_errors = _persist_findings(store, results)

    # Debit the pre-flight upper bound once the module actually ran.
    # We use the upper-bound estimate instead of live usage totals
    # because modules don't bubble up per-call ``response.usage`` —
    # see the M1 note in the CLAUDE.md wiring plan. Missing
    # estimates simply skip the debit silently.
    estimated_usd = per_module_usd.get(module)
    if estimated_usd is not None and module not in debited_modules:
        spend_tracker["accrued_usd"] = round(spend_tracker["accrued_usd"] + float(estimated_usd), 6)
        debited_modules.add(module)

    # Per-conversation errors don't flip the status to ``failed`` —
    # they are partial-pass diagnostics, mirrored into
    # ``error_message`` so resume / triage tooling can see them.
    completion_summary = (
        f"{len(module_errors)} per-conversation error(s); first: {module_errors[0]['error_type']}"
        if module_errors
        else None
    )
    store.mark_module_finished(
        audit_run_id,
        module,
        status="completed",
        error_message=completion_summary,
    )

    progress_log(
        "INFO",
        f"invoke_module({module_name}) completed: stored={stored}, "
        f"idempotent_skips={idempotent}, module_errors={len(module_errors)}.",
    )

    return {
        "module": module_name,
        "status": "completed",
        "findings_stored": stored,
        "idempotent_skips": idempotent,
        "module_errors": module_errors[:10],
        "module_errors_count": len(module_errors),
    }


def build_tool_registry(
    *,
    store: CorpusStore,
    audit_run_id: str,
    progress_log: Callable[[str, str], None] | None = None,
    remaining_budget_usd: float = 0.0,
    anthropic_client: AsyncAnthropic | None = None,
    allow_module_d: bool = False,
    embedding_provider: EmbeddingProvider | None = None,
    cost_estimate: CostEstimate | None = None,
    spend_tracker: dict[str, float] | None = None,
    per_module_usd: dict[ModuleName, float] | None = None,
    debited_modules: set[ModuleName] | None = None,
) -> ToolRegistry:
    """Wire up the 8 custom tools for a given audit run.

    `progress_log(level, message)` receives orchestrator heartbeats the
    CLI can pipe to the user. Defaults to `_LOGGER.info`.

    `anthropic_client` is consumed by ``invoke_module`` when a detection
    module needs LLM access (A, B, C, D, E, F, H). When ``None``, those
    modules return a ``no_client`` status rather than running — useful
    for dry-run exercises and for tests that don't want to instantiate
    the SDK.

    `allow_module_d` gates invocation of Module D (perspective
    sycophancy), which is opt-in per the CLI's ``--include-module-d``
    flag. When ``False``, invoking module D returns ``skipped``.

    `embedding_provider` is consumed by Module H (memory-corpus
    consistency) for corpus chunk indexing and per-claim retrieval.
    When ``None``, Module H returns ``no_embedding_provider`` — the
    orchestrator logs and continues, rather than crashing the audit.

    `cost_estimate` is the pre-flight :class:`~lucid.cost.CostEstimate`
    produced by ``CostEstimator.estimate``. When provided, ``invoke_module``
    increments ``spend_tracker["accrued_usd"]`` by the matching
    per-module upper bound after the module completes successfully —
    enough to make ``estimate_remaining_cost`` reflect actual progress
    without threading per-call ``response.usage`` through every module's
    retry loop. Each module is debited at most once per registry; a
    second successful invocation of the same module reuses the budget
    line and does not double-charge.
    """
    if progress_log is None:

        def _default_progress(level: str, message: str) -> None:
            _LOGGER.log(logging.getLevelName(level), "%s", message)

        progress_log = _default_progress

    # Mutable close-over for the running cost tally. Callers may supply
    # pre-built dicts/sets so that direct-invocation paths (e.g. the
    # deterministic scoring loop in :mod:`lucid.run`) can share spend
    # state with the tool handlers registered here.
    if spend_tracker is None:
        spend_tracker = {"accrued_usd": 0.0, "budget_usd": remaining_budget_usd}

    # Per-module USD upper bounds from the pre-flight estimate. ``None``
    # entries mean "no estimate seen for this module" — typically the
    # estimate covers the modules in the user's --enabled set; modules
    # invoked outside that set (rare) accrue no spend.
    if per_module_usd is None:
        per_module_usd = (
            {mc.module: mc.usd for mc in cost_estimate.per_module} if cost_estimate else {}
        )
    # Track which modules have already been debited so a second
    # invocation (resume / orchestrator retry) does not double-charge.
    if debited_modules is None:
        debited_modules = set()

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
        module_name = args["module"]
        try:
            module_enum = ModuleName(module_name)
        except ValueError:
            # No module_progress row written: an unknown module name is
            # an orchestrator bug, not part of any planned module's
            # lifecycle.
            return {"error": "unknown_module", "module": module_name}
        conv_ids = args.get("conversation_ids") or []
        return await invoke_module_for_run(
            module=module_enum,
            conversation_ids=conv_ids,
            store=store,
            audit_run_id=audit_run_id,
            anthropic_client=anthropic_client,
            embedding_provider=embedding_provider,
            allow_module_d=allow_module_d,
            progress_log=progress_log,
            per_module_usd=per_module_usd,
            debited_modules=debited_modules,
            spend_tracker=spend_tracker,
        )

    registry.register(
        CustomTool(
            name="invoke_module",
            description=(
                "Run a Lucid detection module (A, B, C, D, E, F, G, H) over the "
                "supplied conversation_ids. The module fans out internally with "
                "bounded concurrency and persists every Finding it produces — "
                "the response carries `findings_stored` so you do NOT need to "
                "call `store_finding` afterwards. Pass every sampled "
                "conversation id in one call per module; do not call once per "
                "conversation. Modules return `status='completed'` on success "
                "or one of `skipped` / `no_client` / `no_embedding_provider` "
                "when a precondition is missing — log the latter and continue. "
                "This is the primary tool for actually running the audit; "
                "calling query_corpus alone does not produce findings."
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
                "RESERVED — do not call in the default workflow. Findings are "
                "persisted automatically inside `invoke_module`; calling "
                "`store_finding` after `invoke_module` causes duplicate-key "
                "errors against the (audit_run_id, module, conversation_id, "
                "turn_ids_hash, behavior) idempotency tuple. The tool stays "
                "registered for out-of-band synthesis only (e.g. a derived "
                "finding combining multiple module outputs). Idempotency key "
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
