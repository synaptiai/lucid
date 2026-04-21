"""Non-dry-run audit driver.

Phase 5B shipped the *smoke* path: one live Managed Agents session that
calls `query_corpus` then writes a single placeholder finding. The
module-invocation surface is stubbed until Phase 7 wires the real
modules.

What this file owns:

- Persisting an `AuditRun` row before any LLM call (so every Finding has
  a FK target waiting).
- Acquiring a per-DB `filelock` to prevent two audits racing on the
  same SQLite file (CHECK constraints would still protect integrity
  but the user experience of "audit half-wrote, other audit finished"
  is bad).
- Building the tool registry bound to (store, audit_run_id, budget).
- Running the Managed Agents session with the supplied kickoff message
  and system prompt.
- Transitioning the AuditRun to `completed` / `partial` / `failed`
  based on the session outcome.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from filelock import FileLock, Timeout

from lucid.cost import CostEstimate
from lucid.orchestrator.managed_agent import (
    ManagedAgentsSession,
    OrchestratorConfig,
    SessionOutcome,
)
from lucid.orchestrator.tools import build_tool_registry
from lucid.schemas import (
    AuditRun,
    AuditStatus,
    Conversation,
    CorpusStats,
    ModuleName,
    ModuleTokenUsage,
    SamplingConfigRecord,
    Source,
    TokenUsage,
    Turn,
)
from lucid.store import SCHEMA_VERSION, initialize_db
from lucid.store.sqlite import CorpusStore

if TYPE_CHECKING:  # pragma: no cover
    from lucid.sampling import SamplingConfig

_LOGGER = logging.getLogger(__name__)

FILELOCK_SUFFIX = ".lucid.lock"


# ──────────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────────


class RunError(RuntimeError):
    """Audit-run-level failure. Message is safe to surface to the user."""


class LockHeldError(RunError):
    """Raised when another process holds the per-DB filelock."""


# ──────────────────────────────────────────────────────────────────────────
# Inputs / outputs
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AuditInputs:
    """Everything `run_audit` needs after the CLI has done ingest + sampling."""

    source_paths: dict[Source, Path]
    sampled: list[Conversation]
    turns_by_conv: dict[str, list[Turn]]
    corpus_fingerprint: str
    sampling_config: SamplingConfig
    estimate: CostEstimate
    enabled_modules: list[ModuleName]
    authorized_budget_usd: float


@dataclass
class AuditResult:
    run_id: str
    status: AuditStatus
    outcome: SessionOutcome | None
    findings_written: int
    reason: str


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _db_path_for(data_dir: Path) -> Path:
    return data_dir / "lucid.sqlite3"


def _lock_path_for(data_dir: Path) -> Path:
    return data_dir / FILELOCK_SUFFIX


def _generate_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:12]}"


def _persist_corpus(
    store: CorpusStore, sampled: list[Conversation], turns_by_conv: dict[str, list[Turn]]
) -> None:
    """Store the sampled conversations + turns exactly once.

    Ingest writes to the DB here (rather than in the CLI dry-run path) so
    the cost gate runs before any SQLite disk write. If the user rejects
    the gate, their DB stays pristine.
    """
    # Conversations we haven't seen before in this DB.
    existing_ids = {row["id"] for row in store.fetchall("SELECT id FROM conversations")}
    fresh_convs = [c for c in sampled if c.id not in existing_ids]
    if fresh_convs:
        store.insert_conversations(fresh_convs)

    # Insert only the turns that belong to freshly-inserted convs.
    fresh_turn_rows: list[Turn] = []
    for conv in fresh_convs:
        fresh_turn_rows.extend(turns_by_conv.get(conv.id, []))
    if fresh_turn_rows:
        store.insert_turns(fresh_turn_rows)


def _create_audit_run_row(
    store: CorpusStore,
    *,
    run_id: str,
    inputs: AuditInputs,
    prompt_versions: dict[ModuleName, str],
) -> AuditRun:
    stats = CorpusStats(
        discovered_conversations=len(inputs.sampled),
        sampled_conversations=len(inputs.sampled),
        discovered_turns=sum(len(v) for v in inputs.turns_by_conv.values()),
        sampled_turns=sum(len(inputs.turns_by_conv.get(c.id, [])) for c in inputs.sampled),
        date_range_start=min((c.updated_at for c in inputs.sampled), default=None),
        date_range_end=max((c.updated_at for c in inputs.sampled), default=None),
        sources=list(inputs.source_paths.keys()),
    )
    usage = TokenUsage(
        by_module={
            mc.module: ModuleTokenUsage(
                input_tokens=mc.input_tokens,
                output_tokens=mc.output_tokens,
                cache_read_input_tokens=mc.cached_input_tokens,
                usd_cost=mc.usd,
            )
            for mc in inputs.estimate.per_module
        }
    )
    sampling_record = SamplingConfigRecord(
        n=inputs.sampling_config.n,
        seed=inputs.sampling_config.seed,
        min_turns=inputs.sampling_config.min_turns,
        recency_weight=inputs.sampling_config.recency_weight,
        recency_window_days=inputs.sampling_config.recency_window_days,
        stratify_by_project=inputs.sampling_config.stratify_by_project,
        top_n_projects=inputs.sampling_config.top_n_projects,
    )
    run = AuditRun(
        id=run_id,
        sources=list(inputs.source_paths.keys()),
        source_paths={s: str(p) for s, p in inputs.source_paths.items()},
        started_at=_now(),
        corpus_stats=stats,
        token_usage=usage,
        sampling_config=sampling_record,
        status="running",
        corpus_fingerprint=inputs.corpus_fingerprint,
        prompt_versions=prompt_versions,
        schema_version=SCHEMA_VERSION,
    )
    store.insert_audit_run(run)
    return run


def _update_audit_run_status(
    store: CorpusStore, run_id: str, status: AuditStatus, *, completed: bool
) -> None:
    conn = store.connect()
    if completed:
        conn.execute(
            "UPDATE audit_runs SET status = ?, completed_at = ? WHERE id = ?",
            (status, _now().isoformat(), run_id),
        )
    else:
        conn.execute(
            "UPDATE audit_runs SET status = ? WHERE id = ?",
            (status, run_id),
        )
    conn.commit()


def _count_findings(store: CorpusStore, run_id: str) -> int:
    rows = store.fetchall("SELECT COUNT(*) as n FROM findings WHERE audit_run_id = ?", (run_id,))
    return int(rows[0]["n"]) if rows else 0


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────


def run_audit(
    *,
    inputs: AuditInputs,
    data_dir: Path,
    client: Any,
    system_prompt: str,
    kickoff_message: str,
    prompt_versions: dict[ModuleName, str],
    progress_log: Callable[[str, str], None] | None = None,
    lock_timeout_seconds: float = 0.1,
    session_runner: Callable[[ManagedAgentsSession, str], Coroutine[Any, Any, SessionOutcome]]
    | None = None,
    run_id: str | None = None,
) -> AuditResult:
    """Execute a non-dry-run audit end-to-end.

    `session_runner` is injectable so tests can plug in a fake instead of
    a real Anthropic session. It defaults to `session.run(kickoff_message)`.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = _db_path_for(data_dir)
    initialize_db(db_path)

    lock_path = _lock_path_for(data_dir)
    lock = FileLock(str(lock_path))
    try:
        lock.acquire(timeout=lock_timeout_seconds)
    except Timeout as err:
        raise LockHeldError(
            f"Another Lucid audit appears to be running against {db_path}. "
            f"If you believe this is stale, delete {lock_path} and retry."
        ) from err

    try:
        resolved_run_id = run_id or _generate_run_id()
        with CorpusStore(db_path) as store:
            _persist_corpus(store, inputs.sampled, inputs.turns_by_conv)
            _create_audit_run_row(
                store,
                run_id=resolved_run_id,
                inputs=inputs,
                prompt_versions=prompt_versions,
            )

            registry = build_tool_registry(
                store=store,
                audit_run_id=resolved_run_id,
                progress_log=progress_log,
                remaining_budget_usd=inputs.authorized_budget_usd,
            )
            session = ManagedAgentsSession(
                client=client,
                registry=registry,
                config=OrchestratorConfig(run_id=resolved_run_id),
            )
            # Managed Agents `beta.agents.create` takes `system` as a string
            # (not the messages-API list-of-blocks shape). Inject the
            # smoke-mode prompt at the session level.
            _inject_system_prompt_override(session, system_prompt)

            runner = session_runner or _default_session_runner
            outcome: SessionOutcome | None = None
            status: AuditStatus
            reason: str
            try:
                outcome = asyncio.run(runner(session, kickoff_message))
            except Exception as err:
                _LOGGER.exception("audit session raised")
                status = "failed"
                reason = f"exception: {err}"
            else:
                if outcome.completed:
                    status = "completed"
                    reason = outcome.reason
                else:
                    status = "partial"
                    reason = outcome.reason or "partial"

            _update_audit_run_status(
                store, resolved_run_id, status, completed=(status == "completed")
            )
            findings_written = _count_findings(store, resolved_run_id)

        return AuditResult(
            run_id=resolved_run_id,
            status=status,
            outcome=outcome,
            findings_written=findings_written,
            reason=reason,
        )
    finally:
        lock.release()


async def _default_session_runner(session: ManagedAgentsSession, kickoff: str) -> SessionOutcome:
    return await session.run(kickoff)


def _inject_system_prompt_override(session: ManagedAgentsSession, system_prompt: str) -> None:
    """Swap the system-prompt builder on an existing session.

    Phase 5B uses a short smoke-mode prompt (see
    `lucid.orchestrator.smoke`). Phase 7 will replace this hook with a
    proper `SystemPromptProvider` config kwarg once a third caller
    shows up.
    """

    def _create_agent_with_override() -> str:
        agent = session.client.beta.agents.create(
            name=f"lucid-orchestrator-{session.config.run_id[:8]}",
            model=session.config.orchestrator_model,
            system=system_prompt,
            tools=session.registry.as_agent_tools(),
        )
        return str(agent.id)

    session.create_agent = _create_agent_with_override  # type: ignore[method-assign]


__all__ = [
    "AuditInputs",
    "AuditResult",
    "LockHeldError",
    "RunError",
    "run_audit",
]
