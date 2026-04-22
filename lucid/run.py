"""Non-dry-run audit driver.

What this file owns:

- Persisting an `AuditRun` row before any LLM call (so every Finding has
  a FK target waiting).
- Acquiring a per-DB `filelock` to prevent two audits racing on the
  same SQLite file (CHECK constraints would still protect integrity
  but the user experience of "audit half-wrote, other audit finished"
  is bad).
- Building the tool registry bound to (store, audit_run_id, budget,
  async_client, embedding_provider, allow_module_d, cost_estimate).
- Running the Managed Agents session with the supplied kickoff message
  and system prompt (the prompt is threaded through
  ``OrchestratorConfig`` so there is no monkey-patching of
  ``create_agent``).
- Re-running Module G (deterministic time/model attribution) directly
  against the persisted corpus once the session ends, as a safety net
  in case the orchestrator skipped the in-session attribution call.
- Rendering the HTML report from persisted findings and returning the
  output path.
- Transitioning the AuditRun to `completed` / `partial` / `failed`
  based on the session outcome.

Two clients are required:

- A *sync* ``Anthropic`` client drives :class:`ManagedAgentsSession` —
  the SDK's ``beta.agents.create`` / ``events.send`` / ``events.stream``
  surface is synchronous and the driver bridges to async internally.
- An *async* :class:`AsyncAnthropic` client is handed to the dispatcher
  (and from there to every detection module) so module ``await
  client.messages.create(...)`` calls run as real coroutines instead of
  blocking the event loop.
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
from lucid.orchestrator.tools import build_tool_registry, run_attribution_safety_net
from lucid.report.generator import write_report
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
    from anthropic import Anthropic, AsyncAnthropic

    from lucid.modules.embeddings import EmbeddingProvider
    from lucid.sampling import SamplingConfig

_LOGGER = logging.getLogger(__name__)

FILELOCK_SUFFIX = ".lucid.lock"

DEFAULT_REPORT_DIR = Path("report")


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
    report_path: Path | None = None


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _db_path_for(data_dir: Path) -> Path:
    return data_dir / "lucid.sqlite3"


def _lock_path_for(data_dir: Path) -> Path:
    return data_dir / FILELOCK_SUFFIX


def generate_run_id() -> str:
    """Mint a fresh ``run-<12-hex>`` audit-run id.

    Public so the CLI can pre-allocate the id and weave it into the
    kickoff message before handing the same id back to ``run_audit``
    via ``run_id=``. Without this, the orchestrator has no way to
    reference the run it is working on (the session API does not
    leak the audit-run id to the agent).
    """
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


async def _attribution_safety_net(
    *,
    store: CorpusStore,
    run_id: str,
    sampled_ids: list[str],
    progress_log: Callable[[str, str], None],
) -> int:
    """Bridge :func:`run_attribution_safety_net` + progress logging.

    The orchestrator's system prompt instructs it to call
    ``invoke_module(G)`` last, but Sonnet 4.6 sometimes skips a step
    under token pressure or unusual error paths. Module G is
    deterministic (no LLM, idempotent), so re-running it costs
    nothing and fills any attribution gap. Returns the number of
    *new* findings persisted (zero if the orchestrator already ran G).
    """
    stored, idempotent, errors = await run_attribution_safety_net(
        store=store,
        audit_run_id=run_id,
        conversation_ids=sampled_ids,
    )
    if errors:
        progress_log(
            "WARNING",
            f"post-run Module G surfaced {len(errors)} error(s); the report may have gaps.",
        )
    progress_log(
        "INFO",
        f"post-run Module G safety net: stored={stored}, idempotent_skips={idempotent}.",
    )
    return stored


def _render_report_or_log(
    *,
    store: CorpusStore,
    run_id: str,
    output_dir: Path,
    progress_log: Callable[[str, str], None],
) -> Path | None:
    """Fetch findings + audit_run row, render the HTML, return its path.

    Failures here are logged but do not raise: the audit's findings
    are already persisted in SQLite and the user can re-render from
    them later. Surfacing a stack trace just to lose the report would
    bury the actual data.
    """
    audit_run = store.fetch_audit_run(run_id)
    if audit_run is None:
        progress_log(
            "ERROR",
            f"could not load audit_run row for {run_id}; skipping report render.",
        )
        return None
    findings = store.fetch_findings_for_run(run_id)
    try:
        path = write_report(audit_run, findings, output_dir=output_dir)
    except Exception as err:  # pragma: no cover — logged for operator triage
        _LOGGER.exception("report render failed")
        progress_log("ERROR", f"report render failed: {err}")
        return None
    progress_log("INFO", f"Report written: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────


def run_audit(
    *,
    inputs: AuditInputs,
    data_dir: Path,
    client: Anthropic,
    async_client: AsyncAnthropic,
    system_prompt: str,
    kickoff_message: str,
    prompt_versions: dict[ModuleName, str],
    embedding_provider: EmbeddingProvider | None = None,
    allow_module_d: bool = False,
    progress_log: Callable[[str, str], None] | None = None,
    lock_timeout_seconds: float = 0.1,
    session_runner: Callable[[ManagedAgentsSession, str], Coroutine[Any, Any, SessionOutcome]]
    | None = None,
    run_id: str | None = None,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> AuditResult:
    """Execute a non-dry-run audit end-to-end.

    Parameters
    ----------
    client
        *Sync* Anthropic client used by :class:`ManagedAgentsSession`
        for the agent / environment / session / events transport. The
        SDK's beta.agents surface is synchronous; the driver bridges to
        async internally for the event loop.
    async_client
        :class:`AsyncAnthropic` client used by every detection module
        (A, B, C, D, E, F, H) for ``await client.messages.create(...)``.
        Modules unit-test against ``AsyncMock`` clients of this same
        shape, so passing a sync client here causes silent event-loop
        blocking on every classification call.
    embedding_provider
        Optional :class:`~lucid.modules.embeddings.EmbeddingProvider`
        for Module H. ``None`` makes Module H return
        ``no_embedding_provider`` from the dispatcher.
    allow_module_d
        Whether the dispatcher should run Module D when invoked. Wired
        from ``--include-module-d`` at the CLI.
    session_runner
        Injectable for tests so a fake session can short-circuit the
        Managed Agents transport. Defaults to
        ``session.run(kickoff_message)``.

    Returns
    -------
    AuditResult
        Includes the rendered report path on success
        (``report_dir / "<run_id>.html"``) or ``None`` if rendering
        failed or the run aborted before any module produced output.
    """
    if progress_log is None:

        def _default_progress(level: str, message: str) -> None:
            _LOGGER.log(logging.getLevelName(level), "%s", message)

        progress_log = _default_progress

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
        resolved_run_id = run_id or generate_run_id()
        report_path: Path | None = None
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
                anthropic_client=async_client,
                allow_module_d=allow_module_d,
                embedding_provider=embedding_provider,
                cost_estimate=inputs.estimate,
            )
            session = ManagedAgentsSession(
                client=client,
                registry=registry,
                config=OrchestratorConfig(
                    run_id=resolved_run_id,
                    system_prompt=system_prompt,
                ),
            )

            runner = session_runner or _default_session_runner
            sampled_ids = [c.id for c in inputs.sampled]
            # One asyncio.run for the whole pipeline. Two sequential
            # ``asyncio.run`` calls would tear down a fresh event loop
            # between the session and the safety net, killing any
            # background SDK tasks (HTTP keep-alive, connection pools)
            # the Anthropic client spawned during the session.
            outcome, status, reason = asyncio.run(
                _execute_session_and_safety_net(
                    runner=runner,
                    session=session,
                    kickoff_message=kickoff_message,
                    store=store,
                    audit_run_id=resolved_run_id,
                    sampled_ids=sampled_ids,
                    progress_log=progress_log,
                )
            )

            report_path = _render_report_or_log(
                store=store,
                run_id=resolved_run_id,
                output_dir=report_dir,
                progress_log=progress_log,
            )

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
            report_path=report_path,
        )
    finally:
        lock.release()


async def _execute_session_and_safety_net(
    *,
    runner: Callable[[ManagedAgentsSession, str], Coroutine[Any, Any, SessionOutcome]],
    session: ManagedAgentsSession,
    kickoff_message: str,
    store: CorpusStore,
    audit_run_id: str,
    sampled_ids: list[str],
    progress_log: Callable[[str, str], None],
) -> tuple[SessionOutcome | None, AuditStatus, str]:
    """Drive the orchestrator session, then the Module G safety net.

    Both phases share one event loop — the SDK's background HTTP
    cleanup tasks live inside that loop and would be silently
    destroyed if a fresh ``asyncio.run`` were spun up for the safety
    net. The safety net always runs (even on a session exception) so
    the report has attribution rows for whatever findings did persist.
    """
    outcome: SessionOutcome | None = None
    status: AuditStatus
    reason: str
    try:
        outcome = await runner(session, kickoff_message)
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

    try:
        await _attribution_safety_net(
            store=store,
            run_id=audit_run_id,
            sampled_ids=sampled_ids,
            progress_log=progress_log,
        )
    except Exception as err:
        # Findings are already in the DB; losing the audit_runs row
        # update for a safety-net failure would be strictly worse
        # than the missing attribution rows.
        _LOGGER.exception("post-run Module G safety net failed")
        progress_log("ERROR", f"post-run Module G safety net failed: {err}")

    return outcome, status, reason


async def _default_session_runner(session: ManagedAgentsSession, kickoff: str) -> SessionOutcome:
    return await session.run(kickoff)


__all__ = [
    "DEFAULT_REPORT_DIR",
    "AuditInputs",
    "AuditResult",
    "LockHeldError",
    "RunError",
    "generate_run_id",
    "run_audit",
]
