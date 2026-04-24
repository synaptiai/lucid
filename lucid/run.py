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
  The registry is retained for future synthesis-phase wiring; the
  scoring phase invokes modules directly via
  :func:`invoke_module_for_run` in a deterministic for-loop.
- Running the deterministic scoring loop (:func:`_run_scoring_loop`)
  over every enabled module. Each module returns a status; the audit
  is ``completed`` iff every module returned ``status="completed"``.
- Rendering the HTML report from persisted findings and returning the
  output path.
- Transitioning the AuditRun to `completed` / `partial` / `failed`
  based on the aggregated module outcomes.

Only an *async* :class:`AsyncAnthropic` client is required: every
detection module (A, B, C, D, E, F, H) uses ``await
client.messages.create(...)`` and Module G is deterministic (no LLM).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from filelock import FileLock, Timeout

from lucid.cost import CostEstimate
from lucid.orchestrator.tools import (
    ToolRegistry,
    build_tool_registry,
    invoke_module_for_run,
)
from lucid.report.generator import write_deck, write_report
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

# Per-module statuses that must NOT flip the audit to "partial". These
# are documented, intentional outcomes — running the correct code path
# for the caller's configuration — not failures:
#   - "completed":              module ran end-to-end.
#   - "skipped":                Module D opted out via --no-include-module-d.
#   - "no_embedding_provider":  Module H ran with VOYAGE_API_KEY unset; the
#                               dispatcher short-circuits rather than
#                               fabricate embeddings.
#   - "no_client":              an LLM module ran without ANTHROPIC_API_KEY
#                               (typical in dry-run paths wired through
#                               the scoring loop for validation).
# Anything outside this set (e.g. "failed", exception propagations) flips
# the audit to "partial" and is surfaced in the reason string.
_NON_FAILURE_STATUSES = frozenset({"completed", "skipped", "no_embedding_provider", "no_client"})


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

    Public so the CLI can pre-allocate the id and reference the same id
    it hands back to ``run_audit`` via ``run_id=``.
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


def _behavior_counts_for_run(store: CorpusStore, run_id: str) -> dict[str, int]:
    """Aggregate findings by ``behavior`` label for ``run_id``.

    Feeds the synthesis kickoff's thin-evidence hedging block. Pulls
    from the already-persisted findings table — the scoring loop has
    finished by the time synthesis runs, so the counts are authoritative.
    """
    rows = store.fetchall(
        "SELECT behavior, COUNT(*) AS n FROM findings WHERE audit_run_id = ? GROUP BY behavior",
        (run_id,),
    )
    return {str(row["behavior"]): int(row["n"]) for row in rows}


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
    # Synthesis sections are optional: the report renderer falls back
    # to its deterministic summary blocks when this list is empty, so
    # a fetch failure should never kill the report. We still let a DB
    # error propagate here — the outer except-Exception below swallows
    # and logs it alongside any template/render error.
    report_sections = store.fetch_report_sections_for_run(run_id)
    try:
        path = write_report(
            audit_run,
            findings,
            output_dir=output_dir,
            report_sections=report_sections,
        )
    except Exception as err:  # pragma: no cover — logged for operator triage
        _LOGGER.exception("report render failed")
        progress_log("ERROR", f"report render failed: {err}")
        return None
    progress_log("INFO", f"Report written: {path}")
    # Hackathon demo deck — same data, slide-form companion to the
    # report. A failure here must not bury the report path the user
    # already has, so it logs and continues.
    try:
        deck_path = write_deck(
            audit_run,
            findings,
            output_dir=output_dir,
            report_sections=report_sections,
        )
        progress_log("INFO", f"Deck written: {deck_path}")
    except Exception as err:  # pragma: no cover — logged for operator triage
        _LOGGER.exception("deck render failed")
        progress_log("WARN", f"deck render failed (report still written): {err}")
    return path


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────


def run_audit(
    *,
    inputs: AuditInputs,
    data_dir: Path,
    async_client: AsyncAnthropic,
    prompt_versions: dict[ModuleName, str],
    embedding_provider: EmbeddingProvider | None = None,
    allow_module_d: bool = False,
    progress_log: Callable[[str, str], None] | None = None,
    lock_timeout_seconds: float = 0.1,
    run_id: str | None = None,
    report_dir: Path = DEFAULT_REPORT_DIR,
    client: Anthropic | None = None,
    synthesis_enabled: bool = True,
) -> AuditResult:
    """Execute a non-dry-run audit end-to-end.

    Parameters
    ----------
    async_client
        :class:`AsyncAnthropic` client used by every detection module
        (A, B, C, D, E, F, H) for ``await client.messages.create(...)``.
        Modules unit-test against ``AsyncMock`` clients of this same
        shape; passing a sync client would silently block the event
        loop on every classification call.
    client
        Optional synchronous :class:`Anthropic` client. Required iff
        ``synthesis_enabled`` is True: the Managed Agents beta API
        (``client.beta.agents`` / ``sessions`` / ``environments``) is
        synchronous in the SDK, so the synthesis phase needs a sync
        handle. When ``None`` or ``synthesis_enabled=False``, the
        synthesis phase is skipped cleanly — the scoring findings are
        already persisted and the HTML report still renders.
    synthesis_enabled
        Whether to run the synthesis phase (Opus 4.7 writer) after
        scoring completes. Default True. Gated by ``--no-synthesis``
        at the CLI. Synthesis only runs when scoring finished with
        status ``completed`` or ``partial`` (not ``failed``), and
        when a sync ``client`` is wired.
    embedding_provider
        Optional :class:`~lucid.modules.embeddings.EmbeddingProvider`
        for Module H. ``None`` makes Module H return
        ``no_embedding_provider`` from the dispatcher.
    allow_module_d
        Whether the dispatcher should run Module D when invoked. Wired
        from ``--include-module-d`` at the CLI.

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
        status: AuditStatus = "failed"
        reason = "no run executed"
        with CorpusStore(db_path) as store:
            _persist_corpus(store, inputs.sampled, inputs.turns_by_conv)
            _create_audit_run_row(
                store,
                run_id=resolved_run_id,
                inputs=inputs,
                prompt_versions=prompt_versions,
            )

            # Spend state is shared between the tool registry (for any
            # future synthesis-phase tool handlers) and the scoring
            # loop (which calls ``invoke_module_for_run`` directly).
            # Build it at this scope so both views reference the same
            # mutable state — a module debited once by the scoring loop
            # must not be double-charged by a later tool invocation.
            spend_tracker: dict[str, float] = {
                "accrued_usd": 0.0,
                "budget_usd": inputs.authorized_budget_usd,
            }
            per_module_usd: dict[ModuleName, float] = {
                mc.module: mc.usd for mc in inputs.estimate.per_module
            }
            debited_modules: set[ModuleName] = set()

            registry = build_tool_registry(
                store=store,
                audit_run_id=resolved_run_id,
                progress_log=progress_log,
                remaining_budget_usd=inputs.authorized_budget_usd,
                anthropic_client=async_client,
                allow_module_d=allow_module_d,
                embedding_provider=embedding_provider,
                cost_estimate=inputs.estimate,
                spend_tracker=spend_tracker,
                per_module_usd=per_module_usd,
                debited_modules=debited_modules,
            )

            sampled_ids = [c.id for c in inputs.sampled]

            # Defensively ensure attribution runs. The CLI appends G
            # explicitly (cli.py:244-245), but a non-CLI caller — a
            # future ``lucid ask`` mode, a test harness, or a
            # calibration pipeline reusing this driver — could omit
            # it and produce a report with empty month/model charts.
            # ``AuditInputs`` is frozen, so copy to a local list
            # before appending. Idempotent: skipped if already present.
            enabled_modules = list(inputs.enabled_modules)
            if ModuleName.G_ATTRIBUTION not in enabled_modules:
                enabled_modules.append(ModuleName.G_ATTRIBUTION)
                _LOGGER.debug("Auto-appending Module G (attribution) to enabled_modules")

            try:
                status, reason = asyncio.run(
                    _execute_scoring(
                        store=store,
                        registry=registry,
                        audit_run_id=resolved_run_id,
                        enabled_modules=enabled_modules,
                        sampled_ids=sampled_ids,
                        progress_log=progress_log,
                        anthropic_client=async_client,
                        embedding_provider=embedding_provider,
                        allow_module_d=allow_module_d,
                        per_module_usd=per_module_usd,
                        debited_modules=debited_modules,
                        spend_tracker=spend_tracker,
                    )
                )
            except Exception as err:
                _LOGGER.exception("audit scoring raised")
                status = "failed"
                reason = f"exception: {err}"

            # Synthesis phase — Opus 4.7 writer session. Only fires when
            # scoring didn't hard-fail and the caller wired a sync
            # ``client`` (Managed Agents beta surface is synchronous).
            # The phase's own exceptions are caught inside
            # ``run_synthesis_session``; here we swallow one more layer
            # so the scoring findings + report still ship if the
            # synthesis phase itself raises before returning.
            if synthesis_enabled and client is not None and status in {"completed", "partial"}:
                try:
                    _run_synthesis_phase(
                        client=client,
                        async_client=async_client,
                        store=store,
                        run_id=resolved_run_id,
                        enabled_modules=enabled_modules,
                        sampled_ids=sampled_ids,
                        sampled_turns=sum(
                            len(inputs.turns_by_conv.get(cid, [])) for cid in sampled_ids
                        ),
                        progress_log=progress_log,
                    )
                except Exception as err:  # pragma: no cover — defence in depth
                    _LOGGER.exception("synthesis phase raised outside its error handler")
                    progress_log("WARN", f"synthesis phase crashed: {err}")

            # Finalize the audit_run row BEFORE rendering: the report
            # reads status + completed_at from this row, and rendering
            # first would bake in the stale ``running`` state. The
            # findings are already persisted; updating status is just
            # a metadata flip.
            _update_audit_run_status(
                store, resolved_run_id, status, completed=(status == "completed")
            )

            report_path = _render_report_or_log(
                store=store,
                run_id=resolved_run_id,
                output_dir=report_dir,
                progress_log=progress_log,
            )

            findings_written = _count_findings(store, resolved_run_id)

        return AuditResult(
            run_id=resolved_run_id,
            status=status,
            findings_written=findings_written,
            reason=reason,
            report_path=report_path,
        )
    finally:
        lock.release()


async def _run_scoring_loop(
    *,
    store: CorpusStore,
    audit_run_id: str,
    enabled_modules: list[ModuleName],
    sampled_ids: list[str],
    progress_log: Callable[[str, str], None],
    anthropic_client: AsyncAnthropic | None,
    embedding_provider: EmbeddingProvider | None,
    allow_module_d: bool,
    per_module_usd: dict[ModuleName, float],
    debited_modules: set[ModuleName],
    spend_tracker: dict[str, float],
) -> list[dict[str, Any]]:
    """Deterministically invoke each enabled module over the sampled ids.

    Replaces the Managed Agents orchestrator session. Modules run
    sequentially — each module's own fan-out concurrency is internal
    (bounded by its own semaphores, see e.g. Module H). Returns the
    per-module result dicts that ``invoke_module_for_run`` produced,
    in the order the caller supplied ``enabled_modules``.

    Module G (deterministic attribution; no LLM) runs here as the last
    enabled module by convention from the CLI — not as a post-session
    safety net. The loop does not treat it specially.

    This function performs no error recovery: a raised exception from
    ``invoke_module_for_run`` propagates to the caller (the scoring
    loop's job is to invoke; the audit-run status transition is the
    caller's responsibility). Per-conversation / per-turn failures
    inside modules are already captured as ``ModuleError`` entries in
    ``_persist_findings`` and surface in the result dict's
    ``module_errors`` key — those are NOT raised.

    The ``progress_log("INFO", "Running module X")`` line is emitted
    before each invocation. The module itself will emit additional
    progress messages (start + completion) via
    ``invoke_module_for_run``.
    """
    results: list[dict[str, Any]] = []
    for module in enabled_modules:
        progress_log("INFO", f"Running module {module.value}")
        result = await invoke_module_for_run(
            module=module,
            conversation_ids=sampled_ids,
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
        results.append(result)
    return results


async def _execute_scoring(
    *,
    store: CorpusStore,
    registry: ToolRegistry,
    audit_run_id: str,
    enabled_modules: list[ModuleName],
    sampled_ids: list[str],
    progress_log: Callable[[str, str], None],
    anthropic_client: AsyncAnthropic | None,
    embedding_provider: EmbeddingProvider | None,
    allow_module_d: bool,
    per_module_usd: dict[ModuleName, float],
    debited_modules: set[ModuleName],
    spend_tracker: dict[str, float],
) -> tuple[AuditStatus, str]:
    """Execute the deterministic scoring phase and return (status, reason).

    Wraps :func:`_run_scoring_loop` with outcome aggregation. Each
    module returns a status string; the audit is ``completed`` iff
    every module returned a status in :data:`_NON_FAILURE_STATUSES`
    (successful run or documented intentional gate — Module D opt-out,
    Voyage not configured, dry-run without ANTHROPIC_API_KEY).
    Anything else flips the overall status to ``partial``.
    Per-conversation errors (``module_errors`` in the result dict) do
    NOT flip the overall status — those are documented diagnostics,
    not failures. Any exceptions propagate from the loop and are
    caught by :func:`run_audit`.

    The ``registry`` parameter is retained for future synthesis-phase
    wiring (Phase 5 will thread it through to a second coroutine
    ``_execute_synthesis``). It is unused in the scoring phase.
    """
    _ = registry  # retained for synthesis wiring; see docstring.
    results = await _run_scoring_loop(
        store=store,
        audit_run_id=audit_run_id,
        enabled_modules=enabled_modules,
        sampled_ids=sampled_ids,
        progress_log=progress_log,
        anthropic_client=anthropic_client,
        embedding_provider=embedding_provider,
        allow_module_d=allow_module_d,
        per_module_usd=per_module_usd,
        debited_modules=debited_modules,
        spend_tracker=spend_tracker,
    )
    # Intentional gates — Module D opted out, Voyage not configured, dry-run
    # without ANTHROPIC_API_KEY — are documented successful outcomes, not
    # failures. Conflating them with partial was user-facing wrong: an audit
    # without VOYAGE_API_KEY would exit non-zero even though every module ran
    # its correct path.
    non_failing_count = sum(1 for r in results if r.get("status") in _NON_FAILURE_STATUSES)
    if non_failing_count == len(results):
        return "completed", "all modules completed or intentionally skipped"
    failing = [r for r in results if r.get("status") not in _NON_FAILURE_STATUSES]
    partial_reasons = [f"{r.get('module')}={r.get('status')}" for r in failing]
    return "partial", f"partial: {', '.join(partial_reasons)}"


def _run_synthesis_phase(
    *,
    client: Anthropic,
    async_client: AsyncAnthropic,
    store: CorpusStore,
    run_id: str,
    enabled_modules: list[ModuleName],
    sampled_ids: list[str],
    sampled_turns: int,
    progress_log: Callable[[str, str], None],
) -> None:
    """Invoke :func:`lucid.synthesis.run_synthesis_session` and log results.

    Pulled out of ``run_audit`` so the conditional wiring stays readable
    and the asyncio invocation has one owner. Swallows nothing; errors
    from the synthesis session itself are captured by
    :func:`run_synthesis_session` into its return value, not raised.

    ``async_client`` is forwarded to ``run_synthesis_session`` so the
    Sonnet 4.6 post-processor (Phase 5.6b) can structure each Opus
    section's markdown into :class:`~lucid.schemas.SynthesisBlock` lists.

    The import is local so the synthesis package (which pulls in the
    Managed Agents SDK surface) never loads on dry-run / no-synthesis
    paths.
    """
    from lucid.synthesis.run import run_synthesis_session

    behavior_counts = _behavior_counts_for_run(store, run_id)
    corpus_stats = {
        "sampled_conversations": len(sampled_ids),
        "sampled_turns": sampled_turns,
    }
    synthesis_result = asyncio.run(
        run_synthesis_session(
            client=client,
            async_client=async_client,
            store=store,
            audit_run_id=run_id,
            enabled_modules=[m.value for m in enabled_modules],
            behavior_counts=behavior_counts,
            corpus_stats=corpus_stats,
            progress_log=progress_log,
        )
    )
    progress_log(
        "INFO",
        (
            f"Synthesis: {synthesis_result.sections_written} written, "
            f"{synthesis_result.sections_declined} declined, "
            f"{len(synthesis_result.validation_errors)} validation concerns"
        ),
    )


__all__ = [
    "DEFAULT_REPORT_DIR",
    "AuditInputs",
    "AuditResult",
    "LockHeldError",
    "RunError",
    "generate_run_id",
    "run_audit",
]
