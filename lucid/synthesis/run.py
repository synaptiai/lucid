"""Synthesis phase orchestration.

Runs AFTER the deterministic scoring phase. Spins up one
:class:`SynthesisSession` per audit run, kicks off Opus 4.7 writing
via the ``synthesis/v1`` prompt, collects the persisted
:class:`~lucid.schemas.ReportSection` rows, and applies post-generation
validators (aggregate-claim lockdown, superlative hedging,
uncited-high-intensity audit).

Task 5.3 (this file): integration.

Task 5.4 (this file + :mod:`lucid.synthesis.tools`): regen-cap loop.
The Managed Agents event loop already provides retries implicitly —
when ``write_report_section`` returns ``{"error": "unknown_ids", ...}``,
the payload flows back to the writer via ``user.custom_tool_result``
and the writer can re-issue the call with corrected ids. We enforce a
ceiling in :func:`~lucid.synthesis.tools.make_write_report_section_tool`
so pathological failures don't loop forever.

Phase 5.6b (this file): Sonnet 4.6 post-processor that structures Opus
markdown into :class:`~lucid.schemas.SynthesisSectionOutput` blocks via
``messages.parse()``. Wired into :func:`run_synthesis_session` via the
``async_client`` kwarg — when wired, each populated section gets
enriched with :class:`~lucid.schemas.SynthesisBlock` lists + a
``citation_confidence`` score and upserted back to the DB. When
``async_client`` is ``None`` (tests, dry-run paths) the post-process
is skipped cleanly. Failures degrade gracefully — the report still
renders from raw markdown via the ``markdown_with_citations`` Jinja
filter (Checkpoint B).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lucid.prompts import load_prompt
from lucid.synthesis import SYNTHESIS_PROMPT_VERSION
from lucid.synthesis.post_process import post_process_sections
from lucid.synthesis.session import (
    SynthesisConfig,
    SynthesisOutcome,
    SynthesisSession,
)
from lucid.synthesis.tools import build_synthesis_registry
from lucid.synthesis.validators import (
    ValidationError,
    validate_aggregate_claims,
    validate_superlatives,
    validate_uncited_high_intensity,
)

if TYPE_CHECKING:  # pragma: no cover
    from anthropic import AsyncAnthropic

    from lucid.store.sqlite import CorpusStore

_LOGGER = logging.getLogger(__name__)


# Section ids the v1 synthesis prompt writes. Three fixed hero sections
# plus one narrative per enabled behavior module (A, B, C, D, E, F, H).
# Module G (deterministic attribution) has no narrative — it's metadata
# that shows up in the month/model charts.
_CORE_SECTIONS: tuple[str, ...] = (
    "exec_summary",
    "top_3_actions",
    "headline_findings",
)
_PER_MODULE_TEMPLATE = "module_{letter}_narrative"


@dataclass
class SynthesisRunResult:
    """Outcome of the synthesis phase.

    Intentionally decoupled from :class:`SynthesisOutcome`: the outcome
    is a session-level object (events received, tool-call count, etc.),
    while this result is the phase-level summary the audit caller uses
    for logging + downstream rendering.
    """

    sections_written: int = 0
    sections_declined: int = 0
    validation_errors: list[ValidationError] = field(default_factory=list)
    uncited_high_intensity_ids: list[str] = field(default_factory=list)
    session_outcome: SynthesisOutcome | None = None
    reason: str = ""


def _target_section_ids(enabled_modules: list[str]) -> list[str]:
    """Build the section-id list the writer is asked to produce.

    Core sections + one narrative per enabled behavior module (A, B, C,
    D, E, F, H). Module G gets no narrative — it's attribution metadata
    rendered directly from Module G's deterministic findings.
    """
    per_module = [
        _PER_MODULE_TEMPLATE.format(letter=letter.lower())
        for letter in enabled_modules
        if letter.upper() != "G"
    ]
    return list(_CORE_SECTIONS) + per_module


def _build_kickoff_message(
    *,
    audit_run_id: str,
    target_sections: list[str],
    behavior_counts: dict[str, int],
    corpus_stats: dict[str, Any],
) -> str:
    """Compose the initial user message for the synthesis session.

    Embeds three blocks the prompt hooks into explicitly:

    * **Target sections** — one ``write_report_section`` call per id.
    * **Behavior counts** — the thin-evidence hedging guard uses these.
    * **Corpus stats** — the aggregate-claim lockdown guard uses these.

    Rendering as keyword-listed data (not prose) keeps the guards
    grounded; Opus sees them as a data appendix rather than a
    paraphrasable suggestion.
    """
    lines = [
        f"Write narrative sections for Lucid audit run {audit_run_id}.",
        "",
        "Target sections (write one write_report_section call per section):",
    ]
    for sid in target_sections:
        lines.append(f"  - {sid}")
    lines.append("")
    lines.append("Behavior counts (for thin-evidence hedging):")
    if behavior_counts:
        for label, count in sorted(behavior_counts.items()):
            lines.append(f"  - {label}: {count}")
    else:
        lines.append("  - (no behaviors scored)")
    lines.append("")
    lines.append("Corpus stats (for aggregate validation):")
    for k, v in sorted(corpus_stats.items()):
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append(
        "Start by calling get_findings to understand the run. Then write "
        "each target section via write_report_section with inline [F:id] "
        "and [T:id] citation tokens. If a section has insufficient "
        "evidence (fewer than 3 qualifying findings), call "
        "write_report_section with insufficient_evidence=true and "
        "decline_reason set. When every target section has a "
        "write_report_section call (successful or declined), emit a "
        "final log_progress line and stop."
    )
    return "\n".join(lines)


async def run_synthesis_session(
    *,
    client: Any,
    store: CorpusStore,
    audit_run_id: str,
    enabled_modules: list[str],
    behavior_counts: dict[str, int],
    corpus_stats: dict[str, Any],
    progress_log: Callable[[str, str], None],
    max_regen_attempts: int = 2,
    async_client: AsyncAnthropic | None = None,
) -> SynthesisRunResult:
    """Execute the synthesis phase for an already-scored audit run.

    1. Builds the read-only + ``write_report_section`` tool registry
       with a per-section retry cap (Task 5.4).
    2. Loads ``prompts/synthesis/v1.md`` as the writer's system prompt.
    3. Spins up :class:`SynthesisSession`; runs it until idle/stalled.
    4. Reads back the persisted :class:`~lucid.schemas.ReportSection`
       rows.
    5. (Phase 5.6b) If ``async_client`` is wired, runs the Sonnet 4.6
       post-processor over each populated section to structure the raw
       Opus markdown into :class:`~lucid.schemas.SynthesisBlock` lists +
       a per-section ``citation_confidence`` score. Enriched sections
       are upserted back to the DB; failures log a warning and leave
       the original section untouched (still renderable from markdown).
    6. Applies aggregate + superlative + uncited-high-intensity
       validators over the (possibly enriched) sections + the run's
       findings.
    7. Returns a :class:`SynthesisRunResult` with counters + any
       validation concerns. Never raises: the session may partially
       fail, the validators may emit warnings, neither should abort
       the enclosing audit run.
    """
    result = SynthesisRunResult()

    # Build the tool registry bound to this run. Regen cap flows
    # through to the write_report_section handler.
    registry = build_synthesis_registry(
        store=store,
        audit_run_id=audit_run_id,
        progress_log=progress_log,
        max_regen_attempts=max_regen_attempts,
    )

    # Load the synthesis prompt. A missing/invalid prompt file is a
    # hard stop for the phase, but must not abort the overall audit
    # (the scoring findings are already persisted and render-able).
    try:
        prompt = load_prompt("synthesis", "v1")
    except Exception as err:
        _LOGGER.exception("synthesis prompt load failed")
        result.reason = f"prompt_load_failed: {err}"
        return result

    target_sections = _target_section_ids(enabled_modules)
    kickoff = _build_kickoff_message(
        audit_run_id=audit_run_id,
        target_sections=target_sections,
        behavior_counts=behavior_counts,
        corpus_stats=corpus_stats,
    )

    config = SynthesisConfig(
        run_id=audit_run_id,
        prompt_version=SYNTHESIS_PROMPT_VERSION,
        system_prompt=prompt.padded_body,
    )
    session = SynthesisSession(client=client, registry=registry, config=config)

    progress_log(
        "INFO",
        f"Starting synthesis for {len(target_sections)} sections",
    )
    try:
        session_outcome = await session.run(kickoff)
        result.session_outcome = session_outcome
    except Exception as err:
        _LOGGER.exception("synthesis session raised")
        result.reason = f"session_exception: {err}"
        return result

    # Tally what the session actually persisted. The session's own
    # counters are advisory (may miss late writes); the DB is truth.
    sections = store.fetch_report_sections_for_run(audit_run_id)
    result.sections_written = sum(1 for s in sections if not s.insufficient_evidence)
    result.sections_declined = sum(1 for s in sections if s.insufficient_evidence)
    progress_log(
        "INFO",
        f"Synthesis complete: {result.sections_written} written, "
        f"{result.sections_declined} declined",
    )

    # Two consumers read tool-call counts, each with its own key shape:
    #
    # * The Python ``validate_aggregate_claims`` validator singularizes
    #   the matched noun (``rstrip("sS").lower()``) before the dict
    #   lookup, so it expects SINGULAR keys.
    # * The Sonnet post-processor prompt
    #   (``prompts/synthesis_validator/v1.md``) specifies the Input
    #   shape ``{"conversations": ..., "findings": ...}`` — PLURAL.
    #   Singular keys silently pass type-checks but bypass Sonnet's
    #   aggregate-extraction heuristic, inflating citation_confidence.
    #
    # Build both; hand each consumer the shape it expects.
    sampled_conv_count = int(corpus_stats.get("sampled_conversations") or 0)
    sampled_turn_count = int(corpus_stats.get("sampled_turns") or 0)
    finding_count = sum(behavior_counts.values())
    tool_call_counts: dict[str, int] = {
        "conversation": sampled_conv_count,
        "turn": sampled_turn_count,
        "finding": finding_count,
    }
    sonnet_tool_call_counts: dict[str, int] = {
        "conversations": sampled_conv_count,
        "turns": sampled_turn_count,
        "findings": finding_count,
    }

    # Phase 5.6b — Sonnet 4.6 post-process. Structures each populated
    # section's raw Opus markdown into SynthesisBlocks + a citation-
    # confidence score. Declined sections pass through untouched.
    # Failures log a warning and leave the section unchanged — the
    # report still renders from markdown alone (graceful degradation).
    if async_client is not None and sections:
        findings_for_validation = store.fetch_findings_for_run(audit_run_id)
        valid_finding_ids = [f.id for f in findings_for_validation]
        # valid_turn_ids: all turn ids belonging to conversations that
        # have findings in this audit run. This is the superset Opus
        # could legitimately have cited via get_turn_window — giving it
        # to Sonnet means Sonnet can drop any cite to a turn id outside
        # this set (whether Opus hallucinated it or simply cited a turn
        # from a different run). An earlier approach of union-ing
        # ``section.cited_turn_ids`` was circular: Sonnet could never
        # drop what Opus already claimed.
        sampled_conv_ids = {f.conversation_id for f in findings_for_validation if f.conversation_id}
        if sampled_conv_ids:
            placeholders = ",".join("?" for _ in sampled_conv_ids)
            turn_rows = store.fetchall(
                f"SELECT id FROM turns WHERE conversation_id IN ({placeholders})",
                tuple(sampled_conv_ids),
            )
            valid_turn_ids = [r["id"] for r in turn_rows]
        else:
            # No findings reference a conversation — Sonnet gets an
            # empty cite universe. Post-process is advisory anyway.
            valid_turn_ids = []
        try:
            enriched = await post_process_sections(
                async_client=async_client,
                sections=sections,
                valid_finding_ids=valid_finding_ids,
                valid_turn_ids=valid_turn_ids,
                tool_call_counts=sonnet_tool_call_counts,
            )
            for original, updated in zip(sections, enriched, strict=True):
                if (
                    updated.blocks != original.blocks
                    or updated.citation_confidence != original.citation_confidence
                ):
                    store.upsert_report_section(updated)
            # Downstream validators read the (possibly enriched) list.
            sections = enriched
            progress_log(
                "INFO",
                f"Sonnet post-processed {sum(1 for s in enriched if s.blocks)} sections",
            )
        except Exception as err:  # outer guard; per-section failures are already
            # swallowed inside post_process_sections, but defence in depth keeps
            # one pathological crash from aborting the validators.
            _LOGGER.exception("Sonnet post-process batch failed")
            progress_log("WARN", f"Sonnet post-process failed: {err} (continuing)")
    all_errors: list[ValidationError] = []
    for section in sections:
        if section.insufficient_evidence:
            continue
        all_errors.extend(
            validate_aggregate_claims(
                section.markdown,
                tool_call_counts,
                section_id=section.section_id,
            )
        )
        all_errors.extend(
            validate_superlatives(
                section.markdown,
                behavior_counts,
                section_id=section.section_id,
            )
        )
    result.validation_errors = all_errors

    # Uncited-high-intensity audit reads the full findings list for the
    # run — no section filtering, since a high-intensity finding might
    # have been routed into any section (or missed entirely).
    findings = store.fetch_findings_for_run(audit_run_id)
    uncited = validate_uncited_high_intensity(findings, sections)
    result.uncited_high_intensity_ids = [f.id for f in uncited]
    if uncited:
        progress_log(
            "INFO",
            f"{len(uncited)} high-intensity finding(s) were not cited in any section",
        )

    result.reason = result.session_outcome.reason if result.session_outcome else "no_session"
    return result


__all__ = [
    "SynthesisRunResult",
    "run_synthesis_session",
]
