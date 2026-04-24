"""Post-generation validators for synthesis agent output.

Three checks run after Opus writes prose but before the section is
considered final:

1. Aggregate-claim lockdown — "across N conversations" type phrases
   must be backed by a tool-call result returning that exact count.
2. Superlative hedging — "consistently" / "always" / "frequently" are
   reserved for behavior counts >= THIN_EVIDENCE_THRESHOLD (default 5).
3. Uncited high-intensity audit — findings with intensity >= 2 that
   are NOT cited in any persisted ReportSection get surfaced for the
   session loop to either re-prompt or append to a "Notable uncited"
   subsection.

The aggregate-lockdown and hedging validators are session-local (run
on prose text + session-captured counts). The uncited audit runs
after all sections are persisted (reads the findings table + all
written sections).

See docs/plans/2026-04-24-synthesis-agent-refactor.md Task 5.2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lucid.schemas import Finding, ReportSection

THIN_EVIDENCE_THRESHOLD = 5

# Patterns for aggregate claims. Captures: numeric, noun (plural-or-singular).
# Singularization strips a trailing 's' crudely; good enough for the nouns
# we ship with (conversation(s), session(s), finding(s), turn(s), event(s),
# interaction(s), exchange(s), message(s)).
_AGGREGATE_PATTERN = re.compile(
    r"\b(\d+)\s+(conversations?|sessions?|findings?|turns?|events?|"
    r"interactions?|exchanges?|messages?)\b",
    re.IGNORECASE,
)

# Superlatives / frequency-strong adjectives that claim a behavior
# happens at scale. The writer prompt asks the agent to hedge under
# THIN_EVIDENCE_THRESHOLD; this regex backs up the rule.
_SUPERLATIVE_PATTERN = re.compile(
    r"\b(consistently|always|frequently|repeatedly|invariably|"
    r"persistently|continually|systematically)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ValidationError:
    """One validation finding — not itself a schema ``Finding``; this is
    post-generation analysis of prose."""

    kind: str
    """One of ``aggregate_unsupported``, ``aggregate_mismatch``,
    ``superlative_thin_evidence``, ``uncited_high_intensity``."""

    detail: str
    """Human-readable specifics surfaced to the regen prompt."""

    section_id: str | None = None
    """Section where it was found (``None`` for cross-section audits)."""


def validate_aggregate_claims(
    prose: str,
    tool_call_counts: dict[str, int],
    *,
    section_id: str | None = None,
) -> list[ValidationError]:
    """Reject aggregate claims in prose that aren't backed by a tool-call count.

    Args:
        prose: The markdown the writer produced.
        tool_call_counts: ``{noun: count}`` map captured from the session's
            tool-call results, e.g. ``{"conversation": 42}`` if
            ``query_corpus`` returned 42 rows.
        section_id: Optional ``section_id`` to attach to emitted errors.

    Returns:
        One ``ValidationError`` per unbacked or mismatched aggregate claim.
    """
    errors: list[ValidationError] = []
    for match in _AGGREGATE_PATTERN.finditer(prose):
        claim_n = int(match.group(1))
        raw_noun = match.group(2).rstrip("sS").lower()  # singularize crudely
        observed = tool_call_counts.get(raw_noun)
        if observed is None:
            errors.append(
                ValidationError(
                    kind="aggregate_unsupported",
                    detail=f"{match.group(0)!r} has no tool-call result backing it",
                    section_id=section_id,
                )
            )
        elif observed != claim_n:
            errors.append(
                ValidationError(
                    kind="aggregate_mismatch",
                    detail=(f"{match.group(0)!r} claims {claim_n} but tool count was {observed}"),
                    section_id=section_id,
                )
            )
    return errors


def validate_superlatives(
    prose: str,
    behavior_counts: dict[str, int],
    *,
    section_id: str | None = None,
) -> list[ValidationError]:
    """Reject superlatives in prose when the cited behavior has thin evidence.

    Heuristic: if the prose contains BOTH a superlative AND a mention of
    a behavior label that appears in ``behavior_counts`` with count
    ``< THIN_EVIDENCE_THRESHOLD``, emit an error. The writer prompt
    already tells the agent to hedge under this threshold; this
    validator catches violations.

    Args:
        prose: The markdown the writer produced.
        behavior_counts: ``{behavior_label: count}`` from a pre-session
            aggregate over the findings table.
        section_id: Optional ``section_id`` to attach to emitted errors.

    Returns:
        One ``ValidationError`` per superlative/thin-evidence combo detected.
    """
    errors: list[ValidationError] = []
    superlatives_found = list(_SUPERLATIVE_PATTERN.finditer(prose))
    if not superlatives_found:
        return errors
    for behavior, count in behavior_counts.items():
        if count >= THIN_EVIDENCE_THRESHOLD:
            continue
        if re.search(rf"\b{re.escape(behavior)}\b", prose, re.IGNORECASE):
            for sup in superlatives_found:
                errors.append(
                    ValidationError(
                        kind="superlative_thin_evidence",
                        detail=(
                            f"Superlative {sup.group(0)!r} combined with "
                            f"behavior {behavior!r} which has only {count} "
                            f"finding(s) (threshold {THIN_EVIDENCE_THRESHOLD}); "
                            f"use hedged language."
                        ),
                        section_id=section_id,
                    )
                )
    return errors


def validate_uncited_high_intensity(
    findings: list[Finding], sections: list[ReportSection]
) -> list[Finding]:
    """Return findings with intensity >= 2 that are NOT cited in any section.

    The writer prompt mandates every intensity-2+ finding appear cited
    somewhere; this audit runs after all sections are persisted and
    surfaces the misses. Caller (Task 5.3 / ``run_synthesis_session``)
    decides what to do — re-prompt, append a 'Notable uncited' section,
    or just log.

    Findings with ``intensity=None`` (e.g. Module G attribution, Module
    H memory support) are never eligible; the cutoff is strict ``>= 2``.
    """
    cited_ids: set[str] = set()
    for section in sections:
        cited_ids.update(section.cited_finding_ids)
    return [f for f in findings if (f.intensity or 0) >= 2 and f.id not in cited_ids]


__all__ = [
    "THIN_EVIDENCE_THRESHOLD",
    "ValidationError",
    "validate_aggregate_claims",
    "validate_superlatives",
    "validate_uncited_high_intensity",
]
