"""Disagreement audit flow for the calibration pipeline.

Pipeline:

1. :func:`compute_disagreements` — walks every ``(conversation, turn,
   behavior)`` cell across raters, records each rater's label, and
   assigns a *score* = ``shannon_entropy × rare_behavior_bonus``. High
   entropy = split rater opinion (e.g. 4 yes / 3 no); rare-behavior
   bonus (``1.5×`` when the behavior's overall prevalence is < 10%)
   surfaces cases where even a single dissenter is informative.

2. :func:`export_for_review` — writes the top-N disagreements as
   :class:`ReviewRow` objects to a JSONL. Each row carries the turn's
   assistant content (so the reviewer has context), all raters' labels,
   and a single empty field ``verified_label`` for the human to fill.

3. The human fills in ``verified_label`` ∈ {``absent``, ``present-1``,
   ``present-2``, ``present-3``} per row (~1 min/row × 50 rows = ~45 min).

4. :func:`import_verified` re-reads the completed JSONL.

5. :func:`apply_verified_overrides` materialises a new rater
   ``human_audit`` whose labels on the audited cells mirror the human
   verdicts. Unaudited cells get no labels from this rater — IAA code
   treats that as "rater did not rate this cell", which multi-rater α
   handles via missing-ratings support.

Why one ``human_audit`` rater rather than overriding existing raters:
keeps provenance clean. Every rater's original labels are preserved for
the "before" IAA; the report shows both "pre-audit" and "post-audit"
numbers so readers can see how much the human intervention moved
things.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from lucid.calibration.data import LabeledTurn
from lucid.modules.base import ModuleCorpus
from lucid.schemas import Role

__all__ = [
    "HUMAN_AUDIT_RATER",
    "Disagreement",
    "ReviewRow",
    "apply_verified_overrides",
    "compute_disagreements",
    "export_for_review",
    "import_verified",
    "rank_disagreements",
]


HUMAN_AUDIT_RATER = "human_audit"

VerifiedLabel = Literal["absent", "present-1", "present-2", "present-3", "unverified"]


class Disagreement(BaseModel):
    """One ``(conversation, turn, behavior)`` cell with per-rater disagreement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str
    turn_id: str
    behavior: str
    turn_content: str  # the assistant reply text (for the reviewer's context)
    rater_labels: dict[str, bool]  # rater_name → behavior present?
    rater_intensities: dict[str, int]  # rater_name → intensity (0 if absent)
    score: float  # entropy × rare-behavior bonus (higher = more informative)


class ReviewRow(BaseModel):
    """One row in the reviewer JSONL. The human fills ``verified_label``."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    turn_id: str
    behavior: str
    turn_content: str
    rater_labels: dict[str, bool]
    rater_intensities: dict[str, int]
    score: float
    verified_label: VerifiedLabel = "unverified"
    human_notes: str | None = None


def _shannon_entropy(presence_count: int, total: int) -> float:
    """Binary entropy in bits. H(0) = H(total) = 0."""
    if total == 0 or presence_count == 0 or presence_count == total:
        return 0.0
    p = presence_count / total
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def _turn_content(corpus: ModuleCorpus, conversation_id: str, turn_id: str) -> str:
    turns = corpus.turns_by_conversation.get(conversation_id, ())
    for t in turns:
        if t.id == turn_id:
            return t.content
    return ""


def _assistant_turn_ids(corpus: ModuleCorpus) -> set[tuple[str, str]]:
    """All ``(conv_id, turn_id)`` pairs where the turn is an assistant turn."""
    result: set[tuple[str, str]] = set()
    for conv_id, turns in corpus.turns_by_conversation.items():
        for turn in turns:
            if turn.role == Role.ASSISTANT:
                result.add((conv_id, turn.id))
    return result


def _behavior_prevalence(
    labels_by_rater: Mapping[str, Sequence[LabeledTurn]],
    behavior: str,
    cells: set[tuple[str, str]],
) -> float:
    """Fraction of cells where AT LEAST ONE rater marked the behavior present."""
    if not cells:
        return 0.0
    seen_positive: set[tuple[str, str]] = set()
    for labels in labels_by_rater.values():
        for lt in labels:
            key = (lt.conversation_id, lt.turn_id)
            if key in cells and behavior in lt.present_behaviors:
                seen_positive.add(key)
    return len(seen_positive) / len(cells)


def compute_disagreements(
    labels_by_rater: Mapping[str, Sequence[LabeledTurn]],
    corpus: ModuleCorpus,
    *,
    behaviors: Sequence[str],
    rare_cutoff: float = 0.10,
    rare_bonus: float = 1.5,
) -> list[Disagreement]:
    """Build the full list of per-cell disagreements with scores.

    Every ``(assistant-turn, behavior)`` cell in the corpus gets a
    ``Disagreement`` row — even unanimous ones (score 0.0). Downstream
    :func:`rank_disagreements` filters and sorts.
    """
    raters = list(labels_by_rater.keys())
    if len(raters) < 2:
        raise ValueError("compute_disagreements needs at least 2 raters")

    cells = _assistant_turn_ids(corpus)

    # Index rater labels by (conv, turn) for O(1) lookup.
    by_rater_cell: dict[str, dict[tuple[str, str], LabeledTurn]] = {
        rater: {
            (lt.conversation_id, lt.turn_id): lt for lt in labels_by_rater[rater]
        }
        for rater in raters
    }

    prevalence: dict[str, float] = {
        b: _behavior_prevalence(labels_by_rater, b, cells) for b in behaviors
    }

    out: list[Disagreement] = []
    for behavior in behaviors:
        bonus = rare_bonus if prevalence[behavior] < rare_cutoff else 1.0
        for conv_id, turn_id in cells:
            rater_labels: dict[str, bool] = {}
            rater_intensities: dict[str, int] = {}
            coverage = 0
            for rater in raters:
                lt = by_rater_cell[rater].get((conv_id, turn_id))
                if lt is None:
                    continue  # rater didn't label this cell
                coverage += 1
                present = behavior in lt.present_behaviors
                rater_labels[rater] = present
                rater_intensities[rater] = lt.intensities.get(behavior, 0)
            if coverage < 2:
                continue  # can't disagree without ≥ 2 raters on the same cell

            presence_count = sum(rater_labels.values())
            entropy = _shannon_entropy(presence_count, coverage)
            out.append(
                Disagreement(
                    conversation_id=conv_id,
                    turn_id=turn_id,
                    behavior=behavior,
                    turn_content=_turn_content(corpus, conv_id, turn_id),
                    rater_labels=rater_labels,
                    rater_intensities=rater_intensities,
                    score=entropy * bonus,
                )
            )
    return out


def rank_disagreements(
    disagreements: Sequence[Disagreement],
    *,
    min_score: float = 0.0,
    top_n: int | None = None,
) -> list[Disagreement]:
    """Sort by score descending. ``min_score=0.0`` drops unanimous cells."""
    filtered = [d for d in disagreements if d.score > min_score]
    filtered.sort(key=lambda d: d.score, reverse=True)
    if top_n is not None:
        return filtered[:top_n]
    return filtered


def export_for_review(
    disagreements: Sequence[Disagreement],
    path: Path,
    *,
    top_n: int = 50,
) -> int:
    """Write the top-N disagreements as reviewer JSONL. Returns row count."""
    ranked = rank_disagreements(disagreements, min_score=0.0, top_n=top_n)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in ranked:
            row = ReviewRow(
                conversation_id=d.conversation_id,
                turn_id=d.turn_id,
                behavior=d.behavior,
                turn_content=d.turn_content,
                rater_labels=d.rater_labels,
                rater_intensities=d.rater_intensities,
                score=d.score,
            )
            f.write(row.model_dump_json() + "\n")
    return len(ranked)


def import_verified(path: Path) -> list[ReviewRow]:
    """Load a completed reviewer JSONL.

    Rows where ``verified_label == 'unverified'`` are kept but will not
    produce overrides downstream — the reviewer is permitted to skip
    items they can't judge.
    """
    rows: list[ReviewRow] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(ReviewRow.model_validate_json(line))
        except Exception as err:
            raise ValueError(f"{path}:{lineno}: invalid review row: {err}") from err
    return rows


_INTENSITY_FROM_LABEL: dict[str, int] = {
    "present-1": 1,
    "present-2": 2,
    "present-3": 3,
}


def apply_verified_overrides(
    verified: Sequence[ReviewRow],
    *,
    labeled_at: datetime | None = None,
    rater_name: str = HUMAN_AUDIT_RATER,
) -> list[LabeledTurn]:
    """Materialise a ``human_audit`` rater from completed review rows.

    One LabeledTurn per ``(conv, turn)`` touched by the review — the
    ``present_behaviors`` set is the union of behaviors marked
    ``present-*`` for that cell across the review. Cells where every
    row is ``absent`` or ``unverified`` still produce a LabeledTurn
    (potentially empty) so IAA sees the human's ratings as a complete
    set on the audited items.
    """
    ts = labeled_at or datetime.now(UTC)

    # Group by (conv, turn); track both sets of actioned behaviors.
    per_turn_present: dict[tuple[str, str], set[str]] = {}
    per_turn_intensity: dict[tuple[str, str], dict[str, int]] = {}
    audited_turns: set[tuple[str, str]] = set()

    for row in verified:
        key = (row.conversation_id, row.turn_id)
        if row.verified_label == "unverified":
            continue
        audited_turns.add(key)
        if row.verified_label == "absent":
            continue
        intensity = _INTENSITY_FROM_LABEL.get(row.verified_label)
        if intensity is None:
            continue  # pydantic already validated; defensive
        per_turn_present.setdefault(key, set()).add(row.behavior)
        per_turn_intensity.setdefault(key, {})[row.behavior] = intensity

    results: list[LabeledTurn] = []
    for key in sorted(audited_turns):
        present = frozenset(per_turn_present.get(key, set()))
        intensities = dict(per_turn_intensity.get(key, {}))
        conv_id, turn_id = key
        results.append(
            LabeledTurn(
                conversation_id=conv_id,
                turn_id=turn_id,
                present_behaviors=present,
                intensities=intensities,
                labeler=rater_name,
                labeled_at=ts,
            )
        )
    return results
