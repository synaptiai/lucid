"""Tests for :mod:`lucid.calibration.audit`.

Covers disagreement computation (entropy + rare-behavior bonus), export
round-trip, human-verified import, and override materialisation.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lucid.calibration.audit import (
    HUMAN_AUDIT_RATER,
    Disagreement,
    ReviewRow,
    apply_verified_overrides,
    compute_disagreements,
    export_for_review,
    import_verified,
    rank_disagreements,
)
from lucid.calibration.data import LabeledTurn
from lucid.modules.base import ModuleCorpus
from lucid.schemas import Conversation, Role, Source, Turn


def _turn(conv_id: str, idx: int, role: Role, content: str = "") -> Turn:
    return Turn(
        id=f"{conv_id}-t{idx}",
        conversation_id=conv_id,
        index=idx,
        role=role,
        content=content or f"turn {idx}",
    )


def _conv(cid: str, turns: list[Turn]) -> Conversation:
    return Conversation(
        id=cid,
        source=Source.CLAUDE_AI,
        source_path="/tmp",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        turn_count=len(turns),
    )


def _corpus(conv_ids: list[str], turns_by_conv: dict[str, list[Turn]]) -> ModuleCorpus:
    return ModuleCorpus(
        conversations={cid: _conv(cid, turns_by_conv[cid]) for cid in conv_ids},
        turns_by_conversation=turns_by_conv,
        audit_run_id="run-test",
    )


def _lt(
    conv_id: str,
    turn_id: str,
    *,
    labeler: str,
    present: frozenset[str] = frozenset(),
    intensities: dict[str, int] | None = None,
) -> LabeledTurn:
    return LabeledTurn(
        conversation_id=conv_id,
        turn_id=turn_id,
        present_behaviors=present,
        intensities=intensities or {},
        labeler=labeler,
        labeled_at=datetime(2026, 4, 22, tzinfo=UTC),
    )


# ──────────────────────────────────────────────────────────────────────────
# compute_disagreements
# ──────────────────────────────────────────────────────────────────────────


def test_compute_disagreements_scores_unanimous_cells_as_zero() -> None:
    conv = "c1"
    turns = [_turn(conv, 0, Role.USER), _turn(conv, 1, Role.ASSISTANT, "assistant reply")]
    corpus = _corpus([conv], {conv: turns})
    labels = {
        "a": [_lt(conv, "c1-t1", labeler="a", present=frozenset({"sycophancy"}))],
        "b": [_lt(conv, "c1-t1", labeler="b", present=frozenset({"sycophancy"}))],
        "c": [_lt(conv, "c1-t1", labeler="c", present=frozenset({"sycophancy"}))],
    }
    disagreements = compute_disagreements(labels, corpus, behaviors=["sycophancy"])
    assert len(disagreements) == 1
    assert disagreements[0].score == 0.0  # unanimous


def test_compute_disagreements_splits_max_entropy() -> None:
    """2-yes / 2-no split → entropy = 1.0 bit. No rare-behavior bonus (prev=50%)."""
    conv = "c1"
    turns = [_turn(conv, 0, Role.USER), _turn(conv, 1, Role.ASSISTANT)]
    corpus = _corpus([conv], {conv: turns})
    present = frozenset({"pushback"})
    labels = {
        "a": [_lt(conv, "c1-t1", labeler="a", present=present)],
        "b": [_lt(conv, "c1-t1", labeler="b", present=present)],
        "c": [_lt(conv, "c1-t1", labeler="c", present=frozenset())],
        "d": [_lt(conv, "c1-t1", labeler="d", present=frozenset())],
    }
    disagreements = compute_disagreements(labels, corpus, behaviors=["pushback"])
    assert len(disagreements) == 1
    # prevalence = 1.0 (the single cell has a 'present' label) → no bonus
    assert disagreements[0].score == pytest.approx(1.0)


def test_compute_disagreements_rare_behavior_bonus_applied() -> None:
    """One in-10 cell has the behavior; prevalence = 10% = cutoff → no bonus.
    Make it 1-in-20 → below cutoff → bonus applied."""
    turns_by_conv: dict[str, list[Turn]] = {}
    labels_a: list[LabeledTurn] = []
    labels_b: list[LabeledTurn] = []
    for i in range(20):
        cid = f"c{i}"
        t0 = _turn(cid, 0, Role.USER)
        t1 = _turn(cid, 1, Role.ASSISTANT)
        turns_by_conv[cid] = [t0, t1]
        if i == 0:
            # rater A sees rare behavior; rater B disagrees
            labels_a.append(
                _lt(cid, f"{cid}-t1", labeler="a", present=frozenset({"ritualization"}))
            )
            labels_b.append(_lt(cid, f"{cid}-t1", labeler="b", present=frozenset()))
        else:
            labels_a.append(_lt(cid, f"{cid}-t1", labeler="a", present=frozenset()))
            labels_b.append(_lt(cid, f"{cid}-t1", labeler="b", present=frozenset()))
    corpus = _corpus(list(turns_by_conv.keys()), turns_by_conv)
    labels = {"a": labels_a, "b": labels_b}

    disagreements = compute_disagreements(
        labels, corpus, behaviors=["ritualization"], rare_cutoff=0.10, rare_bonus=1.5
    )
    # One disagreement cell (the one where a said present, b said absent).
    nonzero = [d for d in disagreements if d.score > 0]
    assert len(nonzero) == 1
    # binary entropy at 1/2 = 1.0, × 1.5 bonus = 1.5
    assert nonzero[0].score == pytest.approx(1.5)


def test_compute_disagreements_rejects_single_rater() -> None:
    corpus = _corpus([], {})
    with pytest.raises(ValueError, match="at least 2 raters"):
        compute_disagreements({"a": []}, corpus, behaviors=["x"])


def test_compute_disagreements_skips_cells_with_one_rater() -> None:
    """If only one rater covers a cell, no disagreement computed."""
    conv = "c1"
    turns = [_turn(conv, 0, Role.USER), _turn(conv, 1, Role.ASSISTANT)]
    corpus = _corpus([conv], {conv: turns})
    labels = {
        "a": [_lt(conv, "c1-t1", labeler="a", present=frozenset({"pushback"}))],
        "b": [],  # rater B didn't label anything
    }
    disagreements = compute_disagreements(labels, corpus, behaviors=["pushback"])
    assert disagreements == []


def test_compute_disagreements_ignores_user_turns() -> None:
    conv = "c1"
    turns = [_turn(conv, 0, Role.USER), _turn(conv, 1, Role.USER)]  # no assistant
    corpus = _corpus([conv], {conv: turns})
    labels = {"a": [], "b": []}
    disagreements = compute_disagreements(labels, corpus, behaviors=["pushback"])
    assert disagreements == []


# ──────────────────────────────────────────────────────────────────────────
# rank_disagreements
# ──────────────────────────────────────────────────────────────────────────


def test_rank_disagreements_sorts_descending_and_drops_zero() -> None:
    ds = [
        Disagreement(
            conversation_id="c",
            turn_id="t",
            behavior="b",
            turn_content="x",
            rater_labels={"a": True, "b": True},
            rater_intensities={"a": 2, "b": 2},
            score=0.0,
        ),
        Disagreement(
            conversation_id="c",
            turn_id="t",
            behavior="b",
            turn_content="y",
            rater_labels={"a": True, "b": False},
            rater_intensities={"a": 2, "b": 0},
            score=1.5,
        ),
        Disagreement(
            conversation_id="c",
            turn_id="t",
            behavior="b",
            turn_content="z",
            rater_labels={"a": True, "b": False},
            rater_intensities={"a": 2, "b": 0},
            score=0.8,
        ),
    ]
    ranked = rank_disagreements(ds)
    assert [d.score for d in ranked] == [1.5, 0.8]


def test_rank_disagreements_top_n() -> None:
    ds = [
        Disagreement(
            conversation_id="c",
            turn_id=f"t{i}",
            behavior="b",
            turn_content=f"x{i}",
            rater_labels={"a": True, "b": False},
            rater_intensities={"a": 2, "b": 0},
            score=float(i),
        )
        for i in range(1, 6)
    ]
    ranked = rank_disagreements(ds, top_n=3)
    assert [d.score for d in ranked] == [5.0, 4.0, 3.0]


# ──────────────────────────────────────────────────────────────────────────
# export_for_review + import_verified round-trip
# ──────────────────────────────────────────────────────────────────────────


def test_export_and_import_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "disagreements.jsonl"
    ds = [
        Disagreement(
            conversation_id="c1",
            turn_id="c1-t1",
            behavior="sycophancy",
            turn_content="You're brilliant!",
            rater_labels={"a": True, "b": False, "c": True},
            rater_intensities={"a": 2, "b": 0, "c": 1},
            score=0.918,  # binary entropy 2/3 ≈ 0.918
        ),
    ]
    count = export_for_review(ds, path, top_n=10)
    assert count == 1
    rows = import_verified(path)
    assert len(rows) == 1
    assert rows[0].behavior == "sycophancy"
    assert rows[0].verified_label == "unverified"  # default before human fills
    assert rows[0].turn_content == "You're brilliant!"


def test_import_verified_reports_line_on_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("not-json-at-all\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        import_verified(path)


def test_export_for_review_skips_unanimous_cells(tmp_path: Path) -> None:
    """Score-0 rows should not appear in the review JSONL."""
    path = tmp_path / "d.jsonl"
    ds = [
        Disagreement(
            conversation_id="c",
            turn_id="t",
            behavior="b",
            turn_content="x",
            rater_labels={"a": True, "b": True},
            rater_intensities={"a": 2, "b": 2},
            score=0.0,
        ),
        Disagreement(
            conversation_id="c",
            turn_id="t2",
            behavior="b",
            turn_content="y",
            rater_labels={"a": True, "b": False},
            rater_intensities={"a": 2, "b": 0},
            score=1.0,
        ),
    ]
    count = export_for_review(ds, path)
    assert count == 1  # score-0 row dropped


# ──────────────────────────────────────────────────────────────────────────
# apply_verified_overrides
# ──────────────────────────────────────────────────────────────────────────


def test_apply_overrides_emits_human_audit_rater() -> None:
    verified = [
        ReviewRow(
            conversation_id="c1",
            turn_id="c1-t1",
            behavior="sycophancy",
            turn_content="x",
            rater_labels={"a": True, "b": False},
            rater_intensities={"a": 2, "b": 0},
            score=1.0,
            verified_label="present-3",
        ),
    ]
    overrides = apply_verified_overrides(verified)
    assert len(overrides) == 1
    lt = overrides[0]
    assert lt.labeler == HUMAN_AUDIT_RATER
    assert "sycophancy" in lt.present_behaviors
    assert lt.intensities["sycophancy"] == 3


def test_apply_overrides_unions_multiple_behaviors_on_same_turn() -> None:
    verified = [
        ReviewRow(
            conversation_id="c1",
            turn_id="c1-t1",
            behavior="sycophancy",
            turn_content="x",
            rater_labels={},
            rater_intensities={},
            score=0.0,
            verified_label="present-2",
        ),
        ReviewRow(
            conversation_id="c1",
            turn_id="c1-t1",
            behavior="confident-bullshitting",
            turn_content="x",
            rater_labels={},
            rater_intensities={},
            score=0.0,
            verified_label="present-1",
        ),
    ]
    overrides = apply_verified_overrides(verified)
    assert len(overrides) == 1
    lt = overrides[0]
    assert lt.present_behaviors == frozenset({"sycophancy", "confident-bullshitting"})
    assert lt.intensities == {"sycophancy": 2, "confident-bullshitting": 1}


def test_apply_overrides_absent_label_records_empty_rating() -> None:
    """A turn where every behavior was marked ``absent`` still gets a
    LabeledTurn (empty present_behaviors) — the human DID audit this turn
    and saw nothing."""
    verified = [
        ReviewRow(
            conversation_id="c1",
            turn_id="c1-t1",
            behavior="sycophancy",
            turn_content="x",
            rater_labels={"a": True, "b": False},
            rater_intensities={"a": 2, "b": 0},
            score=1.0,
            verified_label="absent",
        ),
    ]
    overrides = apply_verified_overrides(verified)
    assert len(overrides) == 1
    assert overrides[0].present_behaviors == frozenset()
    assert overrides[0].intensities == {}


def test_apply_overrides_skips_unverified_rows() -> None:
    """A row the human didn't fill in yet produces no override."""
    verified = [
        ReviewRow(
            conversation_id="c1",
            turn_id="c1-t1",
            behavior="sycophancy",
            turn_content="x",
            rater_labels={},
            rater_intensities={},
            score=1.0,
            verified_label="unverified",
        ),
    ]
    overrides = apply_verified_overrides(verified)
    assert overrides == []


def test_apply_overrides_custom_rater_name() -> None:
    verified = [
        ReviewRow(
            conversation_id="c1",
            turn_id="c1-t1",
            behavior="x",
            turn_content="x",
            rater_labels={},
            rater_intensities={},
            score=0.0,
            verified_label="absent",
        ),
    ]
    overrides = apply_verified_overrides(verified, rater_name="daniel")
    assert overrides[0].labeler == "daniel"


# ──────────────────────────────────────────────────────────────────────────
# Entropy math sanity
# ──────────────────────────────────────────────────────────────────────────


def test_shannon_entropy_known_values() -> None:
    from lucid.calibration.audit import _shannon_entropy

    assert _shannon_entropy(0, 3) == 0.0  # unanimous no
    assert _shannon_entropy(3, 3) == 0.0  # unanimous yes
    # 2/4 split = H=1
    assert _shannon_entropy(2, 4) == pytest.approx(1.0)
    # 1/4 split = -0.25*log2(0.25) - 0.75*log2(0.75)
    expected = -(0.25 * math.log2(0.25) + 0.75 * math.log2(0.75))
    assert _shannon_entropy(1, 4) == pytest.approx(expected)
