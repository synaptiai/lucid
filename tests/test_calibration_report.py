"""Tests for :mod:`lucid.calibration.report`.

Covers ``compute_calibration`` (per-behavior metric assembly + primary
metric selection) and the two renderers. No LLM involved; the inputs
are seeded ``LabeledTurn`` objects.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from rich.console import Console

from lucid.calibration.data import LabeledTurn
from lucid.calibration.report import (
    CalibrationReport,
    compute_calibration,
    render_markdown,
    render_rich_table,
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
        labeled_at=datetime(2026, 4, 21, tzinfo=UTC),
    )


def _build_labels(
    n_items: int,
    agreement: float,
    behavior: str,
) -> tuple[list[LabeledTurn], list[LabeledTurn], list[tuple[str, str]]]:
    """Build two rater sets where the given behavior is present on every
    item for the human, and present on ``agreement`` fraction for the judge.
    Returns (human, judge, keys)."""
    human: list[LabeledTurn] = []
    judge: list[LabeledTurn] = []
    keys: list[tuple[str, str]] = []
    for i in range(n_items):
        key = (f"c{i}", f"t{i}")
        keys.append(key)
        human.append(
            _lt(
                *key,
                labeler="human",
                present=frozenset({behavior}),
                intensities={behavior: 2},
            )
        )
        judge_present = i / max(n_items - 1, 1) <= agreement
        judge.append(
            _lt(
                *key,
                labeler="judge",
                present=frozenset({behavior}) if judge_present else frozenset(),
                intensities={behavior: 2} if judge_present else {},
            )
        )
    return human, judge, keys


def test_compute_calibration_returns_report_with_expected_shape() -> None:
    human, judge, keys = _build_labels(n_items=30, agreement=0.9, behavior="sycophancy")

    report = compute_calibration(
        labels_by_rater={"human": human, "judge": judge},
        behaviors=["sycophancy", "pushback"],
        turn_keys=keys,
        module="A",
        prompt_version="v1",
        n_bootstrap=199,
        seed=7,
    )

    assert isinstance(report, CalibrationReport)
    assert report.module == "A"
    assert report.prompt_version == "v1"
    assert report.n_items_total == 30
    assert {b.behavior for b in report.behaviors} == {"sycophancy", "pushback"}


def test_primary_metric_picks_alpha_when_prevalence_is_balanced() -> None:
    """All behaviors roughly balanced → Krippendorff α wins."""
    human, judge, keys = _build_labels(n_items=40, agreement=0.85, behavior="sycophancy")
    # Swap half of both raters to the opposite class for "sycophancy" so
    # prevalence ≈ 0.5 rather than 1.0.
    for i in range(20):
        human[i] = _lt(*keys[i], labeler="human")  # empty
        judge[i] = _lt(*keys[i], labeler="judge")

    report = compute_calibration(
        labels_by_rater={"human": human, "judge": judge},
        behaviors=["sycophancy"],
        turn_keys=keys,
        module="A",
        prompt_version="v1",
        n_bootstrap=99,
        seed=1,
    )
    # One behavior with prevalence ≈ 0.5; not extreme → alpha primary.
    assert report.primary_metric == "alpha"


def test_primary_metric_picks_ac1_when_prevalences_are_skewed() -> None:
    """If 3+ behaviors have extreme (<10% or >90%) prevalence → AC1 primary."""
    keys = [(f"c{i}", f"t{i}") for i in range(40)]
    # Rare behaviors present on only 2 of 40 items for both raters.
    rare = frozenset({"b1"})
    rare_intens = {"b1": 1}
    human: list[LabeledTurn] = []
    judge: list[LabeledTurn] = []
    for i, k in enumerate(keys):
        present_rare = i < 2
        b_present = rare if present_rare else frozenset()
        b_intens = rare_intens if present_rare else {}
        human.append(_lt(*k, labeler="human", present=b_present, intensities=b_intens))
        judge.append(_lt(*k, labeler="judge", present=b_present, intensities=b_intens))

    report = compute_calibration(
        labels_by_rater={"human": human, "judge": judge},
        behaviors=["b1", "b2", "b3", "b4"],
        turn_keys=keys,
        module="A",
        prompt_version="v1",
        n_bootstrap=99,
        seed=2,
    )
    # b1 has prev 2/40 = 0.05 (extreme). b2/b3/b4 have prev 0 (also extreme).
    # 4 skewed behaviors ≥ 3 → AC1 primary.
    assert report.primary_metric == "ac1"


def test_compute_calibration_rejects_single_rater() -> None:
    human, _judge, keys = _build_labels(n_items=10, agreement=1.0, behavior="x")
    with pytest.raises(ValueError, match="at least 2 raters"):
        compute_calibration(
            labels_by_rater={"human": human},
            behaviors=["x"],
            turn_keys=keys,
            module="A",
            prompt_version="v1",
            n_bootstrap=10,
        )


def test_compute_calibration_rejects_empty_keys() -> None:
    with pytest.raises(ValueError, match="turn_keys must be non-empty"):
        compute_calibration(
            labels_by_rater={"h": [], "j": []},
            behaviors=["x"],
            turn_keys=[],
            module="A",
            prompt_version="v1",
            n_bootstrap=10,
        )


def test_render_markdown_contains_all_behaviors_and_primary_metric() -> None:
    human, judge, keys = _build_labels(n_items=20, agreement=0.9, behavior="sycophancy")
    report = compute_calibration(
        labels_by_rater={"human": human, "judge": judge},
        behaviors=["sycophancy", "pushback"],
        turn_keys=keys,
        module="A",
        prompt_version="v1",
        n_bootstrap=99,
        seed=3,
    )
    md = render_markdown(report)
    assert "Module A" in md
    assert "prompt v1" in md
    assert "sycophancy" in md
    assert "pushback" in md
    assert f"**{report.primary_metric}**" in md
    assert "| Behavior |" in md


def test_render_rich_table_does_not_raise(capsys: pytest.CaptureFixture[str]) -> None:
    human, judge, keys = _build_labels(n_items=10, agreement=1.0, behavior="sycophancy")
    report = compute_calibration(
        labels_by_rater={"human": human, "judge": judge},
        behaviors=["sycophancy"],
        turn_keys=keys,
        module="A",
        prompt_version="v1",
        n_bootstrap=49,
        seed=4,
    )
    console = Console(record=True)
    render_rich_table(report, console)
    output = console.export_text()
    assert "sycophancy" in output
    assert "Module" in output
