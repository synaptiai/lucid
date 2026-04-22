"""Integration tests for ``lucid calibrate --module a``.

End-to-end via ``CliRunner``, seeding JSONL files for human and judge
labels, verifying exit codes + output. No LLM involved — this is the
Phase 6A path where both label sets are supplied pre-computed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lucid.calibration.data import LabeledTurn
from lucid.cli import EXIT_USAGE, app

runner = CliRunner()


def _write_labels(path: Path, rows: list[LabeledTurn]) -> None:
    path.write_text(
        "\n".join(r.model_dump_json() for r in rows) + "\n",
        encoding="utf-8",
    )


def _seed_labels(
    tmp_path: Path,
    *,
    n_items: int,
    agreement_fraction: float,
    behaviors: tuple[str, ...],
) -> tuple[Path, Path]:
    """Build human + judge JSONL with a known agreement fraction.

    Every item has the same set of behaviors present in the human labels.
    In the judge labels, ``agreement_fraction`` of items match; the rest
    have the opposite (empty) label set. This gives us deterministic
    fixture inputs with non-trivial but non-perfect IAA.
    """
    human_rows: list[LabeledTurn] = []
    judge_rows: list[LabeledTurn] = []
    for i in range(n_items):
        conv_id = f"c{i}"
        turn_id = f"t{i}"
        matches = i < int(n_items * agreement_fraction)
        human_rows.append(
            LabeledTurn(
                conversation_id=conv_id,
                turn_id=turn_id,
                present_behaviors=frozenset(behaviors),
                intensities={b: 2 for b in behaviors},
                labeler="human",
                labeled_at=datetime(2026, 4, 21, tzinfo=UTC),
            )
        )
        judge_rows.append(
            LabeledTurn(
                conversation_id=conv_id,
                turn_id=turn_id,
                present_behaviors=frozenset(behaviors) if matches else frozenset(),
                intensities={b: 2 for b in behaviors} if matches else {},
                labeler="judge",
                labeled_at=datetime(2026, 4, 21, tzinfo=UTC),
            )
        )

    human_path = tmp_path / "human.jsonl"
    judge_path = tmp_path / "judge.jsonl"
    _write_labels(human_path, human_rows)
    _write_labels(judge_path, judge_rows)
    return human_path, judge_path


# ──────────────────────────────────────────────────────────────────────────
# Usage / error surfaces
# ──────────────────────────────────────────────────────────────────────────


def test_calibrate_module_h_not_yet_supported() -> None:
    result = runner.invoke(app, ["calibrate", "--module", "h"])
    assert result.exit_code == EXIT_USAGE
    assert "later phase" in result.stdout.lower()


def test_calibrate_module_a_requires_both_label_paths() -> None:
    result = runner.invoke(app, ["calibrate", "--module", "a"])
    assert result.exit_code == EXIT_USAGE
    assert "--human-labels" in result.stdout
    assert "--judge-labels" in result.stdout


def test_calibrate_rejects_missing_files(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "calibrate",
            "--module",
            "a",
            "--human-labels",
            str(tmp_path / "does-not-exist.jsonl"),
            "--judge-labels",
            str(tmp_path / "also-missing.jsonl"),
        ],
    )
    assert result.exit_code == EXIT_USAGE


def test_calibrate_rejects_empty_label_files(tmp_path: Path) -> None:
    human = tmp_path / "h.jsonl"
    judge = tmp_path / "j.jsonl"
    human.write_text("", encoding="utf-8")
    judge.write_text("", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "calibrate",
            "--module",
            "a",
            "--human-labels",
            str(human),
            "--judge-labels",
            str(judge),
        ],
    )
    assert result.exit_code == EXIT_USAGE
    assert "empty" in result.stdout.lower()


def test_calibrate_rejects_when_no_shared_turns(tmp_path: Path) -> None:
    human = tmp_path / "h.jsonl"
    judge = tmp_path / "j.jsonl"
    _write_labels(
        human,
        [
            LabeledTurn(
                conversation_id="cA",
                turn_id="t1",
                labeler="h",
                labeled_at=datetime(2026, 4, 21, tzinfo=UTC),
            )
        ],
    )
    _write_labels(
        judge,
        [
            LabeledTurn(
                conversation_id="cB",  # different conversation
                turn_id="t1",
                labeler="j",
                labeled_at=datetime(2026, 4, 21, tzinfo=UTC),
            )
        ],
    )

    result = runner.invoke(
        app,
        [
            "calibrate",
            "--module",
            "a",
            "--human-labels",
            str(human),
            "--judge-labels",
            str(judge),
        ],
    )
    assert result.exit_code == EXIT_USAGE
    assert "overlapping" in result.stdout.lower()


def test_calibrate_rejects_unknown_behavior_name(tmp_path: Path) -> None:
    human, judge = _seed_labels(
        tmp_path, n_items=10, agreement_fraction=1.0, behaviors=("sycophancy",)
    )
    result = runner.invoke(
        app,
        [
            "calibrate",
            "--module",
            "a",
            "--human-labels",
            str(human),
            "--judge-labels",
            str(judge),
            "--behaviors",
            "not-a-real-behavior",
        ],
    )
    assert result.exit_code == EXIT_USAGE
    assert "unknown behaviors" in result.stdout.lower()


# ──────────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────────


def test_calibrate_happy_path_prints_metrics(tmp_path: Path) -> None:
    human, judge = _seed_labels(
        tmp_path,
        n_items=40,
        agreement_fraction=0.9,
        behaviors=("sycophancy",),
    )
    result = runner.invoke(
        app,
        [
            "calibrate",
            "--module",
            "a",
            "--human-labels",
            str(human),
            "--judge-labels",
            str(judge),
            "--behaviors",
            "sycophancy",
            "--n-bootstrap",
            "99",
            "--seed",
            "7",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "sycophancy" in result.stdout
    # Header fields should appear
    assert "Module" in result.stdout
    assert "Primary metric" in result.stdout or "primary metric" in result.stdout.lower()


def test_calibrate_write_markdown_persists_report(tmp_path: Path) -> None:
    human, judge = _seed_labels(
        tmp_path,
        n_items=30,
        agreement_fraction=0.8,
        behaviors=("sycophancy", "pushback"),
    )
    md_path = tmp_path / "calibration.md"

    result = runner.invoke(
        app,
        [
            "calibrate",
            "--module",
            "a",
            "--human-labels",
            str(human),
            "--judge-labels",
            str(judge),
            "--behaviors",
            "sycophancy,pushback",
            "--n-bootstrap",
            "49",
            "--write-markdown",
            str(md_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert md_path.is_file()
    contents = md_path.read_text(encoding="utf-8")
    assert "Module A" in contents
    assert "sycophancy" in contents
    assert "pushback" in contents


def test_calibrate_auto_judge_below_cost_gate_exits_3(tmp_path: Path) -> None:
    """--auto-judge with insufficient --yes-i-authorize-spend-up-to exits 3."""
    from lucid.cli import EXIT_COST_GATE

    result = runner.invoke(
        app,
        [
            "calibrate",
            "--module",
            "a",
            "--auto-judge",
            "--yes-i-authorize-spend-up-to",
            "10",  # below the $50 minimum
            "--chunk-sizes",
            "",  # skip Module A so we don't need an API key
            "--sb-models",
            "",
            "--ollama-models",
            "",
        ],
    )
    assert result.exit_code == EXIT_COST_GATE
    assert "50" in result.stdout


def test_calibrate_import_verified_without_prior_run_exits_usage(tmp_path: Path) -> None:
    verified = tmp_path / "review.jsonl"
    verified.write_text("", encoding="utf-8")
    missing = tmp_path / "nonexistent"

    result = runner.invoke(
        app,
        [
            "calibrate",
            "--module",
            "a",
            "--import-verified",
            str(verified),
            "--output-dir",
            str(missing),
        ],
    )
    assert result.exit_code == EXIT_USAGE


def test_calibrate_mixing_modes_exits_usage(tmp_path: Path) -> None:
    """Combining Phase 6A flags with --auto-judge is a usage error."""
    human, judge = _seed_labels(
        tmp_path, n_items=10, agreement_fraction=1.0, behaviors=("sycophancy",)
    )
    result = runner.invoke(
        app,
        [
            "calibrate",
            "--module",
            "a",
            "--human-labels",
            str(human),
            "--judge-labels",
            str(judge),
            "--auto-judge",
        ],
    )
    assert result.exit_code == EXIT_USAGE
    assert "exactly one mode" in result.stdout.lower()


def test_calibrate_write_markdown_appends_second_run(tmp_path: Path) -> None:
    """Running twice with --write-markdown keeps both reports — useful when
    iterating on prompt versions."""
    human, judge = _seed_labels(
        tmp_path, n_items=20, agreement_fraction=0.85, behaviors=("sycophancy",)
    )
    md_path = tmp_path / "calibration.md"

    for _ in range(2):
        result = runner.invoke(
            app,
            [
                "calibrate",
                "--module",
                "a",
                "--human-labels",
                str(human),
                "--judge-labels",
                str(judge),
                "--behaviors",
                "sycophancy",
                "--n-bootstrap",
                "49",
                "--write-markdown",
                str(md_path),
            ],
        )
        assert result.exit_code == 0, result.stdout

    contents = md_path.read_text(encoding="utf-8")
    # Two run reports means two section headers.
    assert contents.count("## Module A") == 2


# silence unused-import lint
_ = json
_ = pytest
