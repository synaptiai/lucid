"""Tests for :mod:`lucid.calibration.data`.

Covers the JSONL label loader, ratings matrix construction against the
(2, n_items) shape that ``validate.py`` expects, and the seeded train/test
split used by the Day 2 calibration flow (30/70 held-out split).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from lucid.calibration.data import (
    LabeledTurn,
    intensity_matrix,
    load_hand_labels,
    presence_matrix,
    train_test_split,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        b'{"conversation_id":"'
        + str(r["conversation_id"]).encode()
        + b'","turn_id":"'
        + str(r["turn_id"]).encode()
        + b'"}'
        for r in rows
    ]
    path.write_bytes(b"\n".join(lines) + b"\n")


# ──────────────────────────────────────────────────────────────────────────
# load_hand_labels
# ──────────────────────────────────────────────────────────────────────────


def test_load_hand_labels_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    labeled = [
        LabeledTurn(
            conversation_id="c1",
            turn_id="t0",
            present_behaviors=frozenset({"off-ramp-missed", "sycophantic-praise"}),
            intensities={"off-ramp-missed": 2, "sycophantic-praise": 1},
            labeler="daniel",
            labeled_at=datetime(2026, 4, 21, 23, 0, tzinfo=UTC),
        ),
        LabeledTurn(
            conversation_id="c2",
            turn_id="t5",
            present_behaviors=frozenset(),
            intensities={},
            labeler="daniel",
            labeled_at=datetime(2026, 4, 21, 23, 30, tzinfo=UTC),
        ),
    ]
    path.write_text(
        "\n".join(lt.model_dump_json() for lt in labeled) + "\n",
        encoding="utf-8",
    )
    loaded = load_hand_labels(path)
    assert loaded == labeled


def test_load_hand_labels_skips_blank_and_comment_lines(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    lt = LabeledTurn(
        conversation_id="c1",
        turn_id="t0",
        labeler="daniel",
        labeled_at=datetime(2026, 4, 21, tzinfo=UTC),
    )
    path.write_text(
        "# comment line\n\n" + lt.model_dump_json() + "\n  \n",
        encoding="utf-8",
    )
    assert load_hand_labels(path) == [lt]


def test_load_hand_labels_reports_line_on_malformed_row(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text("not json at all\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"labels\.jsonl:1"):
        load_hand_labels(path)


def test_load_hand_labels_rejects_unknown_fields(tmp_path: Path) -> None:
    """`extra='forbid'` surfaces typos in hand-curated JSONL."""
    path = tmp_path / "labels.jsonl"
    path.write_text(
        '{"conversation_id":"c1","turn_id":"t0","labeler":"d",'
        '"labeled_at":"2026-04-21T00:00:00Z","typo_field":"boom"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"labels\.jsonl:1"):
        load_hand_labels(path)


def test_load_hand_labels_rejects_intensity_out_of_range(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(
        '{"conversation_id":"c1","turn_id":"t0","present_behaviors":["x"],'
        '"intensities":{"x":5},"labeler":"d","labeled_at":"2026-04-21T00:00:00Z"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"labels\.jsonl:1"):
        load_hand_labels(path)


def test_load_hand_labels_rejects_intensity_without_presence(tmp_path: Path) -> None:
    """Intensity for a behavior not marked present is a labeling bug."""
    path = tmp_path / "labels.jsonl"
    path.write_text(
        '{"conversation_id":"c1","turn_id":"t0","present_behaviors":[],'
        '"intensities":{"off-ramp-missed":2},"labeler":"d",'
        '"labeled_at":"2026-04-21T00:00:00Z"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"labels\.jsonl:1"):
        load_hand_labels(path)


# ──────────────────────────────────────────────────────────────────────────
# presence_matrix / intensity_matrix
# ──────────────────────────────────────────────────────────────────────────


def _lt(
    conv_id: str,
    turn_id: str,
    *,
    labeler: str,
    present: frozenset[str] | None = None,
    intensities: dict[str, int] | None = None,
) -> LabeledTurn:
    return LabeledTurn(
        conversation_id=conv_id,
        turn_id=turn_id,
        present_behaviors=present or frozenset(),
        intensities=intensities or {},
        labeler=labeler,
        labeled_at=datetime(2026, 4, 21, tzinfo=UTC),
    )


def test_presence_matrix_is_binary_and_row_per_rater() -> None:
    labels = {
        "human": [
            _lt("c1", "t0", labeler="human", present=frozenset({"off-ramp-missed"})),
            _lt("c2", "t0", labeler="human"),
        ],
        "judge": [
            _lt("c1", "t0", labeler="judge", present=frozenset({"off-ramp-missed"})),
            _lt("c2", "t0", labeler="judge", present=frozenset({"off-ramp-missed"})),
        ],
    }
    order = [("c1", "t0"), ("c2", "t0")]
    m = presence_matrix(labels, "off-ramp-missed", order)
    assert m.shape == (2, 2)
    assert np.array_equal(m[0], [1, 0])  # human
    assert np.array_equal(m[1], [1, 1])  # judge


def test_presence_matrix_rejects_incomplete_ratings() -> None:
    labels = {
        "human": [_lt("c1", "t0", labeler="human")],
        "judge": [],
    }
    with pytest.raises(ValueError, match="no label for turn"):
        presence_matrix(labels, "x", [("c1", "t0")])


def test_intensity_matrix_zero_for_absent_behavior() -> None:
    labels = {
        "human": [
            _lt(
                "c1",
                "t0",
                labeler="human",
                present=frozenset({"off-ramp-missed"}),
                intensities={"off-ramp-missed": 2},
            ),
            _lt("c2", "t0", labeler="human"),
        ],
        "judge": [
            _lt(
                "c1",
                "t0",
                labeler="judge",
                present=frozenset({"off-ramp-missed"}),
                intensities={"off-ramp-missed": 3},
            ),
            _lt("c2", "t0", labeler="judge"),
        ],
    }
    order = [("c1", "t0"), ("c2", "t0")]
    m = intensity_matrix(labels, "off-ramp-missed", order)
    assert m.shape == (2, 2)
    assert np.array_equal(m[0], [2, 0])
    assert np.array_equal(m[1], [3, 0])


# ──────────────────────────────────────────────────────────────────────────
# train_test_split
# ──────────────────────────────────────────────────────────────────────────


def test_train_test_split_is_deterministic_under_seed() -> None:
    keys = [(f"c{i}", f"t{i}") for i in range(200)]
    a_train, a_test = train_test_split(keys, test_frac=0.3, seed=42)
    b_train, b_test = train_test_split(keys, test_frac=0.3, seed=42)
    assert a_train == b_train
    assert a_test == b_test


def test_train_test_split_produces_expected_sizes() -> None:
    keys = [(f"c{i}", f"t{i}") for i in range(200)]
    train, test = train_test_split(keys, test_frac=0.3, seed=42)
    assert len(test) == 60
    assert len(train) == 140
    assert len(train) + len(test) == 200


def test_train_test_split_full_coverage_no_overlap() -> None:
    keys = [(f"c{i}", f"t{i}") for i in range(100)]
    train, test = train_test_split(keys, test_frac=0.3, seed=7)
    all_keys = set(train) | set(test)
    assert all_keys == set(keys)
    assert set(train) & set(test) == set()


def test_train_test_split_rejects_zero_or_one_test_frac() -> None:
    keys = [("c", "t")]
    with pytest.raises(ValueError, match=r"test_frac must be in \(0, 1\)"):
        train_test_split(keys, test_frac=0.0)
    with pytest.raises(ValueError, match=r"test_frac must be in \(0, 1\)"):
        train_test_split(keys, test_frac=1.0)


def test_train_test_split_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        train_test_split([], test_frac=0.3)


def test_train_test_split_different_seeds_produce_different_splits() -> None:
    keys = [(f"c{i}", f"t{i}") for i in range(50)]
    _, test_a = train_test_split(keys, test_frac=0.3, seed=1)
    _, test_b = train_test_split(keys, test_frac=0.3, seed=2)
    assert set(test_a) != set(test_b)
