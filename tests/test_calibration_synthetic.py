"""Tests for :mod:`lucid.calibration.synthetic`.

Verifies coverage of the 17 behaviors × 3 intensities cross-grid, plus
clean examples, plus the corpus-build integration.
"""

from __future__ import annotations

from collections import Counter

import pytest

from lucid.calibration.synthetic import (
    SYNTHETIC_CORPUS_V1,
    SYNTHETIC_LABELER,
    build_synthetic_corpus,
    synthetic_labeled_turns,
)
from lucid.modules.module_a_spiralbench import BEHAVIORS
from lucid.schemas import Role


def test_corpus_covers_all_17_behaviors_at_all_3_intensities() -> None:
    """Full cross-grid: every (behavior, intensity) combination is represented."""
    observed = {
        (ex.behavior, ex.intensity) for ex in SYNTHETIC_CORPUS_V1 if ex.behavior is not None
    }
    expected = {(b, i) for b in BEHAVIORS for i in (1, 2, 3)}
    missing = expected - observed
    assert not missing, f"missing (behavior, intensity) combinations: {sorted(missing)}"


def test_corpus_has_clean_examples() -> None:
    """At least 5 clean examples (no behavior) to anchor prevalence calibration."""
    clean_count = sum(1 for ex in SYNTHETIC_CORPUS_V1 if ex.behavior is None)
    assert clean_count >= 5


def test_corpus_slugs_are_unique() -> None:
    slugs = [ex.slug for ex in SYNTHETIC_CORPUS_V1]
    assert len(slugs) == len(set(slugs))


def test_corpus_each_behaviour_has_three_distinct_intensities() -> None:
    by_behavior = Counter()
    for ex in SYNTHETIC_CORPUS_V1:
        if ex.behavior is not None:
            by_behavior[(ex.behavior, ex.intensity)] += 1
    for (behavior, intensity), count in by_behavior.items():
        assert count == 1, (
            f"behavior {behavior!r} at intensity {intensity} appears {count} times; "
            "expected exactly one per (behavior, intensity)."
        )


def test_build_produces_conversation_per_example() -> None:
    corpus, labels = build_synthetic_corpus()
    assert len(corpus.conversations) == len(SYNTHETIC_CORPUS_V1)
    assert len(labels) == len(SYNTHETIC_CORPUS_V1)


def test_build_conversations_have_two_turns_user_assistant() -> None:
    corpus, _ = build_synthetic_corpus()
    for turns in corpus.turns_by_conversation.values():
        assert len(turns) == 2
        assert turns[0].role == Role.USER
        assert turns[1].role == Role.ASSISTANT


def test_build_labels_are_tagged_synthetic_gold() -> None:
    _, labels = build_synthetic_corpus()
    assert all(lt.labeler == SYNTHETIC_LABELER for lt in labels)


def test_build_labels_match_example_ground_truth() -> None:
    _, labels = build_synthetic_corpus()
    by_conv = {lt.conversation_id: lt for lt in labels}
    for ex in SYNTHETIC_CORPUS_V1:
        conv_id = f"syn:v1:{ex.slug}"
        lt = by_conv[conv_id]
        if ex.behavior is None:
            assert lt.present_behaviors == frozenset()
            assert lt.intensities == {}
        else:
            assert ex.behavior in lt.present_behaviors
            assert lt.intensities.get(ex.behavior) == ex.intensity


def test_build_audit_run_id_propagates() -> None:
    corpus, _ = build_synthetic_corpus(audit_run_id="custom-run")
    assert corpus.audit_run_id == "custom-run"


def test_synthetic_labeled_turns_convenience() -> None:
    labels = synthetic_labeled_turns()
    assert len(labels) == len(SYNTHETIC_CORPUS_V1)
    assert all(lt.labeler == SYNTHETIC_LABELER for lt in labels)


def test_assistant_replies_are_non_empty() -> None:
    for ex in SYNTHETIC_CORPUS_V1:
        assert ex.user.strip(), f"{ex.slug}: empty user turn"
        assert ex.assistant.strip(), f"{ex.slug}: empty assistant turn"


def test_intensity_is_none_iff_behavior_is_none() -> None:
    """Construction invariant: clean → intensity None; behavior → intensity 1-3."""
    for ex in SYNTHETIC_CORPUS_V1:
        if ex.behavior is None:
            assert ex.intensity is None, f"{ex.slug}: clean example has intensity"
        else:
            assert ex.intensity in (1, 2, 3), f"{ex.slug}: behavior with bad intensity"


def test_corpus_size_is_60() -> None:
    """17 behaviors × 3 intensities + 9 clean = 60."""
    assert len(SYNTHETIC_CORPUS_V1) == 60


_ = pytest  # silence ruff
