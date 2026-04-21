"""Worked-example tests for :mod:`lucid.calibration.validate`.

All IAA metrics (Gwet AC1, Cohen's κ, Quadratic-Weighted κ) are hand-rolled
since ``irrCAC`` was dropped from the dep set (numpy pin conflict with
voyageai, docs/methodology.md §9) and ``sklearn`` is not in the dep set.
These tests validate the implementations against independently computed
values from a canonical 2×2 cross-tab example and from algebra on small
confusion matrices.

Krippendorff α is thin-wrapped around the ``krippendorff`` package; the
tests confirm the wrapper's invariants (perfect agreement → 1.0, random
data produces a value in the expected range) rather than re-deriving the
library's internals.
"""

from __future__ import annotations

import numpy as np
import pytest

from lucid.calibration.validate import (
    MetricResult,
    bootstrap_metric,
    cohen_kappa,
    gwet_ac1,
    krippendorff_alpha,
    quadratic_weighted_kappa,
)

# ──────────────────────────────────────────────────────────────────────────
# Gwet AC1 — canonical 2×2 cross-tab (Gwet 2014, Handbook of Inter-Rater
# Reliability, 4th ed., Ch. 3 style).
#
# Cross-tab of 150 items:
#            rater2=A  rater2=B
#   rater1=A    118       5
#   rater1=B      7      20
# π_A = 248/300 ≈ 0.82667; π_B = 52/300 ≈ 0.17333
# p_a = 138/150 = 0.92
# p_e_AC1 = π_A(1-π_A) + π_B(1-π_B) = 2·0.14329 = 0.28658
# AC1 = (0.92 - 0.28658) / (1 - 0.28658) = 0.88787
#
# Cohen's κ on the same data:
# p1_A = 123/150 = 0.82; p2_A = 125/150 = 0.83333
# p_e = 0.82·0.83333 + 0.18·0.16667 = 0.71333
# κ = (0.92 - 0.71333) / (1 - 0.71333) = 0.72093
# ──────────────────────────────────────────────────────────────────────────


def _canonical_2x2_ratings() -> np.ndarray:
    """Build the (2, 150) ratings matrix for the worked example above."""
    parts = [
        np.tile(np.array([[0], [0]]), 118),  # 118 × AA
        np.tile(np.array([[0], [1]]), 5),  # 5 × AB
        np.tile(np.array([[1], [0]]), 7),  # 7 × BA
        np.tile(np.array([[1], [1]]), 20),  # 20 × BB
    ]
    return np.concatenate(parts, axis=1)


def test_gwet_ac1_canonical_2x2_matches_expected() -> None:
    ratings = _canonical_2x2_ratings()
    assert gwet_ac1(ratings) == pytest.approx(0.88787, abs=1e-4)


def test_cohen_kappa_canonical_2x2_matches_expected() -> None:
    ratings = _canonical_2x2_ratings()
    assert cohen_kappa(ratings) == pytest.approx(0.72093, abs=1e-4)


# ──────────────────────────────────────────────────────────────────────────
# Edge-case behaviour
# ──────────────────────────────────────────────────────────────────────────


def test_gwet_ac1_perfect_agreement() -> None:
    ratings = np.array([[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]])
    assert gwet_ac1(ratings) == pytest.approx(1.0)


def test_gwet_ac1_single_category_returns_one() -> None:
    """All raters use only one category → undefined in the formula; we return 1.0."""
    ratings = np.array([[1, 1, 1], [1, 1, 1]])
    assert gwet_ac1(ratings) == 1.0


def test_gwet_ac1_rejects_single_rater() -> None:
    with pytest.raises(ValueError, match="at least 2 raters"):
        gwet_ac1(np.array([[0, 1, 0]]))


def test_gwet_ac1_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2D"):
        gwet_ac1(np.array([0, 1, 0]))


def test_cohen_kappa_perfect_agreement() -> None:
    ratings = np.array([[0, 1, 0, 1], [0, 1, 0, 1]])
    assert cohen_kappa(ratings) == pytest.approx(1.0)


def test_cohen_kappa_rejects_non_two_raters() -> None:
    with pytest.raises(ValueError, match=r"\(2, n_items\)"):
        cohen_kappa(np.array([[0, 1], [0, 1], [0, 1]]))


# ──────────────────────────────────────────────────────────────────────────
# Quadratic-Weighted κ
#
# rater1 = [1, 1, 2, 2, 3, 3]
# rater2 = [1, 2, 2, 3, 3, 3]
# confusion matrix O (3×3, rows=rater1, cols=rater2):
#   [[1, 1, 0],
#    [0, 1, 1],
#    [0, 0, 2]]
# row sums (r1 marginals): [2, 2, 2]; col sums (r2): [1, 2, 3]; N=6.
# expected E[i,j] = r1[i]·r2[j]/N → E = [[1/3, 2/3, 1], ×3 rows]
# disagreement weights (k=3): w[i,j] = (i-j)²/4.
# Σ w·O = 1/4 (at (0,1)) + 1/4 (at (1,2)) = 0.5
# Σ w·E = 2 (arithmetic checked by hand).
# QWK = 1 - 0.5/2 = 0.75.
# ──────────────────────────────────────────────────────────────────────────


def test_qwk_worked_example() -> None:
    ratings = np.array([[1, 1, 2, 2, 3, 3], [1, 2, 2, 3, 3, 3]])
    assert quadratic_weighted_kappa(ratings) == pytest.approx(0.75)


def test_qwk_perfect_agreement() -> None:
    ratings = np.array([[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]])
    assert quadratic_weighted_kappa(ratings) == pytest.approx(1.0)


def test_qwk_complete_disagreement_on_endpoints_is_zero() -> None:
    """r1=[1,1,1], r2=[3,3,3]: marginals concentrate on opposite corners → κ=0."""
    ratings = np.array([[1, 1, 1], [3, 3, 3]])
    # Σ w·O and Σ w·E both equal 3, so κ_w = 1 - 3/3 = 0.
    assert quadratic_weighted_kappa(ratings) == pytest.approx(0.0)


def test_qwk_explicit_categories_preserves_ordinal_gaps() -> None:
    """Passing `categories` makes intensity scales with absent values still work."""
    # rater1=[1,3], rater2=[1,3] with categories=[1,2,3].
    # Without the override, the implementation would see only categories
    # {1,3} and treat them as adjacent.
    ratings = np.array([[1, 3], [1, 3]])
    assert quadratic_weighted_kappa(ratings, categories=[1, 2, 3]) == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────────
# Krippendorff α — thin wrapper sanity checks
# ──────────────────────────────────────────────────────────────────────────


def test_krippendorff_alpha_perfect_agreement_nominal() -> None:
    ratings = np.array([[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]])
    assert krippendorff_alpha(ratings) == pytest.approx(1.0)


def test_krippendorff_alpha_rejects_empty_ratings() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        krippendorff_alpha(np.array([[], []]))


# ──────────────────────────────────────────────────────────────────────────
# BCa bootstrap wrapper
# ──────────────────────────────────────────────────────────────────────────


def test_bootstrap_metric_returns_containing_ci() -> None:
    """Point estimate falls inside the 95% CI, n_items recorded, name propagated."""
    rng = np.random.default_rng(12345)
    # Skewed agreement pattern so the statistic is well-defined under resampling.
    n = 60
    r1 = rng.integers(0, 2, size=n)
    r2 = np.where(rng.random(n) < 0.85, r1, 1 - r1)
    ratings = np.stack([r1, r2])

    result = bootstrap_metric(ratings, cohen_kappa, n_resamples=499, rng=rng)

    assert isinstance(result, MetricResult)
    assert result.name == "cohen_kappa"
    assert result.n_items == n
    assert result.ci_low is not None and result.ci_high is not None
    assert result.ci_low <= result.value <= result.ci_high


def test_bootstrap_metric_works_for_gwet_ac1() -> None:
    rng = np.random.default_rng(77)
    n = 100
    ratings = rng.integers(0, 3, size=(2, n))
    result = bootstrap_metric(ratings, gwet_ac1, n_resamples=299, rng=rng)
    assert result.name == "gwet_ac1"
    assert result.ci_low is not None
    assert result.ci_high is not None
