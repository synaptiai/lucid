"""Inter-annotator agreement metrics + BCa bootstrap confidence intervals.

**What's implemented:**

- :func:`krippendorff_alpha` — thin wrapper around ``krippendorff.alpha``.
- :func:`gwet_ac1` — hand-rolled per Gwet 2014, *Handbook of Inter-Rater
  Reliability*, 4th ed., Chapter 3. Works for R ≥ 2 raters, complete
  ratings (no missing values). Validated against worked 2×2 cross-tab.
- :func:`cohen_kappa` — unweighted Cohen's κ, two raters.
- :func:`quadratic_weighted_kappa` — ordinal κ for intensity scales.
- :func:`bootstrap_metric` — BCa 95% CI via ``scipy.stats.bootstrap``
  resampling over items.

**Why hand-rolled:** plan v3 specified ``irrCAC`` + ``sklearn``, but the
former hard-pins ``numpy==1.26.4`` which conflicts with ``voyageai==0.3.7``
(Module H is load-bearing for the demo), and ``sklearn`` would add tens of
MB to the PyInstaller binary for one function. Both formulas are
closed-form and under ~20 lines. See ``docs/methodology.md §9``.

**Shape convention:** ``ratings`` is always ``(n_raters, n_items)`` with
integer category labels. For two-rater metrics the raters dimension is
exactly 2; a ``ValueError`` is raised otherwise so mis-shaped inputs fail
loudly rather than silently producing bogus numbers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import krippendorff
import numpy as np
from scipy import stats

__all__ = [
    "MetricResult",
    "bootstrap_metric",
    "cohen_kappa",
    "gwet_ac1",
    "krippendorff_alpha",
    "quadratic_weighted_kappa",
]

MetricFn = Callable[[np.ndarray], float]
KrippendorffLevel = Literal["nominal", "ordinal", "interval", "ratio"]


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Point estimate + bootstrap CI + bookkeeping for a calibration metric."""

    name: str
    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    n_items: int = 0


# ──────────────────────────────────────────────────────────────────────────
# Krippendorff α
# ──────────────────────────────────────────────────────────────────────────


def krippendorff_alpha(
    ratings: np.ndarray,
    level: KrippendorffLevel = "nominal",
) -> float:
    """Krippendorff's α. ``ratings`` is (n_raters, n_items); NaN = missing."""
    if ratings.size == 0:
        raise ValueError("ratings must be non-empty")
    return float(
        krippendorff.alpha(
            reliability_data=ratings,
            level_of_measurement=level,
        )
    )


# ──────────────────────────────────────────────────────────────────────────
# Gwet AC1
# ──────────────────────────────────────────────────────────────────────────


def gwet_ac1(ratings: np.ndarray) -> float:
    """Gwet 2014, Handbook 4th ed., Eqn 2.4 (paradox-robust IAA).

    Complete ratings assumed — callers that have NaN should drop those items
    before calling (Gwet's handling of missing is a separate, more involved
    formula we don't need for the hackathon's 200-item fully-rated set).
    """
    if ratings.ndim != 2:
        raise ValueError("ratings must be 2D: (n_raters, n_items)")
    raters, items = ratings.shape
    if raters < 2:
        raise ValueError("Gwet AC1 requires at least 2 raters")
    if items < 1:
        raise ValueError("ratings must have at least 1 item")

    categories = np.unique(ratings)
    q = len(categories)
    if q < 2:
        # All raters picked one category: agreement is trivially total.
        return 1.0

    # n_kq[k, q] = number of raters assigning category q to item k.
    n_kq = np.stack([(ratings == c).sum(axis=0) for c in categories], axis=-1)

    # Pairwise-agreement proportion (averaged over items).
    p_a = (n_kq * (n_kq - 1)).sum(axis=1).mean() / (raters * (raters - 1))

    # Category prevalence across all ratings.
    pi = n_kq.sum(axis=0) / (items * raters)

    # Chance-agreement under Gwet's uniform-noise assumption.
    p_e = (pi * (1.0 - pi)).sum() / (q - 1)

    if np.isclose(p_e, 1.0):
        return 1.0 if np.isclose(p_a, 1.0) else float("nan")
    return float((p_a - p_e) / (1.0 - p_e))


# ──────────────────────────────────────────────────────────────────────────
# Cohen's κ (unweighted)
# ──────────────────────────────────────────────────────────────────────────


def cohen_kappa(ratings: np.ndarray) -> float:
    """Unweighted Cohen's κ for exactly two raters."""
    if ratings.ndim != 2 or ratings.shape[0] != 2:
        raise ValueError("cohen_kappa requires ratings of shape (2, n_items)")
    if ratings.shape[1] < 1:
        raise ValueError("ratings must have at least 1 item")

    rater1, rater2 = ratings[0], ratings[1]
    n = rater1.size
    categories = np.unique(np.concatenate([rater1, rater2]))

    p_o = float(np.mean(rater1 == rater2))

    p1 = np.array([np.sum(rater1 == c) / n for c in categories])
    p2 = np.array([np.sum(rater2 == c) / n for c in categories])
    p_e = float((p1 * p2).sum())

    if np.isclose(p_e, 1.0):
        return 1.0 if np.isclose(p_o, 1.0) else float("nan")
    return (p_o - p_e) / (1.0 - p_e)


# ──────────────────────────────────────────────────────────────────────────
# Quadratic-Weighted κ
# ──────────────────────────────────────────────────────────────────────────


def quadratic_weighted_kappa(
    ratings: np.ndarray,
    categories: Sequence[int] | None = None,
) -> float:
    """Quadratic-weighted κ for ordinal intensity labels.

    When ``categories`` is given, the ordinal spacing honours it — useful
    when a rating scale is 1/2/3 but a particular resample happens to miss
    category 2. Without it, the distance between categories is inferred
    from the unique values in the data, which would silently compress an
    intensity-3 label toward an intensity-2 label.
    """
    if ratings.ndim != 2 or ratings.shape[0] != 2:
        raise ValueError("quadratic_weighted_kappa requires ratings of shape (2, n_items)")
    if ratings.shape[1] < 1:
        raise ValueError("ratings must have at least 1 item")

    if categories is None:
        cats = np.unique(np.concatenate([ratings[0], ratings[1]])).tolist()
    else:
        cats = list(categories)
    k = len(cats)
    if k < 2:
        return 1.0

    cat_to_idx = {int(c): i for i, c in enumerate(cats)}
    idx1 = np.array([cat_to_idx[int(v)] for v in ratings[0]])
    idx2 = np.array([cat_to_idx[int(v)] for v in ratings[1]])

    obs = np.zeros((k, k), dtype=float)
    for a, b in zip(idx1, idx2, strict=True):
        obs[a, b] += 1.0
    total = obs.sum()

    r1_margin = obs.sum(axis=1)
    r2_margin = obs.sum(axis=0)
    exp = np.outer(r1_margin, r2_margin) / total

    ks = np.arange(k)
    weights = (ks[:, None] - ks[None, :]) ** 2 / (k - 1) ** 2

    denom = float((weights * exp).sum())
    if np.isclose(denom, 0.0):
        return 1.0 if np.isclose((weights * obs).sum(), 0.0) else float("nan")
    return float(1.0 - (weights * obs).sum() / denom)


# ──────────────────────────────────────────────────────────────────────────
# BCa bootstrap
# ──────────────────────────────────────────────────────────────────────────


def bootstrap_metric(
    ratings: np.ndarray,
    metric_fn: MetricFn,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> MetricResult:
    """Compute ``metric_fn(ratings)`` plus a BCa bootstrap CI over items.

    Resamples the item axis (axis=1) with replacement ``n_resamples`` times;
    ``scipy.stats.bootstrap`` with ``method='BCa'`` handles the
    bias-correction + acceleration internally. ``metric_fn`` must accept
    the same ``(n_raters, n_items)`` shape as ``ratings``.
    """
    if ratings.ndim != 2:
        raise ValueError("ratings must be 2D")
    if ratings.shape[1] < 2:
        raise ValueError("bootstrap requires at least 2 items")

    point = metric_fn(ratings)
    indices = np.arange(ratings.shape[1])

    def _statistic(idx_sample: np.ndarray) -> float:
        # scipy passes `axis=-1` but with vectorized=False it calls scalar-style.
        return metric_fn(ratings[:, idx_sample.astype(int)])

    result = stats.bootstrap(
        (indices,),
        _statistic,
        n_resamples=n_resamples,
        confidence_level=confidence,
        method="BCa",
        vectorized=False,
        random_state=rng,
    )

    return MetricResult(
        name=metric_fn.__name__,
        value=point,
        ci_low=float(result.confidence_interval.low),
        ci_high=float(result.confidence_interval.high),
        n_items=int(ratings.shape[1]),
    )
