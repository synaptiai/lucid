"""Corpus sampling.

`sample_conversations(convs, config)` reduces a corpus to `config.n` items
with the following discipline:

1. **Filter:** drop conversations shorter than `min_turns`.
2. **Stratify (optional):** bucket by `project_slug`. Keep the `top_n_projects`
   with the most conversations (plus the "no project" bucket as one stratum).
   Allocate the target sample size across strata proportional to their size.
3. **Recency weight:** within each stratum, weight items by `exp(-age_days / scale)`
   where `scale` is chosen so items outside `recency_window_days` get ~0.05
   weight. `recency_weight` in [0, 1] blends this exponential with the
   uniform distribution: 0 = uniform, 1 = fully recency-driven.
4. **Sample without replacement** using a seeded `random.Random(seed)`.
5. **Clamp:** if `config.n` exceeds the filtered pool size, return the whole
   pool (callers decide whether to warn).

Determinism: identical inputs + identical `SamplingConfig` always yield the
same output (same ordering). Tests assert this.

The function never mutates its inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from random import Random

from lucid.schemas import Conversation


@dataclass(frozen=True)
class SamplingConfig:
    """Configuration for a single sampling pass.

    Defaults match the PRD §6 / BUILD_GUIDE §6 recommended values.
    """

    n: int = 100
    min_turns: int = 5
    recency_weight: float = 0.7
    recency_window_days: int = 90
    stratify_by_project: bool = True
    top_n_projects: int = 10
    seed: int = 42
    project_filter: tuple[str, ...] | None = None

    def validate(self) -> None:
        if self.n < 0:
            raise ValueError("SamplingConfig.n must be >= 0")
        if self.min_turns < 0:
            raise ValueError("SamplingConfig.min_turns must be >= 0")
        if not 0.0 <= self.recency_weight <= 1.0:
            raise ValueError("SamplingConfig.recency_weight must be in [0.0, 1.0]")
        if self.recency_window_days <= 0:
            raise ValueError("SamplingConfig.recency_window_days must be > 0")
        if self.top_n_projects < 1:
            raise ValueError("SamplingConfig.top_n_projects must be >= 1")


# ──────────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────────


_UNSET_PROJECT_KEY = "__no_project__"


def _filter_by_min_turns(convs: list[Conversation], min_turns: int) -> list[Conversation]:
    return [c for c in convs if c.turn_count >= min_turns]


def _filter_by_project(
    convs: list[Conversation], allow: tuple[str, ...] | None
) -> list[Conversation]:
    if allow is None:
        return convs
    allowed = set(allow)
    return [c for c in convs if c.project_slug in allowed]


def _age_in_days(conv: Conversation, reference: datetime) -> float:
    delta = reference - conv.updated_at
    return max(delta.total_seconds() / 86400.0, 0.0)


def _blended_weight(
    conv: Conversation, *, reference: datetime, config: SamplingConfig
) -> float:
    """Blend uniform and exponential-recency weights via `recency_weight`."""
    if config.recency_weight <= 0.0:
        return 1.0
    # Exponential scale tuned so that an item exactly `recency_window_days` old
    # has weight ~0.05 relative to a brand-new item.
    scale = max(config.recency_window_days / 3.0, 1.0)
    age = _age_in_days(conv, reference)
    recency = math.exp(-age / scale)
    uniform = 1.0
    return (config.recency_weight * recency) + ((1.0 - config.recency_weight) * uniform)


def _allocate_quota(
    buckets: dict[str, list[Conversation]], target: int
) -> dict[str, int]:
    """Allocate `target` across buckets proportional to bucket size.

    Rounding residuals are distributed largest-fractional-part first, which
    guarantees `sum(result.values()) == min(target, sum(sizes))` even when
    `target` is small.
    """
    sizes = {k: len(v) for k, v in buckets.items()}
    total = sum(sizes.values())
    if total == 0 or target == 0:
        return dict.fromkeys(buckets, 0)
    # Never allocate more than is available in a bucket.
    if target >= total:
        return sizes

    raw = {k: target * (s / total) for k, s in sizes.items()}
    floored = {k: min(int(v), sizes[k]) for k, v in raw.items()}
    remaining = target - sum(floored.values())

    # Distribute remainders by fractional descending, skipping buckets that
    # are already at their cap.
    fractions = sorted(
        ((k, raw[k] - floored[k]) for k in raw if floored[k] < sizes[k]),
        key=lambda t: t[1],
        reverse=True,
    )
    i = 0
    while remaining > 0 and i < len(fractions):
        k, _ = fractions[i]
        if floored[k] < sizes[k]:
            floored[k] += 1
            remaining -= 1
        i += 1
        # When we reach the end of the fraction list but still have remaining,
        # loop from the top to top up non-capped buckets.
        if i == len(fractions) and remaining > 0:
            fractions = [(k, 0.0) for k in raw if floored[k] < sizes[k]]
            i = 0
            if not fractions:
                break
    return floored


def _weighted_sample_without_replacement(
    items: list[Conversation], weights: list[float], k: int, rng: Random
) -> list[Conversation]:
    """A* sampling (Efraimidis-Spirakis) — draw the k largest `-log(u)/w` keys."""
    if k <= 0 or not items:
        return []
    if k >= len(items):
        return list(items)
    # Deterministic given `rng`.
    keyed = []
    for item, w in zip(items, weights, strict=True):
        if w <= 0.0:
            continue
        u = rng.random()
        if u == 0.0:
            u = 1e-12
        keyed.append((-math.log(u) / w, item))
    keyed.sort(key=lambda t: t[0])
    return [item for _, item in keyed[:k]]


def _project_key(conv: Conversation) -> str:
    return conv.project_slug or _UNSET_PROJECT_KEY


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────


def sample_conversations(
    conversations: list[Conversation],
    config: SamplingConfig,
    *,
    reference_time: datetime | None = None,
) -> list[Conversation]:
    """Return a deterministic sample of `config.n` conversations.

    `reference_time` is the "now" against which recency is measured; the
    default is the latest `updated_at` across the corpus so tests don't
    change behaviour as wall-clock drifts. Pass a fixed datetime to freeze
    output for golden tests.
    """
    config.validate()
    pool = _filter_by_min_turns(conversations, config.min_turns)
    pool = _filter_by_project(pool, config.project_filter)
    if not pool or config.n == 0:
        return []

    reference = reference_time or max((c.updated_at for c in pool), default=_epoch())

    rng = Random(config.seed)

    if not config.stratify_by_project:
        weights = [_blended_weight(c, reference=reference, config=config) for c in pool]
        flat_chosen = _weighted_sample_without_replacement(pool, weights, config.n, rng)
        return _stable_order(flat_chosen)

    # Stratify: bucket by project, keep top_n_projects plus the no-project bucket.
    buckets: dict[str, list[Conversation]] = {}
    for c in pool:
        buckets.setdefault(_project_key(c), []).append(c)

    # Keep the largest `top_n_projects` project buckets + the no-project bucket
    # if present.
    named = sorted(
        ((k, v) for k, v in buckets.items() if k != _UNSET_PROJECT_KEY),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )[: config.top_n_projects]
    kept: dict[str, list[Conversation]] = dict(named)
    if _UNSET_PROJECT_KEY in buckets:
        kept[_UNSET_PROJECT_KEY] = buckets[_UNSET_PROJECT_KEY]

    quotas = _allocate_quota(kept, config.n)

    stratified_chosen: list[Conversation] = []
    for key, stratum in kept.items():
        q = quotas.get(key, 0)
        if q <= 0:
            continue
        weights = [_blended_weight(c, reference=reference, config=config) for c in stratum]
        stratified_chosen.extend(
            _weighted_sample_without_replacement(stratum, weights, q, rng)
        )

    return _stable_order(stratified_chosen)


def _stable_order(chosen: list[Conversation]) -> list[Conversation]:
    """Sort chosen conversations by (updated_at DESC, id) for a deterministic output order."""
    return sorted(chosen, key=lambda c: (-c.updated_at.timestamp(), c.id))


def _epoch() -> datetime:
    return datetime(1970, 1, 1, tzinfo=UTC)


__all__ = ["SamplingConfig", "sample_conversations"]
