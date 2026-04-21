"""Sampling tests: determinism, stratification, clamping, recency weighting."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from lucid.sampling import SamplingConfig, sample_conversations
from lucid.schemas import Conversation, Source

_UTC = UTC


def _make_conv(
    i: int, *, project: str | None = None, turns: int = 10, age_days: int = 0
) -> Conversation:
    t = datetime(2026, 4, 21, 12, 0, 0, tzinfo=_UTC) - timedelta(days=age_days)
    return Conversation(
        id=f"c-{i}",
        source=Source.CLAUDE_CODE,
        source_path="/tmp",
        created_at=t,
        updated_at=t,
        turn_count=turns,
        project_slug=project,
    )


# ----- determinism -----------------------------------------------------


def test_same_seed_same_output() -> None:
    convs = [_make_conv(i, project=f"p{i % 3}", age_days=i) for i in range(30)]
    config = SamplingConfig(n=10, seed=42)
    a = sample_conversations(convs, config)
    b = sample_conversations(convs, config)
    assert [c.id for c in a] == [c.id for c in b]


def test_different_seed_different_output() -> None:
    convs = [_make_conv(i, project=f"p{i % 3}", age_days=i) for i in range(30)]
    a = sample_conversations(convs, SamplingConfig(n=10, seed=1))
    b = sample_conversations(convs, SamplingConfig(n=10, seed=2))
    assert [c.id for c in a] != [c.id for c in b]


# ----- min_turns filter ------------------------------------------------


def test_min_turns_filters_short_sessions() -> None:
    long_convs = [_make_conv(i, turns=10, age_days=i) for i in range(5)]
    short_convs = [_make_conv(100 + i, turns=2, age_days=i) for i in range(5)]
    convs = long_convs + short_convs
    config = SamplingConfig(n=10, min_turns=5, stratify_by_project=False)
    result = sample_conversations(convs, config)
    # All 5 long conversations selected; no short ones.
    assert len(result) == 5
    assert all(c.turn_count >= 5 for c in result)


# ----- clamping --------------------------------------------------------


def test_clamp_when_n_exceeds_pool() -> None:
    convs = [_make_conv(i, turns=10, age_days=i) for i in range(3)]
    config = SamplingConfig(n=100, stratify_by_project=False)
    result = sample_conversations(convs, config)
    assert len(result) == 3


def test_empty_pool_returns_empty() -> None:
    assert sample_conversations([], SamplingConfig(n=100)) == []


def test_n_zero_returns_empty() -> None:
    convs = [_make_conv(i) for i in range(5)]
    assert sample_conversations(convs, SamplingConfig(n=0)) == []


# ----- stratification --------------------------------------------------


def test_stratification_respects_top_n_projects() -> None:
    # 3 big projects (100 each) + 5 small ones (1 each) = 305 convs.
    # Top-3 with top_n_projects=3 should keep only the big three + no-project.
    big = [
        _make_conv(1000 + i, project=f"big-{i // 100}", age_days=i % 90)
        for i in range(300)
    ]
    small = [_make_conv(2000 + i, project=f"small-{i}", age_days=i) for i in range(5)]
    convs = big + small
    config = SamplingConfig(n=30, top_n_projects=3, stratify_by_project=True, seed=42)
    result = sample_conversations(convs, config)
    selected_projects = {c.project_slug for c in result}
    # 'small-*' projects should be excluded.
    assert not any(p.startswith("small-") for p in selected_projects if p)


def test_stratification_disabled_flattens() -> None:
    convs = [
        _make_conv(i, project=f"p{i % 3}", age_days=i % 30, turns=10)
        for i in range(30)
    ]
    config = SamplingConfig(n=15, stratify_by_project=False, seed=42)
    result = sample_conversations(convs, config)
    assert len(result) == 15


# ----- recency weighting ----------------------------------------------


def test_recency_weight_biases_toward_recent() -> None:
    """With recency_weight=1.0, fresh items should dominate the sample."""
    convs = [_make_conv(i, turns=10, age_days=i * 2) for i in range(50)]
    reference = datetime(2026, 4, 21, 12, 0, 0, tzinfo=_UTC)
    config = SamplingConfig(
        n=10, recency_weight=1.0, recency_window_days=30,
        stratify_by_project=False, seed=42
    )
    result = sample_conversations(convs, config, reference_time=reference)
    avg_age = sum((reference - c.updated_at).days for c in result) / len(result)
    # Mean age should be significantly lower than the uniform expected ~49 days.
    assert avg_age < 40


def test_recency_weight_zero_is_uniform() -> None:
    """With recency_weight=0.0, seeded samples should be stable regardless of age."""
    convs = [_make_conv(i, turns=10, age_days=i * 2) for i in range(50)]
    config = SamplingConfig(
        n=10, recency_weight=0.0, stratify_by_project=False, seed=42
    )
    result_a = sample_conversations(convs, config)
    result_b = sample_conversations(convs, config)
    assert [c.id for c in result_a] == [c.id for c in result_b]


# ----- project filter --------------------------------------------------


def test_project_filter_keeps_only_listed() -> None:
    convs = [_make_conv(i, project=f"p{i % 4}", turns=10, age_days=i) for i in range(40)]
    config = SamplingConfig(
        n=20, project_filter=("p0", "p2"), stratify_by_project=False, seed=42
    )
    result = sample_conversations(convs, config)
    assert all(c.project_slug in {"p0", "p2"} for c in result)


# ----- validation ------------------------------------------------------


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("n", -1),
        ("min_turns", -1),
        ("recency_weight", 1.5),
        ("recency_weight", -0.1),
        ("recency_window_days", 0),
        ("top_n_projects", 0),
    ],
)
def test_invalid_config_raises(field_name: str, value: float) -> None:
    kwargs: dict[str, float] = {"n": 10}
    kwargs[field_name] = value
    with pytest.raises(ValueError):
        SamplingConfig(**kwargs).validate()  # type: ignore[arg-type]
