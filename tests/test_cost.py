"""Cost-estimator tests: module profiles, cache multipliers, $20 gate."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from lucid.cost import (
    COST_GATE_USD,
    DEFAULT_MODULES,
    MODULE_PROFILES,
    PRICES,
    CostEstimator,
    heuristic_counter,
)
from lucid.schemas import Conversation, ModuleName, Role, Source, TextBlock, Turn

_UTC = UTC


def _conv(cid: str = "c-1", turns_n: int = 10) -> tuple[Conversation, list[Turn]]:
    now = datetime(2026, 4, 21, 12, 0, 0, tzinfo=_UTC)
    conv = Conversation(
        id=cid,
        source=Source.CLAUDE_AI,
        source_path="/tmp",
        created_at=now,
        updated_at=now,
        turn_count=turns_n,
    )
    turns = [
        Turn(
            id=f"{cid}-t{i}",
            conversation_id=cid,
            index=i,
            role=Role.USER if i % 2 == 0 else Role.ASSISTANT,
            content=f"turn {i} content — roughly 50 chars of text.",
            blocks=[TextBlock(text=f"turn {i}")],
        )
        for i in range(turns_n)
    ]
    return conv, turns


def test_heuristic_counter_returns_positive() -> None:
    _, turns = _conv()
    n = heuristic_counter("claude-opus-4-7", turns)
    assert n > 0


def test_estimator_returns_per_module_entries() -> None:
    conv, turns = _conv()
    est = CostEstimator().estimate(
        convs=[conv], turns_by_conv={conv.id: turns}
    )
    modules_in_estimate = {mc.module for mc in est.per_module}
    assert modules_in_estimate == set(DEFAULT_MODULES)


def test_estimator_total_is_sum_of_per_module() -> None:
    conv, turns = _conv(turns_n=20)
    est = CostEstimator().estimate(
        convs=[conv], turns_by_conv={conv.id: turns}
    )
    assert est.total_usd == pytest.approx(sum(mc.usd for mc in est.per_module))


def test_estimator_memoizes_counter_per_model_and_conv() -> None:
    """Two modules using the same model should trigger only one count_tokens per conv."""
    calls: list[tuple[str, int]] = []

    def counting_counter(model: str, turns: list[Turn]) -> int:
        calls.append((model, len(turns)))
        return 1000

    conv, turns = _conv()
    estimator = CostEstimator(count_tokens=counting_counter)
    # DEFAULT_MODULES has 6 entries and all use claude-opus-4-7 per MODULE_PROFILES.
    estimator.estimate(convs=[conv], turns_by_conv={conv.id: turns})
    # One call per (model, conv) tuple — not per module.
    assert len(calls) == 1


def test_estimator_module_d_only_included_when_requested() -> None:
    conv, turns = _conv()
    default = CostEstimator().estimate(
        convs=[conv], turns_by_conv={conv.id: turns}
    )
    with_d = CostEstimator().estimate(
        convs=[conv],
        turns_by_conv={conv.id: turns},
        enabled_modules=[*DEFAULT_MODULES, ModuleName.D_PERSPECTIVE],
    )
    assert ModuleName.D_PERSPECTIVE not in default.by_module_dict
    assert ModuleName.D_PERSPECTIVE in with_d.by_module_dict
    assert with_d.total_usd > default.total_usd


def test_cost_gate_boundary() -> None:
    """exceeds_gate() is strict-greater-than COST_GATE_USD."""
    # Fabricate a large estimate via a counter that reports many tokens.
    conv, turns = _conv()

    def many_tokens_counter(model: str, turns: list[Turn]) -> int:
        return 10_000_000  # 10M tokens per conv per model

    big = CostEstimator(count_tokens=many_tokens_counter).estimate(
        convs=[conv], turns_by_conv={conv.id: turns}
    )
    assert big.total_usd > COST_GATE_USD
    assert big.exceeds_gate() is True
    assert big.exceeds_gate(gate_usd=big.total_usd + 1) is False


def test_module_profile_prices_exist_in_price_table() -> None:
    """Every MODULE_PROFILE's model must be in PRICES so estimate() can look it up."""
    for profile in MODULE_PROFILES.values():
        assert profile.model in PRICES, f"Missing price for {profile.model}"


def test_cached_input_tokens_never_exceed_input_tokens() -> None:
    conv, turns = _conv()
    est = CostEstimator().estimate(
        convs=[conv], turns_by_conv={conv.id: turns}
    )
    for mc in est.per_module:
        assert mc.cached_input_tokens <= mc.input_tokens
