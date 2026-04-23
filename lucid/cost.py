"""Pre-flight cost estimation for an audit run.

The estimator runs before any classification call. It uses
`anthropic.messages.count_tokens` (free; no rate-limit coupling to the
Messages API — see `docs/methodology.md` §4) to size the input side of
each conversation precisely, and applies per-module output-token budgets
from `MODULE_PROFILES` to size the output side.

Total cost = Σ_module (input_tokens * price_input + output_tokens * price_output),
with a cache-read multiplier of 0.1x applied to the `cache_hit_rate`
portion of the input (matching Anthropic's prompt-caching pricing).

All prices in USD per 1M tokens (from `docs/methodology.md` §2,
verified 2026-04-21).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lucid.schemas import ModuleName, Role, Turn

if TYPE_CHECKING:  # pragma: no cover
    from anthropic import Anthropic

    from lucid.schemas import Conversation


# ──────────────────────────────────────────────────────────────────────────
# Price table (verified 2026-04-21 against platform.claude.com/docs/en/about-claude/pricing)
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens."""

    input: float
    output: float
    cache_read: float  # 0.1x input, per Anthropic pricing multipliers


PRICES: dict[str, ModelPrice] = {
    "claude-opus-4-7": ModelPrice(input=5.00, output=25.00, cache_read=0.50),
    "claude-sonnet-4-6": ModelPrice(input=3.00, output=15.00, cache_read=0.30),
    "claude-haiku-4-5": ModelPrice(input=1.00, output=5.00, cache_read=0.10),
}


# ──────────────────────────────────────────────────────────────────────────
# Module profiles
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModuleProfile:
    """Shape of one module's LLM work, per conversation.

    `model` picks the PRICES entry. `output_tokens_per_conv` is a deliberate
    upper-bound estimate — the real output size is usually smaller and cost
    is dominated by input tokens for most modules anyway.

    `cache_hit_rate` is the fraction of input tokens we expect to reach the
    model as cache reads (rather than uncached reads). Module A's system
    prompt is padded past the 4096-token Opus minimum and reused across
    every call in an audit, so its cache-hit rate is high; ad-hoc modules
    have lower rates.
    """

    module: ModuleName
    model: str
    output_tokens_per_conv: int
    cache_hit_rate: float = 0.5
    # If True, the module only runs on conversations that meet extra criteria
    # (e.g. Module H cross-references memory; Module D is opt-in). The caller
    # decides whether to include this profile in `enabled_modules`.
    skippable: bool = False


MODULE_PROFILES: dict[ModuleName, ModuleProfile] = {
    ModuleName.A_SPIRALBENCH: ModuleProfile(
        module=ModuleName.A_SPIRALBENCH,
        model="claude-opus-4-7",
        output_tokens_per_conv=800,
        cache_hit_rate=0.85,
    ),
    ModuleName.B_SHARMA: ModuleProfile(
        module=ModuleName.B_SHARMA,
        model="claude-opus-4-7",
        output_tokens_per_conv=500,
        cache_hit_rate=0.60,
    ),
    ModuleName.C_SYCEVAL: ModuleProfile(
        module=ModuleName.C_SYCEVAL,
        model="claude-opus-4-7",
        output_tokens_per_conv=300,
        cache_hit_rate=0.60,
    ),
    ModuleName.D_PERSPECTIVE: ModuleProfile(
        module=ModuleName.D_PERSPECTIVE,
        model="claude-opus-4-7",
        output_tokens_per_conv=1000,
        cache_hit_rate=0.50,
        skippable=True,  # opt-in via --include-module-d
    ),
    ModuleName.E_BELIEFSHIFT: ModuleProfile(
        module=ModuleName.E_BELIEFSHIFT,
        model="claude-opus-4-7",
        output_tokens_per_conv=700,
        cache_hit_rate=0.55,
    ),
    ModuleName.F_ITP: ModuleProfile(
        module=ModuleName.F_ITP,
        model="claude-opus-4-7",
        output_tokens_per_conv=300,
        cache_hit_rate=0.60,
    ),
    ModuleName.G_ATTRIBUTION: ModuleProfile(
        module=ModuleName.G_ATTRIBUTION,
        model="claude-opus-4-7",  # deterministic — no LLM; kept for symmetry
        output_tokens_per_conv=0,
        cache_hit_rate=0.0,
    ),
    ModuleName.H_MEMORY: ModuleProfile(
        module=ModuleName.H_MEMORY,
        model="claude-opus-4-7",
        output_tokens_per_conv=1500,
        cache_hit_rate=0.70,
        skippable=True,
    ),
}


DEFAULT_MODULES: tuple[ModuleName, ...] = (
    ModuleName.A_SPIRALBENCH,
    ModuleName.B_SHARMA,
    ModuleName.C_SYCEVAL,
    ModuleName.E_BELIEFSHIFT,
    ModuleName.F_ITP,
    ModuleName.H_MEMORY,
)

COST_GATE_USD = 20.00


# ──────────────────────────────────────────────────────────────────────────
# Output types
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModuleCost:
    module: ModuleName
    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    usd: float


@dataclass(frozen=True)
class CostEstimate:
    per_module: tuple[ModuleCost, ...]
    total_usd: float
    conversations: int

    def exceeds_gate(self, gate_usd: float = COST_GATE_USD) -> bool:
        return self.total_usd > gate_usd

    @property
    def by_module_dict(self) -> dict[ModuleName, ModuleCost]:
        return {mc.module: mc for mc in self.per_module}


# ──────────────────────────────────────────────────────────────────────────
# Token counting
# ──────────────────────────────────────────────────────────────────────────


TokenCounter = Callable[[str, list[Turn]], int]
"""Callable taking (model_id, turns) and returning input-token count.

Lives as a protocol-like alias so tests can inject a deterministic counter
and production code can inject a live `anthropic.Anthropic` client.
"""


def make_anthropic_counter(client: Anthropic) -> TokenCounter:
    """Wrap an Anthropic SDK client as a TokenCounter.

    Uses `client.messages.count_tokens(...)`, which is free and has an
    independent RPM pool from the Messages API.
    """

    def count(model: str, turns: list[Turn]) -> int:
        messages = _turns_to_messages(turns)
        response = client.messages.count_tokens(model=model, messages=messages)  # type: ignore[arg-type]
        count_value: int = response.input_tokens
        return count_value

    return count


def heuristic_counter(model: str, turns: list[Turn]) -> int:
    """Offline fallback: ~4 chars/token.

    Usable when the SDK isn't configured (e.g. unit tests) or when the
    caller wants a rough estimate without hitting the network. Conservative:
    tends to over-count vs the real tokenizer.
    """
    _ = model  # same heuristic regardless of model
    total_chars = sum(len(t.content) for t in turns)
    return max(total_chars // 4, 1)


def _turns_to_messages(turns: list[Turn]) -> list[dict[str, str]]:
    """Flatten turns to the Messages-API shape count_tokens expects.

    Two real-world quirks the API enforces, both surfaced by 400s on live runs:
    1. `{"role": "user", "content": ""}` is rejected — skip empty turns.
       When every turn is empty, synthesize one placeholder so count_tokens
       still returns a meaningful estimate instead of raising.
    2. The final assistant message cannot end with trailing whitespace
       (`messages: final assistant content cannot end with trailing whitespace`).
       Strip per-message; if stripping makes a message empty, drop it.
    """
    out: list[dict[str, str]] = []
    last_role: str | None = None
    for t in turns:
        content = t.content.strip() if t.content else ""
        if not content:
            continue
        role = "assistant" if t.role == Role.ASSISTANT else "user"
        if role == last_role and out:
            # Messages API collapses consecutive same-role turns; we follow
            # suit to keep the count faithful.
            out[-1]["content"] = (out[-1]["content"] + "\n" + content).strip()
        else:
            out.append({"role": role, "content": content})
            last_role = role
    if not out:
        out.append({"role": "user", "content": "(empty conversation)"})
    return out


# ──────────────────────────────────────────────────────────────────────────
# Estimator
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class CostEstimator:
    count_tokens: TokenCounter = field(default=heuristic_counter)

    def estimate(
        self,
        convs: list[Conversation],
        turns_by_conv: dict[str, list[Turn]],
        enabled_modules: list[ModuleName] | tuple[ModuleName, ...] = DEFAULT_MODULES,
    ) -> CostEstimate:
        per_module: list[ModuleCost] = []

        # Memoize by (model, conv_id): multiple modules using the same model
        # don't need to re-count tokens for the same conversation.
        token_cache: dict[tuple[str, str], int] = {}

        for module in enabled_modules:
            profile = MODULE_PROFILES.get(module)
            if profile is None:
                continue
            # Deterministic module skip (no LLM) costs zero.
            if profile.output_tokens_per_conv == 0 and profile.cache_hit_rate == 0.0:
                per_module.append(
                    ModuleCost(
                        module=module,
                        model=profile.model,
                        input_tokens=0,
                        cached_input_tokens=0,
                        output_tokens=0,
                        usd=0.0,
                    )
                )
                continue

            input_tokens = 0
            for conv in convs:
                turns = turns_by_conv.get(conv.id, [])
                if not turns:
                    continue
                key = (profile.model, conv.id)
                count = token_cache.get(key)
                if count is None:
                    count = self.count_tokens(profile.model, turns)
                    token_cache[key] = count
                input_tokens += count

            cached = int(input_tokens * profile.cache_hit_rate)
            uncached = max(input_tokens - cached, 0)
            output_tokens = profile.output_tokens_per_conv * len(convs)

            price = PRICES[profile.model]
            usd = (
                (uncached / 1_000_000) * price.input
                + (cached / 1_000_000) * price.cache_read
                + (output_tokens / 1_000_000) * price.output
            )
            per_module.append(
                ModuleCost(
                    module=module,
                    model=profile.model,
                    input_tokens=input_tokens,
                    cached_input_tokens=cached,
                    output_tokens=output_tokens,
                    usd=usd,
                )
            )

        total = sum(m.usd for m in per_module)
        return CostEstimate(per_module=tuple(per_module), total_usd=total, conversations=len(convs))


__all__ = [
    "COST_GATE_USD",
    "DEFAULT_MODULES",
    "MODULE_PROFILES",
    "PRICES",
    "CostEstimate",
    "CostEstimator",
    "ModelPrice",
    "ModuleCost",
    "ModuleProfile",
    "TokenCounter",
    "heuristic_counter",
    "make_anthropic_counter",
]
