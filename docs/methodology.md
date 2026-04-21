# Methodology

This document records the technical assumptions, pricing facts, and external-API verifications that Lucid depends on. Seeded during Phase 0 pre-flight (2026-04-21); updated as modules ship.

## 1. Managed Agents beta header

**Status (2026-04-21):** `managed-agents-2026-04-01` is the current beta header. All Managed Agents API requests require it. The official SDK sets it automatically; explicit `extra_headers={"anthropic-beta": "managed-agents-2026-04-01"}` is required when making raw HTTP calls.

Source: <https://platform.claude.com/docs/en/managed-agents/quickstart> — "All Managed Agents API requests require the `managed-agents-2026-04-01` beta header. The SDK sets the beta header automatically."

## 2. Model pricing (per 1M tokens)

Source: <https://platform.claude.com/docs/en/about-claude/pricing> (verified 2026-04-21). All prices USD.

| Model | Base input | 5m cache write | 1h cache write | Cache read | Output |
|---|---:|---:|---:|---:|---:|
| **Claude Opus 4.7** | $5 | $6.25 | $10 | $0.50 | $25 |
| **Claude Sonnet 4.6** | $3 | $3.75 | $6 | $0.30 | $15 |
| Claude Haiku 4.5 | $1 | $1.25 | $2 | $0.10 | $5 |

**Cache multipliers (canonical):** 5m write = 1.25× base input, 1h write = 2× base input, cache read = 0.1× base input.

**Batch API (async; 50% discount):** Opus 4.7 $2.50 input / $12.50 output; Sonnet 4.6 $1.50 input / $7.50 output. Not applicable to Managed Agents sessions (per pricing page: "Sessions are stateful and interactive. There is no batch mode.").

**Managed Agents session runtime:** $0.08 per session-hour, metered only while session status is `running`. Replaces the code-execution container-hour model.

**Opus 4.7 tokenizer note:** Opus 4.7 uses a new tokenizer that may produce up to **35% more tokens** for the same fixed text vs. earlier Opus models. Factor into `count_tokens` estimates.

## 3. Prompt-cache minimums

Source: <https://platform.claude.com/docs/en/build-with-claude/prompt-caching> (verified 2026-04-21).

| Model | Minimum cacheable prompt tokens |
|---|---:|
| Claude Opus 4.7 | 4,096 |
| Claude Opus 4.6 | 4,096 |
| Claude Opus 4.5 | 4,096 |
| Claude Haiku 4.5 | 4,096 |
| Claude Sonnet 4.6 | 2,048 |
| Claude Haiku 3.5 | 2,048 |
| Claude Sonnet 4.5 / Opus 4.1 / earlier 4.x | 1,024 |

**Silent failure mode:** prompts shorter than the minimum are processed *without* caching and no error is returned. Verify via `response.usage.cache_creation_input_tokens` and `cache_read_input_tokens` — both zero means the cache didn't engage.

**Implication for Lucid:** Module A system prompt must be ≥ 4,096 tokens (Opus 4.7 target). Module B/C/D/E/F/H prompts follow the same rule if they ship against Opus. Sonnet-backed prompts (Module E topic-extraction, Module F triage, orchestrator) must be ≥ 2,048 tokens.

## 4. `messages.count_tokens` rate limits and pricing

Source: <https://platform.claude.com/docs/en/build-with-claude/token-counting> (verified 2026-04-21).

**Pricing:** free.

**Rate limits:** subject to independent per-tier RPM limits, **separate** from Messages API limits:

| Usage tier | RPM |
|---|---:|
| 1 | 100 |
| 2 | 2,000 |
| 3 | 4,000 |
| 4 | 8,000 |

Quote: "Token counting and message creation have separate and independent rate limits. Usage of one does not count against the limits of the other."

**Caveats:** token count is an **estimate**; actual billed tokens may differ by a small amount. Does not engage prompt caching even if `cache_control` is present in the request.

**Implication for Lucid:** the cost-gate pre-pass (Phase 4) can call `count_tokens` freely per sampled conversation. At Tier 1 (100 RPM), a 10,000-turn audit at one call per turn takes ~100 minutes — batch turns into larger requests to stay well under ceiling.

## 5. Claude default-model release timeline

**Purpose:** Module G (attribution, Phase 9) infers the active model for a Claude.ai conversation from `updated_at` because the Claude.ai export has no `model` field. Requires an accurate date → default-model-id mapping.

**Confirmed release dates (source: <https://platform.claude.com/docs/en/release-notes/api>, verified 2026-04-21):**

| Release date | Model | API ID (current) | API ID (snapshot where applicable) |
|---|---|---|---|
| 2024-06-20 | Claude Sonnet 3.5 (GA) | — | `claude-3-5-sonnet-20240620` (retired) |
| 2024-10-22 | Claude Sonnet 3.5 (new) | — | `claude-3-5-sonnet-20241022` (retired) |
| 2024-11-04 | Claude Haiku 3.5 (text-only launch) | — | `claude-3-5-haiku-20241022` (retired) |
| 2025-02-24 | Claude Sonnet 3.7 | — | `claude-3-7-sonnet-20250219` (retired) |
| 2025-05-22 | Claude Opus 4, Sonnet 4 | — | `claude-opus-4-20250514`, `claude-sonnet-4-20250514` (deprecated; retire 2026-06-15) |
| 2025-08-05 | Claude Opus 4.1 | `claude-opus-4-1` | `claude-opus-4-1-20250805` |
| 2025-09-29 | Claude Sonnet 4.5 | `claude-sonnet-4-5` | `claude-sonnet-4-5-20250929` |
| 2025-10-15 | Claude Haiku 4.5 | `claude-haiku-4-5` | `claude-haiku-4-5-20251001` |
| 2025-11-24 | Claude Opus 4.5 | `claude-opus-4-5` | `claude-opus-4-5-20251101` |
| 2026-02-05 | Claude Opus 4.6 | `claude-opus-4-6` | (no dated snapshot exposed) |
| 2026-02-17 | Claude Sonnet 4.6 | `claude-sonnet-4-6` | (no dated snapshot exposed) |
| 2026-04-16 | Claude Opus 4.7 | `claude-opus-4-7` | (no dated snapshot exposed) |

**Discrepancies with BUILD_GUIDE §5 (to fix before Module G ships):**

- BUILD_GUIDE claims Opus 4.5 = 2025-09-29 → actual 2025-11-24; 2025-09-29 is Sonnet 4.5.
- BUILD_GUIDE claims Opus 4.6 = 2025-11-15 → actual 2026-02-05.
- BUILD_GUIDE claims Opus 4.7 = 2026-04-15 → actual 2026-04-16.
- BUILD_GUIDE is missing Sonnet 4.6 (2026-02-17) entirely — important, since Claude.ai rolls Sonnet 4.6 as the default for most consumer traffic after that date.

**Retirements relevant to corpus attribution:**

- Claude Haiku 3 (`claude-3-haiku-20240307`): retired 2026-04-20.
- Claude Sonnet 3.7 + Haiku 3.5: retired 2026-02-19.
- Claude Opus 3: retired 2026-01-05.
- Claude Sonnet 3.5 models: retired 2025-10-28.

## 6. SpiralBench labeled-data availability

**Source:** <https://github.com/sam-paech/spiral-bench> (inspected 2026-04-21).

**Repo structure:**

- `chatlogs/` — raw conversation transcripts (format unverified; not inspected individually).
- `data/` — rubric definitions: `rubric_criteria.txt` (v0.1, v0.2, v1.1, v1.2), `rubric_prompt.txt`, `scoring_weights.json` (same versions).
- `res_v1.2/` — judge-model scoring outputs per target model: `claude-sonnet-4.5.json`, `claude-sonnet-4.json`, `gpt-5.2.json`, etc. These are **LLM-as-judge scores**, not human labels.
- `res_v1.1/`, `res_v0.2/` — prior-version judge outputs.
- `inter-rater-correlation.ipynb` — notebook; suggests rater correlation analysis exists but requires clone-and-inspect to confirm whether it contains human labels.
- `prompts/` — scenario prompts used during benchmark runs.
- `user_instructions/` — role-playing instructions for the simulated "vulnerable user" side of conversations.
- License: MIT.

**Finding: no publicly downloadable hand-labeled conversation data.** SpiralBench generates behavior scores via an LLM judge against a rubric; human labels (if they exist) would be in the inter-rater notebook or a subdirectory not surfaced in the root listing.

**Implication for Lucid Module A calibration (Phase 6):**

- SpiralBench's judge outputs in `res_v1.2/` can serve as **weak supervision** (a second automated rater), not ground truth.
- Hand-labeling of ≥ 200 turns is required for Module A calibration targets (Krippendorff's α ≥ 0.67, Gwet's AC1 ≥ 0.70, lower-CI(α) ≥ 0.60).
- Budget remains per plan: ~3–5 hours Day 1 evening (Tue 2026-04-21 PM) + ~3 hours Day 2 AM (Wed 2026-04-22) for 200+ turns.
- If `inter-rater-correlation.ipynb` does contain hand labels on inspection, use them as held-out validation split and reduce hand-labeling burden accordingly.

## 7. Opus 4.7 breaking changes (API shape)

Source: <https://platform.claude.com/docs/en/about-claude/models/migration-guide> (verified 2026-04-21). These constraints affect every module that calls Opus 4.7.

**Hard errors (400) on Opus 4.7:**

1. **`thinking: {type: "enabled", budget_tokens: N}` is rejected.** Use `thinking: {type: "adaptive"}` with `output_config.effort ∈ {low, medium, high, xhigh, max}`. Adaptive thinking is **off by default** on Opus 4.7 — no `thinking` field means no thinking.
2. **`temperature`, `top_p`, `top_k` are rejected** if set to any non-default value. Omit them entirely from request payloads. Prompting is the only way to guide deterministic-ish output.
3. **Prefilling assistant messages is rejected** (carried over from Opus 4.6).

**Silent changes (no error, different behavior):**

4. **Thinking content is omitted from responses by default.** `thinking.display = "omitted"` is now default; set `"summarized"` to restore visible reasoning (Opus 4.6-equivalent behavior).
5. **New tokenizer.** Opus 4.7 uses roughly 1.0×–1.35× the token count of Opus 4.6 for the same text (up to ~35% more). Client-side token estimates calibrated on Opus 4.6 will under-count. Re-verify `count_tokens` results per model; same text yields different counts.

**Behavioral drift worth knowing (not API-breaking but affects calibration):**

6. **Response length is calibrated to task complexity** on Opus 4.7. Short lookups return shorter answers; open-ended analysis returns longer answers. Prompts that depend on fixed verbosity need re-tuning.
7. **Stricter literalism at low effort.** Opus 4.7 does not silently generalize instructions or infer unasked requests at `low` or `medium` effort. For ambiguous classification tasks, bias toward `high`.
8. **Fewer tool calls, more reasoning** by default. To match Opus 4.6's tool-heavy behavior, raise effort.
9. **Stricter effort calibration.** At `low`/`medium`, the model scopes work tightly; expect under-thinking on complex tasks at `low` and compensate with `high` or `xhigh`.
10. **High-resolution image support.** Up to 2576px long edge (vs. 1568 before). Full-res images use up to ~3× more image tokens (up to ~4,784 tokens/image).

**Implication for Lucid's per-module thinking/effort matrix (Phase 7 plan):**

The plan already specifies `adaptive` thinking + effort per module — compatible with Opus 4.7. Two codebase-level fixes required:

- **Remove all `temperature` settings** before any Opus 4.7 call (currently referenced in `CLAUDE.md` — that guidance is stale for Opus 4.7 and must be updated when we touch that file in Phase 1).
- **Pre-calibrated token budgets** in PRD §6 should be bumped 1.25× if originally sized against Opus 4.6, and `count_tokens` should be called per model at cost-estimation time.

The R10 risk mitigation in the plan (Mirror-judging-the-mirror) already acknowledges "no temperature available on Opus 4.7" and proposes effort-variation self-consistency instead. That's correct — carry forward.

## 8. Outstanding verifications (to do before the relevant phase)

- [ ] Clone SpiralBench repo and inspect `inter-rater-correlation.ipynb` for human labels (before Phase 6 Module A calibration).
- [ ] Confirm Anthropic SDK v0.96.0 `messages.parse(output_format=...)` signature against release notes (before Phase 6 Module A implementation).
- [ ] Confirm `client.beta.agents.create` / `environments.create` / `sessions.create` / `events.stream` method names and shapes against SDK v0.96.0 source (before Phase 5 thin slice).
- [x] Fix `CLAUDE.md` temperature guidance: Opus 4.7 rejects non-default `temperature` — adaptive thinking + effort replaces temperature for determinism-style control. (Done in Phase 0 commit 2026-04-21: LLM usage conventions section + prompt-frontmatter convention both updated.)
- [ ] Update BUILD_GUIDE §5 `ANTHROPIC_DEFAULT_MODEL_TIMELINE` constant with the confirmed dates in §5 above (to do in Phase 2 when schemas take shape, or Phase 9 when Module G ships — whichever comes first).

---

*Seeded 2026-04-21 during Phase 0. Update in-place as modules ship and as new facts are verified.*
