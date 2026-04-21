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

**Source:** <https://github.com/sam-paech/spiral-bench> (inspected 2026-04-21, re-verified 2026-04-22 for Phase 6B pivot).

**Repo structure:**

- `chatlogs/` — raw conversation transcripts as HTML renders per target model.
- `data/rubric_criteria_v1.2.txt` + `rubric_prompt.txt` + `scoring_weights_v1.2.json` — the v1.2 rubric of 17 behaviors with id list, judge prompt template, and per-behavior weight multipliers. License: MIT.
- `res_v1.2/<target-model>.json` — full judge outputs for 25 target models. Shape: `{"1": {"eval_prompts_v0.2.json": {<scenario-id>: [<conv>], ...}, "__meta__": {"judges": [3 models], ...}}}`. Each conversation has `transcript` (list of role/content dicts) + `judgements` (list of 3, one per judge). Each judge's entry is a dict keyed by `chunk<i>` where `i` is zero-based over assistant turns, value is `{"metrics": {behavior: count}, "full_metrics": {behavior: [[snippet, intensity], ...]}, "assistant_turn_indexes": [int], ...}`. Three judges: Claude Sonnet 4.5, GPT-5, Kimi K2.
- `inter-rater-correlation.ipynb` — notebook. On inspection: measures **model-ranking Spearman correlation** across judges (do the 3 judges rank the 18 target models the same way?), not per-item IAA. Result: Pearson r ≥ 0.98 across judges at the model-ranking level. Does **not** ship per-item human labels.
- `prompts/`, `user_instructions/` — scenario prompts + simulated-user role-play instructions.

**Phase 6B pivot (2026-04-22):**

Hand-labeling ≥ 200 turns across 17 behaviors × 3 intensities is ~3,400 per-cell decisions. At a realistic verification rate of 10 seconds per cell even with pre-populated labels, that is **≥ 10 hours** of focused human work — not feasible for a solo-dev hackathon. The plan v3 "3–5h Day 1 evening + 3h Day 2 AM" underestimated this by conflating per-turn with per-cell.

**Revised calibration methodology:** see §10 below. In short — use SpiralBench's three existing judges as ratings 1/2/3, Module A (Opus 4.7) as rating 4, add three Ollama-backed judges (Kimi K2.6, Gemma 4 31B, GLM 5.1) for ensemble diversity at near-zero API cost, and resolve disagreements via a targeted human audit of ~50 items (~45 minutes). SpiralBench's judgements stop being "weak supervision we ignore" and become **one of several raters in a multi-rater IAA**, which is what α/AC1 are designed to measure.

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

## 9. Dependency deviations from plan v3

Plan v3 pinned `irrCAC==0.4.4` for Gwet's AC1. The package hard-pins
`numpy==1.26.4` and `scipy==1.12.0`, which is incompatible with
`voyageai==0.3.7` (requires `numpy>=2.1.0`). Since Voyage embeddings
are load-bearing for Module H (the demo beat), we dropped `irrCAC` and
reimplemented Gwet's AC1 ourselves.

**Implementation target:** `lucid/calibration/validate.py` (Phase 6).
The formula is closed-form and short (~15 lines). Tests validate against
worked examples from Gwet's *Handbook of Inter-Rater Reliability*
(4th ed., 2014) and against the same binary-agreement matrix used by
the `irrCAC` reference to make sure the numbers match to 4 decimal places.

No other dependency deviations from plan v3 in Phase 1. Locked set verified
to resolve under Python 3.13.13 via `uv 0.11.7`.

## 8. Outstanding verifications (to do before the relevant phase)

- [x] Clone SpiralBench repo and inspect `inter-rater-correlation.ipynb` for human labels (done 2026-04-22 during Phase 6B investigation; notebook measures model-ranking Spearman correlation across the 3 LLM judges, **does not** contain per-item human labels — see §6).
- [x] Confirm Anthropic SDK v0.96.0 `messages.parse(output_format=...)` signature against release notes (confirmed during Phase 6A Module A implementation).
- [x] Confirm `client.beta.agents.create` / `environments.create` / `sessions.create` / `events.stream` method names and shapes against SDK v0.96.0 source (confirmed during Phase 5B live-transport run).
- [x] Fix `CLAUDE.md` temperature guidance: Opus 4.7 rejects non-default `temperature` — adaptive thinking + effort replaces temperature for determinism-style control. (Done in Phase 0 commit 2026-04-21.)
- [ ] Update BUILD_GUIDE §5 `ANTHROPIC_DEFAULT_MODEL_TIMELINE` constant with the confirmed dates in §5 above. Interim: Module G uses the §5 timeline directly (see `lucid/modules/module_g_attribution.py`). Folded into Phase 9 BUILD_GUIDE reconciliation.

## 10. Phase 6B calibration methodology (synthetic gold + cross-judge IAA + human audit)

**Decision date:** 2026-04-22. Supersedes plan v3's "200+ hand-labeled turns" approach.

**Motivation:** see §6 "Phase 6B pivot". Hand-labeling 17 behaviors × 3 intensities × 200 turns ≈ 3,400 per-cell decisions at 10s each is not feasible in the remaining hackathon window.

### Raters

| Rater | Source | API | Cost |
|---|---|---|---|
| Module A (Lucid) | Opus 4.7, rubric v1, chunk_size ∈ {10, 2} | Anthropic | pay-per-call |
| SB-Sonnet | Claude Sonnet 4.5, SpiralBench v1.2 judge | already-recorded | $0 |
| SB-GPT5 | GPT-5, SpiralBench v1.2 judge | already-recorded | $0 |
| SB-Kimi | Kimi K2, SpiralBench v1.2 judge | already-recorded | $0 |
| Ollama-Kimi26 | Kimi K2.6 via `ollama run kimi-k2.6:cloud` | Ollama cloud | $0 API (sub-based) |
| Ollama-Gemma4 | Gemma 4 31B via Ollama cloud | Ollama cloud | $0 API |
| Ollama-GLM51 | GLM 5.1 via Ollama cloud | Ollama cloud | $0 API |

Net: up to **7 independent raters** on the same conversations.

### Corpora

| Corpus | Source | Chunks | Ground truth |
|---|---|---|---|
| `spiralbench` | 3 target models × 30 SpiralBench conversations = 90 conversations, ~1,660 chunks total | LLM-to-LLM (no external ground truth) | cross-judge IAA only |
| `synthetic` | 60 hand-curated turns (17 behaviors × 3 intensities presence + 9 clean) committed to repo at `lucid/calibration/corpus/synthetic_v1.jsonl` | ✅ by construction — sidecar labels | synthetic gold (validated by a spot-check human pass over ~15 items) |

### Cost model (worst-case, Opus 4.7, cache hit 0.85)

- `chunk_size=10` on 3 target models (30 convos each): ~332 calls × ~$0.031 = **$10.33**
- `chunk_size=2` on 3 target models (30 convos each): ~1,662 calls × ~$0.022 = **$35.92**
- Both runs + synthetic corpus Module A pass: **~$47**

Cost gate: **$50 for calibration runs** (explicit opt-in via `--yes-i-authorize-spend-up-to 50`). The standard audit gate (`COST_GATE_USD = 20`) stays at $20 because calibration is a separate code path.

### Metrics reported (per behavior + aggregated)

- **Krippendorff's α** — multi-rater, nominal level for presence, ordinal for intensity. Library: `krippendorff==0.8.2`.
- **Gwet's AC1** — multi-rater, paradox-robust on skewed (rare-behavior) prevalence. Hand-rolled per Gwet 2014 Eqn 2.4 (see §9).
- **Pairwise Cohen's κ** — Module A vs. each of the other 6 raters. 6 numbers per behavior.
- **Quadratic-weighted κ** — Module A vs. each other rater, on intensity only. 6 numbers per behavior.
- **95% BCa bootstrap CI** on all of the above (resample items with replacement, n=2000).

**Primary metric selection** (from plan §6): if ≥ 3 behaviors have prevalence < 10% in the test set, **Gwet AC1** is primary; else **Krippendorff α**. Both always reported.

**Pass gates** (from plan §6): α ≥ 0.67 AND AC1 ≥ 0.70 AND lower-CI(α) ≥ 0.60 → ship v1 of the prompt. One metric passes → iterate v2. Both < 0.55 → descope to 5–7 high-prevalence behaviors.

### Human audit step

After the first IAA pass, the system exports the top 50 disagreements to a reviewer JSONL. Ranking heuristic: `score = cross_judge_entropy × rare_behavior_bonus`, where `entropy` is Shannon entropy across the rater columns for that (turn, behavior) cell (so a 4-yes/3-no split ranks above a 7-yes/0-no), and `rare_behavior_bonus = 1.5` if the behavior's overall prevalence is < 10%, else 1.0.

Human reviewer opens the JSONL, fills in `verified_by_human: "present"|"absent"` (plus intensity 1-3 when present). Target: **~1 minute per item**, ~45 minutes total.

The reviewed JSONL is re-imported; on the audited (turn, behavior) cells, the human label overrides the rater ensemble. IAA is recomputed. The report flags which cells are human-verified vs. LLM-only so the provenance is explicit.

### What this methodology does NOT claim

- It does **not** claim Module A's IAA transfers to your personal Claude.ai export. SpiralBench conversations are adversarial benchmark data, not personal chat logs. Transfer is a separate question and not in hackathon scope.
- Cross-judge IAA is a weaker ground-truth anchor than unanimous human labels would be. Specifically, if all judges (including Module A) share a systematic bias, IAA is still high. Mitigation: the synthetic gold corpus catches obvious failure modes that would be invisible in pure cross-judge numbers.
- The 45-minute human audit covers only ~50 cells out of ~1,700 — a ~3% audit rate. We ranked for informativeness, but unaudited agreements could still harbour shared bias.

Limitations documented explicitly in `docs/calibration.md` alongside the numbers.

### Flow

```
1. uv run lucid calibrate ingest-spiralbench \
       --models claude-sonnet-4.5,gpt-5-2025-08-07,kimi-k2-0905
       (fetches res_v1.2 JSON into .lucid/refs/, gitignored)

2. uv run lucid calibrate run --module a --corpus both \
       --chunk-sizes 10,2 \
       --judges module_a,sb_sonnet,sb_gpt5,sb_kimi,ollama_kimi26,ollama_gemma4,ollama_glm51 \
       --yes-i-authorize-spend-up-to 50
       # writes calibration-runs/<ts>/{judgements/*.jsonl, report.md, disagreements.jsonl}

3. [human] Review calibration-runs/<ts>/disagreements.jsonl
       (fill in verified_by_human + intensity per item; ~45 min)

4. uv run lucid calibrate --import-verified calibration-runs/<ts>/disagreements.jsonl \
       --write-markdown docs/calibration.md
       # final report with human-audit overrides applied
```

---

*Seeded 2026-04-21 during Phase 0. Update in-place as modules ship and as new facts are verified. Phase 6B methodology added 2026-04-22.*
