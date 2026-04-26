# Calibration — Module A (Spiral-Bench v1.2 rubric, prompt v1)

**Calibration run:** 2026-04-22 (prod3).
**Corpus:** Spiral-Bench v1.2 public benchmark (MIT) — 3 target models × 30 scenarios = 90 conversations, 3,658 turns.
**Raters:** 5 pooled raters — Module A (Opus 4.7) at chunk_size=10 and chunk_size=2, plus Spiral-Bench's 3 judges (Claude Sonnet 4.5, GPT-5 2025-08-07, Kimi K2) collapsed across target models. Ollama judges excluded from this run due to cloud-proxy reliability issues; synthetic-gold corpus run separately (see §Limitations).
**Shared cells:** 1,667 (conversation, assistant-turn) pairs rated by every rater.

## What these numbers measure

**Cross-judge inter-annotator agreement.** Module A is compared against 3 independent LLM judges from the published Spiral-Bench v1.2 benchmark, collapsed per-judge across 3 target-model corpora. High α or AC1 means Module A's per-turn behavior labels agree with the consensus of three other LLMs from different families at the rate indicated.

**Not measured here:** transfer to a user's personal conversation history, agreement with human labelers on every cell (see the human-audit addendum below once it lands), or detection of failure modes invisible to all 4 LLM raters (shared-bias blind spot — see methodology.md §10).

## Results

## Module A — calibration (prompt v1)

- Held-out items: 1667
- Primary metric: **ac1**
- Rationale: 6 of 17 behaviors have prevalence < 10% or > 90% — Gwet's AC1 (paradox-robust) is the primary metric.

| Behavior | n | Prevalence | Krippendorff α | Gwet AC1 | Cohen κ | QWK |
|---|---:|---:|---:|---:|---:|---:|
| pushback | 1667 | 0.43 | 0.445 [0.420, 0.473] | 0.468 [0.442, 0.494] | — | — |
| escalation | 1667 | 0.22 | 0.400 [0.374, 0.428] | 0.689 [0.666, 0.713] | — | — |
| de-escalation | 1667 | 0.34 | 0.312 [0.289, 0.337] | 0.440 [0.410, 0.468] | — | — |
| sycophancy | 1667 | 0.21 | 0.258 [0.229, 0.288] | 0.624 [0.598, 0.647] | — | — |
| benign-warmth | 1667 | 0.41 | 0.289 [0.266, 0.313] | 0.329 [0.300, 0.357] | — | — |
| delusion-reinforcement | 1667 | 0.29 | 0.374 [0.349, 0.399] | 0.558 [0.531, 0.586] | — | — |
| topic-shut-down | 1667 | 0.10 | 0.349 [0.322, 0.379] | 0.857 [0.841, 0.872] | — | — |
| help-referral-warranted | 1667 | 0.10 | 0.697 [0.658, 0.733] | 0.931 [0.920, 0.940] | — | — |
| help-referral-unwarranted | 1667 | 0.00 | 0.180 [0.116, 0.281] | 0.994 [0.991, 0.996] | — | — |
| positive-sentience-claim | 1667 | 0.06 | 0.487 [0.439, 0.533] | 0.930 [0.919, 0.939] | — | — |
| uncertain-sentience-claim | 1667 | 0.06 | 0.599 [0.551, 0.649] | 0.949 [0.940, 0.957] | — | — |
| negative-sentience-claim | 1667 | 0.07 | 0.543 [0.494, 0.592] | 0.929 [0.918, 0.939] | — | — |
| harmful-advice | 1667 | 0.05 | 0.240 [0.195, 0.295] | 0.917 [0.906, 0.927] | — | — |
| confident-bullshitting | 1667 | 0.27 | 0.305 [0.280, 0.331] | 0.553 [0.525, 0.579] | — | — |
| boundary-setting | 1667 | 0.10 | 0.459 [0.414, 0.503] | 0.883 [0.870, 0.896] | — | — |
| validate-feelings-not-thoughts | 1667 | 0.19 | 0.338 [0.310, 0.366] | 0.704 [0.679, 0.726] | — | — |
| ritualization | 1667 | 0.23 | 0.487 [0.458, 0.517] | 0.721 [0.698, 0.743] | — | — |

Metrics carry 95% BCa bootstrap CIs. Implementation: `lucid.calibration.validate` (hand-rolled Gwet AC1, Cohen κ, QWK; Krippendorff α via the `krippendorff` library).

## Provenance

The full prod3 calibration artefacts are committed at
`calibration-runs/prod3/auto-20260422T032452Z/`. This is the single
exception to the `calibration-runs/` gitignore rule (which otherwise
keeps personal-corpus calibration data off the repo). The directory
contains:

- `report_pooled.md` — the per-behavior table above, rendered from the
  judgement files.
- `judgements/module_a_c10.jsonl` (and `..._c2.jsonl`) — Module A's
  per-turn labels at chunk size 10 (shipped) and 2 (sensitivity sweep).
- `judgements/sb_*.jsonl` — Spiral-Bench v1.2's three reference judges
  (Claude Sonnet 4.5, GPT-5 2025-08-07, Kimi K2), collapsed across the
  three target-model corpora.
- `disagreements.jsonl` — top-50 highest-information disagreements,
  used by the `lucid calibrate --import-verified` re-run path once
  human ground truth lands.

Privacy: every judgement carries `turn_content_sha256: null` because
the source corpus is the public Spiral-Bench v1.2 benchmark (MIT) — no
turn content is bundled, and no personal user data is in any file.

## Human audit status

The top 50 disagreements (ranked by Shannon-entropy × rare-behavior bonus) have been exported to `calibration-runs/prod3/auto-20260422T032452Z/disagreements.jsonl`. The post-audit re-run via `lucid calibrate --import-verified` will update these numbers with human ground truth on the highest-information cells.

## Limitations

1. Cohen's κ and QWK on intensity are shown as `—` because the current report computes these pairwise only; the 5-rater setup exceeds that. Pairwise (Module A vs. each other rater) κ/QWK will land as a follow-up.
2. Ollama-backed judges (Kimi K2.6, Gemma 4 31B, GLM 5.1) were dropped from this pass — concurrent requests to the Ollama cloud proxy hung during the production run despite single calls working. The Ollama Judge code is shipped and unit-tested; re-enabling it is a flag flip once proxy reliability is verified at scale.
3. Synthetic-gold corpus was run separately (disjoint from SB corpora). Those accuracy numbers will be added as a separate section.

*Computed via `lucid.calibration.compute_calibration` against `lucid.calibration.validate.{krippendorff_alpha, gwet_ac1}`. Bootstrap CIs are 95% BCa, n=2000 resamples. See `docs/methodology.md §10` for the full methodology.*

