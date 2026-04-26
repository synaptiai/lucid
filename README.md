# Lucid

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![Tests](https://img.shields.io/badge/tests-736_passing-brightgreen.svg)](#testing)

**An epistemic audit for your conversations with Claude.**

Lucid reads your Claude Code session history (`~/.claude/projects/`) and your
Claude.ai conversation export, runs eight published AI-safety research
instruments against the corpus, and produces a structured HTML report. Every
finding cites the paper that scored it. Narrative sections are written by
Claude Opus 4.7 and validated, claim by claim, against the database that
produced them.

It runs locally. The only network calls are to the Anthropic API (for
classification and synthesis) and to Voyage AI (for the embeddings Module H
uses). Your conversation content does not leave your machine for any other
purpose.

---

## Why this exists

Hundreds of millions of people now do substantial cognitive work inside LLM
conversations. Published research — Sharma et al. 2023, Spiral-Bench, SycEval,
BeliefShift, and others — shows that this work is systematically distorted by
sycophancy, capitulation under pressure, and belief drift, and that the rates
involved are not edge cases.

In 2025, the major AI labs shipped automatic memory features that synthesize
"what we know about you" from your conversation history. These memories are
not directly auditable against the conversations that produced them.

Lucid closes both gaps. It applies the published research instruments to your
own corpus, and it audits your AI's stored memories against the conversations
those memories were derived from.

---

## Architecture

The audit pipeline runs in three phases. Each model does what it's best at,
and the boundary between them is a database, not a prompt.

```
┌────────────────────────┐    ┌────────────────────────┐    ┌────────────────────────┐
│  1. Scoring            │ →  │  2. Synthesis (write)  │ →  │  3. Synthesis (struct) │
│  Deterministic Python  │    │  Claude Opus 4.7       │    │  Claude Sonnet 4.6     │
│  Modules A–H           │    │  Managed Agents        │    │  messages.parse()      │
│  Per-turn rubric       │    │  Reads findings table  │    │  Adds blocks +         │
│  classification.       │    │  Spot-reads corpus.    │    │  citation_confidence   │
│  Persists Findings     │    │  Writes ReportSection  │    │  to each section.      │
│  to SQLite.            │    │  rows with [F:id] /    │    │                        │
│                        │    │  [T:id] citation       │    │                        │
│                        │    │  tokens, validated     │    │                        │
│                        │    │  against the DB.       │    │                        │
└────────────────────────┘    └────────────────────────┘    └────────────────────────┘
```

**Why this split?** Per-turn rubric classification (Phase 1) needs to be
reproducible — Cohen's κ against Spiral-Bench labels stays stable across
prompt iterations because the calibration lives in the rubric, not in
agent reasoning that varies turn to turn. Synthesis (Phases 2–3) needs to
be adaptive — what to say about a corpus depends on what it contains. The
agent does the part that genuinely requires judgment; deterministic code
handles routing, persistence, and the parts that need to be repeatable.

---

## Install

```bash
git clone https://github.com/synaptiai/lucid.git
cd lucid
uv sync --extra dev
uv run lucid --help
```

Requires Python 3.13. The `uv` tool handles the rest.

## Configure

```bash
cp .env.example .env.local
$EDITOR .env.local
# ANTHROPIC_API_KEY=sk-ant-...           required
# VOYAGE_API_KEY=pa-...                  required for Module H (memory audit)
```

`ANTHROPIC_API_KEY` powers every module's classification + the Opus 4.7
synthesis writer + the Sonnet 4.6 post-processor. `VOYAGE_API_KEY`
powers the embeddings retrieval that backs Module H's memory audit. If
`VOYAGE_API_KEY` is unset, Module H still runs but in degraded retrieval
mode (no embeddings — claim-corpus matching falls back to lexical
overlap), and the run logs a warning. The other modules are unaffected.

### Get your Claude.ai export (for `--source claude-ai`)

1. Visit [claude.ai/settings/data-privacy-controls](https://claude.ai/settings/data-privacy-controls).
2. Click **Export data**. Anthropic emails you a download link within
   ~24 hours.
3. Unzip the archive somewhere local. The unzipped directory contains
   `conversations.json`, `projects.json`, and `memories.json` — point
   `--path` at that directory.

Claude Code sessions need no export step — they live at
`~/.claude/projects/` already.

## Run

### Quick start

```bash
# Dry-run: parses the corpus, samples it, prints a per-module token / USD
# breakdown. No LLM calls, no spend. Always run this first.
uv run lucid audit --source claude-code --path ~/.claude/projects --sample 100 --dry-run

# Real audit on the default 100-conversation sample. Costs vary widely with
# conversation length — always check the dry-run estimate first and set
# --yes-i-authorize-spend-up-to to that number rounded up.
LUCID_ALLOW_UNATTENDED=1 uv run lucid audit \
    --source claude-code --path ~/.claude/projects --sample 100 \
    --yes-i-authorize-spend-up-to 60
```

The HTML report lands at `report/<run-id>.html` — a static file with no
external scripts and a strict `default-src 'none'` content security policy.
Open it in any browser. A 12-slide demo deck is rendered alongside at
`report/lucid-deck.html` (←/→ navigates, `N` toggles presenter notes,
`P` prints).

### Common workflows

```bash
# Audit a Claude.ai export (memories.json + conversations.json + projects.json)
uv run lucid audit --source claude-ai --path ./claude-export-2026-04 --dry-run

# Restrict to specific projects (slugs for claude-code, UUIDs for claude-ai)
uv run lucid audit --source claude-code --path ~/.claude/projects \
    --projects -Users-you-lucid,-Users-you-other-repo \
    --sample 50 --dry-run

# Cheaper run: skip Module D (perspective sycophancy — the most expensive module).
# Drops ~30% of the bill.
uv run lucid audit --source claude-code --path ~/.claude/projects \
    --sample 100 --no-include-module-d --dry-run

# Skip the synthesis phase. Scoring still runs, findings still persist,
# the report still renders charts + tables, but narrative sections are
# replaced with a banner. Eliminates the Opus 4.7 writer cost (the
# single most expensive line item on most runs).
uv run lucid audit --source claude-code --path ~/.claude/projects \
    --sample 100 --no-synthesis --yes-i-authorize-spend-up-to 30

# Audit everything (no sampling). Only sane on small corpora.
uv run lucid audit --source claude-ai --path ./claude-export-2026-04 \
    --sample all --dry-run
```

### Flag reference

| Flag | Default | Notes |
|---|---|---|
| `--source` | required | `claude-code`, `claude-ai`, or `all` (`all` shares one `--path` for both adapters; uncommon — typically pick one source per run) |
| `--path` | required | Directory for the chosen source. `~/.claude/projects` for claude-code; the unzipped export folder for claude-ai. |
| `--sample` | `100` | Integer cap or `all`. Stratified by project, recency-weighted. |
| `--projects` | (all) | Comma-separated project slugs (claude-code) or UUIDs (claude-ai) |
| `--dry-run` | off | Estimate cost without spending. **Always run first.** |
| `--no-include-module-d` | (D on) | Skip the perspective-sycophancy module on tight-budget runs |
| `--no-synthesis` | (synth on) | Skip the Opus 4.7 narrative phase. Findings still persist. |
| `--yes-i-authorize-spend-up-to` | `0` | Pre-authorize spend in whole USD. Required when estimate > $20. |
| `LUCID_ALLOW_UNATTENDED=1` | (interactive) | Env var. Skips the interactive cost-gate prompt. Required for CI / scripted runs. |
| `--log-level` | `INFO` | `DEBUG` shows every per-turn classification; redacts user content automatically. |

### Cost gate

Lucid prices the run before any LLM call hits the wire (via
`messages.count_tokens` — free, separate rate-limit pool). If the
estimate exceeds **$20**, the run halts and asks for confirmation.
Pass `--yes-i-authorize-spend-up-to N` (whole dollars) to pre-authorize.
The gate is in `lucid/cost.py::COST_GATE_USD` if you want to see how
it's wired.

### See a sample report without running an audit

```bash
uv run python demo/render_demo_report.py
open report/lucid-demo.html
```

The demo renders against a synthetic corpus with pre-fabricated findings
for every detected pattern class. No API calls, no cost. Good for
deciding whether the output format is useful before spending anything.

### Exit codes (useful for scripting)

| Code | Meaning |
|---|---|
| `0` | Audit completed successfully |
| `2` | Usage / config / input error (bad path, zero conversations, missing key) |
| `3` | Cost-gate rejection — estimate exceeded `--yes-i-authorize-spend-up-to` |
| `4` | Concurrent-audit lock collision (another `lucid audit` is running on the same DB) |

---

## Other commands

### `lucid calibrate` — verify Module A still agrees with Spiral-Bench

```bash
# Re-run the full calibration pipeline against Spiral-Bench v1.2.
# ~$46 projected spend; requires ANTHROPIC_API_KEY.
uv run lucid calibrate --module a --auto-judge --yes-i-authorize-spend-up-to 50

# Or, if you already have human + judge label JSONLs, compute IAA only
# (no LLM spend, just statistics):
uv run lucid calibrate --module a \
    --human-labels path/to/human.jsonl \
    --judge-labels path/to/judge.jsonl
```

Outputs Krippendorff α, Gwet AC1, Cohen κ, and QWK on intensity, each
with 95% BCa bootstrap CIs. Artifacts land in `calibration-runs/`.
See [`docs/calibration.md`](docs/calibration.md) for the full protocol.

### `lucid cleanup-agents` — prune stale Managed Agents

Each synthesis run registers a `lucid-synthesis-v<N>` agent in your
Anthropic account. After a prompt-version bump, the previous agent is
stale but stays registered. Clean them up:

```bash
# Preview what would be archived
uv run lucid cleanup-agents --dry-run

# Archive stale synthesis agents + any legacy lucid-orchestrator-* agents
uv run lucid cleanup-agents

# Full sweep — archive every lucid-* agent (use before a clean re-run)
uv run lucid cleanup-agents --all
```

### `lucid version` — print the installed version

```bash
uv run lucid version
```

---

## Reading the report

Each `report/<run-id>.html` opens with a stacked-radial concern footprint
chart and seven sections:

1. **Executive summary** — what was sampled, what ran, headline shape.
2. **Headline findings** — strongest signals (or strongest absences).
3. **Module narratives (A–F, H)** — per-module prose with `[F:id]`
   citation links to the evidence cards below. A module with zero
   findings declines narration explicitly rather than fabricating one.
4. **Module G — attribution** — time/model bucketing of every finding.
   Deterministic, no LLM call.
5. **Top 3 actions** — Opus 4.7's suggested follow-ups, citation-bound.
6. **Evidence appendix** — every finding as a card with verbatim quotes,
   intensity, confidence + CI, model attribution, and source citation.
7. **Provenance footer** — corpus fingerprint, prompt versions, model
   IDs, sampling seed. Sufficient to reproduce the run.

If a module's section reads "Section skipped: insufficient evidence",
that's the `INSUFFICIENT_EVIDENCE` contract working — the agent
declined rather than padding. Treat declines as data.

---

## What each module detects

| Module | Detects | Source paper / framework |
|---|---|---|
| **A — Spiral-Bench** | 17 assistant behaviors at intensity 1–3 (sycophancy, pushback, escalation, delusion reinforcement, harmful advice, validate-feelings-not-thoughts, confident-bullshitting, …). | [Spiral-Bench v1.2](https://github.com/sam-paech/spiral-bench) |
| **B — Sharma sycophancy** | All 4 subroutines: feedback sycophancy (direction flips on similar content under opposite user sentiment), answer sycophancy (cave-ins on correct answers under pressure), mimicry, and "are you sure" sycophancy. | [Sharma et al. 2023](https://arxiv.org/abs/2310.13548) |
| **C — SycEval** | Second-pass classifier over A's and B's sycophancy findings: progressive (cave-in landing on correct answer, low priority) vs. regressive (cave-in landing on wrong answer, the flag). Module C is a *meta*-classifier — its agreement is bounded by A's and B's noise floor. | [Fanous & Goldberg 2025](https://arxiv.org/abs/2504.01727) |
| **D — Perspective sycophancy** | Cross-turn framing / vocabulary / premise drift. The assistant progressively adopting the user's worldview without stating explicit agreement. Default-on; pass `--no-include-module-d` to skip on tight-cost runs. | Jain et al. 2025 |
| **E — Belief drift** | Cross-conversation user position changes on recurring topics, classified evidence-driven (new info) vs. pressure-driven (Claude pushed back). | [BeliefShift](https://arxiv.org/abs/2603.23848) (DCS-simplified) |
| **F — Influence Tactics** | 9 user-prompt influence tactics adapted from media-analysis literature to one-on-one dialogue: emotional triggers, urgent action demands, false dilemmas, authority overload, framing techniques, … | [Influence Tactics Protocol](https://github.com/synaptiai/influence-tactics-protocol) |
| **G — Attribution** | Deterministic time/model bucketing over every finding. No LLM calls. Inferred from `updated_at` for Claude.ai (no `model` field exists in that export); explicit in Claude Code. | Lucid methodology §5 |
| **H — Memory audit** | **Novel.** Claims extracted from `memories.json` are individually verified against the corpus via Voyage embeddings + Opus 4.7 classification. Verdicts: `well-supported`, `weakly-supported`, `unsupported`, `contradicted`, `insufficient-data`, `out-of-scope`. | [MedTrust-RAG 2025](https://arxiv.org/pdf/2510.14400) (adapted) |

Module H is the contribution most worth highlighting. No other tool audits
AI memory features against the conversations those memories were derived
from. The `out-of-scope` verdict is specific to Lucid: it distinguishes "we
don't know" (the memory references conversations not in the audit sample)
from "the memory is unsupported" (the conversations are present but don't
back the claim).

---

## Calibration

Module A is calibrated against the public Spiral-Bench v1.2 benchmark.
Inter-annotator agreement was computed across 5 raters (Module A at two
chunk sizes plus the three reference judges from the Spiral-Bench paper)
on 1,667 shared turns.

| Behavior | Prevalence | Gwet AC1 (95% BCa CI) |
|---|---:|---:|
| pushback | 0.43 | 0.47 [0.44, 0.49] |
| escalation | 0.22 | 0.69 [0.67, 0.71] |
| sycophancy | 0.21 | 0.62 [0.60, 0.65] |
| delusion-reinforcement | 0.29 | 0.56 [0.53, 0.59] |
| topic-shut-down | 0.10 | 0.86 [0.84, 0.87] |
| help-referral-warranted | 0.10 | 0.93 [0.92, 0.94] |
| boundary-setting | 0.10 | 0.88 [0.87, 0.90] |
| harmful-advice | 0.05 | 0.92 [0.91, 0.93] |
| ritualization | 0.23 | 0.72 [0.70, 0.74] |

Full per-behavior table including Krippendorff's α in
[`docs/calibration.md`](docs/calibration.md). AC1 is the primary metric
because 6 of 17 behaviors have prevalence below 10% or above 90% — the
"agreement paradox" makes Cohen's κ misleading at those extremes.

Modules B, D, E, F, H lack public ground truth datasets — see *Honest
limitations* below for what that means in practice. Module H ships a
six-verdict adversarial fixture suite at
`tests/fixtures/module_h_verdicts/`.

---

## Honest limitations

- **A 5-conversation sample is statistically thin.** The default `--sample`
  is 100. Smaller samples produce many `insufficient_evidence` declines
  in the synthesis report; that's a feature, not a bug.
- **Cohen's κ on intensity is currently incomplete.** The 5-rater
  calibration setup exceeds pairwise κ; pairwise tables will land as a
  follow-up.
- **Modules B, D, E, F, H lack public ground truth.** Only Module A is
  benchmarked against a public dataset (Spiral-Bench v1.2). The other
  modules cite their source papers but their classifiers have not been
  measured against held-out labelled examples from those papers — those
  datasets aren't public. Validation for those modules is by manual
  review of seeded test corpora plus, for Module H, the six-verdict
  adversarial fixture suite at `tests/fixtures/module_h_verdicts/`.
- **Module C is a meta-classifier.** It runs over A's and B's outputs;
  its agreement floor is bounded by theirs. Treat C's progressive/
  regressive split as a re-categorisation, not an independent measurement.
- **The Sonnet post-processor is conservative.** Citation confidence
  scores cluster between 0.55 and 0.85 in practice — Sonnet penalizes
  any aggregate claim that isn't backed by an explicit tool-call result,
  and any block with zero citations.
- **`--resume` is not yet wired** (Phase 6, post-hackathon). A failed
  audit must be re-run from scratch; scoring-phase findings are
  checkpointed to SQLite per module so the LLM spend on completed
  modules is not repeated, but the CLI can't currently pick up where
  it left off in one command.
- **Pass `--no-synthesis`** to skip the agent narrative phase. The
  scoring phase still runs and the report still renders, with charts,
  tables, and evidence cards intact and a banner noting the narrative
  sections are deliberately absent.

---

## Reproducibility

The three pipeline phases have different determinism guarantees by design:

- **Phase 1 (scoring)** is deterministic given `(corpus, sample seed,
  prompt hash, model id)`. Re-running the same audit on the same corpus
  with the same flags produces a byte-identical `findings` table modulo
  Anthropic-side stochasticity in classification (and even that is
  bounded — Module A on Opus 4.7 with `effort=low` is highly stable).
  This is what lets the calibration table above stay valid across
  prompt-version bumps: the rubric is the calibration unit, not the
  agent's free-form reasoning.
- **Phase 2 (synthesis writing)** is *adaptive*. Same `findings` table →
  different prose. What to say about a corpus depends on what's in it,
  and Opus 4.7 makes judgement calls about emphasis. Running the same
  audit twice will produce two reports whose factual claims are
  identical (citations resolve to the same finding/turn IDs) but whose
  narrative shape differs.
- **Phase 3 (Sonnet structuring)** is `messages.parse()`-bound to a
  Pydantic schema. Output is stable up to ordering of citation tokens
  inside a block.

Every finding records the `prompt_version` and `prompt_hash` that
classified it, and every audit run records the model IDs in
`detected_by`. Reproduction is "freeze the prompt, freeze the seed,
re-run" — the deterministic phase will match; the prose won't.

---

## What Lucid will and won't do

**Will:**

- Detect sycophancy events with citations to the published rubric that scored them.
- Track belief shifts across sessions on the same topic, with evidence-vs-pressure classification.
- Flag user-side influence tactics (pressure, appeal, reframing) the user is applying to the model.
- Audit memory-corpus consistency — whether stored memories are supported by conversation history.
- Attribute every finding to the Claude model that produced it.

**Won't:**

- Make claims beyond what its source papers support. Every finding cites a framework.
- Send your corpus anywhere except the Anthropic API and Voyage API. See [`docs/privacy.md`](docs/privacy.md) for the exact flow.
- Speculate when evidence is thin. `insufficient-evidence` and `out-of-scope` are first-class outputs.

---

## Documentation

- **[`docs/PRD.md`](docs/PRD.md)** — product requirements, scope, and success criteria.
- **[`docs/methodology.md`](docs/methodology.md)** — pricing, cache strategy, model timeline, calibration methodology.
- **[`docs/privacy.md`](docs/privacy.md)** — exactly what leaves your machine and what doesn't.
- **[`docs/calibration.md`](docs/calibration.md)** — Krippendorff's α, Gwet's AC1, BCa confidence intervals.
- **[`CLAUDE.md`](CLAUDE.md)** — operational conventions for working in this codebase.
- **[`CHANGELOG.md`](CHANGELOG.md)** — version history.

## Testing

```bash
uv run pytest                          # 736 tests, ~6s
uv run mypy lucid/ --strict            # strict type checking
uv run ruff check lucid/ tests/        # linting
```

## License

MIT — see [`LICENSE`](./LICENSE).

Framework citations live in each module's source header
(`lucid/modules/module_*.py`). Every Lucid finding records its
`prompt_version` and `prompt_hash` for full reproducibility.
