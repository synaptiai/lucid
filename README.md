# Lucid

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![Tests](https://img.shields.io/badge/tests-730_passing-brightgreen.svg)](#testing)

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

## Run

```bash
# Estimate cost first — parses the corpus, samples it, and prints a per-module
# token / USD breakdown. No LLM calls, no spend.
uv run lucid audit --source claude-code --path ~/.claude/projects --sample 5 --dry-run

# Real audit on a 5-conversation sample. Typical cost: $2–4.
uv run lucid audit --source claude-code --path ~/.claude/projects --sample 5 \
    --yes-i-authorize-spend-up-to 10
```

The HTML report lands at `report/<run-id>.html` — a static file with no
external scripts and a strict `default-src 'none'` content security policy.
Open it in any browser.

A 12-slide demo deck is rendered alongside at `report/lucid-deck.html`.
Press `←`/`→` to navigate, `N` for presenter notes, `P` to print.

### See a sample report without running an audit

```bash
uv run python demo/render_demo_report.py
open report/lucid-demo.html
```

The demo renders against a synthetic corpus with pre-fabricated findings
for every detected pattern class. No API calls, no cost.

---

## What each module detects

| Module | Detects | Source paper / framework |
|---|---|---|
| **A — Spiral-Bench** | 17 assistant behaviors at intensity 1–3 (sycophancy, pushback, escalation, delusion reinforcement, harmful advice, validate-feelings-not-thoughts, confident-bullshitting, …). | [Spiral-Bench v1.2](https://github.com/sam-paech/spiral-bench) |
| **B — Sharma sycophancy** | All 4 subroutines: feedback sycophancy (direction flips on similar content under opposite user sentiment), answer sycophancy (cave-ins on correct answers under pressure), mimicry, and "are you sure" sycophancy. | [Sharma et al. 2023](https://arxiv.org/abs/2310.13548) |
| **C — SycEval** | Second-pass classifier over A's and B's sycophancy findings: progressive (cave-in landing on correct answer, low priority) vs. regressive (cave-in landing on wrong answer, the flag). | [Fanous & Goldberg 2025](https://arxiv.org/abs/2504.01727) |
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

Modules B, D, E, F, H lack public ground truth datasets. Validation for
those modules is by manual review of seeded test corpora; numbers will
land in [`docs/calibration.md`](docs/calibration.md) as they're collected.

---

## Honest limitations

- **A 5-conversation sample is statistically thin.** The default `--sample`
  is 100. Smaller samples produce many `insufficient_evidence` declines
  in the synthesis report; that's a feature, not a bug.
- **Cohen's κ on intensity is currently incomplete.** The 5-rater
  calibration setup exceeds pairwise κ; pairwise tables will land as a
  follow-up.
- **The Sonnet post-processor is conservative.** Citation confidence
  scores cluster between 0.55 and 0.85 in practice — Sonnet penalizes
  any aggregate claim that isn't backed by an explicit tool-call result,
  and any block with zero citations.
- **Pass `--no-synthesis`** to skip the agent narrative phase. The
  scoring phase still runs and the report still renders, with charts,
  tables, and evidence cards intact and a banner noting the narrative
  sections are deliberately absent.

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
uv run pytest                          # 730 tests, ~5s
uv run mypy lucid/ --strict            # strict type checking
uv run ruff check lucid/ tests/        # linting
```

## License

MIT — see [`LICENSE`](./LICENSE).

Framework citations live in each module's source header
(`lucid/modules/module_*.py`). Every Lucid finding records its
`prompt_version` and `prompt_hash` for full reproducibility.
