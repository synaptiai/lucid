# Lucid

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)

**First-draft interface for auditing your thinking-with-AI.**

Lucid ingests your Claude Code sessions (`~/.claude/projects/`) and Claude.ai conversation export, runs a composition of published AI-safety research frameworks (Spiral-Bench, Sharma sycophancy, SycEval, Jain perspective sycophancy, BeliefShift, Influence Tactics Protocol, MedTrust-RAG) against the corpus, and produces a structured HTML report surfacing sycophancy events, belief drift, reinforcement spirals, user-side influence tactics, memory-corpus consistency, and time/model attribution.

Everything runs locally. API calls go out only to Anthropic (for classification) and Voyage AI (for Module H embeddings).

## Status

Alpha. In-progress submission for the Anthropic Opus 4.7 hackathon
(2026-04-21 → 2026-04-26). Not yet published to PyPI — install from
source until a v0.1.0 tag lands.

The audit pipeline runs end-to-end: ingest → cost gate → Managed Agents
orchestrator (with direct-invocation backfill) → modules A/B/C/D/E/F/H
plus deterministic G → static HTML report + hackathon slide deck.
Module D (Jain perspective sycophancy) runs by default per PRD §4.4; use
`--no-include-module-d` on tight-cost runs. Calibration numbers against
Spiral-Bench v1.2 live in [`docs/calibration.md`](docs/calibration.md).

## Install

```bash
# Install from source (only path until a v0.1.0 tag is published)
git clone https://github.com/synaptiai/lucid.git
cd lucid
uv sync --extra dev
uv run lucid --help
```

## Quickstart

```bash
# Configure API keys
cp .env.example .env.local
$EDITOR .env.local    # set ANTHROPIC_API_KEY and VOYAGE_API_KEY (optional; for Module H)

# Dry-run first (parses + samples + estimates cost; no LLM spend)
uv run lucid audit --source claude-code --path ~/.claude/projects --sample 20 --dry-run

# Real run
uv run lucid audit --source claude-code --path ~/.claude/projects --sample 100
```

The report lands at `report/<run-id>.html` — a static HTML file with no
external scripts. A companion hackathon slide deck is written alongside at
`report/lucid-deck.html` (press `N` in-deck for speaker notes, `P` to print).
Open either in any browser.

### See a sample report without running an audit

```bash
uv run python demo/render_demo_report.py
open report/lucid-demo.html
```

This renders Lucid's HTML format against a synthetic seeded corpus
(`demo/corpus/`) using pre-fabricated findings for every detected pattern
class. No API calls, no cost.

## Modules

| Module | What it detects | Citation |
|---|---|---|
| A | 17 Spiral-Bench behaviors on assistant turns (intensity 1-3) | [Spiral-Bench v1.2](https://github.com/sam-paech/spiral-bench) |
| B.1 | Feedback sycophancy: direction flips on similar content under opposite user sentiment | [Sharma et al. 2023](https://arxiv.org/abs/2310.13548) |
| B.2 | Answer sycophancy: cave-in on a correct answer under low-info user pressure | Sharma et al. 2023 |
| C | Progressive/regressive classifier on A+B sycophancy events | [Fanous & Goldberg 2025 (SycEval)](https://arxiv.org/abs/2504.01727) |
| D | Perspective sycophancy: cross-turn framing / premise / vocabulary drift | Jain et al. 2025 |
| E | Cross-conversation belief drift with evidence-vs-pressure attribution | [BeliefShift arxiv:2603.23848](https://arxiv.org/abs/2603.23848) |
| F | 9-category user-prompt influence tactics analyzer | [Influence Tactics Protocol](https://github.com/synaptiai/influence-tactics-protocol) |
| G | Deterministic time/model attribution | Lucid methodology §5 |
| H | Memory-corpus consistency via retrieval + two-stage verification, with source-aware routing (user-level vs. project-scoped memories) and an `out-of-scope` verdict for claims about projects not in the audit sample | [MedTrust-RAG 2025](https://arxiv.org/pdf/2510.14400) |

Each module's prompts and rubric live under `prompts/module_<letter>/`. Every
finding in the report cites the framework that scored it. Module D (Jain
perspective sycophancy) runs by default; it is Opus 4.7 at `effort=xhigh`
and typically pushes a 100-conversation audit past the default $20 cost
gate — real runs want `--yes-i-authorize-spend-up-to 50`. Pass
`--no-include-module-d` to skip Module D on a tight-cost run.

## What Lucid does and doesn't do

**Does:**
- Detect sycophancy events with citations to the published rubric that scored them.
- Track belief shifts across sessions on the same topic, with evidence-vs-pressure classification.
- Flag user-side influence tactics (pressure, appeal, reframing) the user is applying to the model.
- Check memory-corpus consistency — whether stored memories are supported by conversation history.
- Attribute findings to the Claude model that was active at the time (inferred from `updated_at` for Claude.ai; explicit in Claude Code).

**Does not:**
- Make claims beyond what its source papers support. Every finding cites a framework.
- Send your corpus to anyone other than Anthropic and Voyage, for the narrow API calls documented in [`docs/privacy.md`](docs/privacy.md).
- Speculate when evidence is thin. `"insufficient-data"` is a first-class output.

## Calibration numbers

See [`docs/calibration.md`](docs/calibration.md) for the latest Krippendorff's α, Gwet's AC1, per-label Cohen's κ, QWK, and BCa confidence intervals for each module with ground-truth labels.

## Methodology

See [`docs/methodology.md`](docs/methodology.md) for the pricing, cache, and model-timeline facts the implementation is calibrated against.

## License

MIT — see [`LICENSE`](./LICENSE).

Framework citations live in each module's source header (`lucid/modules/module_*.py`).
