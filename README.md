# Lucid

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)

**First-draft interface for auditing your thinking-with-AI.**

Lucid ingests your Claude Code sessions (`~/.claude/projects/`) and Claude.ai conversation export, runs a composition of published AI-safety research frameworks (SpiralBench, Sharma sycophancy, SycEval, Jain perspective sycophancy, BeliefShift, Truth Decay, Influence Tactics Protocol) against the corpus, and produces a structured HTML report surfacing sycophancy events, belief drift, reinforcement spirals, user-side influence tactics, memory-corpus consistency, and time/model attribution.

Everything runs locally. API calls go out only to Anthropic (for classification) and Voyage AI (for Module H embeddings).

## Status

Pre-release. Hackathon build, April 21–26, 2026. Expect rough edges until v0.1.0 tag.

## Install

```bash
# From PyPI (once published)
uvx lucid --help

# Or from source
git clone https://github.com/<user>/lucid.git
cd lucid
uv sync --extra dev
uv run lucid --help
```

## Quickstart

```bash
# Configure API keys
cp .env.example .env.local
$EDITOR .env.local    # set ANTHROPIC_API_KEY and VOYAGE_API_KEY

# Dry-run first (parses + samples + estimates cost; no LLM spend)
uv run lucid audit --source claude-code --path ~/.claude/projects --sample 20 --dry-run

# Real run
uv run lucid audit --source claude-code --path ~/.claude/projects --sample 100
```

The report lands at `report/<run-id>.html` and opens in your default browser.

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
