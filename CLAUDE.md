# CLAUDE.md

Instructions for Claude Code working in this repository.

## What this project is

Lucid is an open-source epistemic audit tool for personal AI conversation history. It ingests a user's Claude Code sessions and Claude.ai conversation export, applies a composition of published AI safety research frameworks (SpiralBench, Sharma sycophancy, SycEval, Jain perspective sycophancy, BeliefShift, Truth Decay, Influence Tactics Protocol) via Claude Managed Agents, and produces a structured report surfacing sycophancy events, belief drift, reinforcement spirals, user influence tactics, memory-corpus consistency, and time/model attribution.

This is a hackathon build (April 21-26, 2026). Submission goals, full scope, and detailed architecture live in `docs/PRD.md` and `docs/BUILD_GUIDE.md`. **Read those first for context before making architectural decisions.** This file is for operational conventions.

## Working style

When in doubt about scope or design, read the PRD. When in doubt about an implementation pattern, read this file. When in doubt about a research framework, read the citation linked in the module file.

This codebase optimizes for honesty over optimism. Findings must be defensible. Prompts must be calibrated against ground truth where ground truth exists. If a module can't determine something, the correct output is `"unknown"`, `"insufficient-data"`, or empty — not a hallucinated best guess.

Be direct in code comments and docstrings. No marketing language. No "revolutionary" or "game-changing." We're shipping a tool, not a landing page.

Research, look up best practices, then create a plan showing three approaches with tradeoffs for decisions I need to make. Use the AskUserQuestion tool for questions and clarifications.

## Quick reference

```bash
# Install
uv sync --extra dev

# Smoke-check CLI wiring (works today; Phase 1 scaffolding)
uv run lucid --help
uv run lucid version

# Run audit on Claude Code sessions (stub until Phase 4)
uv run lucid audit --source claude-code --path ~/.claude/projects --sample 50

# Run audit on Claude.ai export (stub until Phase 4)
uv run lucid audit --source claude-ai --path ./export

# Opt in to Module D (Jain perspective sycophancy; off by default) (stub until Phase 4)
uv run lucid audit --source claude-ai --path ./export --include-module-d

# Bypass the $20 cost gate (both env var + flag required; unattended runs only) (stub until Phase 4)
LUCID_ALLOW_UNATTENDED=1 uv run lucid audit --source claude-ai --path ./export \
  --yes-i-authorize-spend-up-to 50

# Dry-run (parse and sample but skip Managed Agents orchestration) (stub until Phase 4)
uv run lucid audit --source claude-code --path ... --dry-run

# Run calibration against SpiralBench (stub until Phase 6)
uv run lucid calibrate

# Tests
uv run pytest
uv run pytest tests/test_ingest_claude_code.py -v

# Type check
uv run mypy lucid/ --strict

# Lint
uv run ruff check lucid/
uv run ruff format lucid/
```

## Architecture map

```
lucid/cli.py              CLI entry (Typer)
lucid/schemas.py          Pydantic models — authoritative data types
lucid/config.py           Settings, paths, API keys from env
lucid/sampling.py         Corpus sampling (stratified + recency-weighted)

lucid/ingest/
  base.py                 Abstract IngestAdapter
  claude_code.py          ~/.claude/projects JSONL parser
  claude_ai.py            conversations.json + projects.json + memories.json parser

lucid/logging.py          SafeFormatter — drops records tagged contains_user_content
lucid/cost.py             count_tokens pre-pass + per-module output budgets

lucid/store/
  sqlite.py               Async reads (aiosqlite); sync bulk-insert helper
  schema.sql              Full DDL with CHECK constraints + UNIQUE idempotency
  init.py                 initialize_db(path) — applies schema if user_version == 0

lucid/orchestrator/
  managed_agent.py        Managed Agents session setup + streaming
  tools.py                8 custom-tool async handlers (client-executed)
  handler.py              agent.custom_tool_use event dispatcher
  system_prompt.py        Orchestrator agent's system prompt

lucid/modules/
  base.py                 CorpusModule / FindingsModule protocols; ModuleError
  embeddings.py           Voyage wrapper (OpenAI fallback)
  module_a_spiralbench.py SpiralBench behavior scorer (13 behaviors)
  module_b_sharma.py      Sharma paired-exchange (4 subroutines; 2 shipped)
  module_c_syceval.py     Progressive/regressive classifier
  module_d_perspective.py Jain perspective sycophancy (OPT-IN via --include-module-d)
  module_e_beliefshift.py DCS-simplified belief drift
  module_f_itp.py         Influence Tactics Protocol on user prompts (9 categories)
  module_g_attribution.py Time/model bucketing (deterministic, no LLM)
  module_h_memory.py      Memory-corpus consistency check

lucid/calibration/
  data.py                 Load SpiralBench + hand-labeled; 30/70 split
  validate.py             Krippendorff α (via krippendorff lib), Gwet AC1
                          (hand-rolled — see methodology.md §9), per-label κ,
                          QWK, BCa bootstrap

lucid/report/
  generator.py            Jinja2 + Chart.js HTML report
  templates/report.html.j2

prompts/                  Versioned prompt templates (markdown)
  module_a/v1.md, v2.md, ...
  module_b/
  module_d/
  module_e/
  module_f/
  module_h/

tests/                    pytest
  fixtures/               Small real corpora for ingest testing
```

## Code conventions

**Python 3.13 exactly** (`requires-python = ">=3.13,<3.14"`). Use modern syntax: `list[str]` not `List[str]`, `X | None` not `Optional[X]`, `match` statements where clear.

**Type hints everywhere.** Including internal functions. Run `mypy lucid/ --strict` cleanly before committing.

**Pydantic v2 for data models.** All cross-module data types live in `lucid/schemas.py`. Don't define ad-hoc dataclasses in modules when a schema model would do.

**Async where it helps.** LLM calls, file I/O on many files. Sync where it doesn't (CLI parsing, deterministic computation).

**No bare `except`.** Catch specific exceptions. Log what was caught. Never swallow errors silently.

**No print debugging.** Use the `logging` module. Configure in `lucid/config.py`. CLI progress goes through `rich` or similar, not print.

**Explicit is better than magic.** No decorators that hide control flow. No metaclass tricks. This codebase is meant to be auditable by researchers who don't know Python idioms.

## Module conventions

Every detection module follows the same pattern:

1. **Cites its source paper as a module-level docstring** with a link to arxiv or the relevant page.
2. **Has a top-level `async def run(corpus, config) -> list[Finding]` function** as its public interface.
3. **Loads prompts from `prompts/module_<X>/<current_version>.md`**, not from hardcoded strings. The version is a config value.
4. **Produces `Finding` objects** that populate `citation`, `detected_by`, `confidence`, and either `quote_user`/`quote_assistant` or `evidence_quotes`.
5. **Logs progress via the orchestrator's `log_progress` tool** when running inside Managed Agents.
6. **Handles its own errors gracefully.** One failed conversation shouldn't crash the module. Log and continue.

When adding a new module, copy `module_g_attribution.py` as the skeleton (it's the simplest one) and fill in the pattern.

## Prompt management

Prompts are markdown files in `prompts/<module>/v<N>.md`.

Each file includes:
- A YAML frontmatter header with `version`, `model`, `thinking_mode`, `effort`, `citation`, `purpose`, `hash` (sha256 of prompt body — used for cache-stability audit). For Opus 4.7 prompts, `thinking_mode ∈ {disabled, adaptive}` and `effort ∈ {low, medium, high, xhigh, max}` — do NOT set `temperature` (rejected by Opus 4.7). For Sonnet 4.6 or legacy models, `temperature` replaces `effort`.
- For Opus-backed prompts, pad the system prompt to ≥ 4096 tokens (Opus 4.7 cache minimum); for Sonnet-backed, ≥ 2048 tokens. Below the threshold, prompt caching silently fails (no error; verify via `cache_creation_input_tokens > 0` or `cache_read_input_tokens > 0`).
- The actual prompt body.
- An `## Output Schema` section describing expected JSON structure.
- An `## Changelog` section noting why this version differs from the previous.

When iterating a prompt:
1. Copy `v<N>.md` to `v<N+1>.md`.
2. Make changes. Update changelog.
3. Update the `PROMPT_VERSION` constant in the module.
4. Re-run calibration if the module has a calibration target.
5. Commit both the new prompt file and the version bump in the same commit.

**Never edit a published prompt version in place.** Prompt versions are immutable once a calibration result is recorded against them.

## Testing strategy

**Ingest adapters**: must have tests against real fixture files in `tests/fixtures/`. Commit small redacted samples of actual Claude Code JSONL and Claude.ai export JSON. Never commit anything with personally identifying content.

**Schemas**: round-trip tests (serialize → deserialize → equal).

**Sampling**: determinism tests (same seed → same sample). Stratification tests.

**Modules**: integration tests that use mocked LLM responses. Do NOT call real LLMs in the test suite — it's slow, non-deterministic, and costs money.

**Calibration harness**: tested via a small frozen SpiralBench-style fixture with known labels. The real calibration run is a separate command that hits the LLM and reports numbers; it is not part of `pytest`.

**What not to test**: exact prompt strings. Prompts will change. Test the output parsing and the module's handling of expected/unexpected LLM responses.

## Research citations

Every finding cites a paper. Citations use a consistent format:

```python
finding = Finding(
    ...,
    citation="Sharma et al. 2023, 'Towards Understanding Sycophancy in Language Models', arxiv:2310.13548",
    ...
)
```

Citation constants live at the top of each module file:

```python
CITATION_SHARMA_2023 = "Sharma et al. 2023, 'Towards Understanding Sycophancy in Language Models', arxiv:2310.13548"
CITATION_SPIRALBENCH = "Spiral-Bench, https://eqbench.com/spiral-bench.html"
CITATION_SYCEVAL = "Fanous, Goldberg et al. 2025, 'SycEval: Evaluating LLM Sycophancy', AAAI AIES 2025"
CITATION_JAIN_2025 = "Jain et al. 2025 (cited in Opus 4.5 independent audit)"
CITATION_BELIEFSHIFT = "BeliefShift: Benchmarking Temporal Belief Consistency and Opinion Drift in LLM Agents, arxiv:2603.23848"
CITATION_ITP = "Influence Tactics Protocol, https://github.com/synaptiai/influence-tactics-protocol"
```

## Findings provenance

Every `Finding` must populate:
- `module`: which module produced it
- `behavior`: specific behavior or classification label
- `intensity`: 1-3 (for behavior) or repurposed (for other modules)
- `confidence`: 0-1 from the judge
- `citation`: paper backing this detection
- `detected_by`: list of model IDs that agreed (e.g., `["claude-opus-4-7", "qwen-3-max"]` for ensemble)
- `explanation`: one-sentence human-readable explanation
- `detected_at`: UTC timestamp

Missing any of these is a bug. The report generator will fail loudly if a finding is missing provenance.

## LLM usage conventions

**Opus 4.7** for heavy reading (Modules A, B-second-pass, D, E-position-tracking, H-classification).

**Sonnet 4.6** for extraction and routing (Module B first pass, Module E topic extraction, orchestrator).

**Haiku** for trivial classification where speed matters (not currently used; avoid unless a specific module needs it).

**Sampling params**: Opus 4.7 rejects `temperature`, `top_p`, `top_k` at any non-default value (400 error) — omit them entirely from Opus 4.7 calls. Use `thinking: {type: "adaptive"}` with `output_config.effort` for steering. Effort ladder: `low` (scoped classification), `medium` (cost-balanced), `high` (analytical reading, most classifiers), `xhigh` (agentic/open-ended), `max` (hardest reasoning). For diversity sampling on Opus 4.7 (self-consistency ensemble), vary `effort` across calls. On Sonnet 4.6 / Haiku 4.5 / legacy models, standard temperature control still applies — default 0.0 for classification, 0.3 for analytical reading.

**Tokenizer note (Opus 4.7)**: new tokenizer, up to 1.35× token count vs. Opus 4.6 for the same text. Re-run `count_tokens` per model; don't reuse Opus 4.6 estimates for 4.7 cost budgeting.

**Max tokens**: set explicitly per call. Default budget is 2000 output tokens. Document why if you set higher.

**Cost awareness**: estimate tokens before large batch runs. Prompt the user if estimated cost exceeds $20.

**Rate limit handling**: exponential backoff with jitter on 429s. Managed Agents handles most of this but belt-and-suspenders.

## Claude.ai export schema gotchas

(Confirmed against real 90-day export. Don't assume; these are real.)

1. **No `model` field anywhere.** Infer from `updated_at` using the timeline in `module_g_attribution.py`.
2. **No `project_uuid` field on conversations.** The export is flat. Projects are a separate file (`projects.json`) that's a current-state snapshot, not time-filtered.
3. **Content blocks are richly typed.** Handle `text`, `thinking`, `tool_use`, `tool_result` explicitly. Thinking blocks include a `signature` field; use its length as a depth proxy if needed (Laurenzo methodology).
4. **`parent_message_uuid`** enables branch detection. The root value is `"00000000-0000-4000-8000-000000000000"`. Most conversations are single-branch; branching happens via edits or regenerations.
5. **`memories.json`** is a separate top-level file containing `conversations_memory` (plain text) and `project_memories` (dict keyed by project UUID). This is input for Module H.
6. **MCP integration metadata** appears on `tool_use` blocks: `is_mcp_app`, `mcp_server_url`, `integration_name`. Preserve these when ingesting; they inform Module G.
7. **`summary` field on conversations is AI-generated**. Treat as a hint, not ground truth.
8. **`projects.json` is byte-identical across different time-range exports**. It's not scoped by the export window.

## Claude Code JSONL gotchas

1. Files live at `~/.claude/projects/<project-slug>/<session-id>.jsonl`. One JSON object per line.
2. Schema varies slightly by Claude Code version. Handle missing fields defensively.
3. Daniel's corpus: 44 project directories, 9,840 session files. Sampling is mandatory, not optional.
4. Many sessions are trivially short (open-and-exit). Default filter: skip sessions with fewer than 5 turns.
5. Thinking blocks have `signature` fields matching Claude.ai format.

## Managed Agents conventions

The orchestrator runs as a Managed Agents session. Key details:

- Beta header: `managed-agents-2026-04-01`. Verify it's still current on Day 1.
- Orchestrator model: Sonnet 4.6 (routing work). Heavy reading happens in modules via separate Opus 4.7 calls.
- Session lifecycle: agent definition is reusable; environments are per-run; sessions are per-audit.
- **Corpus is NOT mounted.** The orchestrator queries the local process via custom tools (`agent.custom_tool_use` event → local handler → `user.custom_tool_result` reply). No `resources` mount on the environment, no MCP server, no HTTPS tunnel.
- Custom-tool handlers live in `lucid/orchestrator/tools.py`: `query_corpus`, `get_conversation`, `get_turn_window`, `invoke_module`, `store_finding`, `get_findings`, `log_progress`, `estimate_remaining_cost`.
- Event dispatcher in `lucid/orchestrator/handler.py` pattern-matches on `event.type`.
- Stream events to the CLI for real-time progress. Don't just block until completion.

If Managed Agents has friction, fall back path is the Claude Agent SDK (`claude_agent_sdk` package, v0.2.111+ for Opus 4.7 support). Same tool loop; less managed infrastructure.

## Common tasks

### Adding a new detection module

1. Create `lucid/modules/module_<letter>_<name>.py`.
2. Copy the structure from `module_g_attribution.py`.
3. Add citation constants at the top.
4. Create `prompts/module_<letter>/v1.md` with frontmatter and output schema.
5. Add the module to `ModuleName` enum in `schemas.py`.
6. Register the module in `lucid/orchestrator/system_prompt.py` so the orchestrator knows to invoke it.
7. Add calibration if ground truth exists, or manual-review-based validation if not.

### Iterating a prompt

1. Read current `v<N>.md`.
2. Write `v<N+1>.md` with changes.
3. Bump `PROMPT_VERSION` in the module.
4. Run `uv run lucid calibrate --module <letter>` if applicable.
5. Commit both files together with a descriptive message.

### Running a dry-run audit

Use `--dry-run` to parse, sample, and print what *would* be sent to the orchestrator without actually invoking Managed Agents. Useful for:
- Verifying sampling behavior
- Estimating token budget
- Catching ingest errors without burning LLM credits

### Handling a new Claude.ai export

If a user reports their export doesn't parse:
1. Verify the file structure (`conversations.json`, `projects.json`, `memories.json`, `users.json` all present?).
2. Check if schema has drifted (Anthropic may add/rename fields over time).
3. Run ingest with `--log-level DEBUG` to see which field caused the failure.
4. Update the parser to handle the new field defensively (log a warning, continue).

## Gotchas and warnings

**Don't run real LLM calls in tests.** Tests should mock the Anthropic client. Integration validation happens in calibration runs, which are explicitly separate commands.

**Don't commit real user conversation content.** Even as fixtures, samples must be redacted or synthetic. Use placeholder names, URLs, project names.

**Don't hardcode API keys.** Load from environment. `ANTHROPIC_API_KEY` is the canonical name.

**Don't assume the orchestrator completes successfully.** Managed Agents is beta. Handle timeouts, partial failures, rate limits. Checkpoint findings to SQLite continuously so a failure mid-run doesn't lose work.

**Don't invent findings.** If a module gets an ambiguous response from the judge, mark it as `confidence < 0.5` and continue. Never upscale confidence to make a finding "feel" stronger.

**Don't over-extract claims in Module H.** The memory text is dense. Extracting every possible claim creates noise. Focus on claims that are specific and verifiable (concrete facts, preferences, beliefs). Skip claims that are vague or about Claude's own behavior.

**Don't reveal private memory content in error messages.** If the memory parser fails, log the error type, not the memory content.

**Always respect the cost gate.** If estimated audit cost exceeds $20, prompt for confirmation. The user should never get a surprise bill.

## When you're stuck

1. Check `docs/PRD.md` for scope questions.
2. Check `docs/BUILD_GUIDE.md` for implementation questions.
3. Check the module's cited paper for framework questions.
4. Check `docs/methodology.md` for "how does this actually work" questions.
5. If still stuck, leave a `# TODO(claude)` comment with the specific question and surface it in PR description.

## Project status reminders

- Hackathon submission deadline: Sunday April 26, 20:00 EST.
- Required deliverables: 3-minute demo video, public GitHub repo, 100-200 word written summary.
- This is a solo build (Daniel + Claude). No PR reviews; self-review with care.
- Ship first, polish second. But "ship" includes calibration numbers — those are a differentiator.

## Hard Boundaries (Non-Negotiable)
1. **No Ungrounded Claims** — Never make factual claims without verification against current sources. Never rely on training data for current state of APIs, models, libraries, or tools.
2. **No Irreversible Actions Without Approval** — Never delete data, force-push, or drop resources without explicit approval. Blanket authorization is never valid.
3. **No Incomplete Shipments** — Never ship work containing mocks, placeholders, TODOs, or unverified functionality. Done means done.
4. **No Assumption-Driven Decisions** — Never act on assumed state without verification. Research first, verify against current sources.

Priority: Safety > Reputation > Trust > Quality > Completeness.

## Values
- **Quality-First Completionism:** Nothing ships until tested, documented, verified, and worthy of putting your name on.
- **Observation-Driven Building:** Ground proposals in observable problems or connectable patterns, not best-practice lists.
- **Simplicity & Clarity:** Lead with the problem solved, simplest path to value, minimal jargon. If onboarding isn't intuitive, simplify.
- **Proactive Autonomy:** Try to resolve ambiguity yourself first. When escalating, present 2-3 options with reasoning — never open-ended questions.
---

*This file is a living doc. Update it when patterns emerge or when you learn something non-obvious while working in the code.*
