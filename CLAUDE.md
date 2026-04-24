# CLAUDE.md

Instructions for Claude Code working in this repository.

## What this project is

Lucid is an open-source epistemic audit tool for personal AI conversation history. It ingests a user's Claude Code sessions and Claude.ai conversation export, applies a composition of published AI safety research frameworks (SpiralBench, Sharma sycophancy, SycEval, Jain perspective sycophancy, BeliefShift, Truth Decay, Influence Tactics Protocol) via a two-phase pipeline — deterministic Python scoring, then a Managed Agents synthesis session that writes narrative report sections — and produces a structured report surfacing sycophancy events, belief drift, reinforcement spirals, user influence tactics, memory-corpus consistency, and time/model attribution.

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

# Dry-run (parse + sample + count_tokens cost estimate; no LLM spend)
# Fully wired since Phase 4 — returns a per-module USD breakdown.
uv run lucid audit --source claude-code --path ~/.claude/projects --sample 10 --dry-run
uv run lucid audit --source claude-ai --path ./export --sample 20 --dry-run

# Run audit on Claude Code sessions (stub until Phase 5; exits 2 with hint)
uv run lucid audit --source claude-code --path ~/.claude/projects --sample 50

# Run audit on Claude.ai export (stub until Phase 5)
uv run lucid audit --source claude-ai --path ./export

# Module D (Jain perspective sycophancy) ships ON by default per PRD §4.4.
# Pass --no-include-module-d to skip it on a tight-cost run; the $20 gate
# typically trips with D enabled, so real runs want --yes-i-authorize-spend-up-to 50.
uv run lucid audit --source claude-ai --path ./export --dry-run
uv run lucid audit --source claude-ai --path ./export --no-include-module-d --dry-run

# Bypass the $20 cost gate for unattended runs (flag + env both required; Phase 5)
LUCID_ALLOW_UNATTENDED=1 uv run lucid audit --source claude-ai --path ./export \
  --yes-i-authorize-spend-up-to 50

# Skip the synthesis phase (scoring still runs; report renders deterministic
# scaffolding + a muted banner noting narrative sections are absent).
uv run lucid audit --source claude-ai --path ./export --no-synthesis

# Run calibration against SpiralBench (stub until Phase 6)
uv run lucid calibrate --module a

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

lucid/run.py              Two-phase pipeline: _run_scoring_loop (det.) +
                          run_synthesis_session (agent). AuditResult
                          aggregates scoring + synthesis status.

lucid/orchestrator/
  tools.py                Tool-schema registry + scoring-side custom tools
                          (invoke_module, store_finding, query_corpus,
                          get_conversation, get_turn_window, get_findings,
                          log_progress, estimate_remaining_cost). Kept here
                          for Phase 3.4 migration into lucid/synthesis/.

lucid/synthesis/
  __init__.py             SYNTHESIS_PROMPT_VERSION constant; re-exports.
  session.py              SynthesisSession + SynthesisConfig/Handles/Outcome,
                          MANAGED_AGENTS_BETA_HEADER.
  run.py                  run_synthesis_session() — section loop, regen
                          retries on unknown-id errors, upserts into
                          report_sections.
  handler.py              agent.custom_tool_use dispatcher + HeartbeatMonitor.
  tools.py                build_synthesis_registry — read-only tools for the
                          writer (get_findings, get_conversation, spot-read
                          turns) + write_report_section (DB upsert).
  validators.py           validate_aggregate_claims, validate_superlatives,
                          validate_uncited_high_intensity (post-generation).
  lifecycle.py            get_or_create_synthesis_agent +
                          prune_stale_synthesis_agents (covers legacy
                          lucid-orchestrator-* names for backward-compat).

lucid/modules/
  base.py                 CorpusModule / FindingsModule protocols; ModuleError
  embeddings.py           Voyage wrapper (OpenAI fallback)
  module_a_spiralbench.py SpiralBench behavior scorer (13 behaviors)
  module_b_sharma.py      Sharma paired-exchange (4 subroutines; 2 shipped)
  module_c_syceval.py     Progressive/regressive classifier
  module_d_perspective.py Jain perspective sycophancy (default-on; --no-include-module-d to skip)
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
  synthesis/              Opus 4.7 writer prompt (narrative sections).
                          v2 is current; v1 preserved (frozen once shipped).
  synthesis_validator/    Sonnet 4.6 post-processor prompt. Wired — reads
                          Opus markdown and emits SynthesisSectionOutput
                          JSON via AsyncAnthropic.messages.parse().

tests/                    pytest
  fixtures/               Small real corpora for ingest testing
```

## Code conventions

**Python 3.13 exactly** (`requires-python = ">=3.13,<3.14"`). Use modern syntax: `list[str]` not `List[str]`, `X | None` not `Optional[X]`, `match` statements where clear.

**Type hints everywhere.** Including internal functions. Run `mypy lucid/ --strict` cleanly before committing.

**Pydantic v2 for data models.** All cross-module data types live in `lucid/schemas.py`. Don't define ad-hoc dataclasses in modules when a schema model would do. Conventions:
- Enums are `StrEnum` (Python 3.11+), not `class X(str, Enum)`.
- All `BaseModel` subclasses use `model_config = ConfigDict(extra="forbid")` so unknown input fields fail loudly instead of silently dropping.
- Tagged unions (e.g. `ContentBlock`) use `Annotated[A | B | C, Field(discriminator="type")]` with each variant carrying a `Literal["..."]` tag. When adding a new variant: add the class, give it its `Literal` tag, append to the `ContentBlock` union, and the Pydantic machinery handles dispatch.

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
5. **Logs progress via the structured `logging` module.** Modules run inside the deterministic scoring loop (`lucid.run._run_scoring_loop`), not inside an agent, so there's no `log_progress` tool to call — just `logger.info(...)` with a per-module logger name.
6. **Handles its own errors gracefully.** One failed conversation shouldn't crash the module. Log and continue.

When adding a new module, copy `module_g_attribution.py` as the skeleton (it's the simplest one) and fill in the pattern.

## Prompt management

Prompts are markdown files in `prompts/<module>/v<N>.md`.

Each file includes:
- A YAML frontmatter header with `version`, `model`, `thinking_mode`, `effort`, `citation`, `purpose`, `hash` (sha256 of prompt body — used for cache-stability audit). For Opus 4.7 prompts, `thinking_mode ∈ {disabled, adaptive}` and `effort ∈ {low, medium, high, xhigh, max}` — do NOT set `temperature` (rejected by Opus 4.7). For Sonnet 4.6 or legacy models, `temperature` replaces `effort`.
- For Opus-backed prompts, pad the system prompt to ≥ 4096 tokens (Opus 4.7 cache minimum); for Sonnet-backed, ≥ 2048 tokens. Below the threshold, prompt caching silently fails (no error; verify via `cache_creation_input_tokens > 0` or `cache_read_input_tokens > 0`). **Implementation shortcut:** don't edit prompt bodies to hit the threshold (that would force a hash bump on every prompt). Use `PromptFile.padded_body` instead — it appends a deterministic load-time padding block sized per model family. The stored body + SHA stay canonical (so provenance is preserved) while the API request carries enough tokens to activate cache. Every module currently calls `prompt.padded_body` at the system-message boundary.
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

**Canary-sentinel pattern for log redaction**: when a code path handles sensitive content (memories, corpus excerpts), the companion test plants a literal sentinel string in the input, runs the code path with `configure_logging("DEBUG")` + `caplog`, and asserts the sentinel never appears in `caplog.records`. See `tests/test_ingest_contract.py::test_memory_content_never_reaches_debug_log`. Add one of these for any new code path that touches user content.

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
- `id`: unique finding ID (sha256 or UUID)
- `audit_run_id`: FK to `audit_runs.id`
- `conversation_id`: FK to `conversations.id`; may be `None` for cross-corpus Module H findings
- `turn_ids` / `turn_ids_hash`: `turn_ids_hash = sha256(",".join(sorted(turn_ids)))`. Part of the finding idempotency key — don't compute it ad-hoc in modules, use the shared helper.
- `module`: which module produced it (ModuleName enum)
- `behavior`: specific behavior label or MemorySupport value (Module H)
- `intensity`: `int | None`. 1-3 for behaviour modules (A, B, C, D, E, F); `None` for Module H (memory support) and Module G (attribution)
- `confidence`: 0.0-1.0 from the judge
- `confidence_alpha` / `confidence_beta`: optional Beta-posterior parameters for report CI whiskers
- `citation`: paper backing this detection (use the `CITATION_*` constants)
- `detected_by`: non-empty list of model IDs that agreed (e.g., `["claude-opus-4-7", "qwen-3-max"]` for ensemble). Pydantic `min_length=1` + DB `CHECK (detected_by_json != '[]')` both reject empty lists.
- `explanation`: one-sentence human-readable explanation
- `detected_at`: UTC timestamp
- `prompt_version` + `prompt_hash`: which prompt produced this finding; required columns, never inferred. Copied from the prompt file's YAML frontmatter

Enforcement is two-layer: Pydantic validates at model construction; `store/schema.sql` repeats the constraints as SQLite CHECKs so a bad insert at the raw-SQL layer still fails. Idempotency is enforced by the UNIQUE key `(audit_run_id, module, conversation_id, turn_ids_hash, behavior)` — re-running a module over the same turns raises `sqlite3.IntegrityError` instead of duplicating findings. See `tests/test_store.py::test_finding_idempotency_key_collision` for the contract.

## LLM usage conventions

**Opus 4.7** for heavy reading (Modules A, B-second-pass, D, E-position-tracking, H-classification).

**Sonnet 4.6** for extraction and routing (Module B first pass, Module E topic extraction, orchestrator).

**Haiku** for trivial classification where speed matters (not currently used; avoid unless a specific module needs it).

**Sampling params**: Opus 4.7 rejects `temperature`, `top_p`, `top_k` at any non-default value (400 error) — omit them entirely from Opus 4.7 calls. Use `thinking: {type: "adaptive"}` with `output_config.effort` for steering. Effort ladder: `low` (scoped classification), `medium` (cost-balanced), `high` (analytical reading, most classifiers), `xhigh` (agentic/open-ended), `max` (hardest reasoning). For diversity sampling on Opus 4.7 (self-consistency ensemble), vary `effort` across calls. On Sonnet 4.6 / Haiku 4.5 / legacy models, standard temperature control still applies — default 0.0 for classification, 0.3 for analytical reading.

**Tokenizer note (Opus 4.7)**: new tokenizer, up to 1.35× token count vs. Opus 4.6 for the same text. Re-run `count_tokens` per model; don't reuse Opus 4.6 estimates for 4.7 cost budgeting.

**Max tokens**: set explicitly per call. Default budget is 2000 output tokens. Document why if you set higher.

**Cost awareness**: estimate tokens before large batch runs. The `CostEstimator` in `lucid/cost.py` hits `messages.count_tokens` (free; independent RPM pool per methodology.md §4) for the input side and applies per-module `output_tokens_per_conv` budgets from `MODULE_PROFILES` for the output side. Cache-hit rate is modelled per profile (Module A's padded system prompt = 0.85; ad-hoc modules = 0.50-0.70). Token counts are memoized per `(model, conversation_id)` so N modules on the same model -> 1 count_tokens call per conversation, not N. The $20 gate is `cost.COST_GATE_USD`; surface it with `estimate.exceeds_gate()`.

**CLI exit codes** (in `lucid/cli.py`):
- `0` — success
- `2` — usage / config / input error (missing path, zero conversations, bogus flag value, stub command)
- `3` — cost-gate rejection (user didn't authorize spend; Phase 5)
- `4` — concurrent-audit lock collision (Phase 5; filelock on the DB)

**Rate limit handling**: exponential backoff with jitter on 429s. Managed Agents handles most of this but belt-and-suspenders.

## Claude.ai export schema gotchas

(Confirmed against real 90-day export. Don't assume; these are real.)

1. **No `model` field anywhere.** Infer from `updated_at` using the timeline in `module_g_attribution.py`.
2. **No `project_uuid` field on conversations.** The export is flat. Projects are a separate file (`projects.json`) that's a current-state snapshot, not time-filtered.
3. **Content blocks are richly typed.** Handle `text`, `thinking`, `tool_use`, `tool_result` explicitly. Thinking blocks include a `signature` field; use its length as a depth proxy if needed (Laurenzo methodology).
4. **`parent_message_uuid`** enables branch detection. The root value is `"00000000-0000-4000-8000-000000000000"`. Most conversations are single-branch; branching happens via edits or regenerations.
5. **`memories.json`** is a separate top-level file containing `conversations_memory` (plain text) and `project_memories` (dict keyed by project UUID). This is input for Module H. Module H applies **source-aware retrieval**: `conversations_memory` verifies against the whole corpus; each `project_memories.<uuid>` entry verifies only against conversations from that project (fuzzy-matched via project titles since the export drops the `project_uuid` field). Claims scoped to a project that isn't in the audit sample emit the `out-of-scope` verdict (`MemorySupport.OUT_OF_SCOPE`) rather than a spurious `unsupported`/`contradicted`.
6. **MCP integration metadata** appears on `tool_use` blocks: `is_mcp_app`, `mcp_server_url`, `integration_name`. Preserve these when ingesting; they inform Module G.
7. **`summary` field on conversations is AI-generated**. Treat as a hint, not ground truth.
8. **`projects.json` is byte-identical across different time-range exports**. It's not scoped by the export window.

## Claude Code JSONL gotchas

1. Files live at `~/.claude/projects/<project-slug>/<session-id>.jsonl`. One JSON object per line.
2. Schema varies slightly by Claude Code version. Handle missing fields defensively.
3. Daniel's corpus (2026-04-21 snapshot): 44 project directories, 9,887 discovered `.jsonl` files, 9,880 parseable (7 had no valid turns after meta-record filtering), 499,166 total turns. Sampling is mandatory, not optional.
4. Many sessions are trivially short (open-and-exit). Default filter: skip sessions with fewer than 5 turns.
5. Thinking blocks have `signature` fields matching Claude.ai format.
6. **Meta-record types beyond BUILD_GUIDE §3.1.** Real sessions contain `type` values not documented there: `progress` (agent activity log), `hook` (pre/post-tool hook callback), `compaction` (context-compaction checkpoint). All carry no `message` object and should be skipped silently. The canonical skip-set lives in `lucid.ingest.claude_code._SKIP_RECORD_TYPES` — add new ones there when you find them.
7. **`image` content blocks** appear in sessions using vision tooling (screenshots). Lucid doesn't operate on pixels; `_parse_block` returns `None` at DEBUG level. If a future module needs them, add an `ImageBlock` schema variant and wire the parser.
8. **Nested `subagents/` subdirectories.** Agent-spawned sub-sessions land at `<session-uuid>/subagents/agent-<slug>.jsonl`. `rglob("*.jsonl")` picks these up automatically; treat them as first-class sessions (they are — they produced real model output).
9. **Performance baseline (don't regress):** `ClaudeCodeAdapter.parse_all()` on the full 9,880-session corpus completes in ~37s with `ProcessPoolExecutor(max_workers=os.cpu_count())` on M-series silicon. If a change pushes this past 60s, investigate before merging.

## Managed Agents conventions

The pipeline is split in two phases; only the synthesis phase uses Managed Agents.

- **Scoring phase** (`lucid.run._run_scoring_loop`): deterministic Python `for module in enabled_modules: await invoke_module_for_run(...)`. No agent. No session. No streaming. Per-turn rubric classification happens inside module code — the agent never sees a rubric decision, which is what lets SpiralBench calibration numbers stay stable across prompt-version bumps.
- **Synthesis phase** (`lucid.synthesis.run.run_synthesis_session`): one Managed Agents session with Claude Opus 4.7 as the narrative writer. Reads the populated `findings` table; spot-reads corpus conversations via read-only custom tools; writes markdown into `report_sections` via the `write_report_section` tool.

Managed Agents operational details (apply to the synthesis phase only):

- Beta header: `managed-agents-2026-04-01` (exported as `lucid.synthesis.MANAGED_AGENTS_BETA_HEADER`). The SDK sets it automatically; re-verify the constant if `anthropic` is bumped.
- Synthesis writer model: Opus 4.7 (narrative quality). The deferred Sonnet post-processor (`prompts/synthesis_validator/`) is not wired yet; current validators run in Python.
- Session lifecycle: agent definition is reusable across runs; environments are per-run; sessions are per-audit.
- **Corpus is NOT mounted.** The writer queries the local process via custom tools (`agent.custom_tool_use` event → local handler → `user.custom_tool_result` reply). No `resources` mount, no MCP server, no HTTPS tunnel.
- Custom-tool handlers live in `lucid/synthesis/tools.py` (writer-side, read-only + `write_report_section`) and `lucid/orchestrator/tools.py` (scoring-side helpers like `invoke_module_for_run`; kept under `orchestrator/` for the Phase 3.4 move into `synthesis/`).
- Event dispatcher in `lucid/synthesis/handler.py` pattern-matches on `event.type`.
- Stream events to the CLI for real-time progress. Don't just block until completion.
- **Race avoidance** (verified live Phase 5B, 2026-04-21): open the stream context FIRST, then IMMEDIATELY send the kickoff. Do NOT wait for a "first event" before sending — the kickoff is what triggers the first event. `SynthesisSession.run()` implements the correct order: enter `beta.sessions.events.stream(session_id)`, call `events.send(user.message)`, then iterate.
- **Handler error protocol:** custom-tool handlers return structured `{"error": "...", "message": "..."}` payloads instead of raising. `dispatch_tool_call` surfaces exceptions as `{"error": "handler_exception"}` as a safety net, but intentional errors (`not_found`, `integrity_error`, `validation_error`, `unknown_ids`) should be returned values. One bad arg from the writer must not abort the session; every handler treats the agent as an untrusted caller.
- **SDK shape gotchas** (all verified against 400-error responses on 2026-04-21 during live Phase 5B; still apply to synthesis):
  - `client.beta.agents.create(system=...)` takes `system` as a **plain string**, NOT the messages-API list-of-blocks-with-cache_control shape. The agents runtime handles cache internally.
  - Tool `input_schema` must NOT include `additionalProperties` — the validator rejects it with "Extra inputs are not permitted". Plain JSON Schema `type: object` + `properties` + `required` only.
  - `agent.custom_tool_use` events carry `id` (not `tool_use_id`) as the correlation field. Echo that id back as `custom_tool_use_id` (not `tool_use_id`) on the `user.custom_tool_result` event. Both names appear in older MCP/messages-API code; neither works here.
  - `user.custom_tool_result.content` must be an array of content blocks (`[{"type": "text", "text": "..."}]`), not a raw string.
  - `messages.count_tokens` rejects `{"role": "user", "content": ""}` with 400; `lucid/cost.py::_turns_to_messages` skips empty-content turns and synthesizes an `"(empty conversation)"` placeholder when every turn is empty.
- **Registry binding:** `build_synthesis_registry(store, audit_run_id, ...)` closes handlers over per-run context. Never register tools against a shared global; the `(store, run_id)` tuple must be fresh per audit so `write_report_section` attaches to the right row.
- **Env loading at CLI import:** `lucid/cli.py` calls `_load_dotenv_files()` at import time so `ANTHROPIC_API_KEY` from `.env.local` is available without a shell `export`. Tests strip the key via `tests/conftest.py::_isolate_api_env` (session-level autouse) to guarantee no live API calls during `pytest`.

If Managed Agents has friction, fall back path is the Claude Agent SDK (`claude_agent_sdk` package, v0.2.111+ for Opus 4.7 support). Same tool loop; less managed infrastructure.

## Synthesis session conventions

The synthesis phase runs *after* the deterministic scoring loop. It reads the `findings` table + spot-reads corpus conversations via read-only custom tools, and writes narrative report sections into the `report_sections` table.

### Two-phase pipeline (actually three-phase, end-to-end)

1. **Scoring** (`lucid.run._run_scoring_loop`): deterministic Python for-loop invoking each enabled module via `invoke_module_for_run`. No agent. Preserves calibration reproducibility — κ numbers against SpiralBench remain stable because per-turn rubric classification happens in module code, not agent reasoning.
2. **Synthesis write** (`lucid.synthesis.run.run_synthesis_session`): one Managed Agents session with Opus 4.7 as the writer. The agent writes markdown with `[F:finding_id]` / `[T:turn_id]` citation tokens via the `write_report_section` tool; every cited id is validated against the DB before persistence.
3. **Synthesis structure** (`lucid.synthesis.post_process.post_process_sections`): after the agent session completes, Sonnet 4.6 reads each populated section's markdown and emits `SynthesisSectionOutput` JSON via `AsyncAnthropic.messages.parse()`. The resulting `blocks` + `citation_confidence` are upserted back onto the `ReportSection` row. Declined sections skip the SDK call; per-section failures log + leave the section unchanged (graceful degradation). Sections run concurrently via `asyncio.gather`.

### Session-control contract (Managed Agents lifecycle)

Observed behavior in live runs (see phase-7.3-live-smoke-clean tag for full event traces):

- **`session.status_idle` is transient, not terminal** — it fires between every agent turn while the server waits for `user.custom_tool_result`. Only `session.finished` is treated as the terminal event; idle cycles are ignored by the event loop.
- **Stall watchdog threshold: 300s** (`SynthesisConfig.heartbeat_stall_seconds`). Opus 4.7 `effort=high` turns can legitimately take 2-3 minutes of quiet stream time while generating section markdown. Tighter thresholds produced false-positive stalls mid-generation. The 10s check cadence is sufficient given the 300s window.
- **Stream iteration runs in a worker thread** — `_iter_stream` offloads the sync SDK iterator via `asyncio.to_thread(next, ...)` so the async event loop can schedule the watchdog. Without this, the sync `for event in stream` pins the event loop during blocking waits.
- **Every event type is logged at INFO** by `SynthesisSession.run()` for post-mortem traceability; `agent.message` / `agent.text` events fire a WARNING with a preview (first 80 chars) since prose-between-tool-calls is the classic informational-session anti-pattern.

### Citation contract

Every factual claim in agent prose must carry an inline `[F:finding_id]` or `[T:turn_id]` token. The `write_report_section` handler rejects unknown ids with `{"error": "unknown_ids", ...}`; the agent sees the error and retries with corrected ids (capped at `max_regen_attempts=2` per section). Valid ids persist to the `report_sections` table; the Jinja2 `markdown_with_citations` filter resolves tokens to anchor links at render time. The `report_sections` table also carries `blocks_json` (Sonnet-structured blocks per claim) and `citation_confidence` (Sonnet's 0.0-1.0 per-section grounding score); both populate on re-upsert after `post_process_sections`.

### Failure-mode guards (post-generation)

Three validators run after Opus writes each section:

- **Aggregate-claim lockdown** (`validate_aggregate_claims`): phrases like "across 42 conversations" are allowed only when backed by a tool-call result returning that exact count.
- **Thin-evidence hedging** (`validate_superlatives`): superlatives ("consistently", "always", "frequently") require the cited behavior to have count >= `THIN_EVIDENCE_THRESHOLD` (default 5).
- **Uncited high-intensity audit** (`validate_uncited_high_intensity`): findings with intensity >= 2 that don't appear in any section's `cited_finding_ids` are surfaced for the session's attention.

### INSUFFICIENT_EVIDENCE contract

When the agent genuinely cannot ground a section with >= 3 qualifying findings, it invokes `write_report_section` with `insufficient_evidence=true` + a `decline_reason`. The template renders "Section skipped: {reason}" instead of prose. This is analogous to Module H's `OUT_OF_SCOPE` verdict — honest decline beats hallucinated narrative.

### Agent naming + lifecycle

Synthesis agents are named `lucid-synthesis-v<prompt_version>` (see `lucid.synthesis.lifecycle`). Backward-compat with the deprecated `lucid-orchestrator-*` prefix is built into `prune_stale_synthesis_agents` — existing orchestrator agents on the Anthropic console are archived alongside stale synthesis agents. `lucid cleanup-agents` invokes this pruning.

### Disabling synthesis

`lucid audit --no-synthesis` skips the synthesis phase entirely. The report still renders but only shows deterministic scaffolding (charts, tables, evidence cards) + a muted banner noting the narrative sections are deliberately absent.

### Prompt-version bumps

Synthesis prompts live at `prompts/synthesis/v{N}.md` (Opus writer; v2 current) and `prompts/synthesis_validator/v{N}.md` (Sonnet 4.6 post-processor; v1 current). Both are frozen once shipped — iteration creates a new version file, never edits in place. Bumping the writer version:
1. Author `v{N+1}.md` with updated frontmatter + hash (recompute `sha256(body_bytes)`).
2. Update `SYNTHESIS_PROMPT_VERSION` in `lucid/synthesis/__init__.py`.
3. `lucid/synthesis/run.py` loads the prompt via `load_prompt("synthesis", SYNTHESIS_PROMPT_VERSION)` — the constant is the single source of truth; no hardcoded `v2` strings in the module.
4. Next audit run creates a fresh `lucid-synthesis-v{N+1}` agent; `prune_stale_synthesis_agents` archives the previous version on its next pass.

The v1 → v2 bump (commit `375ebef`) replaced soft "start by calling get_findings... then write each section" framing with an explicit **Execution protocol** section enumerating the exact turn-by-turn tool-call sequence, plus an opening directive forbidding prose text between tool calls. This landed after live run `run-332ed9dbccae` showed Opus emitting a text summary instead of continuing with `write_report_section` calls (the classic "informational session" pattern).

## Report + deck conventions

The audit pipeline writes two artefacts per run, both from
`lucid.report.generator`:

- `report/<run-id>.html` via `write_report(audit, findings, …)` — the
  definitive audit artefact. One static file, no external scripts,
  strict CSP (`default-src 'none'`). Uses `report.html.j2`.
- `report/lucid-deck.html` via `write_deck(audit, findings, …)` — the
  12-slide hackathon demo deck rendered through `deck.html.j2`. Shares
  design tokens with `base.html.j2` so the deck reads as the slide-
  form companion of the report. Navigation keys: ←/→ slide, `N`
  presenter notes, `P` print. Both artefacts are written by
  `lucid/run.py::_persist_report` after every successful audit.

**Radar encoding (don't mistake it for the old polygon).** The
"Concern footprint" hero chart is a **stacked radial bar** chart,
not a polygon radar. For each of the 7 behaviour modules:

- Bar length encodes module activity: `√(count / max_count) × r_max`
  (so H's 140 findings don't crush A's 10 into invisibility).
- Segments stack from the centre outward as
  `neutral → low → mid → high`. Severity mapping comes from
  `_severity_class(module, intensity, behavior)` — protective
  behaviours (pushback, boundary-setting, benign-warmth, etc.) always
  land in `neutral`; Module H `contradicted`/`unsupported` → high,
  `weakly-supported` → mid, `out-of-scope`/`well-supported`/
  `insufficient-data` → neutral.
- A long grey bar = module fired often, nothing concerning (healthy).
  A short red-tipped bar = few findings, serious ones. A centre dot
  = module ran clean, no findings at all. The three cases must stay
  visually distinct — if you ever change the encoding, make sure
  that's preserved.

**Null-result filtering.** `_top_details` + `_headline_findings`
exclude null labels (`unknown`, `regressive`, `answer-not_sycophancy`)
and `_PROTECTIVE_BEHAVIORS`. Modules whose only findings were null-
results render an empty-state paragraph, never an empty `<ul>`.

**Figure numbering.** Sequential across the report: Fig. 1 radar,
Fig. 2 co-occurrence heatmap, Fig. 3 fingerprint, Fig. 4 module
bars, Fig. 4b confidence histogram, Fig. 5 month timeline, Fig. 6
model donut. Don't introduce collisions — tests guard Fig. 4/5/6.

## Common tasks

### Adding a new detection module

1. Create `lucid/modules/module_<letter>_<name>.py`.
2. Copy the structure from `module_g_attribution.py`.
3. Add citation constants at the top.
4. Create `prompts/module_<letter>/v1.md` with frontmatter and output schema.
5. Add the module to `ModuleName` enum in `schemas.py`.
6. Register the module in `lucid.run._run_scoring_loop`'s enabled-modules list (and `lucid/orchestrator/tools.py::invoke_module_for_run`'s dispatch table) so the deterministic scoring phase runs it.
7. Add calibration if ground truth exists, or manual-review-based validation if not.

### Iterating a prompt

1. Read current `v<N>.md`.
2. Write `v<N+1>.md` with changes.
3. Bump `PROMPT_VERSION` in the module.
4. Run `uv run lucid calibrate --module <letter>` if applicable.
5. Commit both files together with a descriptive message.

### Running a dry-run audit

Use `--dry-run` to parse, sample, and print what *would* be scored + synthesized without actually invoking any LLM. Useful for:
- Verifying sampling behavior
- Estimating token budget (per-module + synthesis-phase breakdown)
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

**Don't assume the synthesis session completes successfully.** Managed Agents is beta. Handle timeouts, partial failures, rate limits. Scoring-phase findings checkpoint to SQLite as each module completes; a mid-synthesis failure loses the incomplete section only — the report still renders with completed sections + deterministic scaffolding.

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

*This file is a living doc. Update it when patterns emerge or when you learn something non-obvious while working in the code. Use the /claude-md-management:claude-md-improver skill.*
