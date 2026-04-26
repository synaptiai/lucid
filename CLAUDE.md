# CLAUDE.md

Operational rules for Claude Code working in this repository. Read `docs/PRD.md` for scope, `docs/BUILD_GUIDE.md` for architecture context, `docs/methodology.md` for "how does this actually work."

## What this project is

Lucid is an open-source epistemic audit tool for personal AI conversation history. It ingests Claude Code sessions and Claude.ai conversation exports, applies a composition of published AI safety frameworks (SpiralBench, Sharma sycophancy, SycEval, Jain perspective sycophancy, BeliefShift, Truth Decay, Influence Tactics Protocol) via a two-phase pipeline — deterministic Python scoring, then a Managed Agents synthesis session that writes narrative report sections — and produces a structured report surfacing sycophancy events, belief drift, reinforcement spirals, user influence tactics, memory-corpus consistency, and time/model attribution.

This is a hackathon build (April 21–26, 2026), now in its final-day polish phase.

## Working style

Optimize for honesty over optimism. Findings must be defensible. Prompts must be calibrated against ground truth where ground truth exists. If a module can't determine something, the correct output is `"unknown"`, `"insufficient-data"`, or empty — not a hallucinated best guess.

Be direct in code comments and docstrings. No marketing language. No "revolutionary" or "game-changing." We're shipping a tool, not a landing page.

For decisions with multiple viable approaches: research first, then present 2–3 options with tradeoffs. Use the AskUserQuestion tool for clarifications.

## Quick reference

```bash
# Install
uv sync --extra dev

# CLI smoke
uv run lucid --help
uv run lucid version

# Dry-run (parse + sample + count_tokens cost estimate; no LLM spend)
uv run lucid audit --source claude-code --path ~/.claude/projects --sample 10 --dry-run
uv run lucid audit --source claude-ai --path ./export --dry-run

# Module D ships ON by default. --no-include-module-d skips it.
# Real runs typically need --yes-i-authorize-spend-up-to 50 because D pushes past the $20 gate.
LUCID_ALLOW_UNATTENDED=1 uv run lucid audit --source claude-ai --path ./export \
  --yes-i-authorize-spend-up-to 50

# Skip synthesis (scoring still runs; report renders deterministic scaffolding only)
uv run lucid audit --source claude-ai --path ./export --no-synthesis

# Calibration (real LLM call, separate from pytest)
uv run lucid calibrate --module a

# Tests / lint / type-check
uv run pytest
uv run mypy lucid/ --strict
uv run ruff check lucid/ && uv run ruff format lucid/
```

## Architecture map

```
lucid/cli.py              CLI entry (Typer)
lucid/schemas.py          Pydantic models — authoritative data types
lucid/config.py           Settings, paths, API keys from env
lucid/sampling.py         Corpus sampling (stratified + recency-weighted)
lucid/cost.py             count_tokens pre-pass + per-module output budgets
lucid/logging.py          SafeFormatter — drops records tagged contains_user_content
lucid/prompts.py          Prompt loader: parses frontmatter, verifies sha256, applies cache padding
lucid/run.py              Two-phase pipeline: _run_scoring_loop (deterministic) + run_synthesis_session

lucid/ingest/             IngestAdapter ABC + claude_code.py + claude_ai.py
lucid/store/              schema.sql + aiosqlite reads + initialize_db
lucid/orchestrator/       tools.py — scoring-side custom tools (invoke_module, log_progress, etc.)

lucid/synthesis/
  __init__.py             SYNTHESIS_PROMPT_VERSION constant
  session.py              SynthesisSession + MANAGED_AGENTS_BETA_HEADER
  run.py                  run_synthesis_session() — section loop + regen retries
  handler.py              custom_tool_use dispatcher + HeartbeatMonitor
  tools.py                build_synthesis_registry — read-only tools + write_report_section
  validators.py           validate_aggregate_claims, validate_superlatives, validate_uncited_high_intensity
  lifecycle.py            get_or_create_synthesis_agent + prune_stale_synthesis_agents
  post_process.py         Sonnet structuring of agent prose into SynthesisSectionOutput

lucid/modules/
  base.py                 CorpusModule / FindingsModule protocols
  embeddings.py           Voyage wrapper (OpenAI fallback)
  module_a_spiralbench.py SpiralBench behavior scorer (17 behaviors)
  module_b_sharma.py      Sharma paired-exchange (all 4 subroutines)
  module_c_syceval.py     Progressive/regressive classifier
  module_d_perspective.py Jain perspective sycophancy (default-on)
  module_e_beliefshift.py DCS-simplified belief drift
  module_f_itp.py         Influence Tactics Protocol on user prompts (9 categories)
  module_g_attribution.py Time/model bucketing (deterministic, no LLM)
  module_h_memory.py      Memory-corpus consistency check

lucid/calibration/        SpiralBench loading + Krippendorff α + Gwet AC1 + per-label κ + QWK + BCa bootstrap
lucid/report/             Jinja2 + Chart.js HTML report (templates/report.html.j2, deck.html.j2)

prompts/
  module_<letter>/v<N>.md Versioned, immutable once shipped
  synthesis/v<N>.md       Opus 4.7 writer prompt (v2 current)
  synthesis_validator/    Sonnet 4.6 post-processor prompt (v1 current)

tests/                    pytest with mocked Anthropic clients; real-LLM calibration is a separate command
```

## Code conventions

**Python 3.13 exactly** (`requires-python = ">=3.13,<3.14"`). Modern syntax: `list[str]` not `List[str]`, `X | None` not `Optional[X]`, `match` where clear.

**Type hints everywhere.** `mypy lucid/ --strict` must pass before commit.

**Pydantic v2 for data models.** All cross-module types live in `lucid/schemas.py`. Conventions:
- Enums are `StrEnum` (Python 3.11+).
- All `BaseModel` use `model_config = ConfigDict(extra="forbid")` so unknown input fields fail loudly.
- Tagged unions use `Annotated[A | B | C, Field(discriminator="type")]` with each variant carrying a `Literal["..."]` tag.

**Async where it helps** (LLM calls, many-file I/O). Sync where it doesn't (CLI parsing, deterministic computation).

**No bare `except`.** Catch specific exceptions; log; never swallow silently.

**No print debugging.** Use `logging`. CLI progress goes through `rich`.

**Explicit > magic.** No decorators that hide control flow. This codebase must be auditable by researchers who don't read Python idioms.

## Module conventions

Every detection module follows the same pattern:

1. **Cites its source paper** as a module-level docstring with arxiv link.
2. **Top-level `async def run(corpus, config) -> list[Finding]`** as public interface.
3. **Loads prompts via `load_prompt("<letter>", PROMPT_VERSION)`** — never hardcoded strings.
4. **Produces `Finding` objects** populating `citation`, `detected_by`, `confidence`, `quote_user`/`quote_assistant` or `evidence_quotes`.
5. **Logs progress via `logging`** (a per-module logger). Modules run inside the deterministic scoring loop, not inside an agent — there's no `log_progress` tool to call.
6. **Handles its own errors gracefully.** One failed conversation must not crash the module.

To add a new module: use the `add-detection-module` skill (`/add-detection-module`). It scaffolds the module file, schema enum entry, prompt v1 stub, scoring-loop registration, orchestrator dispatch entry, cost profile, and test scaffold.

## Prompt management

Prompts live at `prompts/<name>/v<N>.md`. Each file has YAML frontmatter (`version`, `model`, `thinking_mode`, `effort`, `citation`, `purpose`, `hash`) and a body. The loader (`lucid/prompts.py`) verifies `sha256(body)` matches the `hash` field — mismatches raise loudly so audit provenance never claims a prompt version that doesn't match what ran.

**Cache padding.** Anthropic's prompt cache silently no-ops below the model-family threshold (Opus 4.7: 4096 tokens; Sonnet 4.6: 2048 tokens). Lucid prompts sit below the threshold; `PromptFile.padded_body` appends a deterministic load-time padding block to clear the threshold. The stored body + SHA stay canonical (provenance preserved) while the API request carries enough tokens to activate cache. Every module passes `prompt.padded_body` to the API.

**Iterating a prompt:** use the `bump-prompt-version` skill (`/bump-prompt-version <module>`). It copies `v<N>.md` → `v<N+1>.md`, recomputes the hash, bumps the `PROMPT_VERSION` constant, and stages both files. After editing the body, re-seal with `--rehash-only`. **Never edit a published prompt version in place** — the `protect_files.py` PreToolUse hook enforces this.

After a bump on a module with a calibration target (A, B, C, D, E, H), invoke the `calibration-regression-checker` agent to verify κ/α didn't regress against the previous run.

## Testing strategy

- **Ingest adapters:** real fixture files in `tests/fixtures/`. Fixtures must be redacted/synthetic — never real user content.
- **Schemas:** round-trip serialize → deserialize → equal.
- **Sampling:** determinism (same seed → same sample) + stratification.
- **Modules:** mocked LLM responses. **Never call real LLMs in pytest** — slow, non-deterministic, costs money.
- **Calibration harness:** tested via a small frozen SpiralBench-style fixture with known labels. The real calibration run hits the LLM and is a separate command.
- **Canary-sentinel pattern for log redaction:** when a code path handles sensitive content, the companion test plants a literal sentinel in the input, runs with `configure_logging("DEBUG")` + `caplog`, and asserts the sentinel never appears in `caplog.records`. See `tests/test_ingest_contract.py::test_memory_content_never_reaches_debug_log`. Add one for any new path that touches user content.
- **Don't test exact prompt strings.** Test output parsing and the module's handling of expected/unexpected LLM responses.

## Findings provenance (enforcement-critical)

Every `Finding` must populate:

- `id`, `audit_run_id` (FK), `conversation_id` (FK; `None` for cross-corpus Module H findings)
- `turn_ids` + `turn_ids_hash` = `sha256(",".join(sorted(turn_ids)))`. Part of the idempotency key — use the shared helper, don't compute ad-hoc.
- `module` (ModuleName enum), `behavior` (specific label or MemorySupport for Module H)
- `intensity`: 1–3 for behaviour modules (A, B, C, D, E, F); `None` for Module H + Module G.
- `confidence` (0.0–1.0) + optional `confidence_alpha`/`confidence_beta` for report CI whiskers
- `citation` — use the `CITATION_*` constants at the top of each module file
- `detected_by`: non-empty list of model IDs (Pydantic `min_length=1` + DB `CHECK` both enforce)
- `explanation` — one-sentence human-readable
- `detected_at` — UTC timestamp
- `prompt_version` + `prompt_hash` — copied from the prompt YAML frontmatter; required, never inferred

Enforcement is two-layer: Pydantic validates at construction; `store/schema.sql` repeats constraints as SQLite CHECKs so a raw-SQL bad insert still fails. Idempotency UNIQUE key: `(audit_run_id, module, conversation_id, turn_ids_hash, behavior)`. Re-running a module over the same turns raises `sqlite3.IntegrityError` instead of duplicating. See `tests/test_store.py::test_finding_idempotency_key_collision`.

Citation constants:

```python
CITATION_SHARMA_2023 = "Sharma et al. 2023, 'Towards Understanding Sycophancy in Language Models', arxiv:2310.13548"
CITATION_SPIRALBENCH = "Spiral-Bench, https://eqbench.com/spiral-bench.html"
CITATION_SYCEVAL = "Fanous, Goldberg et al. 2025, 'SycEval: Evaluating LLM Sycophancy', AAAI AIES 2025"
CITATION_JAIN_2025 = "Jain et al. 2025 (cited in Opus 4.5 independent audit)"
CITATION_BELIEFSHIFT = "BeliefShift: Benchmarking Temporal Belief Consistency and Opinion Drift in LLM Agents, arxiv:2603.23848"
CITATION_ITP = "Influence Tactics Protocol, https://github.com/synaptiai/influence-tactics-protocol"
```

## LLM usage conventions

**Model assignment:**
- **Opus 4.7** — heavy reading (Modules A, B-second-pass, D, E-position-tracking, H-classification, synthesis writer).
- **Sonnet 4.6** — extraction and routing (Module B first pass, Module E topic extraction, synthesis post-processor).
- **Haiku** — currently unused; avoid unless a specific module needs it.

**Sampling params (Opus 4.7):** rejects `temperature`, `top_p`, `top_k` at any non-default value (400 error) — omit them entirely. Use `thinking: {type: "adaptive"}` with `output_config.effort` for steering. Effort ladder: `low` → `medium` → `high` → `xhigh` → `max`. For self-consistency ensemble, vary `effort` across calls.

**Sampling params (Sonnet 4.6 / Haiku 4.5):** standard temperature control still applies. Default 0.0 for classification, 0.3 for analytical reading.

**Tokenizer note:** Opus 4.7 uses a new tokenizer (up to 1.35× Opus 4.6 for the same text). Re-run `count_tokens` per model; don't reuse Opus 4.6 estimates for 4.7 cost budgeting.

**Max tokens:** explicit per call. Default 2000. Document why if higher.

**Cost awareness:** `CostEstimator` in `lucid/cost.py` hits `messages.count_tokens` (free; independent RPM pool) for input and applies `MODULE_PROFILES[*].output_tokens_per_conv` for output. Cache-hit rate modelled per profile (Module A's padded prompt = 0.85; ad-hoc modules = 0.50–0.70). Token counts memoized per `(model, conversation_id)`. Gate is `cost.COST_GATE_USD = $20`; surface with `estimate.exceeds_gate()`.

**Rate limit handling:** exponential backoff with jitter on 429s. Managed Agents handles most; belt-and-suspenders for direct calls.

## CLI exit codes (`lucid/cli.py`)

- `0` — success
- `2` — usage / config / input error (missing path, zero conversations, bogus flag, stub command)
- `3` — cost-gate rejection
- `4` — concurrent-audit lock collision (filelock on the DB)

## Claude.ai export schema gotchas

Confirmed against real 90-day export. Don't assume.

1. **No `model` field anywhere.** Infer from `updated_at` using `module_g_attribution.py`.
2. **No `project_uuid` on conversations.** The export is flat. `projects.json` is a separate current-state snapshot, not time-filtered.
3. **Content blocks are richly typed.** Handle `text`, `thinking`, `tool_use`, `tool_result` explicitly. Thinking blocks have a `signature` field; use its length as a depth proxy if needed (Laurenzo methodology).
4. **`parent_message_uuid`** enables branch detection. Root value: `"00000000-0000-4000-8000-000000000000"`. Most conversations are single-branch.
5. **`memories.json`** is a separate top-level file with `conversations_memory` (plain text) and `project_memories` (dict keyed by project UUID). Module H applies **source-aware retrieval**: `conversations_memory` verifies against the whole corpus; each `project_memories.<uuid>` entry verifies only against conversations from that project (fuzzy-matched via project titles since the export drops `project_uuid`). Claims scoped to a project not in the audit sample emit `MemorySupport.OUT_OF_SCOPE` rather than a spurious `unsupported`/`contradicted`.
6. **MCP integration metadata** appears on `tool_use` blocks: `is_mcp_app`, `mcp_server_url`, `integration_name`. Preserve when ingesting; informs Module G.
7. **`summary` field is AI-generated.** Treat as a hint, not ground truth.
8. **`projects.json` is byte-identical across different time-range exports** — not scoped by export window.

## Claude Code JSONL gotchas

1. Files at `~/.claude/projects/<project-slug>/<session-id>.jsonl`. One JSON object per line.
2. Schema varies slightly by Claude Code version — handle missing fields defensively.
3. Daniel's corpus (2026-04-21 snapshot): 44 project dirs, 9,887 discovered `.jsonl`, 9,880 parseable, 499,166 total turns. Sampling is mandatory.
4. Many sessions are trivially short (open-and-exit). Default filter: skip < 5 turns.
5. Thinking blocks have `signature` fields matching Claude.ai format.
6. **Meta-record types beyond BUILD_GUIDE §3.1.** Real sessions contain `type` values not documented there: `progress`, `hook`, `compaction`. All carry no `message` object — skip silently. Canonical skip-set: `lucid.ingest.claude_code._SKIP_RECORD_TYPES`. Add new ones there as you find them.
7. **`image` content blocks** appear in vision-tool sessions. Lucid doesn't operate on pixels; `_parse_block` returns `None` at DEBUG. Add `ImageBlock` schema variant if a future module needs them.
8. **Nested `subagents/` subdirectories.** Agent-spawned sub-sessions land at `<session-uuid>/subagents/agent-<slug>.jsonl`. `rglob("*.jsonl")` picks them up; treat as first-class sessions.
9. **Performance baseline (don't regress):** `ClaudeCodeAdapter.parse_all()` on the full 9,880-session corpus completes in ~37s with `ProcessPoolExecutor(max_workers=os.cpu_count())` on M-series silicon. Investigate before merging if a change pushes past 60s.

## Synthesis phase (Managed Agents)

The pipeline is two-phase. **Scoring** (`lucid.run._run_scoring_loop`) is deterministic Python — no agent, no session, no streaming. Per-turn rubric classification happens inside module code, which is what lets SpiralBench calibration numbers stay stable across prompt-version bumps. **Synthesis** (`lucid.synthesis.run.run_synthesis_session`) opens one Managed Agents session with Opus 4.7 as the writer; reads the populated `findings` table; spot-reads conversations via read-only custom tools; writes markdown into `report_sections` via `write_report_section`. After the session completes, Sonnet 4.6 (`post_process.post_process_sections`) reads each section's markdown and emits structured `blocks` + `citation_confidence` via `messages.parse()`.

**Operational details:**

- Beta header: `managed-agents-2026-04-01` (exported as `lucid.synthesis.MANAGED_AGENTS_BETA_HEADER`). SDK sets it; re-verify on `anthropic` bumps.
- Session lifecycle: agent definition reusable across runs; environments per-run; sessions per-audit.
- **Corpus is NOT mounted.** Writer queries the local process via custom tools (`agent.custom_tool_use` event → local handler → `user.custom_tool_result` reply). No `resources` mount, no MCP server, no HTTPS tunnel.
- Custom-tool handlers: `lucid/synthesis/tools.py` (writer-side, read-only + `write_report_section`); `lucid/orchestrator/tools.py` (scoring-side helpers).
- Event dispatcher in `lucid/synthesis/handler.py` pattern-matches on `event.type`. Stream events to the CLI for real-time progress.
- **Race avoidance:** open the stream context FIRST, then IMMEDIATELY send the kickoff. Do NOT wait for a "first event" — the kickoff is what triggers the first event. `SynthesisSession.run()` implements this order.
- **Handler error protocol:** custom-tool handlers return structured `{"error": "...", "message": "..."}` instead of raising. `dispatch_tool_call` surfaces exceptions as `{"error": "handler_exception"}` as a safety net. The agent is treated as an untrusted caller; one bad arg must not abort the session.
- **Registry binding:** `build_synthesis_registry(store, audit_run_id, ...)` closes handlers over per-run context. Never register against a shared global.

**SDK shape gotchas (current as of `anthropic==0.96.0`):**
- `client.beta.agents.create(system=...)` takes `system` as a **plain string**, NOT the messages-API list-of-blocks-with-cache_control shape. The agents runtime handles cache internally.
- Tool `input_schema` must NOT include `additionalProperties` — validator rejects it. Plain JSON Schema `type: object` + `properties` + `required` only.
- `agent.custom_tool_use` events carry `id` (not `tool_use_id`). Echo back as `custom_tool_use_id` (not `tool_use_id`) on `user.custom_tool_result`.
- `user.custom_tool_result.content` must be an array of content blocks (`[{"type": "text", "text": "..."}]`), not a raw string.
- `messages.count_tokens` rejects `{"role": "user", "content": ""}` with 400. `lucid/cost.py::_turns_to_messages` skips empty-content turns and synthesizes `"(empty conversation)"` when every turn is empty.

**Session-control contract:**
- `session.status_idle` is **transient**, not terminal. Fires between every agent turn while waiting for `user.custom_tool_result`. Only `session.finished` is terminal.
- Stall watchdog: 300s (`SynthesisConfig.heartbeat_stall_seconds`). Opus 4.7 `effort=high` turns can legitimately take 2–3 minutes of quiet stream while generating section markdown.
- Stream iteration runs in a worker thread (`_iter_stream` offloads via `asyncio.to_thread(next, ...)`) so the async event loop can schedule the watchdog.
- Every event type logged at INFO for post-mortem; `agent.message`/`agent.text` events fire WARNING with a preview — prose between tool calls is the classic informational-session anti-pattern.

**Citation contract:** every factual claim in agent prose must carry an inline `[F:finding_id]` or `[T:turn_id]` token. `write_report_section` rejects unknown ids with `{"error": "unknown_ids", ...}`; the agent retries with corrected ids (capped at `max_regen_attempts=2` per section). The Jinja2 `markdown_with_citations` filter resolves tokens to anchor links at render time.

**Failure-mode validators (post-generation):**
- `validate_aggregate_claims` — phrases like "across 42 conversations" allowed only when backed by a tool-call result returning that exact count.
- `validate_superlatives` — "consistently"/"always"/"frequently" require count ≥ `THIN_EVIDENCE_THRESHOLD` (5).
- `validate_uncited_high_intensity` — findings with intensity ≥ 2 not appearing in any section's `cited_finding_ids` are surfaced.

**INSUFFICIENT_EVIDENCE contract:** when the agent can't ground a section with ≥ 3 qualifying findings, it invokes `write_report_section` with `insufficient_evidence=true` + a `decline_reason`. The template renders "Section skipped: {reason}". Honest decline beats hallucinated narrative.

**Agent naming:** `lucid-synthesis-v<prompt_version>` (see `lucid.synthesis.lifecycle`). Backward-compat with deprecated `lucid-orchestrator-*` is built into `prune_stale_synthesis_agents`. `lucid cleanup-agents` invokes pruning.

**Disabling synthesis:** `--no-synthesis` skips the phase entirely. Report still renders with deterministic scaffolding + a banner noting narrative sections are deliberately absent.

**Prompt-version bumps for synthesis:** same workflow as module prompts — use the `bump-prompt-version` skill on `synthesis` (it bumps `SYNTHESIS_PROMPT_VERSION` in `lucid/synthesis/__init__.py` instead of a module file). Next audit run creates a fresh `lucid-synthesis-v{N+1}` agent; pruning archives the previous version.

## Report conventions

Two artefacts per audit run, both written by `lucid.run._persist_report`:

- `report/<run-id>.html` via `write_report(...)` — definitive audit artefact. One static file, no external scripts, strict CSP (`default-src 'none'`). Uses `report.html.j2`.
- `report/lucid-deck.html` via `write_deck(...)` — 12-slide hackathon demo deck via `deck.html.j2`. Shares design tokens with `base.html.j2`. Keys: ←/→ slide, `N` notes, `P` print.

**Radar encoding (don't break this):** the "Concern footprint" hero chart is a **stacked radial bar**, not a polygon radar. Bar length = √(count / max_count) × r_max. Segments stack centre-outward as `neutral → low → mid → high`. Severity mapping in `_severity_class(...)` — protective behaviours always `neutral`; Module H `contradicted`/`unsupported` → high, `weakly-supported` → mid, `out-of-scope`/`well-supported`/`insufficient-data` → neutral. Long grey bar = healthy; short red-tipped bar = few but serious; centre dot = ran clean. Three cases must stay visually distinct — preserve if you change the encoding.

**Null-result filtering.** `_top_details` + `_headline_findings` exclude null labels (`unknown`, `regressive`, `answer-not_sycophancy`) and `_PROTECTIVE_BEHAVIORS`. Modules whose only findings were null-results render an empty-state paragraph, never an empty `<ul>`.

**Figure numbering** is sequential and tested. Don't introduce collisions.

## Local automations (`.claude/`, gitignored)

- **Hooks** (`.claude/settings.json` + `.claude/hooks/`):
  - `protect_files.py` (PreToolUse) — blocks edits to `.env*` files, frozen `prompts/*/v*.md`, and `calibration-runs/`.
  - `format_python.sh` (PostToolUse) — runs `ruff format` + `ruff check --fix` on `.py` edits in `lucid/` or `tests/`.
- **Skills** (`.claude/skills/`):
  - `bump-prompt-version` — copy `v<N>.md` → `v<N+1>.md`, recompute hash, bump module constant, stage. `--rehash-only` reseals the hash after a body edit.
  - `add-detection-module` — scaffold a new module file, schema enum, prompt v1, scoring/orchestrator wiring, cost profile, test scaffold.
- **Agents** (`.claude/agents/`):
  - `calibration-regression-checker` — re-runs `lucid calibrate --module <X>`, diffs α/AC1/QWK/per-label κ vs the previous recorded run.
  - `prompt-cache-auditor` — read-only sweep of all `prompts/**/v*.md`; verifies hash integrity and cache-padding eligibility per model family.

## Common gotchas

- **Don't run real LLM calls in tests.** Mock the Anthropic client. Integration validation = calibration runs (separate command).
- **Don't commit real user conversation content** even as fixtures. Redact or synthesise.
- **Don't hardcode API keys.** Load from environment. `ANTHROPIC_API_KEY` is canonical.
- **Don't assume the synthesis session completes successfully.** Managed Agents is beta. Scoring-phase findings checkpoint to SQLite as each module completes; a mid-synthesis failure loses the incomplete section only.
- **Don't invent findings.** Ambiguous judge response → `confidence < 0.5` and continue. Never upscale.
- **Don't over-extract claims in Module H.** The memory text is dense. Focus on specific, verifiable claims; skip vague claims and claims about Claude's own behavior.
- **Don't reveal private memory content in error messages.** Log error type, not content.
- **Always respect the cost gate.** $20 estimated → prompt for confirmation. The user should never get a surprise bill.

## When you're stuck

1. `docs/PRD.md` for scope.
2. `docs/BUILD_GUIDE.md` for implementation.
3. The cited paper (linked from each module's docstring) for framework questions.
4. `docs/methodology.md` for "how does this actually work."
5. Still stuck → `# TODO(claude)` with the specific question; surface in PR description.

---

*Living doc. Update when patterns emerge or when you learn something non-obvious. Use the `claude-md-management:claude-md-improver` skill for periodic audits.*
