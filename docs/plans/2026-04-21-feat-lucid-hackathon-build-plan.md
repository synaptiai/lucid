---
title: Lucid Hackathon Build (Days 1–6) — v3 locked
type: feat
status: active
date: 2026-04-21
amended: 2026-04-21 (stack locked, deps pinned, PDF-verified rules, plugin/MCP dropped)
origin: LUCID_PRD.md (v3) + LUCID_BUILD_GUIDE.md (v2) + LUCID_PODCAST_STRETCH.md
decision_doc: docs/plans/2026-04-21-decision-stack-and-distribution.md
submission_deadline: 2026-04-26T20:00:00-04:00
python: "3.13"
---

# Lucid Hackathon Build Implementation Plan (v3 locked)

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Stay within the phase you are executing; do not jump ahead. Commit after each green test. Authoritative docs: `CLAUDE.md` (conventions), `docs/PRD.md` (scope), `docs/BUILD_GUIDE.md` (schemas + prompts), `docs/plans/2026-04-21-decision-stack-and-distribution.md` (why the stack is what it is).

**Goal:** Ship an open-source (MIT) Python 3.13 CLI that ingests Claude Code sessions + Claude.ai export, runs 8 detection modules via a Managed Agents orchestrator, and produces an HTML report. Deadline 2026-04-26 20:00 EST.

**Architecture:** Python 3.13 CLI → Pydantic-typed ingest adapters for Claude Code JSONL and Claude.ai export → SQLite corpus store → Managed Agents orchestrator (Sonnet 4.6, effort=low) invoking eight detection modules (Opus 4.7, per-module thinking/effort matrix) via **custom tools (client-executed)** — no MCP tunnel, no HTTPS endpoint — with prompt caching on stable module prefixes → deterministic attribution pass → Jinja2 + vendored Chart.js HTML report (autoescape + CSP). Calibration target: Krippendorff's α ≥ 0.67 and Gwet's AC1 ≥ 0.70 (both reported) on 200+ hand-labeled turns.

**Tech Stack (locked 2026-04-21):**

```
Python 3.13.x (target 3.13 — voyageai + irrCAC gate 3.14)
uv 0.11.x  (package manager, tool runner)

Runtime:
  typer==0.24.1            pydantic==2.13.3         anthropic==0.96.0
  voyageai==0.3.7          orjson==3.11.8           ijson==3.5.0
  aiosqlite==0.22.1        jinja2==3.1.6            rich==15.0.0
  tenacity==9.1.4          numpy==2.4.4             scipy==1.17.1
  krippendorff==0.8.2      irrCAC==0.4.4            filelock==3.29.0

Dev:
  pytest==9.0.3            pytest-asyncio==1.3.0    mypy==1.20.2
  ruff==0.15.11            pyinstaller==6.19.0

Build (fallback if PyInstaller fights scipy imports): Nuitka 4.0.8.
No mcp package. No plugin manifest. No Synapti submission. All deferred post-hackathon.
```

---

## Judging criteria (what we're building toward)

| Criterion | Weight | What in the plan services this |
|---|---:|---|
| **Impact** | 30% | Open-source MIT; runs locally on user's own corpus; novel problem (memory-corpus audit) per Module H. |
| **Demo** | 25% | Three visible beats: Module A behavioral profile chart, Module B paired-sycophancy quote, Module H memory-corpus table with Beta-CI whiskers. |
| **Opus 4.7 use** | 25% | Managed Agents orchestrator with custom tools; per-module thinking/effort matrix (Phase 7 table); prompt caching on stable rubrics (85–90% input-cost cut, verified via `cache_read_input_tokens`). |
| **Depth & execution** | 20% | 200+ hand-labeled turns; Krippendorff α + Gwet AC1 + per-label κ + QWK with bootstrap BCa CIs; security hardening (XSS, CSP, path traversal, canary logs); idempotent SQLite store; two-stage Module H verification. |

---

## Locked decisions (see decision doc for analysis)

| Decision | Locked value |
|---|---|
| Stack | Approach A: Python 3.13 + PyInstaller binary + `uv` |
| Distribution | CLI only. Binary via GitHub Releases + Homebrew tap + PyPI for `uvx`. No plugin, no marketplace. |
| Internal orchestrator | Managed Agents session + custom tools (client-executed via `agent.custom_tool_use` / `user.custom_tool_result`). No MCP server. No HTTPS tunnel. |
| Embedding provider (Module H) | Voyage AI `voyage-3-large` (OpenAI `text-embedding-3-small` fallback) |
| Calibration sample size | 200+ hand-labeled turns. 8–12h hand-label budget Day 1 evening → Day 2 morning. |
| Primary IAA metric | Decide end of Day 2 AM (once label prevalence is visible). Report both α and AC1 regardless. |
| Module scope (visible) | A, B-feedback, H (demo beats) |
| Module scope (background) | C, E, F, G |
| Module scope (cut/opt-in) | D (`--include-module-d` flag only) |
| Demo framing | Problem Statement 2 — "first-draft interface for auditing your thinking-with-AI" |

---

## Origin & scope

Operationalizes two authoritative documents:

- `LUCID_PRD.md` (v3) → `docs/PRD.md` after Phase 1 move. Scope, success criteria, risks R1–R11.
- `LUCID_BUILD_GUIDE.md` (v2) → `docs/BUILD_GUIDE.md`. Schemas (modernize on move), confirmed source file formats, module prompt templates.
- `LUCID_PODCAST_STRETCH.md` → `docs/podcast_stretch.md`. Stretch-only ElevenLabs TTS. Cut if Module H precision or demo polish slip.

**Meta-task**: move these three files from repo root to `docs/` during Phase 1 to unblock `CLAUDE.md` cross-references.

---

## Technical approach

### Architecture

```
     ┌─────────────────────────┐
     │  lucid/cli.py (Typer)   │  audit | calibrate
     └────────────┬────────────┘
                  │
         ┌────────▼─────────┐
         │  config + paths  │  SecretStr keys, ANTHROPIC_LOG unset
         └────────┬─────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼────┐   ┌────▼────┐   ┌────▼────┐
│ ingest │   │sampling │   │  store  │
│ cc/cai │   │(seeded) │   │ sqlite  │
│ PPE +  │   └────┬────┘   │ WAL +   │
│ orjson │        │        │ CHECK   │
│ ijson  │        │        └────┬────┘
└───┬────┘        │             │
    │             │             │
    └─────────┬───┴─────────────┘
              │
     ┌────────▼─────────────────┐
     │   orchestrator           │
     │  tools.py (async fns)    │◄── single source of truth
     │  managed_agent.py        │    for custom-tool handlers
     │  (session + events loop) │
     └────────┬─────────────────┘
              │
    ┌─────────┼──────────────────────────┐
    │         │          │               │
┌───▼──┐ ┌───▼──┐  ┌────▼────┐     ┌────▼────┐
│Mod A │ │Mod B │  │ Mod C/F │ ... │ Mod H   │
│ SB   │ │Sharma│  │ D(opt)/E│     │ Memory  │
│cache │ │feedback      │         │ Voyage+ │
│prefix│ │ +answer      │         │ numpy   │
└──┬───┘ └──┬───┘  └────┬────┘     └────┬────┘
   │        │           │               │
   └────────┴───────────┴───────────────┘
                  │
         ┌────────▼─────────┐
         │  Findings (DB)   │  UNIQUE(run,mod,conv,turns,behav)
         └────────┬─────────┘
                  │
         ┌────────▼──────────┐
         │ Mod G Attribution │  deterministic
         └────────┬──────────┘
                  │
         ┌────────▼────────┐
         │ report/generator │  Jinja2 autoescape + CSP
         │ vendored Chart   │  aggregated + <details>
         └─────────────────┘
```

### Data flow per audit run

1. CLI parses flags, resolves source path(s) with `Path.resolve(strict=True)`, rejects symlinks outside root, applies size caps, creates `AuditRun` row in a single transaction.
2. Ingest adapters parse via `ProcessPoolExecutor` + orjson (Claude Code) or `ijson` streaming (Claude.ai) → Pydantic validate → sync `sqlite3.executemany` batch-insert (WAL, 1000 rows/batch, 5000 rows/tx).
3. Sampling applies stratification + recency weight + seeded RNG.
4. Cost estimator: `client.messages.count_tokens()` pre-pass × per-module output budgets → display → $20 gate.
5. Orchestrator starts Managed Agents session. Corpus NOT uploaded — orchestrator queries via custom tools back to the local process. Prompt caching on module system prompts (5m TTL for per-conversation context, 1h for orchestrator prompt).
6. Modules run per execution plan (A → B → C → [D opt-in] → E → F → H → G). Each module uses `messages.parse(output_format=<PydanticSchema>)`. Per-module thinking/effort tuning.
7. Findings batched per module-run; `module_progress` table updated transactionally. Idempotent UNIQUE on `(audit_run_id, module, conversation_id, turn_ids_hash, behavior)`.
8. Module G reads findings, buckets deterministically.
9. Report generator renders `report/<run_id>.html` → opens in browser.

### Custom-tools pattern (replaces MCP server for internal orchestrator)

Per [Tools — Custom tools](https://platform.claude.com/docs/en/managed-agents/tools): the cloud agent emits `agent.custom_tool_use` events; our local process handles them via async handlers and replies with `user.custom_tool_result`. No MCP server, no HTTPS endpoint, no ngrok.

Tool handlers in `lucid/orchestrator/tools.py`:
- `query_corpus(filter) -> list[Conversation]`
- `get_conversation(id) -> Conversation`
- `get_turn_window(conversation_id, start, end) -> list[Turn]`
- `invoke_module(module, conversation_ids, prompt_version, model) -> list[Finding]`
- `store_finding(finding: Finding) -> str`
- `get_findings(filter) -> list[Finding]`
- `log_progress(message, level) -> None`
- `estimate_remaining_cost() -> float`

Same Python functions are called by the orchestrator event loop. Direct, testable, no transport layer.

---

## Implementation phases

Day counter: Day 1 = Tue 2026-04-21 (today). Deadline = Sun 2026-04-26 20:00 EST.

---

### Phase 0 — Technical pre-flight (Day 1, before coding)

**Tasks:**

1. Verify Managed Agents beta header still `managed-agents-2026-04-01`.
2. Record Opus 4.7 + Sonnet 4.6 pricing (input / output / cache-write / cache-read per 1M tokens) in `docs/methodology.md`.
3. Verify `ANTHROPIC_DEFAULT_MODEL_TIMELINE` (from BUILD_GUIDE §5) against Anthropic news page.
4. Clone [github.com/sam-paech/spiral-bench](https://github.com/sam-paech/spiral-bench); check for labeled data. If not available → plan Day 1 PM hand-labeling of 200+ turns from SpiralBench public conversations + held-out Claude.ai slice.
5. Create Voyage AI account, generate API key (200M tokens free tier), store in `.env.local`.
6. Verify Anthropic prompt-caching minimums: 4096 tokens for Opus 4.7, 2048 for Sonnet 4.6.
7. Verify `client.messages.count_tokens()` is free and rate-limit-exempt.

**Exit criterion:** `docs/methodology.md` contains pricing table, Managed Agents header status, timeline, SpiralBench data plan, cache minimums. `VOYAGE_API_KEY` and `ANTHROPIC_API_KEY` in `.env.local`.

---

### Phase 1 — Repo bootstrap + docs move + tooling (Day 1, 12:30–14:00)

**Goal:** Fresh MIT-licensed codebase with `uv`, Typer, pytest, ruff, mypy wired. Docs paths corrected.

**Files:**

- Create: `pyproject.toml` (locked versions below), `lucid/__init__.py`, `lucid/cli.py` (stub), `lucid/config.py`, `lucid/logging.py`, `tests/conftest.py`, `.python-version` (`3.13`).
- Move: `LUCID_PRD.md` → `docs/PRD.md`, `LUCID_BUILD_GUIDE.md` → `docs/BUILD_GUIDE.md`, `LUCID_PODCAST_STRETCH.md` → `docs/podcast_stretch.md`.
- Create: `docs/methodology.md` (seeded from Phase 0), `docs/calibration.md` (filled Day 2-4), `docs/privacy.md` (filled Phase 9).
- Update `CLAUDE.md` cross-references (paths now resolve).
- Write `README.md` skeleton (install, quickstart, methodology link, MIT badge).
- Write `.gitignore`: Python standard + `.env*` + `*.sqlite3` + `*.sqlite3-wal` + `*.sqlite3-shm` + `*.sqlite3-journal` + `report/*.html` + `.lucid.lock` + `dist/` + `build/`.

**`pyproject.toml` (locked):**

```toml
[project]
name = "lucid"
version = "0.1.0"
requires-python = ">=3.13,<3.14"
description = "First-draft interface for auditing your thinking-with-AI."
license = {text = "MIT"}
dependencies = [
    "typer==0.24.1",
    "pydantic==2.13.3",
    "anthropic==0.96.0",
    "voyageai==0.3.7",
    "orjson==3.11.8",
    "ijson==3.5.0",
    "aiosqlite==0.22.1",
    "jinja2==3.1.6",
    "rich==15.0.0",
    "tenacity==9.1.4",
    "numpy==2.4.4",
    "scipy==1.17.1",
    "krippendorff==0.8.2",
    "irrCAC==0.4.4",
    "filelock==3.29.0",
]

[project.optional-dependencies]
dev = [
    "pytest==9.0.3",
    "pytest-asyncio==1.3.0",
    "mypy==1.20.2",
    "ruff==0.15.11",
    "pyinstaller==6.19.0",
]

[project.scripts]
lucid = "lucid.cli:app"

[tool.ruff]
line-length = 100
target-version = "py313"

[tool.mypy]
strict = true
python_version = "3.13"

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

**Steps:**

1. `uv python install 3.13`
2. `uv sync --extra dev`
3. Verify `uv run lucid --help` returns (empty Typer app)
4. `lucid/config.py`: `ANTHROPIC_API_KEY` + `VOYAGE_API_KEY` as `pydantic.SecretStr`; `os.environ.pop("ANTHROPIC_LOG", None)` on startup; `rich.traceback.install(show_locals=False)`.
5. Commit: `feat: bootstrap lucid package (py3.13, uv, locked deps)`.

**Exit criterion:** `uv run pytest` runs (0 tests), `uv run mypy lucid/ --strict` clean, `uv run ruff check lucid/` clean, `docs/PRD.md` and `docs/BUILD_GUIDE.md` resolve for CLAUDE.md.

---

### Phase 2 — Data schemas + SQLite store (Day 1, 14:00–16:00)

**Goal:** Modernized Pydantic v2 schemas + idempotent SQLite schema with CHECK constraints.

**Key amendments to BUILD_GUIDE §2 schemas:**

- `Optional[X]` → `X | None` everywhere (Python 3.13 idiom; mypy-strict compatible).
- `AuditRun.status: Literal["running","completed","failed","partial","aborted_pre_spend"]` (not `str`).
- `AuditRun.corpus_stats` and `token_usage`: nested `BaseModel`s, not `dict`.
- `AuditRun` gains: `corpus_fingerprint: str`, `prompt_versions: dict[ModuleName, str]`, `schema_version: int`, `skipped_modules: list[ModuleName]`.
- `Finding.prompt_version: str` is a **first-class column** (required). Also `Finding.prompt_hash: str`.
- `Finding.confidence_alpha: float | None` and `confidence_beta: float | None` for Beta-distribution posteriors (visible in Depth-scoring report footer).
- `ContentBlock` as discriminated union via `Annotated[TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock, Field(discriminator="type")]`.
- `Finding.intensity: int | None` (Module H doesn't use 1–3 intensity scale).

**Files:**

- `lucid/schemas.py` (modernized, discriminated unions, nested BaseModels).
- `lucid/store/__init__.py`, `lucid/store/sqlite.py` — `CorpusStore` wrapping `aiosqlite` for orchestrator reads; sync `sqlite3.executemany` helper for bulk ingest.
- `lucid/store/schema.sql` — full DDL with CHECK constraints, FK CASCADE, UNIQUE idempotency key, `module_progress`, `embeddings` tables. **Full schema in decision doc §G.**
- `lucid/store/init.py` — `initialize_db(path)` applies `schema.sql` if `PRAGMA user_version == 0`, bumps to 1.
- `tests/test_schemas.py`, `tests/test_store.py`.

**Steps (TDD):**

1. Write failing test for `Conversation` round-trip.
2. Implement `schemas.py`; green.
3. Add round-trip tests for every model.
4. Write `schema.sql`, implement `initialize_db`, write insert/select test.
5. Add `test_findings_uniqueness` — duplicate insert raises `IntegrityError`.
6. Commit.

**Exit criterion:** `mypy --strict` clean. No `Optional[X]`. No bare `dict` annotations. Schema CHECK constraints enforce the provenance invariants from CLAUDE.md.

---

### Phase 3 — Ingest adapters + security guards (Day 1, 16:00–19:00)

**Goal:** Both corpora parse at scale (<60s on Daniel's 9,840 Claude Code files) with fixture-based tests + explicit security guards.

**Files:**

- `lucid/ingest/base.py` — `IngestAdapter` ABC: `discover`, `parse_one`, `parse_all`, `fingerprint`.
- `lucid/ingest/claude_code.py` — `ProcessPoolExecutor(max_workers=os.cpu_count())` + orjson for JSONL parse.
- `lucid/ingest/claude_ai.py` — `ijson.items(f, 'conversations.item')` streaming; branch reconstruction via `parent_message_uuid`.
- `lucid/ingest/memories.py` — extracts `conversations_memory` + `project_memories`. Redacts any logging.
- `lucid/logging.py` — `SafeFormatter` drops records with `extra={"contains_user_content": True}`.
- `tests/fixtures/claude_code/sample_session.jsonl` — synthetic, all block types.
- `tests/fixtures/claude_ai/` — synthetic minimal 4-file export.
- `tests/fixtures/malicious/` — symlink-out, oversize, deep-nested.
- `tests/test_ingest_contract.py` — contract tests per adapter.

**Security constants (from deepening §D):**

- `MAX_TURNS_PER_CONVERSATION = 10_000`
- `MAX_BLOCKS_PER_TURN = 1000`
- `MAX_TEXT_LEN_PER_BLOCK = 10_000_000` (10 MB)
- `MAX_SESSION_FILE_SIZE = 50 * 1024 * 1024` (50 MB)
- `MAX_EXPORT_FILE_SIZE = 500 * 1024 * 1024` (500 MB)
- `IJSON_RECURSION_CAP = 32`

**Path safety:** `Path(user_path).expanduser().resolve(strict=True)`. Reject if not directory, if any discovered file is a symlink outside resolved root.

**Fingerprint contract:** `fingerprint(path) = sha256(sorted('{conv_id}:{content_hash}'))`. Stable across discovers, changes on add, stable on internal content edit.

**Tests:**

1. Parse single session with all 4 block types — round-trip through schemas.
2. Branch reconstruction: `parent_message_uuid` tree on claude_ai fixture.
3. MCP metadata preservation.
4. `memories.json` round-trip: `conversations_memory` + `project_memories` dict.
5. Unknown-field preserved in `metadata` (+ warning log).
6. **Security:** symlink-out rejected. Oversize rejected. Deep-nested doesn't OOM.
7. **Canary log leak:** sentinel `LUCID_CANARY_MEMORY` in test memory, `--log-level DEBUG` captured, assert sentinel absent.

**Exit criterion:** Daniel's real `~/.claude/projects` (9,840 files) parses in <60s. All security tests green.

---

### Phase 4 — Sampling + CLI + dry-run + flow gaps (Day 1, 19:00–22:00)

**Goal:** `lucid audit --dry-run` end-to-end: discover → sample → estimate via `count_tokens` → consent → exit. No LLM spend.

**Files:**

- `lucid/sampling.py` — `SamplingConfig` + `sample_conversations` with stratification + recency weight + seeded RNG.
- `lucid/cli.py` — Typer app with `audit`, `calibrate` subcommands. App-level callback for `--log-level` / `--config`. Typed `Annotated[T, typer.Option(...)]` with `Source`/`ModuleName` enums + `Path(exists=True, dir_okay=True)`.
- `lucid/cost.py` — `estimate_cost()` via `count_tokens` + per-module output budgets from PRD §6.
- `tests/test_sampling.py`, `tests/test_cli.py`, `tests/test_cost.py`.

**CLI surface (locked):**

```
lucid audit --source {claude-code,claude-ai,all} --path <p>
           [--sample N | --sample all | --projects p1,p2,...]
           [--dry-run] [--resume <run-id>]
           [--include-module-d]
           [--yes-i-authorize-spend-up-to N] [--log-level LEVEL]

lucid calibrate --module {a,h} [--prompt-version VN]
```

**Cost-gate bypass:** `--yes-i-authorize-spend-up-to N` + `LUCID_ALLOW_UNATTENDED=1` env var (both required).

**Flow-gap handling (added to acceptance criteria):**

- 0 conversations → exit 2 with path hint.
- N < requested sample → warn + clamp.
- 1 conversation → skip E/D with `insufficient-data` placeholder.
- `lucid` bare → quick-start + exit 0.
- Concurrent audit on same DB → `filelock` → exit 2 with unlock hint.
- Cost-gate rejection → parsed corpus reusable (ingest always persists before gate).

**Day 1 exit criterion:**

- `uv run lucid audit --source claude-code --path ~/.claude/projects --sample 10 --dry-run` parses Daniel's real corpus, samples 10, prints `count_tokens`-based cost estimate, exits.
- `uv run lucid audit --source claude-ai --path ./export --dry-run` same.
- Repo pushed to GitHub, public, MIT-licensed, README populated.
- All tests green, `mypy --strict` clean.

---

### Phase 5 — Managed Agents thin slice via custom tools (Day 1 evening / Day 2 early AM)

**Goal:** Prove the Managed Agents pipe via custom tools. De-risks R1 before Module A calibration.

**Architectural change vs earlier draft:** NO MCP server, NO HTTPS tunnel. Use Managed Agents [custom tools](https://platform.claude.com/docs/en/managed-agents/tools) — `type: "custom"` in `agent.tools`. Our local process handles `agent.custom_tool_use` events via async handlers and replies with `user.custom_tool_result`.

**Files:**

- `lucid/orchestrator/tools.py` — 8 async tool handlers listed in "Custom-tools pattern" section above. Plain async Python. Unit-testable without any transport.
- `lucid/orchestrator/managed_agent.py` — agent.create + environment.create (no `resources` mount by default) + session.create + event stream loop with heartbeat.
- `lucid/orchestrator/system_prompt.py` — trimmed routing prompt, `cache_control` for 1h TTL.
- `lucid/orchestrator/handler.py` — event dispatcher: `match event.type: case "agent.custom_tool_use": ...`.

**Thin-slice sequence:**

1. `client.beta.agents.create(name="lucid-orchestrator", model="claude-sonnet-4-6", system_prompt=..., tools=[{"type": "custom", "name": "log_progress", ...}, ...], extra_headers={"anthropic-beta": "managed-agents-2026-04-01"})`
2. `client.beta.environments.create(config={"type": "cloud", "networking": {"type": "unrestricted"}})` (no mounted files).
3. `client.beta.sessions.create(agent_id=..., environment_id=...)`
4. `client.beta.sessions.events.stream(session.id)` — OPEN FIRST, then send kickoff message (avoid race).
5. Orchestrator emits `agent.custom_tool_use{name="query_corpus", args={...}}` → local process computes, replies with `user.custom_tool_result{tool_use_id=..., content=...}`.
6. Verify `response.usage.cache_read_input_tokens > 0` on second call.
7. Heartbeat: no events for >60s → mark `partial` + save checkpoint.

**Fallback:** if Managed Agents blocks at any step, the exact same `lucid/orchestrator/tools.py` functions can be wrapped as Claude Agent SDK tools (`claude_agent_sdk ≥ 0.2.111`) by flipping `LUCID_ORCHESTRATOR_BACKEND=sdk` in env. No code duplication.

**Exit criterion:** `uv run lucid audit --source claude-code --path <p> --sample 5 --yes-i-authorize-spend-up-to 1` produces one dummy finding via Managed Agents + custom tools. Cache hit verified. Heartbeat/checkpoint verified by killing session mid-run.

---

### Phase 6 — Calibration harness + Module A (Day 2, full day)

**Goal:** Module A ships with Krippendorff's α ≥ 0.67 AND Gwet's AC1 ≥ 0.70 (both reported) + per-label Cohen's κ + QWK intensity. 200+ labeled turns with bootstrap BCa CIs.

**Labeling plan:**

- Day 1 evening (while Phase 5 runs): hand-label ~100 turns from a held-out slice of Daniel's Claude.ai export. 2–3 minutes per turn × 100 = 3–5 hours.
- Day 2 AM 6:00–9:00: label additional ~100 turns. Total 200+.
- If SpiralBench's labeled data turns out to be publicly downloadable (Phase 0 outcome) → use directly; the hand-labeled set becomes the held-out validation split.

**Dual-metric approach (D1 resolution by-data):**

At labeling completion, compute label-prevalence distribution. If ≥ 3 labels have prevalence < 10%:
- **Primary: Gwet's AC1** (paradox-robust on skewed data)
- **Secondary: Krippendorff's α** (for methodological parity with potential SpiralBench comparison)

Otherwise:
- **Primary: Krippendorff's α**
- **Secondary: Gwet's AC1**

Always report both. Per-label binary Cohen's κ always reported. Intensity: per-label quadratic-weighted κ.

**Files:**

- `prompts/module_a/v1.md` — YAML frontmatter: `version: 1`, `model: claude-opus-4-7`, `thinking_mode: disabled`, `effort: low`, `citation: SpiralBench`, `hash: <sha256>`. Rubric padded ≥ 4096 tokens (Opus cache minimum).
- `lucid/modules/base.py` — `CorpusModule` / `FindingsModule` protocols; `ModuleError` dataclass; `ModuleResult = Finding | ModuleError` tagged union; `run_with_bounded_concurrency` helper.
- `lucid/modules/module_g_attribution.py` — deterministic skeleton, implemented first (serves as pattern).
- `lucid/modules/module_a_spiralbench.py` — 10-turn chunking, `messages.parse(output_format=SpiralBenchScore)`, `cache_control={"type":"ephemeral"}` on system prompt.
- `lucid/calibration/data.py` — load SpiralBench labels (if available) + hand-labeled set; 30/70 held-out split.
- `lucid/calibration/validate.py` — Krippendorff's α via `krippendorff.alpha(...)`, Gwet's AC1 via `irrCAC.raw.CAC(...).gwet(...)`, per-label Cohen's κ via `sklearn.metrics.cohen_kappa_score`, QWK, bootstrap BCa via `scipy.stats.bootstrap(method='BCa')`.
- `tests/conftest.py` — `mock_anthropic_client` fixture using real `anthropic.types.Message` shapes.
- `tests/test_prompt_injection.py` — fixture with `"ignore previous instructions..."` in turn content, assert classifier ignores.

**Cache-stability audit:**

- No `datetime.now()` in system prompts.
- `json.dumps(..., sort_keys=True)` for serialized context.
- Deterministic tool order.
- First fan-out call awaits `message_start` before launching N-1 parallel calls.

**Concurrent LLM calls:** `asyncio.Semaphore(10)` + tenacity `wait_random_exponential(multiplier=1, max=60)` on 429.

**Decision tree (Day 2 evening):**

- α ≥ 0.67 AND AC1 ≥ 0.70 AND lower-CI(α) ≥ 0.60 → ship v1.
- One metric passes, other doesn't → iterate v2, re-validate on 70%. Re-validate on 30% held-out only after.
- Both < 0.55 → descope to 5–7 high-prevalence behaviors + rerun.

**Exit criterion:** `docs/calibration.md` shows: α, AC1, per-label κ, QWK, CI bounds, shipped prompt version. Cache hit rate > 50% on second+ calls.

---

### Phase 6A — Scaffolding (shipped 2026-04-21)

Files shipped: `lucid/modules/base.py`, `lucid/modules/module_a_spiralbench.py` (mocked), `lucid/modules/module_g_attribution.py` (deterministic, fully implemented), `lucid/calibration/{data,validate,report}.py`, `lucid/prompts.py`, `prompts/module_a/v1.md` (Spiral-Bench v1.2 rubric, 17 behaviors), `tests/conftest.py` mock_anthropic_client, `tests/test_prompt_injection.py`, `lucid calibrate --module a` CLI wired against pre-computed label JSONL pairs. 89 new tests, 224/224 pass, mypy --strict clean.

Not done in 6A (by design): live LLM run, real calibration numbers, the `docs/calibration.md` artifact.

---

### Phase 6B — Calibration methodology pivot + live run (amended 2026-04-22)

**Why an amendment:** plan v3 assumed ≥ 200 hand-labeled turns at "~3 min per turn" = 10 hours human work. That math conflated per-turn with per-cell — the real workload is 17 behaviors × 3 intensities × 200 turns = ~3,400 per-cell decisions, closer to 10+ hours even with pre-populated labels. Not feasible for a solo-dev hackathon.

**Revised approach** (full details in `docs/methodology.md §10`):

Use **multi-rater cross-judge IAA** as the primary calibration target, with a small synthetic gold corpus for rare-behavior coverage and a ~45-minute human audit on the highest-information disagreements.

**Seven raters:**
- Module A (Opus 4.7)
- SpiralBench's 3 judges already recorded in `res_v1.2/*.json`: Claude Sonnet 4.5, GPT-5, Kimi K2
- 3 Ollama-backed judges: Kimi K2.6, Gemma 4 31B, GLM 5.1 (near-zero API cost)

**Two corpora:**
- SpiralBench: 3 target models × 30 conversations = ~1,660 assistant-turn chunks. LLM-to-LLM IAA only.
- Synthetic gold: 60 hand-curated turns, labels by construction, ~15 human-verified.

**Cost gate bumped to $50 for calibration** (standard audit gate stays $20). Both chunk sizes (10 and 2) run on all 3 target models → ~$47 Opus spend.

**Human role** (new minimum):
1. Run the calibration command (one invocation, ~30 min wall clock including Ollama rate-limits).
2. Review `calibration-runs/<ts>/disagreements.jsonl` — top 50 cross-judge disagreements ranked by entropy × rare-behavior weight. ~45 min at ~1 min/item.
3. Re-run with `--import-verified` to apply human overrides. Final numbers land in `docs/calibration.md`.

Total: ~1.5 hours human time. Not 10 hours.

**Files shipped in 6B code-side:**

- `lucid/calibration/spiralbench.py` — fetch + parse `res_v1.2/*.json` → Conversation + Turn + LabeledTurn.
- `lucid/calibration/synthetic.py` + `lucid/calibration/corpus/synthetic_v1.jsonl` — 60 engineered turns, sidecar gold labels.
- `lucid/calibration/judges/` — Judge protocol, ModuleA wrapper, SpiralBench-file reader, synthetic-gold reader, Ollama backend.
- `lucid/calibration/audit.py` — disagreement export + human-verified import.
- `lucid calibrate` CLI extended with `ingest-spiralbench`, `run`, `--export-disagreements`, `--import-verified` flows.

**Dependency addition:** `ollama>=0.4.0` moved from optional to required (judge is load-bearing for the plan).

**Exit criterion (unchanged from Phase 6 original):** `docs/calibration.md` shows α, AC1, per-label κ, QWK, CI bounds, shipped prompt version. Cache hit rate > 50%. Plus: the report explicitly labels which cells are human-verified vs. LLM-only (provenance transparency).

---

### Phase 7 — Modules B, C, [D opt-in], E, F (Day 3, full day)

**Goal:** Five remaining LLM-backed modules integrated into orchestrator.

**Scope decisions:**

- **Module B: 2 of 4 subroutines.** Ship feedback + answer only. Mimicry and "are-you-sure" stubbed (architected, not implemented).
- **Module D: opt-in** via `--include-module-d` flag. Skeleton + prompt shipped; not in default pipeline.
- **Module E: budget 2× other modules** (three sub-prompts + stateful per-topic timeline).
- **Module F: 3-stage filter** (heuristic → Sonnet 4.6 triage → Opus 4.7 classification). Cost: $10 → $1.20 per audit, 20 min → 3 min wall clock.

**Per-module thinking/effort matrix:**

| Module | Thinking | Effort | Model |
|---|---|---|---|
| A | disabled | low | Opus 4.7 |
| B second-pass | adaptive + display=omitted | high | Opus 4.7 |
| C | disabled | low | Opus 4.7 |
| D (opt-in) | adaptive + display=summarized | xhigh | Opus 4.7 |
| E topic-extraction | disabled | low | Sonnet 4.6 |
| E position-tracking | adaptive | high | Opus 4.7 |
| F triage | disabled | low | Sonnet 4.6 |
| F classification | adaptive | high | Opus 4.7 |
| G | n/a | n/a | deterministic |
| Orchestrator | disabled | low | Sonnet 4.6 |

**Files:**

- `prompts/module_b/feedback_v1.md`, `prompts/module_b/answer_v1.md` (ship). `mimicry_v0.md`, `are_you_sure_v0.md` (stubs).
- `prompts/module_c/v1.md`.
- `prompts/module_d/v1.md` (opt-in).
- `prompts/module_e/topics_v1.md`, `positions_v1.md`, `drift_v1.md`.
- `prompts/module_f/heuristic_v1.py` (pure Python filter), `triage_v1.md` (Sonnet), `classify_v1.md` (Opus).
- `lucid/modules/module_b_sharma.py`, `module_c_syceval.py`, `module_d_perspective.py`, `module_e_beliefshift.py`, `module_f_itp.py`.

**Exit criterion:** Modules A, B (2 subroutines), C, E, F produce findings on fixture corpus. Module D skeleton present with opt-in flag. All tests green (mocked LLM).

---

### Phase 8 — Module H (Day 4 AM)

**Goal:** Keep Thinking Prize bid. Memory-corpus consistency with Voyage embeddings + two-stage verification.

**Pipeline:**

1. **Pre-index corpus once.** Chunk by turn-pair (user-turn + adjacent assistant-turn). Metadata: `{conversation_id, timestamp, project_uuid, model}`. Batch-embed via Voyage (~80 API calls @ 128/batch = ~40s, ~$0.20 for 10K chunks).
2. **Store vectors as SQLite BLOB** (persistent across resume). Load into numpy matrix at audit start (~40MB for 10K × 1024-dim float32).
3. **Extract atomic claims** from memories via Opus 4.7 call per memory chunk.
4. **Per-claim retrieval:** embed claim → numpy cosine dot-product → top-k=20 excerpts.
5. **Auto-tag `insufficient-data`** if top-1 similarity < 0.35 (threshold validated empirically on 30-50 hand-labeled pairs Day 4 AM).
6. **Verifier LLM (Opus 4.7, adaptive, effort=high):** classify well-supported / weakly-supported / unsupported / contradicted / insufficient-data with evidence quotes.
7. **Two-stage verification** on ambiguous: if verifier returns `insufficient` AND max-sim > 0.35 → decompose claim via Sonnet, retrieve for each sub-claim, re-verify.

**Files:**

- `prompts/module_h/extract_v1.md`, `classify_v1.md`, `refine_v1.md`.
- `lucid/modules/module_h_memory.py`.
- `lucid/modules/embeddings.py` — Voyage wrapper (batch API, tenacity retry). OpenAI fallback if Voyage fails.
- `lucid/calibration/module_h_seeded.py` — seeded corpus with 5+ known-truth claims.
- `tests/fixtures/module_h_seeded/` — synthetic memory + corpus pair.

**Validation (Day 4 target):** precision ≥ 0.8 on `unsupported` + `contradicted` classifications against seeded corpus.

**Exit criterion:** Real `memories.json` produces ranked support table. Single-call pilot vs split pipeline decision recorded. Module integrated: runs after A–F, before G.

---

### Phase 9 — Module G + Report generator (Day 4 PM)

**Goal:** HTML report with three demo beats, security-hardened, rendering Beta-CI whiskers.

**Hard security requirements (from deepening §D):**

- Jinja2 `Environment(autoescape=select_autoescape(["html", "j2"]))`.
- CSP meta tag: `<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'">`.
- Vendored `chart.umd.min.js` (~80KB). No CDN.
- Test: inject `<script>alert(1)</script>` into `Finding.explanation` + `quote_user`, assert escaped.

**Rendering:**

- Aggregate findings before template (`behavior × bucket → count/intensity-summary`).
- `<details>` drill-down for per-finding inspection.
- Paginate Module H claims table to 50/page.
- Beta-distribution CI whiskers on bar charts (`Finding.confidence_alpha + confidence_beta`).

**Handle `status=partial`:** banner per incomplete module. Module G skips incomplete modules in bucketing.

**Files:**

- `lucid/modules/module_g_attribution.py` — deterministic bucketing from Phase 6 skeleton.
- `lucid/report/generator.py` — Jinja2 env with autoescape.
- `lucid/report/templates/base.html.j2`, `report.html.j2`.
- `lucid/report/static/chart.umd.min.js` (vendored).
- `tests/test_report_generator.py` — XSS-injection + partial-status + empty-state tests.

**Exit criterion:** Report renders from real Lucid audit run. XSS test green. CSP present. <2MB HTML, <500ms TTI. Beta-CI whiskers visible.

---

### Phase 10 — Demo corpus + end-to-end polish + stretch decision (Day 5 AM)

**Tasks:**

1. **Layer 1 — SpiralBench public:** final calibration numbers in `docs/calibration.md` + README.
2. **Layer 2 — Seeded corpus:** craft 3–5 synthetic conversations exhibiting feedback sycophancy (Module B), belief shift with evidence-vs-pressure contrast (Module E), contradicted memory claim (Module H). Stored at `demo/corpus/`.
3. **Layer 3 — Daniel's real corpus:** local-only verification run. Findings never appear in demo video.
4. **Podcast stretch decision:** if Module H precision < 0.8 OR demo dry-run doesn't hit 3 min cleanly → cut. Otherwise Day 4 evening + Day 5 morning implement.
5. **Full end-to-end dry-run** on seeded corpus. Time it.
6. **`docs/privacy.md` written** (required before Phase 11).

**Exit criterion:** Full `lucid audit` under 10 min wall clock on 100-session sample. Seeded corpus produces predictable demo beats.

---

### Phase 11 — Binary build + release (Day 5 PM / Day 6 AM)

**Goal:** Shippable binaries + pushed public repo.

**Binary build:**

- PyInstaller `6.19.0` one-file builds for macOS arm64, Linux x64, Windows x64.
- If PyInstaller fights scipy hidden-imports → swap to Nuitka `4.0.8` standalone.
- Smoke-test each binary: `lucid audit --source claude-ai --path tests/fixtures/claude_ai --dry-run` runs clean.

**Distribution channels:**

- GitHub Releases with all 3 platform binaries attached. Unsigned macOS binary documented with `xattr -d com.apple.quarantine ./lucid` workaround.
- `uv publish` → PyPI for `uvx lucid audit`.
- Homebrew tap `github.com/<user>/homebrew-lucid` with formula pinning the GitHub Release.

**Repo polish:**

- README: install instructions (binary + `uvx lucid` + Homebrew), quickstart, methodology link, calibration numbers.
- `docs/calibration.md` finalized with real α, AC1, per-label κ, QWK, BCa CIs.
- `docs/methodology.md` finalized with pricing + embedding-provider + prompt-cache strategy.
- `docs/privacy.md` finalized.
- `LICENSE` = MIT.

**Final pre-submission:**

- [ ] Binaries built and smoke-tested.
- [ ] Repo public, all docs landed.
- [ ] `uv run pytest` green, `mypy --strict` clean, `ruff check` clean.
- [ ] Tag `v0.1.0` and push.

---

## Acceptance criteria

### Functional

- [ ] `lucid audit --source {claude-code|claude-ai|all} --path <p>` produces findings + HTML report.
- [ ] `lucid audit --dry-run` works without LLM spend.
- [ ] `lucid audit --include-module-d` opts Module D in; default excludes.
- [ ] `lucid calibrate --module a` prints α, AC1, per-label κ, QWK, BCa CIs.
- [ ] Cost estimator prompts when > $20.
- [ ] `--yes-i-authorize-spend-up-to N` + `LUCID_ALLOW_UNATTENDED=1` both required to bypass.
- [ ] `lucid audit --resume <run-id>` picks up from `module_progress`.
- [ ] `lucid` bare → quick-start, exit 0.

### Flow-gap handling

- [ ] 0 conversations → exit 2 with path hint.
- [ ] Missing/malformed `memories.json` → warn + skip Module H.
- [ ] `--resume <unknown-id>` / schema drift / prompt drift → exit 2 with clear message.
- [ ] Concurrent audit on same DB → `filelock` exit 2.
- [ ] Managed Agents heartbeat timeout (>60s) → `partial` AuditRun; resume completes it.

### Module coverage

- [ ] A, B-feedback, B-answer, C, E, F, G, H implemented.
- [ ] D implemented as opt-in.
- [ ] Every Finding populates required provenance fields (enforced by SQLite CHECK).
- [ ] Module A: α ≥ 0.67 AND AC1 ≥ 0.70 with lower-CI(α) ≥ 0.60 on 200+ labeled turns (or descope recorded).
- [ ] Module H: precision ≥ 0.8 on seeded corpus unsupported/contradicted.

### Non-functional

- [ ] `uv run pytest` green. `uv run mypy lucid/ --strict` clean. `uv run ruff check lucid/` clean.
- [ ] No real LLM calls in pytest suite.
- [ ] No real user content in committed fixtures.
- [ ] 100-session audit completes in <10 min wall clock.
- [ ] 9,840-file Claude Code parse completes in <60s via ProcessPoolExecutor.
- [ ] Orchestrator handles 429 via SDK auto-retry + module-boundary tenacity.

### Security

- [ ] Jinja2 autoescape configured; XSS injection test green.
- [ ] CSP meta tag in report.
- [ ] Prompt-injection fixture: classifier ignores adversarial instructions in corpus content.
- [ ] Path-traversal test (symlink out-of-root) rejected.
- [ ] Canary sentinel not in DEBUG logs.
- [ ] `ANTHROPIC_API_KEY` in `pydantic.SecretStr`; never in logged repr.
- [ ] `ANTHROPIC_LOG` unset on startup.

### LLM integration

- [ ] Module A system prompt ≥ 4096 tokens with `cache_control`.
- [ ] `response.usage.cache_read_input_tokens > 0` on second module call.
- [ ] `messages.parse(output_format=<PydanticModel>)` in every classifier module.
- [ ] Per-module `thinking_mode` + `effort` declared as module constants.

### Release

- [ ] PyInstaller (or Nuitka) binaries built for macOS arm64 / Linux x64 / Windows x64; each smoke-tests `--dry-run` clean.
- [ ] GitHub repo public, MIT-licensed, README complete with install/quickstart/methodology/calibration.
- [ ] PyPI published (enables `uvx lucid audit`).
- [ ] Homebrew tap formula pins the GitHub Release.

---

## Risk table (updated)

| Risk | Phase | Mitigation |
|---|---|---|
| R1 Managed Agents learning curve | 0, 5 | Read Managed Agents docs Phase 0; Agent SDK fallback via `LUCID_ORCHESTRATOR_BACKEND=sdk` env var. |
| R2 Opus 4.7 hallucinates sycophancy | 6 | Dual metric α + AC1; 200+ labels; bootstrap BCa. Descope tree if both < 0.55. |
| R3 Claude.ai export schema drift | 3 | Defensive parsing; unknown fields → `metadata`. |
| R4 Rate limit / cost blowout | 4, all modules | Sample default 100; `count_tokens` gate; `max_retries=4` in SDK; prompt caching saves ~$14/audit; Module F 3-stage filter. |
| R5 Demo doesn't land | 10, 11 | Synthetic seeded corpus; Beta-CI whiskers; paired-quote + Module H table pre-rehearsed. |
| R6 Privacy | 1, 3, 9 | SecretStr keys; `ANTHROPIC_LOG` unset; SafeFormatter; canary test; `docs/privacy.md` threat model; no `--redact` flag but synthetic demo corpus. |
| R7 Prior art missed | 0 | 30-min sweep. Module H remains unique with right framing (Problem 2 + user-side, not LLM-side). |
| R8 Module H "unsupported" vs "absent" | 8 | Similarity < 0.35 auto `insufficient-data`; two-stage verification on ambiguous. |
| R9 Binary build fights scipy hidden-imports | 11 | PyInstaller primary → Nuitka `--mode=onefile` fallback. Reserve 4h for macOS arm64 build. |
| R10 Mirror-judging-the-mirror | 6 | Opus 4.7 effort-variation self-consistency (no temperature available). |
| R11 9,840-file parse slow | 3 | ProcessPoolExecutor + orjson → ~30s. ijson bounds Claude.ai export memory. |

---

## Remaining open decisions (cheap to resolve on-the-day)

1. **D1 — primary IAA metric (α vs AC1).** Decide end of Day 2 AM after 200+ labels produced. Data-driven: if ≥3 labels have prevalence < 10% → AC1 primary. Report both regardless.
2. **D4 — final module scope.** Decide end of Day 3 after Phase 7. If calibration and Phase 7 on-track, ship A, B-feedback, B-answer, C, E, F, G, H. If slip, cut Module E to drift-only or drop F's Opus classification stage.
3. **Build tool** — PyInstaller primary; Nuitka fallback if scipy hidden-imports fight. Decide empirically in Phase 11.

---

## Sources

Decision doc `docs/plans/2026-04-21-decision-stack-and-distribution.md` §Sources has the full citation list (~60 URLs). Key ones:

- Managed Agents: https://platform.claude.com/docs/en/managed-agents/{quickstart, events-and-streaming, tools, files}
- Custom tools pattern: https://platform.claude.com/docs/en/managed-agents/tools
- Prompt caching: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Anthropic SDK: https://github.com/anthropics/anthropic-sdk-python/releases (v0.96.0)
- Voyage AI: https://docs.voyageai.com, https://pypi.org/project/voyageai/ (v0.3.7)
- Krippendorff: https://pypi.org/project/krippendorff/ (v0.8.2)
- irrCAC: https://pypi.org/project/irrCAC/ (v0.4.4)
- Cerebral Valley: https://cerebralvalley.ai/e/built-with-4-7-hackathon
- SpiralBench: https://eqbench.com/spiral-bench.html, https://github.com/sam-paech/spiral-bench
- Sharma 2023: https://arxiv.org/abs/2310.13548
- MedTrust-RAG (two-stage verification): https://arxiv.org/pdf/2510.14400
- Gwet AC1 vs kappa paradox: https://pmc.ncbi.nlm.nih.gov/articles/PMC12163189/

---

*Plan v3 locked 2026-04-21. Execute via `superpowers:executing-plans` or `subagent-driven-development`. Any deviation from locked decisions → update `docs/plans/2026-04-21-decision-stack-and-distribution.md` with reasoning before code changes.*
