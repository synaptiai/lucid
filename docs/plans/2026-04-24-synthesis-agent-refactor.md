# Synthesis Agent Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move Managed Agents from a vestigial for-loop orchestrator into a synthesis session that reads findings + spot-reads corpus and writes the narrative sections of the report with grounded, validated citations; keep deterministic code responsible for scoring and routing.

**Architecture:** Two-phase pipeline. Phase 1 — deterministic Python scoring loop invokes modules A-H directly (no agent, same module contracts, same findings persisted). Phase 2 — new `SynthesisSession` Managed Agent reads the findings table (cached in system prompt) + spot-reads conversations via read-only tools, writes section prose with `[F:finding_id]` inline citation tokens, a Sonnet 4.6 post-processor structures the prose into validated `{text, citations, aggregate_support}` blocks, validator drops uncited/invalid prose, persisted blocks inline into Jinja2 template. Honest `INSUFFICIENT_EVIDENCE` verdict when the agent cannot ground a section.

**Tech Stack:** Python 3.13, Pydantic v2, SQLite + `aiosqlite`, Anthropic SDK (Managed Agents beta: `managed-agents-2026-04-01`), Claude Opus 4.7 (synthesis write) + Sonnet 4.6 (post-process), Jinja2 (report), tenacity (retries), filelock (concurrency).

---

## Phase overview & timeline

| Phase | Scope | Est. hours | Depends on | Parallelizable within |
|---|---|---|---|---|
| 1 | Foundation: `report_sections` table, Pydantic models, CRUD | 2-3 | — | Tasks 1.1 & 1.2 parallel |
| 2 | Orchestrator deletion + deterministic scoring loop | 2-3 | 1 | Tasks 2.1-2.3 sequential, 2.4 parallel with 2.5 |
| 3 | `lucid/synthesis/` package + session + tool registry | 3-4 | 1, 2 | Tasks 3.2 & 3.3 parallel |
| 4 | Synthesis schemas, prompts v1, output contracts | 2-3 | 3 | Tasks 4.1, 4.2, 4.3 parallel |
| 5 | Two-phase write + validator + failure-mode guards | 3-4 | 4 | Tasks 5.1 → 5.2 → 5.3 sequential; 5.4 parallel |
| 6 | Report template integration + demo renderer | 2-3 | 1, 5 | Tasks 6.1-6.3 sequential; 6.4 parallel |
| 7 | Docs, CHANGELOG, end-to-end smoke run | 1-2 | 6 | Tasks 7.1 & 7.2 parallel |
| 8 | Stretch: `lucid ask` interactive mode | 2-3 | 3, 5 | Optional |

**Total: 17-25 hours.** Deadline: Sunday 2026-04-26 20:00 EST.

## Minimum shippable subset

If time collapses, ship in this priority order. Each checkpoint leaves the project in a shippable state:

1. **Checkpoint A — "synthesis single-section" (~10 hours)**: Phase 1 + 2 + 3 + a compressed Phase 4/5 that produces ONLY the `exec_summary` section with citation validation + Phase 6's template hook for `exec_summary`. Keep all other report sections templated as today. Pitch: "Lucid's narrative exec summary is agent-written with finding-level citations; the rest remains deterministic."
2. **Checkpoint B — "synthesis full narrative" (+6 hours)**: Add per-module narratives + `top_3_actions` + `headline_findings` hybrid mode. This is the full product vision.
3. **Checkpoint C — "synthesis + ask" (+3 hours)**: Phase 8. Stretch.

**Backstop:** the current Managed Agents orchestrator + backfill still works today and passes tests. We can revert to `main` at any phase checkpoint and ship what's there.

## Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Synthesis agent hallucinates finding_ids that don't exist | High | High | Validator rejects; regen loop (max 2); explicit "you may only cite IDs shown above" in system prompt |
| R2 | `messages.parse()` doesn't work with Managed Agents tool loop | Medium | High | Fallback: Sonnet post-process call does `messages.parse()` *outside* the agent session; agent stays on unstructured markdown |
| R3 | Prompt caching fails silently (below 4096-token threshold) | Low | Medium | Log `cache_creation_input_tokens`/`cache_read_input_tokens` from every session; alert in test if both are zero |
| R4 | Large findings tables (1000+) break in-context synthesis | Low | Medium | Short-circuit: if `len(findings) > 1000`, fall back to `query_findings` tool instead of full-table inject. Not shipping in v1; document threshold |
| R5 | Refactor breaks existing module tests (A-H) | Medium | High | Run `uv run pytest tests/test_module_*.py` after every phase commit. Modules themselves are untouched |
| R6 | Template changes regress the existing report's visual design | Medium | Medium | Snapshot a pre-refactor report (`run-9b7031f168cf.html`) and diff visually at every template-touching commit |
| R7 | SQLite schema migration corrupts existing DBs | Low | High | `schema_version` bump + `ALTER TABLE` path tested against a fixture DB; `report_sections` is additive (no data migration) |
| R8 | Synthesis session stalls past session timeout | Low | Medium | Heartbeat watchdog (already exists in `HeartbeatMonitor`); session_timeout_seconds=3600 default; partial results persist per-section |
| R9 | Opus 4.7 output quality varies between sections | High | Low | Accept variance for v1; Phase 2 Sonnet post-processor catches format drift; calibration against human eyeball at smoke test |
| R10 | Demo renderer (`demo/render_demo_report.py`) breaks after template changes | Medium | Low | Template degrades gracefully when `report_sections` empty; demo renders without prose + footer note |

## Commit hygiene

- **One task = one commit.** Commit messages follow existing repo style (check `git log --oneline -20` for tone).
- Every commit must leave `uv run pytest -x` green on the full module test suite (`tests/test_module_*.py`, `tests/test_ingest_*.py`, `tests/test_schemas.py`, `tests/test_store.py`). Orchestrator tests will be deleted in Phase 2 — that deletion is allowed to red them mid-phase as long as the phase-end commit is clean.
- Skip the `--no-verify` escape hatch on commit hooks. If a hook fails, fix the root cause.
- Every phase ends with a "phase checkpoint" commit that runs the full smoke: `uv run lucid audit --source claude-code --path tests/fixtures/claude-code --sample 5 --dry-run` + a 2-conversation real run against `tests/fixtures/`.

---

# Phase 1 — Foundation (report_sections table + models + CRUD)

**Goal:** Land the persistence layer that Phases 5 and 6 write to/read from. No user-visible change yet. Fully testable in isolation.

## Task 1.1 — Add `ReportSection` Pydantic model

**Files:**
- Modify: `lucid/schemas.py` (append after `TokenUsage` around line ~280, or wherever the last model is)
- Test: `tests/test_schemas.py` (append)

**Step 1: Write the failing test**

Append to `tests/test_schemas.py`:

```python
from lucid.schemas import ReportSection

def test_report_section_roundtrip():
    section = ReportSection(
        audit_run_id="run-abc123",
        section_id="exec_summary",
        markdown="Across [F:f001] findings, the corpus shows a pattern of pushback.",
        cited_finding_ids=["f001", "f002"],
        cited_turn_ids=["t001"],
        insufficient_evidence=False,
        created_at=datetime(2026, 4, 24, 10, 0, tzinfo=UTC),
    )
    roundtrip = ReportSection.model_validate_json(section.model_dump_json())
    assert roundtrip == section

def test_report_section_rejects_empty_section_id():
    with pytest.raises(ValidationError):
        ReportSection(
            audit_run_id="run-abc",
            section_id="",
            markdown="x",
            cited_finding_ids=[],
            cited_turn_ids=[],
            insufficient_evidence=False,
            created_at=datetime.now(tz=UTC),
        )

def test_report_section_insufficient_evidence_allows_empty_prose():
    """When the agent declines a section, markdown may be empty and citations may be empty."""
    section = ReportSection(
        audit_run_id="run-abc",
        section_id="top_3_actions",
        markdown="",
        cited_finding_ids=[],
        cited_turn_ids=[],
        insufficient_evidence=True,
        decline_reason="fewer than 5 qualifying findings",
        created_at=datetime.now(tz=UTC),
    )
    assert section.insufficient_evidence is True
```

**Step 2: Run to verify failure**

`uv run pytest tests/test_schemas.py::test_report_section_roundtrip -v`
Expected: `ImportError: cannot import name 'ReportSection' from 'lucid.schemas'`.

**Step 3: Implement the model**

Append to `lucid/schemas.py` (after the last model, before `__all__` if present):

```python
class ReportSection(BaseModel):
    """One agent-written section of the HTML report.

    Populated by the synthesis phase; read by the report renderer.
    ``markdown`` contains inline ``[F:finding_id]`` / ``[T:turn_id]``
    citation tokens that the renderer resolves to hover-links. Every
    id in ``cited_finding_ids`` / ``cited_turn_ids`` MUST exist in the
    run's findings/turns tables — the synthesis validator enforces this
    before the row lands here.

    When ``insufficient_evidence`` is True, the agent declined the
    section. ``markdown`` is empty, citation lists are empty, and
    ``decline_reason`` carries a human-readable explanation. The template
    renders "Section skipped: <decline_reason>" instead of prose.
    """

    model_config = ConfigDict(extra="forbid")

    audit_run_id: str = Field(min_length=1)
    section_id: str = Field(min_length=1, max_length=64)
    markdown: str = Field(default="")
    cited_finding_ids: list[str] = Field(default_factory=list)
    cited_turn_ids: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = False
    decline_reason: str | None = None
    created_at: datetime
```

**Step 4: Run to verify pass**

`uv run pytest tests/test_schemas.py -k report_section -v`
Expected: 3/3 PASS.

**Step 5: Run full schema tests + mypy strict**

`uv run pytest tests/test_schemas.py -v && uv run mypy lucid/schemas.py --strict`
Expected: all green.

**Step 6: Commit**

```bash
git add lucid/schemas.py tests/test_schemas.py
git commit -m "feat(schemas): add ReportSection for synthesis-layer prose persistence"
```

## Task 1.2 — Add `report_sections` SQL table + schema bump

**Files:**
- Modify: `lucid/store/schema.sql` (append new table at end)
- Modify: `lucid/store/init.py` (bump `SCHEMA_VERSION` if it exists there, or verify the initialize path handles new tables via `IF NOT EXISTS`)
- Test: `tests/test_store.py` (new test for schema presence)

**Step 1: Write the failing test**

Append to `tests/test_store.py`:

```python
def test_report_sections_table_exists(tmp_path):
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        rows = store.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='report_sections'"
        )
        assert len(rows) == 1

def test_report_sections_schema_columns(tmp_path):
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        rows = store.fetchall("PRAGMA table_info(report_sections)")
        columns = {row["name"] for row in rows}
        expected = {
            "id", "audit_run_id", "section_id", "markdown",
            "cited_finding_ids_json", "cited_turn_ids_json",
            "insufficient_evidence", "decline_reason", "created_at",
        }
        assert expected.issubset(columns)

def test_report_sections_unique_key(tmp_path):
    """(audit_run_id, section_id) is unique — re-run overwrites, does not dup."""
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        conn = store.connect()
        # Need a parent audit_run row for FK.
        # Use the existing fixture helper or inline a minimal insert.
        # (Adapt to whatever test_store.py's helpers provide.)
        _insert_minimal_audit_run(conn, run_id="run-ut1")
        conn.execute(
            "INSERT INTO report_sections(id, audit_run_id, section_id, markdown, "
            "cited_finding_ids_json, cited_turn_ids_json, insufficient_evidence, "
            "decline_reason, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            ("rs-1", "run-ut1", "exec_summary", "x", "[]", "[]", 0, None, "2026-04-24T10:00:00+00:00"),
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO report_sections(id, audit_run_id, section_id, markdown, "
                "cited_finding_ids_json, cited_turn_ids_json, insufficient_evidence, "
                "decline_reason, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                ("rs-2", "run-ut1", "exec_summary", "y", "[]", "[]", 0, None, "2026-04-24T10:01:00+00:00"),
            )
            conn.commit()
```

**Step 2: Run to verify failure**

`uv run pytest tests/test_store.py -k report_sections -v`
Expected: all 3 fail with "no such table".

**Step 3: Add table DDL**

Append to `lucid/store/schema.sql`:

```sql
-- ------------------------------------------------------------------
-- Report sections (synthesis-layer agent prose)
-- ------------------------------------------------------------------
--
-- One row per (audit_run_id, section_id). Populated by the synthesis
-- phase; read by the report renderer. Re-running synthesis against
-- the same audit_run_id MUST upsert (ON CONFLICT DO UPDATE), not
-- duplicate — hence the unique index on (audit_run_id, section_id).
--
-- `markdown` carries inline `[F:finding_id]` / `[T:turn_id]` citation
-- tokens. The CHECK below mirrors the Pydantic `ReportSection.insufficient_evidence`
-- semantics: when the agent declined the section, markdown is empty
-- and citation lists are empty; when the section is populated,
-- markdown is non-empty.
CREATE TABLE IF NOT EXISTS report_sections (
    id                       TEXT PRIMARY KEY,
    audit_run_id             TEXT NOT NULL REFERENCES audit_runs(id) ON DELETE CASCADE,
    section_id               TEXT NOT NULL CHECK (length(section_id) > 0),
    markdown                 TEXT NOT NULL DEFAULT '',
    cited_finding_ids_json   TEXT NOT NULL DEFAULT '[]',
    cited_turn_ids_json      TEXT NOT NULL DEFAULT '[]',
    insufficient_evidence    INTEGER NOT NULL DEFAULT 0 CHECK (insufficient_evidence IN (0, 1)),
    decline_reason           TEXT,
    created_at               TEXT NOT NULL,
    CHECK (
        (insufficient_evidence = 1 AND length(markdown) = 0)
        OR (insufficient_evidence = 0 AND length(markdown) > 0)
    ),
    UNIQUE (audit_run_id, section_id)
);

CREATE INDEX IF NOT EXISTS idx_report_sections_run
    ON report_sections(audit_run_id);
```

**Step 4: Check whether a SCHEMA_VERSION bump is needed**

`grep -n "SCHEMA_VERSION" lucid/store/*.py`

If `SCHEMA_VERSION` exists and is a pinned integer, bump it (e.g. 1 → 2). If migration logic branches on it, add a no-op branch for old → new since `report_sections` is additive (`IF NOT EXISTS`) and doesn't need data migration. If migration is handled purely by `IF NOT EXISTS` DDL, leave the version alone.

**Step 5: Run tests**

`uv run pytest tests/test_store.py -k report_sections -v`
Expected: 3/3 PASS.

**Step 6: Re-run full store tests**

`uv run pytest tests/test_store.py -v`
Expected: all existing tests still green.

**Step 7: Commit**

```bash
git add lucid/store/schema.sql lucid/store/init.py tests/test_store.py
git commit -m "feat(store): add report_sections table for synthesis-layer prose"
```

## Task 1.3 — Add `CorpusStore` CRUD helpers for `report_sections`

**Files:**
- Modify: `lucid/store/sqlite.py` (add `insert_report_section`, `upsert_report_section`, `fetch_report_sections_for_run`)
- Test: `tests/test_store.py`

**Step 1: Write failing tests**

```python
def test_upsert_report_section_roundtrip(tmp_path):
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _insert_minimal_audit_run(store.connect(), run_id="run-ut2")
        section = ReportSection(
            audit_run_id="run-ut2",
            section_id="exec_summary",
            markdown="One paragraph with [F:f001].",
            cited_finding_ids=["f001"],
            cited_turn_ids=[],
            insufficient_evidence=False,
            created_at=datetime.now(tz=UTC),
        )
        store.upsert_report_section(section)
        rows = store.fetch_report_sections_for_run("run-ut2")
        assert len(rows) == 1
        assert rows[0].markdown == section.markdown

def test_upsert_report_section_is_idempotent(tmp_path):
    """Re-running synthesis replaces the row rather than duplicating."""
    # ... populate once with markdown="v1", upsert again with markdown="v2",
    # assert only one row and markdown == "v2"
```

**Step 2: Run to verify failure**

Expected: `AttributeError: 'CorpusStore' object has no attribute 'upsert_report_section'`.

**Step 3: Implement helpers**

Add to `lucid/store/sqlite.py`:

```python
def upsert_report_section(self, section: ReportSection) -> None:
    """Insert or replace a report section (idempotent by (run_id, section_id))."""
    conn = self.connect()
    conn.execute(
        """
        INSERT INTO report_sections (
            id, audit_run_id, section_id, markdown,
            cited_finding_ids_json, cited_turn_ids_json,
            insufficient_evidence, decline_reason, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(audit_run_id, section_id) DO UPDATE SET
            markdown = excluded.markdown,
            cited_finding_ids_json = excluded.cited_finding_ids_json,
            cited_turn_ids_json = excluded.cited_turn_ids_json,
            insufficient_evidence = excluded.insufficient_evidence,
            decline_reason = excluded.decline_reason,
            created_at = excluded.created_at
        """,
        (
            f"rs-{uuid.uuid4().hex[:12]}",
            section.audit_run_id,
            section.section_id,
            section.markdown,
            orjson.dumps(section.cited_finding_ids).decode(),
            orjson.dumps(section.cited_turn_ids).decode(),
            1 if section.insufficient_evidence else 0,
            section.decline_reason,
            section.created_at.isoformat(),
        ),
    )
    conn.commit()

def fetch_report_sections_for_run(self, audit_run_id: str) -> list[ReportSection]:
    rows = self.fetchall(
        "SELECT audit_run_id, section_id, markdown, cited_finding_ids_json, "
        "cited_turn_ids_json, insufficient_evidence, decline_reason, created_at "
        "FROM report_sections WHERE audit_run_id = ? ORDER BY section_id",
        (audit_run_id,),
    )
    return [
        ReportSection(
            audit_run_id=r["audit_run_id"],
            section_id=r["section_id"],
            markdown=r["markdown"],
            cited_finding_ids=orjson.loads(r["cited_finding_ids_json"]),
            cited_turn_ids=orjson.loads(r["cited_turn_ids_json"]),
            insufficient_evidence=bool(r["insufficient_evidence"]),
            decline_reason=r["decline_reason"],
            created_at=datetime.fromisoformat(r["created_at"]),
        )
        for r in rows
    ]
```

**Step 4: Run tests**

`uv run pytest tests/test_store.py -k report_section -v`
Expected: all PASS.

**Step 5: Commit**

```bash
git add lucid/store/sqlite.py tests/test_store.py
git commit -m "feat(store): add ReportSection CRUD helpers (upsert, fetch)"
```

## Task 1.4 — Phase 1 checkpoint

**Step 1:** Full test suite green.

`uv run pytest -x`

**Step 2:** mypy strict green.

`uv run mypy lucid/ --strict`

**Step 3:** Dry-run still works (no behavior change yet).

`uv run lucid audit --source claude-code --path tests/fixtures/claude-code --sample 3 --dry-run`

Expected: exits 0, prints sampling + cost estimate, no errors.

**Step 4:** Tag a safety point.

```bash
git tag phase-1-foundation
```

---

# Phase 2 — Orchestrator deletion & deterministic scoring loop

**Goal:** Replace the Managed Agents orchestrator + its backfill with a 10-line Python scoring loop. The pipeline is end-to-end the same from the user's perspective — just without the useless agent session in the middle. This is the "rip off the band-aid" phase.

## Task 2.1 — Extract `invoke_module_handler` as a standalone async function

**Files:**
- Modify: `lucid/orchestrator/tools.py` — extract the body of the inline `invoke_module` handler (around line 665-832) into a module-level `async def invoke_module_for_run(...)` that the scoring loop can call directly.
- Keep the existing `invoke_module` closure registering the tool for the moment — it just delegates to the new function. (We'll delete the closure in Task 2.4.)
- Test: `tests/test_orchestrator_tools.py` — add a direct unit test for `invoke_module_for_run`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_invoke_module_for_run_direct_call(tmp_path, monkeypatch):
    """Calling invoke_module_for_run without the orchestrator registry
    wrapper produces the same result shape."""
    from lucid.orchestrator.tools import invoke_module_for_run
    # ... build store, persist minimal run + conversation + turns,
    # call invoke_module_for_run(module=G_ATTRIBUTION, conversation_ids=[...], ...)
    # assert result["status"] == "completed"
```

**Step 2: Run to verify failure**

Expected: ImportError.

**Step 3: Extract the function**

Lift the body of the inline `invoke_module` closure into a module-level function:

```python
async def invoke_module_for_run(
    *,
    module: ModuleName,
    conversation_ids: list[str],
    store: CorpusStore,
    audit_run_id: str,
    anthropic_client: AsyncAnthropic | None,
    embedding_provider: EmbeddingProvider | None,
    allow_module_d: bool,
    progress_log: Callable[[str, str], None],
    per_module_usd: dict[ModuleName, float],
    debited_modules: set[ModuleName],
    spend_tracker: dict[str, float],
) -> dict[str, Any]:
    """Standalone version of the invoke_module tool handler.

    Used by the deterministic scoring loop in lucid/run.py and (for
    now) also by the inline `invoke_module` closure in
    ``build_tool_registry``. Once the orchestrator is fully removed,
    this is the only entry point.
    """
    # body moved from the closure, unchanged
```

Update the closure in `build_tool_registry` to simply:

```python
async def invoke_module(args: dict[str, Any]) -> dict[str, Any]:
    module_name = args["module"]
    try:
        module_enum = ModuleName(module_name)
    except ValueError:
        return {"error": "unknown_module", "module": module_name}
    return await invoke_module_for_run(
        module=module_enum,
        conversation_ids=args.get("conversation_ids") or [],
        store=store,
        audit_run_id=audit_run_id,
        anthropic_client=anthropic_client,
        embedding_provider=embedding_provider,
        allow_module_d=allow_module_d,
        progress_log=progress_log,
        per_module_usd=per_module_usd,
        debited_modules=debited_modules,
        spend_tracker=spend_tracker,
    )
```

**Step 4: Run tests**

`uv run pytest tests/test_orchestrator_tools.py -v`
Expected: old tests pass (delegation is behavior-preserving); new direct-call test passes.

**Step 5: Commit**

```bash
git add lucid/orchestrator/tools.py tests/test_orchestrator_tools.py
git commit -m "refactor(orchestrator): extract invoke_module_for_run as standalone async fn"
```

## Task 2.2 — Write `_run_scoring_loop` in `lucid/run.py`

**Files:**
- Modify: `lucid/run.py` — add new `async def _run_scoring_loop(...)` near `_execute_session_and_safety_net`
- Test: `tests/test_run.py` or `tests/test_run_scoring_loop.py` (new)

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_run_scoring_loop_invokes_every_enabled_module(tmp_path, monkeypatch):
    """The loop invokes each enabled module exactly once with the full id list,
    in the order given, and returns a summary of per-module outcomes."""
    # Stub invoke_module_for_run to a spy that records calls + returns "completed"
    calls = []
    async def _spy(*, module, conversation_ids, **_):
        calls.append((module, tuple(conversation_ids)))
        return {"module": module.value, "status": "completed", "findings_stored": 0}
    monkeypatch.setattr("lucid.run.invoke_module_for_run", _spy)
    # call _run_scoring_loop with enabled_modules=[A, B, G], ids=["c1", "c2"]
    # assert calls == [(A, ("c1","c2")), (B, ("c1","c2")), (G, ("c1","c2"))]
```

**Step 2: Run to verify failure**

Expected: AttributeError (`_run_scoring_loop` not defined).

**Step 3: Implement**

Add to `lucid/run.py`:

```python
async def _run_scoring_loop(
    *,
    store: CorpusStore,
    registry: ToolRegistry,      # kept for the per-module debit context
    audit_run_id: str,
    enabled_modules: list[ModuleName],
    sampled_ids: list[str],
    progress_log: Callable[[str, str], None],
    anthropic_client: AsyncAnthropic | None,
    embedding_provider: EmbeddingProvider | None,
    allow_module_d: bool,
    per_module_usd: dict[ModuleName, float],
    debited_modules: set[ModuleName],
    spend_tracker: dict[str, float],
) -> list[dict[str, Any]]:
    """Deterministically invoke each enabled module over the sampled ids.

    Replaces the Managed Agents orchestrator session + its backfill.
    Modules run sequentially (bounded concurrency is internal to each
    module). Returns the per-module result dicts for logging.
    """
    results: list[dict[str, Any]] = []
    for module in enabled_modules:
        progress_log("INFO", f"Running module {module.value}")
        result = await invoke_module_for_run(
            module=module,
            conversation_ids=sampled_ids,
            store=store,
            audit_run_id=audit_run_id,
            anthropic_client=anthropic_client,
            embedding_provider=embedding_provider,
            allow_module_d=allow_module_d,
            progress_log=progress_log,
            per_module_usd=per_module_usd,
            debited_modules=debited_modules,
            spend_tracker=spend_tracker,
        )
        results.append(result)
    return results
```

**Step 4: Run tests**

Expected: new test PASS. Existing orchestrator tests still pass (we haven't deleted anything yet).

**Step 5: Commit**

```bash
git add lucid/run.py tests/test_run_scoring_loop.py
git commit -m "feat(run): add _run_scoring_loop replacing orchestrator backfill"
```

## Task 2.3 — Rewire `run_audit` to use `_run_scoring_loop` (orchestrator path still present as fallback)

**Files:**
- Modify: `lucid/run.py` — `run_audit` now calls `_run_scoring_loop`; `_execute_session_and_safety_net` is reserved for a `--legacy-orchestrator` flag (kept one version to allow A/B debugging)

Decision: **don't keep the legacy path.** It's debt. Remove it outright in this task.

**Step 1: Modify `run_audit` body**

In `lucid/run.py::run_audit`, replace the `asyncio.run(_execute_session_and_safety_net(...))` call with:

```python
outcome, status, reason = asyncio.run(
    _execute_scoring(
        store=store,
        registry=registry,
        audit_run_id=resolved_run_id,
        enabled_modules=inputs.enabled_modules,
        sampled_ids=sampled_ids,
        progress_log=progress_log,
        anthropic_client=async_client,
        embedding_provider=embedding_provider,
        allow_module_d=allow_module_d,
        per_module_usd=per_module_usd,
        debited_modules=debited_modules,
        spend_tracker=spend_tracker,
    )
)
```

And define a new thin coroutine `_execute_scoring` that:
1. Calls `_run_scoring_loop(...)`.
2. Returns `(None, status, reason)` — no `SessionOutcome` anymore. Update `AuditResult` to tolerate `outcome: SessionOutcome | None = None` (already nullable; verify).
3. status = "completed" if every module returned `status="completed"`; else "partial".

**Step 2: Delete the now-unreachable functions**

Delete from `lucid/run.py`:
- `_execute_session_and_safety_net` (lines ~487-563)
- `_backfill_unfinished_modules` (lines ~566-607)
- `_attribution_safety_net` (lines ~254-285)
- `_default_session_runner` (line ~609)

Delete imports of `ManagedAgentsSession`, `OrchestratorConfig`, `SessionOutcome`, `run_attribution_safety_net` that are no longer referenced. Keep `ToolRegistry`, `build_tool_registry` — those stay for the spend tracker context.

**Step 3: Module G is now "just another module"**

The for-loop runs G like anything else. Delete any code that treats G as a safety net. Add a comment at the G-invocation point in `_run_scoring_loop`: *"G is deterministic (no LLM). It runs here as the last enabled module by convention from the CLI."*

Update CLI's `enabled_modules` construction in `lucid/cli.py` to ensure G is always appended last when `--no-include-module-g` isn't present (if such a flag exists; if not, G is always in the list). Verify.

**Step 4: Update signature of `run_audit` — remove `system_prompt` + `kickoff_message` + `session_runner` params**

```python
def run_audit(
    *,
    inputs: AuditInputs,
    data_dir: Path,
    async_client: AsyncAnthropic,   # sync Anthropic no longer needed
    # Removed: client, system_prompt, kickoff_message, session_runner
    prompt_versions: dict[ModuleName, str],
    embedding_provider: EmbeddingProvider | None = None,
    allow_module_d: bool = False,
    progress_log: Callable[[str, str], None] | None = None,
    lock_timeout_seconds: float = 0.1,
    run_id: str | None = None,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> AuditResult:
```

**Step 5: Update `lucid/cli.py` call site**

In `lucid/cli.py` (search for `run_audit(` around line 321):
- Delete `system_prompt=SYSTEM_PROMPT` kwarg.
- Delete `kickoff_message=...` kwarg (and the `_build_kickoff_message` helper if unused).
- Delete the sync `Anthropic` client instantiation (async is the only one needed now).
- Delete `from lucid.orchestrator.system_prompt import SYSTEM_PROMPT` import.

**Step 6: Run the full test suite**

`uv run pytest -x`

Expected failures at this step:
- `tests/test_run.py` — any test that mocked `session_runner` or asserted on `SessionOutcome` fields now breaks.
- `tests/test_cli.py` — if it asserts `SYSTEM_PROMPT` was passed.

Fix these tests: replace `session_runner=` fakes with spy on `invoke_module_for_run`; drop `SessionOutcome` field assertions; assert on `AuditResult.status` instead.

**Step 7: Commit**

```bash
git add lucid/run.py lucid/cli.py tests/
git commit -m "refactor(run): replace orchestrator session with deterministic scoring loop

BREAKING: run_audit() no longer accepts system_prompt, kickoff_message,
or session_runner kwargs. The CLI is the only caller; downstream
consumers must update. The Managed Agents session is removed from the
scoring path — every module is invoked directly via
invoke_module_for_run in a for loop. Module G runs last by convention
rather than as a post-session safety net."
```

## Task 2.4 — Delete dead orchestrator files

**Step 1:** Inventory what's still imported.

```bash
grep -rn "from lucid.orchestrator" lucid/ tests/ demo/ scripts/
grep -rn "import lucid.orchestrator" lucid/ tests/ demo/ scripts/
```

Expected remaining importers:
- Things importing `build_tool_registry` / `ToolRegistry` (kept — Phase 3 reuses them renamed).
- Things importing `dispatch_tool_call` / `HeartbeatMonitor` (kept — Phase 3 moves them).
- Things importing `run_attribution_safety_net` (should be zero after Task 2.3).
- Things importing `SYSTEM_PROMPT` / `PROMPT_VERSION` / `ManagedAgentsSession` / `OrchestratorConfig` / `SessionOutcome` / `SessionHandles` (should be zero after Task 2.3).

If non-zero on the last group, fix.

**Step 2:** Delete these files:

```bash
git rm lucid/orchestrator/system_prompt.py
git rm lucid/orchestrator/managed_agent.py
```

**Step 3:** Trim `lucid/orchestrator/__init__.py` — remove exports that point at deleted symbols.

**Step 4:** Delete orchestrator-session tests:

```bash
git rm tests/test_orchestrator_managed_agent.py  # if it exists
git rm tests/test_run_audit_live_path.py         # if it exists
```

(Keep `tests/test_orchestrator_tools.py`, `tests/test_orchestrator_handler.py`, `tests/test_orchestrator_lifecycle.py` for now — Phase 3 repurposes them.)

**Step 5:** Run full test suite.

`uv run pytest -x`
Expected: green. Any test file referencing deleted symbols must be deleted or fixed.

**Step 6:** Commit.

```bash
git add -A
git commit -m "chore(orchestrator): delete session + system prompt + related tests

The scoring loop replaces the orchestrator session entirely; these
files are unreachable. Handler + lifecycle + tool registry remain
for Phase 3 to repurpose into the synthesis package."
```

## Task 2.5 — Phase 2 checkpoint

**Step 1:** Full suite green.
`uv run pytest`

**Step 2:** Smoke — dry run.
`uv run lucid audit --source claude-code --path tests/fixtures/claude-code --sample 3 --dry-run`

**Step 3:** Smoke — real run (small).
`uv run lucid audit --source claude-ai --path tests/fixtures/claude-ai --sample 2 --yes-i-authorize-spend-up-to 5`

Expected: completes end-to-end without the Managed Agents session appearing in logs; HTML report renders; `report_sections` table exists but is empty.

**Step 4:** Count LOC delta.
```bash
git diff --stat phase-1-foundation..HEAD -- lucid/
```
Expected: net negative (~1200 LOC removed from orchestrator + run + cli).

**Step 5:** Tag.
```bash
git tag phase-2-scoring-loop
```

---

# Phase 3 — `lucid/synthesis/` package + session infrastructure

**Goal:** Stand up the new home for the synthesis agent. Move reusable pieces (`HeartbeatMonitor`, `dispatch_tool_call`, `lifecycle` renamed) into `lucid/synthesis/` so the package is coherent. The session class here is the mirror of the deleted `ManagedAgentsSession`, minus the vestigial complexity.

## Task 3.1 — Scaffold `lucid/synthesis/` package

**Files:**
- Create: `lucid/synthesis/__init__.py`
- Create: `lucid/synthesis/session.py` (empty stub for now)
- Create: `lucid/synthesis/tools.py` (empty stub)
- Create: `lucid/synthesis/lifecycle.py` (will move from orchestrator)
- Create: `lucid/synthesis/handler.py` (will move from orchestrator)

**Step 1:** Create the directory + empty stubs.

```bash
mkdir -p lucid/synthesis
touch lucid/synthesis/__init__.py
```

Scaffold `lucid/synthesis/__init__.py`:

```python
"""Synthesis session: Managed Agents narrative writer.

Runs *after* the deterministic scoring phase. Reads the findings table,
spot-reads the corpus via read-only custom tools, writes narrative
sections of the HTML report with inline ``[F:finding_id]`` citation
tokens. A Sonnet 4.6 post-processor structures the prose into
validated blocks; uncited or invalid prose is dropped at render time.

See docs/plans/2026-04-24-synthesis-agent-refactor.md for the design.
"""
```

**Step 2:** Commit.

```bash
git add lucid/synthesis/
git commit -m "chore(synthesis): scaffold lucid/synthesis/ package"
```

## Task 3.2 — Move `handler.py` (dispatch + heartbeat) into `lucid/synthesis/`

**Files:**
- Move: `lucid/orchestrator/handler.py` → `lucid/synthesis/handler.py`
- Update: `lucid/synthesis/handler.py` — no logic change; just the module path
- Delete: `lucid/orchestrator/handler.py`
- Update: any importers

**Step 1:** Move the file.

```bash
git mv lucid/orchestrator/handler.py lucid/synthesis/handler.py
```

**Step 2:** Update its test.

```bash
git mv tests/test_orchestrator_handler.py tests/test_synthesis_handler.py
```

In `tests/test_synthesis_handler.py`, replace `from lucid.orchestrator.handler import ...` with `from lucid.synthesis.handler import ...`.

**Step 3:** Re-run tests.

`uv run pytest tests/test_synthesis_handler.py -v`
Expected: green.

**Step 4:** Commit.

```bash
git add -A
git commit -m "refactor(synthesis): move handler (dispatch + heartbeat) from orchestrator"
```

## Task 3.3 — Move + rename `lifecycle.py` (with backcompat for old prefix)

**Files:**
- Move: `lucid/orchestrator/lifecycle.py` → `lucid/synthesis/lifecycle.py`
- Modify: replace `LUCID_ORCHESTRATOR_PREFIX = "lucid-orchestrator-"` with `LUCID_SYNTHESIS_PREFIX = "lucid-synthesis-"`; add a `LUCID_LEGACY_PREFIXES = ("lucid-orchestrator-",)` constant so the prune helper archives old agents too.
- Update: `prune_stale_synthesis_agents` searches both current and legacy prefixes.

**Step 1:** Move.

```bash
git mv lucid/orchestrator/lifecycle.py lucid/synthesis/lifecycle.py
git mv tests/test_orchestrator_lifecycle.py tests/test_synthesis_lifecycle.py
```

**Step 2:** Rename constants + functions in `lucid/synthesis/lifecycle.py`.

```python
LUCID_SYNTHESIS_PREFIX = "lucid-synthesis-"
LUCID_LEGACY_PREFIXES: tuple[str, ...] = ("lucid-orchestrator-",)

def get_or_create_synthesis_agent(
    client,
    *,
    model: str,
    system_prompt: str,
    tools: list[dict[str, Any]],
    prompt_version: str,
) -> str:
    """Find-or-create the synthesis agent named ``lucid-synthesis-v{prompt_version}``."""
    ...

def prune_stale_synthesis_agents(client) -> None:
    """Archive agents matching the current synthesis prefix at a stale
    version, plus any agent matching a legacy prefix (migration from
    lucid-orchestrator-*)."""
    prefixes = (LUCID_SYNTHESIS_PREFIX,) + LUCID_LEGACY_PREFIXES
    ...
```

Update the test to assert `prune_stale_synthesis_agents` archives both `lucid-orchestrator-v2` and `lucid-synthesis-v0` (when the current version is v1).

**Step 3:** Update `lucid/cli.py` cleanup-agents command to import from the new path; update its help text.

**Step 4:** Run tests.

`uv run pytest tests/test_synthesis_lifecycle.py -v`

**Step 5:** Commit.

```bash
git add -A
git commit -m "refactor(synthesis): move lifecycle, rename agent prefix to lucid-synthesis-

Adds backcompat archival of legacy lucid-orchestrator-* agents so a
deployed orchestrator agent from a prior version is cleaned up on
the next run's prune pass."
```

## Task 3.4 — Read-only `ToolRegistry` for synthesis

**Files:**
- Create: `lucid/synthesis/tools.py` — new `build_synthesis_registry(...)` that exposes only the read-only tools + `write_report_section` (stub for now)
- Modify: `lucid/orchestrator/tools.py` — keep existing `build_tool_registry` for scoring spend-tracker context; both live in peace

**Step 1: Write the failing test**

```python
def test_synthesis_registry_exposes_only_read_only_tools():
    from lucid.synthesis.tools import build_synthesis_registry
    registry = build_synthesis_registry(
        store=mock_store,
        audit_run_id="run-abc",
        progress_log=lambda *_: None,
    )
    names = set(registry.names)
    assert names == {
        "query_corpus",
        "get_conversation",
        "get_turn_window",
        "get_findings",
        "log_progress",
        "write_report_section",
    }
    # Specifically: no invoke_module, no store_finding, no estimate_remaining_cost
    assert "invoke_module" not in names
    assert "store_finding" not in names
```

**Step 2: Run to verify failure.**

**Step 3: Implement.**

In `lucid/synthesis/tools.py`, define `build_synthesis_registry(...)` that:
- Reuses the read-only handlers from `lucid/orchestrator/tools.py` (import them; they become effectively "shared infrastructure"). Later we'll move those imports up to `lucid/synthesis/tools.py` directly.
- Adds a stub `write_report_section` handler — implementation lands in Phase 5 Task 5.2.

```python
from lucid.orchestrator.tools import (  # shared handlers
    ToolRegistry,
    CustomTool,
    # the closures for query_corpus / get_conversation / get_turn_window /
    # get_findings / log_progress live inside build_tool_registry today;
    # extract them into module-level factory functions in a prerequisite
    # refactor or inline-copy them here.
)
```

Prerequisite refactor: to share the handlers cleanly, extract each inline closure from `lucid/orchestrator/tools.py::build_tool_registry` into a module-level factory — e.g., `def make_query_corpus_tool(store) -> CustomTool: ...`. Do this as part of Task 3.4.

**Step 4: Run tests.**

`uv run pytest tests/test_synthesis_tools.py -v`
Expected: PASS. Existing orchestrator tests still green.

**Step 5: Commit.**

```bash
git add -A
git commit -m "feat(synthesis): add build_synthesis_registry with read-only + write_report_section stub"
```

## Task 3.5 — `SynthesisSession` class (mirror of the deleted ManagedAgentsSession)

**Files:**
- Modify: `lucid/synthesis/session.py`
- Test: `tests/test_synthesis_session.py`

**Step 1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_synthesis_session_runs_to_completion(fake_managed_agents_client):
    """A fake client that streams a simulated session should round-trip
    through SynthesisSession.run without errors."""
    ...

@pytest.mark.asyncio
async def test_synthesis_session_cleans_up_ephemeral_env_session(fake_client):
    """After run(), the environment + session are deleted unless keep_ephemeral=True."""
    ...

@pytest.mark.asyncio
async def test_synthesis_session_stall_watchdog_fires(fake_stalling_client):
    """If the stream goes silent past heartbeat_stall_seconds, the session
    unwinds as partial."""
    ...
```

**Step 2: Implement `SynthesisSession`**

Copy the structure from the deleted `ManagedAgentsSession`. Simplify:

- Drop `continuation_nudges` (always off — the state machine rejected them anyway).
- Drop `keep_ephemeral` unless needed for debugging (default off).
- Drop the synth agent's `PROMPT_VERSION` import from the session; pass it in via `SynthesisConfig.prompt_version` instead.

```python
@dataclass(frozen=True)
class SynthesisConfig:
    run_id: str
    model: str = "claude-opus-4-7"
    prompt_version: str  # e.g. "v1"
    system_prompt: str
    session_timeout_seconds: float = 3600.0
    heartbeat_stall_seconds: float = 60.0
    heartbeat_check_interval_seconds: float = 5.0
    keep_ephemeral: bool = False

class SynthesisSession:
    def __init__(self, *, client, registry, config):
        ...

    async def run(self, kickoff_message: str) -> SynthesisOutcome:
        ...
```

Where `SynthesisOutcome` carries: `completed`, `reason`, `sections_written`, `sections_declined`, `events_received`, `tool_calls`, `cache_read_tokens`, `cache_write_tokens`, `diagnostics`.

**Step 3: Run tests**

`uv run pytest tests/test_synthesis_session.py -v`
Expected: all PASS.

**Step 4: Commit**

```bash
git add lucid/synthesis/session.py tests/test_synthesis_session.py
git commit -m "feat(synthesis): add SynthesisSession class (Managed Agents driver)"
```

## Task 3.6 — Phase 3 checkpoint

**Step 1:** Full suite green.
**Step 2:** Package-level import sanity: `python -c "from lucid.synthesis import SynthesisSession, build_synthesis_registry"` returns 0.
**Step 3:** Tag.
```bash
git tag phase-3-synthesis-scaffold
```

---

# Phase 4 — Synthesis schemas, prompts, output contracts

**Goal:** Lock the agent's output contract and write v1 prompts for both the Opus writer and the Sonnet post-processor.

## Task 4.1 — Pydantic models for synthesis outputs

**Files:**
- Modify: `lucid/schemas.py` — add `SynthesisSectionOutput`, `SynthesisUnsupportedSection`, `SynthesisSectionError`, `SynthesisBlock`
- Test: `tests/test_schemas.py`

**Step 1: Write failing tests** (round-trip + validation-failure cases)

**Step 2: Implement**

```python
class SynthesisBlock(BaseModel):
    """One coherent block of narrative prose, parsed by the Sonnet
    post-processor into a claim-plus-citations unit."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000)
    citations: list[str] = Field(default_factory=list)  # finding_ids or turn_ids
    aggregate_claim: str | None = None  # e.g. "across 42 conversations"
    aggregate_support: int | None = None  # validated integer matching the claim


class SynthesisSectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(min_length=1, max_length=64)
    blocks: list[SynthesisBlock] = Field(default_factory=list)
    raw_markdown: str = Field(default="")   # pre-post-process Opus output
    citation_confidence: float = Field(ge=0.0, le=1.0)
    cited_finding_ids: list[str] = Field(default_factory=list)
    cited_turn_ids: list[str] = Field(default_factory=list)


class SynthesisUnsupportedSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str
    reason: Literal["insufficient_evidence", "retrieval_failure", "generation_error"]
    decline_message: str = Field(min_length=1, max_length=500)
    retrieval_top_similarity: float | None = None
    retrieval_excerpt_count: int | None = None


class SynthesisSectionError(BaseModel):
    """Mirror of lucid.modules.base.ModuleError for synthesis-phase failures."""
    model_config = ConfigDict(extra="forbid")

    section_id: str
    error_type: str   # e.g. "parse_failed", "validation_failed"
    message: str = Field(max_length=500)


SynthesisSectionResult = SynthesisSectionOutput | SynthesisUnsupportedSection | SynthesisSectionError
```

**Step 3: Tests green, mypy strict green.**

**Step 4: Commit.**

```bash
git add lucid/schemas.py tests/test_schemas.py
git commit -m "feat(schemas): add SynthesisSectionOutput / Unsupported / Error models"
```

## Task 4.2 — Opus writer prompt v1

**Files:**
- Create: `prompts/synthesis/v1.md`

**Step 1: Write the prompt**

Follow the repo's existing prompt frontmatter conventions (from `prompts/module_a/v1.md` or similar). Required frontmatter: `version`, `model: claude-opus-4-7`, `thinking_mode: adaptive`, `effort: high`, `citation: "Lucid methodology §N"`, `purpose`, `hash: (sha256 of body)`.

Prompt body must:

1. **Open with role**: *"You are the synthesis agent for a Lucid epistemic audit..."*
2. **Enumerate the sections it will write** (exec_summary, top_3_actions, module_X_narrative for X in enabled modules, memory_audit_summary).
3. **State the output format**: Markdown prose with inline `[F:finding_id]` for finding citations and `[T:turn_id]` for direct turn quotes. NO JSON during generation.
4. **Enforce the three failure-mode guards**:
   - **Uncited high-intensity**: *"Every finding with intensity >= 2 must appear in at least one section. If you cannot ground one, list it in the `uncited_high_intensity` section instead."*
   - **Aggregate lockdown**: *"Never write phrases like 'across N conversations' or 'N% of sessions' unless you have just received a tool result containing that exact count. If you need an aggregate, call `query_corpus` or `get_findings` first."*
   - **Thin-evidence hedging**: The system prompt injects `{behavior_label: count}` map; instruct the agent: *"For any behavior label where count < 5, use language like 'in a handful of cases' rather than 'consistently'."*
5. **Include the `INSUFFICIENT_EVIDENCE` escape hatch**: *"If you cannot ground a section (fewer than 3 qualifying findings), do not write filler prose. Instead, invoke `write_report_section(section_id, insufficient_evidence=True, decline_reason=\"...\")` and move to the next section."*
6. **Citation rule**: *"You may cite only finding_ids present in the FINDINGS table below, and only turn_ids that you have fetched via `get_turn_window` in the current session. Citations outside these sets will be rejected at validation."*
7. **Output size**: Soft target 150-300 words per section; hard cap 500.
8. **Padding to ≥ 4096 tokens** for Opus cache activation (use a deterministic padding block as other prompts do — see `PromptFile.padded_body` usage in existing modules).

Frontmatter `hash` is computed from the body; the loader recomputes and validates.

**Step 2: Add the prompt's SHA to a manifest**

If the repo has a `prompts/MANIFEST.toml` or similar, add an entry. Otherwise ensure `lucid.prompts.load_prompt("synthesis", "v1")` resolves.

**Step 3: Write a schema-compliance test**

```python
def test_synthesis_v1_prompt_has_required_frontmatter():
    prompt = load_prompt("synthesis", "v1")
    assert prompt.model == "claude-opus-4-7"
    assert prompt.thinking_mode in {"disabled", "adaptive"}
    assert prompt.effort in {"low", "medium", "high", "xhigh", "max"}
    assert "effort" in prompt.frontmatter  # not temperature (Opus 4.7 rejects it)
    assert "temperature" not in prompt.frontmatter
    # Padded body meets cache minimum.
    body = prompt.padded_body
    assert estimate_tokens(body) >= 4096
```

**Step 4: Commit.**

```bash
git add prompts/synthesis/v1.md tests/test_prompts.py
git commit -m "feat(prompts): synthesis v1 — Opus 4.7 writer prompt with failure-mode guards"
```

## Task 4.3 — Sonnet post-processor prompt v1

**Files:**
- Create: `prompts/synthesis_validator/v1.md`

**Step 1:** Write a Sonnet 4.6 prompt that takes the Opus markdown output and one target section_id, and emits a `SynthesisSectionOutput` JSON conforming to the Pydantic schema. Temperature 0.0; `messages.parse()` with the schema.

Body key points:
- **Input**: the Opus section markdown + the list of valid finding_ids and turn_ids for the run.
- **Output contract**: JSON matching `SynthesisSectionOutput`. Each block pairs one coherent sentence (or 2-3 closely bound sentences) with its citations extracted from `[F:...]` / `[T:...]` tokens.
- **Rules**:
  - Drop any `[F:x]` where `x` is not in the valid finding_id set — append it to `_dropped_citations` metadata (schema extension: optional `dropped_citations: list[str]`).
  - For every `aggregate_claim` string (e.g. "across 42 conversations"), fill `aggregate_support` with the integer. The runtime validator checks this against the DB.
  - Preserve prose verbatim — do NOT rewrite.

**Step 2:** Schema-compliance test analogous to 4.2.

**Step 3:** Commit.

```bash
git add prompts/synthesis_validator/v1.md tests/test_prompts.py
git commit -m "feat(prompts): synthesis_validator v1 — Sonnet 4.6 structure pass"
```

## Task 4.4 — Phase 4 checkpoint

Full suite green; tag `phase-4-schemas-prompts`.

---

# Phase 5 — Synthesis execution: two-phase write + validators

**Goal:** Wire the session to the prompts, implement the phase-1 Opus write + phase-2 Sonnet structure pass + DB validator, and persist results to `report_sections`.

## Task 5.1 — `write_report_section` tool handler (the real one)

**Files:**
- Modify: `lucid/synthesis/tools.py`

**Step 1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_write_report_section_persists_to_db(tmp_path):
    """Calling the tool with valid markdown + citations persists a ReportSection row."""
    ...

@pytest.mark.asyncio
async def test_write_report_section_validates_citations_against_db():
    """Citations referencing unknown finding_ids cause the handler to return
    {"error": "unknown_finding_ids", "unknown": [...]} WITHOUT persisting."""
    ...

@pytest.mark.asyncio
async def test_write_report_section_insufficient_evidence_path():
    """When insufficient_evidence=True + decline_reason supplied, persists
    a ReportSection with empty markdown + the reason, citation lists empty."""
    ...
```

**Step 2: Implement the handler.**

Pseudocode:

```python
async def write_report_section(args: dict[str, Any]) -> dict[str, Any]:
    section_id = args["section_id"]
    insufficient = bool(args.get("insufficient_evidence", False))
    decline_reason = args.get("decline_reason")
    markdown = str(args.get("markdown") or "")
    cited_finding_ids = list(args.get("cited_finding_ids") or [])
    cited_turn_ids = list(args.get("cited_turn_ids") or [])

    if insufficient:
        if not decline_reason:
            return {"error": "missing_decline_reason"}
        section = ReportSection(
            audit_run_id=audit_run_id,
            section_id=section_id,
            markdown="",
            cited_finding_ids=[],
            cited_turn_ids=[],
            insufficient_evidence=True,
            decline_reason=decline_reason,
            created_at=_now(),
        )
        store.upsert_report_section(section)
        return {"ok": True, "insufficient_evidence": True}

    # Validate citations against DB before persist.
    unknown_findings = _validate_finding_ids(store, audit_run_id, cited_finding_ids)
    unknown_turns = _validate_turn_ids(store, cited_turn_ids)
    if unknown_findings or unknown_turns:
        return {
            "error": "unknown_ids",
            "unknown_finding_ids": unknown_findings,
            "unknown_turn_ids": unknown_turns,
        }

    section = ReportSection(
        audit_run_id=audit_run_id,
        section_id=section_id,
        markdown=markdown,
        cited_finding_ids=cited_finding_ids,
        cited_turn_ids=cited_turn_ids,
        insufficient_evidence=False,
        decline_reason=None,
        created_at=_now(),
    )
    store.upsert_report_section(section)
    return {"ok": True, "section_id": section_id}
```

**Step 3: Register the tool with proper input_schema (no `additionalProperties` per CLAUDE.md).**

**Step 4: Tests green. Commit.**

```bash
git add lucid/synthesis/tools.py tests/test_synthesis_tools.py
git commit -m "feat(synthesis): implement write_report_section tool with citation validation"
```

## Task 5.2 — Aggregate-claim lockdown validator

**Files:**
- Create: `lucid/synthesis/validators.py`
- Test: `tests/test_synthesis_validators.py`

**Step 1: Write tests for `validate_aggregate_claims(prose, tool_history) -> list[ValidationError]`**

Cases:
- Prose contains "across 40 conversations" and session history includes a `query_corpus` call returning `count=40` → PASS.
- Prose contains "across 40 conversations" and session history contains no such count → FAIL with `error="aggregate_unsupported"`.
- Prose contains "across 100 sessions" and session history says `count=50` → FAIL with `error="aggregate_mismatch"`.
- Prose contains "sometimes" (hedged, non-aggregate) → PASS.

**Step 2: Implement**

```python
_AGGREGATE_RE = re.compile(r"\b(\d+)\s+(conversations?|sessions?|findings?|turns?)\b", re.I)

def validate_aggregate_claims(
    prose: str, tool_history: list[ToolCallRecord]
) -> list[ValidationError]:
    """Return errors for any '\d+ <noun>' claim not backed by a tool call result."""
    errors = []
    for m in _AGGREGATE_RE.finditer(prose):
        claim_n = int(m.group(1))
        noun = m.group(2).rstrip("s").lower()
        observed = _find_tool_count(tool_history, noun)
        if observed is None:
            errors.append(ValidationError("aggregate_unsupported", m.group(0)))
        elif observed != claim_n:
            errors.append(
                ValidationError("aggregate_mismatch", f"{m.group(0)} (actual {observed})")
            )
    return errors
```

**Step 3: Add `validate_superlatives(prose, behavior_counts) -> list[ValidationError]`**

Cases:
- "consistently" + behavior count < 5 → FAIL.
- "occasionally" + any count → PASS.
- No superlatives → PASS.

**Step 4: Add `validate_uncited_high_intensity(findings, cited_ids) -> list[Finding]`**

Cases:
- Finding with intensity=3 NOT in any `cited_finding_ids` across all report_sections → returned.
- Finding with intensity=1 → not returned regardless.

**Step 5: Tests green. Commit.**

```bash
git add lucid/synthesis/validators.py tests/test_synthesis_validators.py
git commit -m "feat(synthesis): aggregate-claim + superlative + uncited-high-intensity validators"
```

## Task 5.3 — `run_synthesis_session` orchestration function

**Files:**
- Modify: `lucid/synthesis/session.py` or create `lucid/synthesis/run.py`
- Test: `tests/test_synthesis_run.py`

**Step 1: Write failing tests (mocked clients)**

```python
@pytest.mark.asyncio
async def test_run_synthesis_session_writes_all_sections(mock_clients, fixture_findings):
    """Given a fake Managed Agents client that streams section writes for
    exec_summary + 3 module narratives, run_synthesis_session persists
    4 ReportSection rows, none flagged insufficient."""

@pytest.mark.asyncio
async def test_run_synthesis_session_honors_insufficient_evidence():
    """An agent declining one section persists a row with insufficient_evidence=True."""

@pytest.mark.asyncio
async def test_run_synthesis_session_runs_sonnet_post_process():
    """After Opus writes markdown, Sonnet post-process call is made per section
    and the resulting structured blocks populate ReportSection.cited_finding_ids."""
```

**Step 2: Implement**

```python
async def run_synthesis_session(
    *,
    async_client: AsyncAnthropic,
    sync_client,             # Managed Agents SDK is sync at the events layer
    store: CorpusStore,
    audit_run_id: str,
    enabled_modules: list[ModuleName],
    progress_log: Callable[[str, str], None],
) -> SynthesisOutcome:
    """Execute the synthesis agent session for an already-scored run.

    1. Builds the read-only + write_report_section tool registry.
    2. Constructs the kickoff message with section list + behavior counts.
    3. Runs the Managed Agents session (Opus 4.7).
    4. For each section the agent wrote (markdown form), runs a
       post-process Sonnet 4.6 call to structure it into
       SynthesisSectionOutput blocks.
    5. Applies validators (aggregate lockdown, superlative check).
    6. On validator failure: drop the section and log.
    7. Finally, runs the uncited-high-intensity audit and appends a
       "Notable uncited findings" section if any findings were missed.
    """
    ...
```

**Step 3: Integrate into `run_audit`**

In `lucid/run.py`, after `_run_scoring_loop` completes, and before `_render_report_or_log`, add:

```python
if synthesis_enabled:
    await run_synthesis_session(
        async_client=async_client,
        sync_client=client,
        store=store,
        audit_run_id=resolved_run_id,
        enabled_modules=inputs.enabled_modules,
        progress_log=progress_log,
    )
```

Gate on a new CLI flag `--no-synthesis` (default: synthesis ON). Add the flag in `lucid/cli.py`.

**Step 4: Tests green. Commit.**

```bash
git add lucid/synthesis/ lucid/run.py lucid/cli.py tests/
git commit -m "feat(synthesis): run_synthesis_session wires Opus writer + Sonnet post-process

Called after the scoring loop. Produces ReportSection rows for every
enabled section, with per-section error isolation and
insufficient_evidence decline support."
```

## Task 5.4 — Failure-mode guard: regen loop for invalid citations

**Files:**
- Modify: `lucid/synthesis/session.py` — if `write_report_section` returns `{"error": "unknown_ids"}`, the session captures it and re-prompts the agent once (max 2 attempts) with an explicit list of the invalid ids.

This is parallelizable with 5.3 but easier to do after 5.3 exists. Test with a fake client that emits invalid ids on first try and valid ids on retry.

Commit:

```bash
git commit -m "feat(synthesis): single-retry regen loop when write_report_section rejects citations"
```

## Task 5.5 — Phase 5 checkpoint

**Step 1:** End-to-end smoke with real API against `tests/fixtures/claude-ai` + 2 conversations.

```bash
export ANTHROPIC_API_KEY=...
uv run lucid audit --source claude-ai --path tests/fixtures/claude-ai --sample 2 \
    --yes-i-authorize-spend-up-to 5
```

Check the resulting SQLite: `sqlite3 data/lucid.sqlite3 'SELECT section_id, length(markdown), insufficient_evidence FROM report_sections'`.

Expected: at least `exec_summary` has markdown; sections are populated; cache metrics logged.

**Step 2:** Tag `phase-5-synthesis-live`.

---

# Phase 6 — Report template integration

**Goal:** Inline the agent-written sections into `report.html.j2`, degrade gracefully when empty, strip redundant helpers.

## Task 6.1 — Thread `report_sections` into `ReportContext`

**Files:**
- Modify: `lucid/report/generator.py`
  - Extend `ReportContext` (dataclass or dict) with `report_sections: dict[str, ReportSection]`.
  - `write_report(audit_run, findings, ...)` now also fetches `store.fetch_report_sections_for_run(audit_run.id)`, keys by `section_id`, passes to the template.
- Test: `tests/test_report_generator.py` — assert rendered HTML contains agent prose when `report_sections` is populated.

**Step 1:** Failing test (expects template output to contain prose when sections provided).
**Step 2:** Implement.
**Step 3:** Commit.

## Task 6.2 — Template hook for `exec_summary` (Checkpoint A)

**Files:**
- Modify: `lucid/report/templates/report.html.j2` — where `actionable-summary` section renders, add a conditional: if `report_sections.exec_summary` present, render its markdown (via a safe `markdown` filter) as the hero paragraph.

```jinja
{% if report_sections.exec_summary and not report_sections.exec_summary.insufficient_evidence %}
  <section class="exec-summary agent-prose" aria-labelledby="exec-h">
    <p class="eyebrow" id="exec-h">Executive summary</p>
    <h2 class="serif">What this audit found</h2>
    <div class="agent-markdown">
      {{ report_sections.exec_summary.markdown | markdown_with_citations | safe }}
    </div>
  </section>
{% elif report_sections.exec_summary and report_sections.exec_summary.insufficient_evidence %}
  <section class="exec-summary agent-declined" aria-labelledby="exec-h">
    <p class="eyebrow">Executive summary</p>
    <p class="muted">Section skipped: {{ report_sections.exec_summary.decline_reason }}</p>
  </section>
{% endif %}
```

Implement `markdown_with_citations` as a Jinja2 filter in `generator.py`:
- Converts `[F:abc123]` into `<a href="#finding-abc123" class="citation">[F]</a>`.
- Converts `[T:xyz]` into `<a href="#turn-xyz" class="citation">[T]</a>`.
- Renders other markdown via existing markdown library.

**Step 1-5:** Failing test → implement filter → render → commit.

```bash
git commit -m "feat(report): template hook + markdown_with_citations filter for exec_summary"
```

At this point **Checkpoint A (minimum shippable subset) is met.** Tag:

```bash
git tag checkpoint-A-synthesis-single-section
```

## Task 6.3 — Template hooks for per-module narratives + top_3_actions + headline_findings

**Files:**
- Modify: `lucid/report/templates/report.html.j2` — similar conditional blocks for each agent-driven section_id.

For each module section, replace the existing `<p>` where the top interpretations render with the agent's `module_<X>_narrative` prose, falling back to the existing templated version when `report_sections` is empty.

**Step 1: Identify every target section in `report.html.j2` (from Phase 0 research)**: `exec_summary` (done in 6.2), `top_3_actions`, `headline_findings`, `module_A_narrative` through `module_H_narrative` (skip G).

**Step 2: Add conditional blocks for each.**

**Step 3: Visual diff pre/post**: render the sample audit, open in browser, compare to `report/run-9b7031f168cf.html`.

**Step 4: Commit.**

```bash
git add lucid/report/templates/ lucid/report/generator.py
git commit -m "feat(report): template hooks for all agent-driven sections (hybrid per-module + headlines)"
```

## Task 6.4 — Remove redundant `_compute_top_actions` + `_headline_findings`

**Files:**
- Modify: `lucid/report/generator.py`

Delete `_compute_top_actions` and `_headline_findings` derivation code. Ensure the template's fallback (non-agent) branches still render something sensible — e.g., the "At a glance" module bars + top 3 findings by intensity without the fancy recommendation prose.

**Step 1:** Identify fallback rendering for the templated branches (when `report_sections` is empty). These were the old output shape; leave them intact.

**Step 2:** Delete the derivation code only; keep the dependent `aggregate` fields or return stubs if they're referenced by the fallback template.

**Step 3:** Commit.

```bash
git commit -m "chore(report): drop _compute_top_actions / _headline_findings derivation; agent drives these sections now"
```

## Task 6.5 — Demo renderer graceful degradation

**Files:**
- Modify: `demo/render_demo_report.py` — already doesn't write `report_sections`, so template degrades automatically. Add a footer note to the demo output: *"This demo report omits agent-written narrative (exec summary, per-module prose, headlines). In a production `lucid audit` run those sections are written by the synthesis agent and validated for citation grounding."*

Add this as a template partial or a demo-specific Jinja variable.

**Step 1-3:** Trivial change + test render + commit.

## Task 6.6 — Phase 6 checkpoint

**Step 1:** Visual diff check. Open the pre-refactor + post-refactor reports side-by-side in a browser.

**Step 2:** Smoke run → inspect HTML.

**Step 3:** Tag `phase-6-report-integration`.

---

# Phase 7 — Docs + CHANGELOG + smoke test

**Goal:** Don't ship silent changes. Update operator-facing docs and do one full end-to-end verification.

## Task 7.1 — Update CLAUDE.md + README

**Files:**
- Modify: `CLAUDE.md` — new section `## Synthesis session conventions` documenting the two-phase write, the citation contract, the validators, and the `PROMPT_VERSION` key.
- Modify: `CLAUDE.md` — remove / rewrite the `## Managed Agents conventions` section to describe the new split (scoring=code, synthesis=agent).
- Modify: `README.md` — update "Status" line (line ~17-23) to describe the synthesis layer; update the architecture bullet.

**Step 1-3:** Write → review diff → commit.

```bash
git commit -m "docs: document synthesis session, update README to reflect scoring/synthesis split"
```

## Task 7.2 — CHANGELOG + runbook note

**Files:**
- Modify: `CHANGELOG.md` if exists, else create one.

Entry should call out: breaking change to `run_audit(...)` signature; new `--no-synthesis` flag; new `report_sections` table; agent naming migrated from `lucid-orchestrator-*` to `lucid-synthesis-*` (existing agents auto-pruned on next run).

**Step 1-2:** Write → commit.

## Task 7.3 — Smoke run — end-to-end on real corpus sample

```bash
uv run lucid audit --source claude-ai --path ~/lucid-export --sample 10 \
    --yes-i-authorize-spend-up-to 20 --include-module-d
```

Expected: completes; `report/run-<id>.html` opens in browser; agent prose visible in exec_summary + module narratives; citations hoverable/clickable.

Examine `report_sections` in SQLite:

```bash
sqlite3 data/lucid.sqlite3 "SELECT section_id, length(markdown), insufficient_evidence, decline_reason FROM report_sections WHERE audit_run_id='run-...'"
```

Expected: all target sections present; 0-2 may be `insufficient_evidence`; markdown lengths 150-500 chars for normal sections.

Verify `cache_read_input_tokens > 0` in logs (prompt caching active).

## Task 7.4 — Phase 7 checkpoint

Tag `phase-7-docs-and-smoke`. At this point the full feature ships.

---

# Phase 8 — Stretch: `lucid ask` interactive mode

**Goal:** User-driven investigation on a completed audit. Same tools as the synthesis session, no `write_report_section`, streams agent response to terminal.

## Task 8.1 — CLI command

`lucid ask <run-id> "why did module B flag conversation X?"` — resolve run by id prefix, load read-only registry (no `write_report_section`), spin up a `SynthesisSession` variant or a dedicated `InvestigationSession` with a different system prompt.

## Task 8.2 — Prompt `prompts/investigation/v1.md`

Short prompt: "You are investigating a completed Lucid audit. Answer the user's question using only the findings + corpus tools available. Cite every factual claim with [F:id] or [T:id]."

## Task 8.3 — Streaming output

Implement a minimal stream handler that prints agent text to stdout as it arrives.

## Task 8.4 — Commit + tag

```bash
git tag phase-8-lucid-ask
```

---

# Execution checklist (quick reference)

- [ ] Phase 1.1: `ReportSection` model
- [ ] Phase 1.2: `report_sections` table
- [ ] Phase 1.3: CRUD helpers
- [ ] Phase 1.4: checkpoint + tag
- [ ] Phase 2.1: extract `invoke_module_for_run`
- [ ] Phase 2.2: `_run_scoring_loop`
- [ ] Phase 2.3: rewire `run_audit`
- [ ] Phase 2.4: delete dead orchestrator files
- [ ] Phase 2.5: checkpoint + tag
- [ ] Phase 3.1: scaffold `lucid/synthesis/`
- [ ] Phase 3.2: move `handler.py`
- [ ] Phase 3.3: move + rename `lifecycle.py`
- [ ] Phase 3.4: `build_synthesis_registry`
- [ ] Phase 3.5: `SynthesisSession` class
- [ ] Phase 3.6: checkpoint + tag
- [ ] Phase 4.1: synthesis Pydantic models
- [ ] Phase 4.2: Opus prompt v1
- [ ] Phase 4.3: Sonnet validator prompt v1
- [ ] Phase 4.4: checkpoint + tag
- [ ] Phase 5.1: `write_report_section` handler
- [ ] Phase 5.2: aggregate/superlative/uncited validators
- [ ] Phase 5.3: `run_synthesis_session`
- [ ] Phase 5.4: regen loop
- [ ] Phase 5.5: checkpoint + tag
- [ ] Phase 6.1: `report_sections` in `ReportContext`
- [ ] Phase 6.2: `exec_summary` template hook → **Checkpoint A shippable**
- [ ] Phase 6.3: remaining template hooks
- [ ] Phase 6.4: drop redundant helpers
- [ ] Phase 6.5: demo renderer note
- [ ] Phase 6.6: checkpoint + tag → **Checkpoint B shippable**
- [ ] Phase 7.1: CLAUDE.md + README
- [ ] Phase 7.2: CHANGELOG
- [ ] Phase 7.3: smoke run
- [ ] Phase 7.4: final tag
- [ ] Phase 8 (optional): `lucid ask` → **Checkpoint C shippable**

---

# Glossary / conventions

- **Scoring phase**: deterministic Python loop invoking modules A-H. Produces `Finding` rows. No LLM reasoning at the routing layer.
- **Synthesis phase**: Managed Agents session invoked after scoring completes. Produces `ReportSection` rows with grounded narrative.
- **Citation token**: `[F:finding_id]` or `[T:turn_id]` in agent markdown. Validated at tool-call time + render time.
- **`INSUFFICIENT_EVIDENCE`**: explicit agent decline. Persists a row with `insufficient_evidence=True` + `decline_reason`. Template renders "section skipped".
- **Two-phase write**: Opus writes markdown naturally; Sonnet 4.6 post-processes into `SynthesisSectionOutput` JSON via `messages.parse()`.
- **Failure-mode guards**: aggregate-claim lockdown, superlative hedging, uncited-high-intensity audit. All run post-generation, before `ReportSection` row is persisted.

---

*End of plan.*
