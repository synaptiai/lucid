# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Synthesis phase** — agent-driven narrative writer for report
  sections. Claude Opus 4.7 writes `exec_summary`, `top_3_actions`,
  `headline_findings`, and per-module narratives (A, B, C, D, E, F, H)
  with inline `[F:finding_id]` / `[T:turn_id]` citation tokens. Every
  citation validated against the DB; unknown ids trigger regen (cap: 2
  retries per section).
- **Sonnet 4.6 post-processor** — after Opus writes each section,
  `lucid/synthesis/post_process.py` runs Sonnet via
  `AsyncAnthropic.messages.parse()` with `SynthesisSectionOutput` as the
  `output_format` schema. Extracts per-claim blocks + a 0.0–1.0
  `citation_confidence` score and upserts them onto the `ReportSection`
  row. Sections run concurrently via `asyncio.gather`; per-section
  failures degrade gracefully (section unchanged, warning logged).
- **`ReportSection.blocks` + `citation_confidence`** fields for the
  Sonnet output. New SQL columns `blocks_json` (DEFAULT `'[]'`) and
  `citation_confidence` (REAL, nullable, CHECK `0.0-1.0`) added via
  migration `m_003.sql`; `SCHEMA_VERSION` bumped to 3.
- **Synthesis writer prompt v2** (`prompts/synthesis/v2.md`) replacing
  v1's soft framing with an explicit **Execution protocol** section
  enumerating the turn-by-turn tool-call sequence, plus an opening
  directive forbidding prose text between tool calls. v1 preserved
  (prompts are frozen once shipped).
- **Event-type diagnostic logging** in `SynthesisSession.run()` — every
  stream event type is logged at INFO; `agent.message` / `agent.text`
  events fire a WARNING with a 80-char preview for post-mortems.
  Closes with a `synthesis outcome: ...` summary line.
- **`report_sections` table** — new SQLite table persisting agent prose
  keyed by `(audit_run_id, section_id)` with upsert semantics. CHECK
  constraint enforces the `insufficient_evidence` ↔ empty-markdown
  invariant.
- **`markdown_with_citations` Jinja filter** — escapes HTML then
  converts `[F:id]` / `[T:id]` tokens to anchor links. Paragraph
  breaks on blank lines.
- **Three post-generation validators** — `validate_aggregate_claims`
  (enforces N-count aggregate claims match actual tool-call results),
  `validate_superlatives` (hedges "consistently"/"frequently" when
  behavior count < 5), `validate_uncited_high_intensity` (surfaces
  intensity >= 2 findings not cited in any section).
- **`--no-synthesis` CLI flag** — opt out of the synthesis phase;
  report still renders with deterministic scaffolding.
- **`SYNTHESIS_PROMPT_VERSION`** constant in `lucid.synthesis` for
  agent-naming + cache-key stability across prompt iterations.
- **Backward-compat agent pruning** — `prune_stale_synthesis_agents`
  archives both `lucid-synthesis-*` (current) and `lucid-orchestrator-*`
  (legacy) agents so existing consoles stay clean.

### Changed

- **BREAKING**: `run_audit()` signature. Removed `system_prompt`,
  `kickoff_message`, `session_runner` kwargs. Added
  `synthesis_enabled: bool = True` kwarg. `client: Anthropic` kwarg
  remains for synthesis's Managed Agents surface (sync SDK is required
  for `beta.agents` lifecycle calls).
- **BREAKING**: `AuditResult.outcome` field removed. Replaced by
  `status` + `reason` on the result + cache/token counters in the
  underlying `SynthesisOutcome` during the synthesis phase.
- **Agent naming migrated** from `lucid-orchestrator-v*` to
  `lucid-synthesis-v*`. Legacy agents on the Anthropic console are
  archived automatically on the next `lucid cleanup-agents` pass.
- `_execute_scoring` now treats `skipped`, `no_embedding_provider`,
  and `no_client` as successful intentional gates (not failures).
  A user running an audit without `VOYAGE_API_KEY` now exits 0, not 2.
- **`SynthesisSession` event-loop termination** — `session.status_idle`
  is no longer treated as terminal. It fires between every agent turn
  while the server waits for `user.custom_tool_result`; only
  `session.finished` ends the loop now. Caught by live run
  `run-0cc74fc5cdd2` where the session was cut off after the first
  agent turn's idle cycle. Genuine hangs are caught by the stall
  watchdog (300s default).
- **`_iter_stream` offloaded to `asyncio.to_thread`** — the SDK's sync
  stream iterator previously blocked the event loop in C-land between
  events, preventing the stall watchdog from ever running. The sync
  `next()` now runs in a thread pool so the async loop stays
  schedulable. A sentinel bridges `StopIteration` through
  `asyncio.to_thread`'s RuntimeError coercion.
- **Stall threshold raised 60s → 300s** (`SynthesisConfig.heartbeat_stall_seconds`).
  Opus 4.7 `effort=high` turns can legitimately take 2-3 minutes of
  quiet stream time mid-generation (verified in `run-d049b5ac04df`).
  Check cadence coarsened 5s → 10s to match.
- `SYNTHESIS_PROMPT_VERSION` bumped `v1` → `v2`. `lucid/synthesis/run.py`
  now loads via the constant (previously hardcoded `"v1"` — fixed as
  part of the v2 bump).

### Removed

- `lucid/orchestrator/system_prompt.py` (`SYSTEM_PROMPT`,
  `PROMPT_VERSION` constants).
- `lucid/orchestrator/managed_agent.py` (`ManagedAgentsSession`,
  `OrchestratorConfig`, `SessionOutcome`, `SessionHandles`,
  `MANAGED_AGENTS_BETA_HEADER`). Successor:
  `lucid/synthesis/session.py::SynthesisSession`.
- `_execute_session_and_safety_net`, `_backfill_unfinished_modules`,
  `_attribution_safety_net`, `_default_session_runner` from `lucid/run.py`.
  The deterministic `_run_scoring_loop` replaces all four.
- `tests/test_orchestrator_managed_agent.py` (16 tests). Coverage
  for session-style lifecycle now lives in `tests/test_synthesis_session.py`.
- `tests/test_run_audit_live_path.py` (10 tests). The legacy
  session-based audit path it tested no longer exists.

### Deprecated

Nothing — all removed items are gone outright, not shadowed with
deprecation shims.

### Migration notes

Downstream callers of `run_audit` must:
1. Remove `system_prompt=`, `kickoff_message=`, `session_runner=` kwargs.
2. Add `synthesis_enabled=False` (to preserve old behavior) or `True`
   (to opt into the new narrative phase — requires a sync Anthropic
   client passed via `client=`).
3. Stop reading `AuditResult.outcome` — the field is gone. Use
   `result.status` + `result.reason` instead.

Existing `lucid-orchestrator-v*` agents on the Anthropic console will
be archived automatically on the next `lucid cleanup-agents` run.
No manual cleanup required.
