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
