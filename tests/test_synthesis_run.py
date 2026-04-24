"""Tests for :func:`lucid.synthesis.run.run_synthesis_session` orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from lucid.synthesis.run import (
    _build_kickoff_message,
    _target_section_ids,
    run_synthesis_session,
)


def _seed_audit_run_inline(store, run_id: str) -> None:
    """Minimal audit-run seed for synthesis run tests.

    Duplicated from tests/test_synthesis_tools.py rather than imported
    (cross-file import in test collection has subtle ordering effects;
    a tiny duplication is cheaper than coupling).
    """
    from lucid.schemas import (
        AuditRun,
        CorpusStats,
        SamplingConfigRecord,
        Source,
        TokenUsage,
    )
    from lucid.store import SCHEMA_VERSION

    run = AuditRun(
        id=run_id,
        sources=[Source.CLAUDE_AI],
        source_paths={Source.CLAUDE_AI: "/tmp/export"},
        started_at=datetime(2026, 4, 21, 12, 0, 0, tzinfo=UTC),
        completed_at=None,
        corpus_stats=CorpusStats(
            discovered_conversations=10,
            sampled_conversations=5,
            discovered_turns=50,
            sampled_turns=25,
        ),
        token_usage=TokenUsage(),
        sampling_config=SamplingConfigRecord(
            n=5,
            seed=42,
            min_turns=5,
            recency_weight=0.7,
            recency_window_days=90,
            stratify_by_project=True,
            top_n_projects=10,
        ),
        status="running",
        corpus_fingerprint="abc",
        prompt_versions={},
        schema_version=SCHEMA_VERSION,
    )
    store.insert_audit_run(run)


def test_target_section_ids_includes_core_and_per_module():
    sections = _target_section_ids(["A", "B", "G", "H"])
    assert "exec_summary" in sections
    assert "top_3_actions" in sections
    assert "headline_findings" in sections
    assert "module_a_narrative" in sections
    assert "module_b_narrative" in sections
    assert "module_h_narrative" in sections
    # Module G (attribution) has no narrative — it's metadata.
    assert "module_g_narrative" not in sections


def test_build_kickoff_mentions_target_sections_and_counts():
    kickoff = _build_kickoff_message(
        audit_run_id="run-1",
        target_sections=["exec_summary", "module_b_narrative"],
        behavior_counts={"sycophancy": 12, "pushback": 3},
        corpus_stats={"sampled_conversations": 45, "sampled_turns": 838},
    )
    assert "exec_summary" in kickoff
    assert "module_b_narrative" in kickoff
    assert "sycophancy: 12" in kickoff
    assert "pushback: 3" in kickoff
    assert "sampled_conversations: 45" in kickoff
    assert "sampled_turns: 838" in kickoff


async def test_run_synthesis_session_handles_session_exception(tmp_path, monkeypatch):
    """An exception raised inside ``SynthesisSession.run`` is captured
    in ``result.reason`` rather than propagated to the caller."""
    from lucid.store.init import initialize_db
    from lucid.store.sqlite import CorpusStore

    async def _raise(*_args, **_kwargs):
        raise RuntimeError("fake session failure")

    monkeypatch.setattr("lucid.synthesis.run.SynthesisSession.run", _raise)

    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run_inline(store, run_id="run-ex1")
        result = await run_synthesis_session(
            client=MagicMock(),
            store=store,
            audit_run_id="run-ex1",
            enabled_modules=["A"],
            behavior_counts={},
            corpus_stats={"sampled_conversations": 0, "sampled_turns": 0},
            progress_log=lambda *_: None,
        )
    assert "session_exception" in result.reason
    assert "fake session failure" in result.reason
    # No sections were persisted.
    assert result.sections_written == 0
    assert result.sections_declined == 0


async def test_run_synthesis_session_counts_written_and_declined(tmp_path, monkeypatch):
    """When the session completes, pre-persisted ``ReportSection`` rows
    are counted correctly into ``SynthesisRunResult``."""
    from lucid.schemas import ReportSection
    from lucid.store.init import initialize_db
    from lucid.store.sqlite import CorpusStore
    from lucid.synthesis.session import SynthesisHandles, SynthesisOutcome

    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run_inline(store, run_id="run-count1")

        # Pre-populate report_sections as if the session had written them.
        store.upsert_report_section(
            ReportSection(
                audit_run_id="run-count1",
                section_id="exec_summary",
                markdown="Some narrative body.",
                cited_finding_ids=[],
                cited_turn_ids=[],
                insufficient_evidence=False,
                decline_reason=None,
                created_at=datetime.now(tz=UTC),
            )
        )
        store.upsert_report_section(
            ReportSection(
                audit_run_id="run-count1",
                section_id="top_3_actions",
                markdown="",
                cited_finding_ids=[],
                cited_turn_ids=[],
                insufficient_evidence=True,
                decline_reason="too thin",
                created_at=datetime.now(tz=UTC),
            )
        )

        fake_outcome = SynthesisOutcome(
            handles=SynthesisHandles(
                agent_id="a",
                environment_id="e",
                session_id="s",
            ),
            completed=True,
            reason="session.status_idle",
        )

        async def _ok(*_args, **_kwargs):
            return fake_outcome

        monkeypatch.setattr("lucid.synthesis.run.SynthesisSession.run", _ok)

        result = await run_synthesis_session(
            client=MagicMock(),
            store=store,
            audit_run_id="run-count1",
            enabled_modules=["A"],
            behavior_counts={"sycophancy": 1},
            corpus_stats={"sampled_conversations": 1, "sampled_turns": 5},
            progress_log=lambda *_: None,
        )
    assert result.sections_written == 1
    assert result.sections_declined == 1
    assert result.reason == "session.status_idle"
    assert result.session_outcome is fake_outcome


async def test_run_synthesis_session_calls_sonnet_post_process(tmp_path, monkeypatch):
    """When async_client is provided and sections are populated,
    run_synthesis_session invokes post_process_sections and persists
    the enriched sections back to the DB."""
    from lucid.schemas import ReportSection, SynthesisBlock
    from lucid.store.init import initialize_db
    from lucid.store.sqlite import CorpusStore
    from lucid.synthesis.session import SynthesisHandles, SynthesisOutcome

    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run_inline(store, run_id="run-pp-int")

        # Pre-populate a populated section (as if the agent just wrote it).
        store.upsert_report_section(
            ReportSection(
                audit_run_id="run-pp-int",
                section_id="exec_summary",
                markdown="Prose about something.",
                cited_finding_ids=[],
                cited_turn_ids=[],
                insufficient_evidence=False,
                decline_reason=None,
                created_at=datetime.now(tz=UTC),
            )
        )

        # Mock SynthesisSession.run() to return an idle outcome.
        fake_outcome = SynthesisOutcome(
            handles=SynthesisHandles(agent_id="a", environment_id="e", session_id="s"),
            completed=True,
            reason="session.status_idle",
        )

        async def _ok(*_args, **_kwargs):
            return fake_outcome

        monkeypatch.setattr("lucid.synthesis.run.SynthesisSession.run", _ok)

        # Mock post_process_sections to return enriched section.
        async def _fake_post_process(*, async_client, sections, **_kwargs):
            enriched = []
            for s in sections:
                if s.insufficient_evidence:
                    enriched.append(s)
                else:
                    enriched.append(
                        s.model_copy(
                            update={
                                "blocks": [
                                    SynthesisBlock(text="Prose about something.", citations=[])
                                ],
                                "citation_confidence": 0.75,
                            }
                        )
                    )
            return enriched

        monkeypatch.setattr("lucid.synthesis.run.post_process_sections", _fake_post_process)

        async_client_mock = MagicMock()
        result = await run_synthesis_session(
            client=MagicMock(),
            async_client=async_client_mock,
            store=store,
            audit_run_id="run-pp-int",
            enabled_modules=["A"],
            behavior_counts={},
            corpus_stats={"sampled_conversations": 1, "sampled_turns": 5},
            progress_log=lambda *_: None,
        )
        assert result.sections_written == 1

        # Verify the enriched section was upserted.
        rows = store.fetch_report_sections_for_run("run-pp-int")
        assert len(rows) == 1
        assert len(rows[0].blocks) == 1
        assert rows[0].citation_confidence == 0.75


async def test_run_synthesis_session_sends_plural_tool_call_counts_keys(tmp_path, monkeypatch):
    """Guards against silent-bug regression: the Sonnet prompt's Input
    shape specifies plural keys (conversations, turns, findings).
    Singular keys silently pass type-checks but bypass Sonnet's
    aggregate-extraction heuristic, inflating citation_confidence.
    """
    from lucid.schemas import ReportSection
    from lucid.store.init import initialize_db
    from lucid.store.sqlite import CorpusStore
    from lucid.synthesis.session import SynthesisHandles, SynthesisOutcome

    captured_calls: list[dict[str, int]] = []

    async def _fake_post_process_sections(*, tool_call_counts, sections, **_kwargs):
        captured_calls.append(dict(tool_call_counts or {}))
        return list(sections)

    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run_inline(store, run_id="run-keys1")

        # At least one populated section triggers the post-process branch.
        store.upsert_report_section(
            ReportSection(
                audit_run_id="run-keys1",
                section_id="exec_summary",
                markdown="Narrative body.",
                cited_finding_ids=[],
                cited_turn_ids=[],
                insufficient_evidence=False,
                decline_reason=None,
                created_at=datetime.now(tz=UTC),
            )
        )

        fake_outcome = SynthesisOutcome(
            handles=SynthesisHandles(agent_id="a", environment_id="e", session_id="s"),
            completed=True,
            reason="session.status_idle",
        )

        async def _ok(*_args, **_kwargs):
            return fake_outcome

        monkeypatch.setattr("lucid.synthesis.run.SynthesisSession.run", _ok)
        monkeypatch.setattr(
            "lucid.synthesis.run.post_process_sections",
            _fake_post_process_sections,
        )

        await run_synthesis_session(
            client=MagicMock(),
            async_client=MagicMock(),
            store=store,
            audit_run_id="run-keys1",
            enabled_modules=["A"],
            behavior_counts={"sycophancy": 1},
            corpus_stats={"sampled_conversations": 3, "sampled_turns": 15},
            progress_log=lambda *_: None,
        )

    assert len(captured_calls) > 0
    keys = set(captured_calls[0].keys())
    assert "conversations" in keys
    assert "turns" in keys
    assert "findings" in keys
    # Explicitly guard against the regression: singular keys silently
    # bypass Sonnet's citation_confidence heuristic.
    for singular in ("conversation", "turn", "finding"):
        assert singular not in keys, f"singular key {singular!r} — Sonnet prompt expects plural"


async def test_run_synthesis_session_runs_validators_over_persisted_sections(tmp_path, monkeypatch):
    """After a successful session, the three validators run over the
    persisted sections (aggregate / superlative / uncited-high-intensity)."""
    from lucid.schemas import ReportSection
    from lucid.store.init import initialize_db
    from lucid.store.sqlite import CorpusStore
    from lucid.synthesis.session import SynthesisHandles, SynthesisOutcome

    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run_inline(store, run_id="run-val1")
        # Aggregate-claim mismatch (claims 99 but tool count was 3) +
        # superlative with a thin-evidence behavior (sycophancy, 1).
        store.upsert_report_section(
            ReportSection(
                audit_run_id="run-val1",
                section_id="exec_summary",
                markdown=(
                    "Across 99 conversations the assistant consistently "
                    "exhibited sycophancy in every turn."
                ),
                cited_finding_ids=[],
                cited_turn_ids=[],
                insufficient_evidence=False,
                decline_reason=None,
                created_at=datetime.now(tz=UTC),
            )
        )

        fake_outcome = SynthesisOutcome(
            handles=SynthesisHandles(agent_id="a", environment_id="e", session_id="s"),
            completed=True,
            reason="session.status_idle",
        )

        async def _ok(*_args, **_kwargs):
            return fake_outcome

        monkeypatch.setattr("lucid.synthesis.run.SynthesisSession.run", _ok)

        result = await run_synthesis_session(
            client=MagicMock(),
            store=store,
            audit_run_id="run-val1",
            enabled_modules=["A"],
            behavior_counts={"sycophancy": 1},
            corpus_stats={"sampled_conversations": 3, "sampled_turns": 15},
            progress_log=lambda *_: None,
        )

    # Validators produced concerns: aggregate mismatch + superlative
    # with thin evidence.
    kinds = {e.kind for e in result.validation_errors}
    assert "aggregate_mismatch" in kinds
    assert "superlative_thin_evidence" in kinds
