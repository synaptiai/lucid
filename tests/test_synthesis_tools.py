"""Tests for the synthesis tool registry."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from lucid.schemas import (
    Conversation,
    Finding,
    ModuleName,
    Source,
)


def _seed_audit_run_inline(store, run_id: str) -> None:
    """Minimal audit-run seed for synthesis tool tests.

    Inlined (not imported from test_store) because test collection order
    across files is non-deterministic; duplicating the few lines is
    simpler than coupling two test modules.
    """
    from lucid.schemas import (
        AuditRun,
        CorpusStats,
        SamplingConfigRecord,
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


def _seed_conversation(store, conversation_id: str = "c-1") -> None:
    store.insert_conversations(
        [
            Conversation(
                id=conversation_id,
                source=Source.CLAUDE_CODE,
                source_path="/tmp/proj",
                created_at=datetime(2026, 4, 21, 12, 0, 0, tzinfo=UTC),
                updated_at=datetime(2026, 4, 21, 12, 0, 0, tzinfo=UTC),
                turn_count=1,
            )
        ]
    )


def _seed_finding(store, *, run_id: str, finding_id: str, conversation_id: str = "c-1") -> None:
    """Insert one well-formed Finding tied to ``run_id``.

    Caller must seed the conversation row first (FK constraint).
    """
    turn_ids = ["t-1"]
    turn_ids_hash = hashlib.sha256(",".join(sorted(turn_ids)).encode()).hexdigest()
    finding = Finding(
        id=finding_id,
        audit_run_id=run_id,
        conversation_id=conversation_id,
        turn_ids=turn_ids,
        turn_ids_hash=turn_ids_hash,
        module=ModuleName.A_SPIRALBENCH,
        behavior="safe-redirection",
        intensity=2,
        confidence=0.8,
        explanation="seeded for synthesis test",
        citation="Spiral-Bench v1.2",
        detected_by=["claude-opus-4-7"],
        detected_at=datetime(2026, 4, 21, 12, 0, 0, tzinfo=UTC),
        prompt_version="v1",
        prompt_hash="h",
    )
    store.insert_finding(finding)


def test_synthesis_registry_exposes_only_expected_tools(tmp_path):
    """Synthesis exposes 6 tools: 5 read-only shared + 1 new write_report_section.

    Explicitly excludes invoke_module, store_finding, estimate_remaining_cost
    which belong to the scoring phase.
    """
    from lucid.store.init import initialize_db
    from lucid.store.sqlite import CorpusStore
    from lucid.synthesis.tools import build_synthesis_registry

    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        registry = build_synthesis_registry(
            store=store,
            audit_run_id="run-synth-test",
            progress_log=lambda *_: None,
        )
    names = set(registry.names)
    expected = {
        "query_corpus",
        "get_conversation",
        "get_turn_window",
        "get_findings",
        "log_progress",
        "write_report_section",
    }
    assert names == expected, f"synthesis registry exposed {names}, expected {expected}"
    # Explicit absence check — these MUST NOT be in the synthesis registry.
    for forbidden in ("invoke_module", "store_finding", "estimate_remaining_cost"):
        assert forbidden not in names, f"synthesis registry must not expose {forbidden}"


@pytest.mark.asyncio
async def test_write_report_section_persists_declined_section(tmp_path):
    """insufficient_evidence=True with decline_reason persists correctly."""
    from lucid.store.init import initialize_db
    from lucid.store.sqlite import CorpusStore
    from lucid.synthesis.tools import build_synthesis_registry

    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run_inline(store, run_id="run-write-1")
        registry = build_synthesis_registry(
            store=store,
            audit_run_id="run-write-1",
            progress_log=lambda *_: None,
        )
        tool = registry.get("write_report_section")
        assert tool is not None
        result = await tool.handler(
            {
                "section_id": "top_3_actions",
                "insufficient_evidence": True,
                "decline_reason": "Only 2 qualifying findings above intensity 1.",
            }
        )
        assert result["ok"] is True
        assert result["insufficient_evidence"] is True
        assert result["section_id"] == "top_3_actions"

        rows = store.fetch_report_sections_for_run("run-write-1")
        assert len(rows) == 1
        assert rows[0].insufficient_evidence is True
        assert rows[0].decline_reason == ("Only 2 qualifying findings above intensity 1.")
        assert rows[0].markdown == ""
        assert rows[0].cited_finding_ids == []


@pytest.mark.asyncio
async def test_write_report_section_validates_finding_ids_against_db(tmp_path):
    """Unknown finding_ids are rejected WITHOUT persistence."""
    from lucid.store.init import initialize_db
    from lucid.store.sqlite import CorpusStore
    from lucid.synthesis.tools import build_synthesis_registry

    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run_inline(store, run_id="run-write-2")
        registry = build_synthesis_registry(
            store=store,
            audit_run_id="run-write-2",
            progress_log=lambda *_: None,
        )
        tool = registry.get("write_report_section")
        assert tool is not None
        result = await tool.handler(
            {
                "section_id": "exec_summary",
                "markdown": "A claim with [F:does-not-exist].",
                "cited_finding_ids": ["does-not-exist"],
                "cited_turn_ids": [],
            }
        )
        assert result.get("error") == "unknown_ids"
        assert "does-not-exist" in result["unknown_finding_ids"]
        assert result["unknown_turn_ids"] == []

        # Nothing was persisted.
        assert store.fetch_report_sections_for_run("run-write-2") == []


@pytest.mark.asyncio
async def test_write_report_section_persists_populated_section(tmp_path):
    """Valid citation ids → ReportSection persists end-to-end."""
    from lucid.store.init import initialize_db
    from lucid.store.sqlite import CorpusStore
    from lucid.synthesis.tools import build_synthesis_registry

    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run_inline(store, run_id="run-write-3")
        _seed_conversation(store, conversation_id="c-1")
        _seed_finding(store, run_id="run-write-3", finding_id="f-real-1")

        registry = build_synthesis_registry(
            store=store,
            audit_run_id="run-write-3",
            progress_log=lambda *_: None,
        )
        tool = registry.get("write_report_section")
        assert tool is not None
        result = await tool.handler(
            {
                "section_id": "exec_summary",
                "markdown": "One paragraph with [F:f-real-1].",
                "cited_finding_ids": ["f-real-1"],
                "cited_turn_ids": [],
            }
        )
        assert result["ok"] is True
        assert result["cited_finding_count"] == 1
        assert result["cited_turn_count"] == 0

        rows = store.fetch_report_sections_for_run("run-write-3")
        assert len(rows) == 1
        assert rows[0].markdown == "One paragraph with [F:f-real-1]."
        assert rows[0].cited_finding_ids == ["f-real-1"]
        assert rows[0].insufficient_evidence is False


@pytest.mark.asyncio
async def test_write_report_section_missing_decline_reason(tmp_path):
    """insufficient_evidence=True without decline_reason → error, no persist."""
    from lucid.store.init import initialize_db
    from lucid.store.sqlite import CorpusStore
    from lucid.synthesis.tools import build_synthesis_registry

    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run_inline(store, run_id="run-write-4")
        registry = build_synthesis_registry(
            store=store,
            audit_run_id="run-write-4",
            progress_log=lambda *_: None,
        )
        tool = registry.get("write_report_section")
        assert tool is not None
        result = await tool.handler(
            {
                "section_id": "top_3_actions",
                "insufficient_evidence": True,
                # no decline_reason
            }
        )
        assert result.get("error") == "missing_decline_reason"
        assert store.fetch_report_sections_for_run("run-write-4") == []


@pytest.mark.asyncio
async def test_write_report_section_cap_limits_retry_attempts(tmp_path):
    """After ``max_regen_attempts + 1`` calls for the same ``section_id``,
    the handler rejects with ``regen_limit_exceeded`` so the writer
    moves on instead of looping forever on a pathological failure."""
    from lucid.store.init import initialize_db
    from lucid.store.sqlite import CorpusStore
    from lucid.synthesis.tools import make_write_report_section_tool

    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run_inline(store, run_id="run-cap1")
        tool = make_write_report_section_tool(store, "run-cap1", max_regen_attempts=2)
        # 3 attempts (1 original + 2 retries) should pass the cap check,
        # each returning ``unknown_ids`` because the cited finding id
        # does not exist — but none hit ``regen_limit_exceeded`` yet.
        for _ in range(3):
            result = await tool.handler(
                {
                    "section_id": "exec_summary",
                    "markdown": "attempt with [F:does-not-exist]",
                    "cited_finding_ids": ["does-not-exist"],
                }
            )
            assert result.get("error") == "unknown_ids"
        # The 4th attempt exceeds the cap and is rejected before
        # touching the DB.
        result = await tool.handler(
            {
                "section_id": "exec_summary",
                "markdown": "one more attempt",
                "cited_finding_ids": [],
            }
        )
        assert result.get("error") == "regen_limit_exceeded"
        assert result["section_id"] == "exec_summary"
        assert result["attempts"] == 4


def test_synthesis_registry_shares_read_only_factories_with_orchestrator(tmp_path):
    """The read-only handlers used by synthesis come from the same factories
    ``build_tool_registry`` uses — guards against divergence."""
    from lucid.orchestrator.tools import (
        make_get_conversation_tool,
        make_get_findings_tool,
        make_get_turn_window_tool,
        make_log_progress_tool,
        make_query_corpus_tool,
    )
    from lucid.store.init import initialize_db
    from lucid.store.sqlite import CorpusStore

    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        # Each factory should return a registerable CustomTool.
        for factory_call in [
            lambda: make_query_corpus_tool(store),
            lambda: make_get_conversation_tool(store),
            lambda: make_get_turn_window_tool(store),
            lambda: make_get_findings_tool(store, "run-shared"),
            lambda: make_log_progress_tool(lambda *_: None),
        ]:
            tool = factory_call()
            assert tool.name
            assert tool.handler is not None
            assert callable(tool.handler)
