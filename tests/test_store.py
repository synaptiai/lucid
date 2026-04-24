"""CorpusStore tests — initialize_db, insert/select, idempotency."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lucid.schemas import (
    AuditRun,
    Conversation,
    CorpusStats,
    Finding,
    ModuleName,
    ReportSection,
    Role,
    SamplingConfigRecord,
    Source,
    TextBlock,
    TokenUsage,
    Turn,
)
from lucid.store import SCHEMA_VERSION, initialize_db
from lucid.store.sqlite import CorpusStore

_UTC = UTC


def _now() -> datetime:
    return datetime(2026, 4, 21, 12, 0, 0, tzinfo=_UTC)


def _turn_ids_hash(ids: list[str]) -> str:
    return hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()


def _seed_audit_run(store: CorpusStore, run_id: str = "run-1") -> AuditRun:
    run = AuditRun(
        id=run_id,
        sources=[Source.CLAUDE_AI],
        source_paths={Source.CLAUDE_AI: "/tmp/export"},
        started_at=_now(),
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
    return run


# ----- initialize_db ----------------------------------------------------


def test_initialize_db_fresh(tmp_path: Path) -> None:
    db = tmp_path / "lucid.sqlite3"
    returned = initialize_db(db)
    assert returned == db.resolve()
    assert db.exists()

    conn = sqlite3.connect(db)
    try:
        (version,) = conn.execute("PRAGMA user_version;").fetchone()
        assert version == SCHEMA_VERSION
        # Sanity: a core table exists.
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "audit_runs" in tables
        assert "findings" in tables
        assert "module_progress" in tables
        assert "embeddings" in tables
    finally:
        conn.close()


def test_initialize_db_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    # Second call must not raise and must not re-apply schema.
    initialize_db(db)
    conn = sqlite3.connect(db)
    try:
        (version,) = conn.execute("PRAGMA user_version;").fetchone()
        assert version == SCHEMA_VERSION
    finally:
        conn.close()


def test_initialize_db_rejects_newer_version(tmp_path: Path) -> None:
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    # Pretend a future Lucid bumped the DB.
    conn = sqlite3.connect(db)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1};")
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="Upgrade Lucid"):
        initialize_db(db)


# ----- basic insert / select -------------------------------------------


def test_insert_and_count_conversations(tmp_path: Path) -> None:
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run(store)
        convs = [
            Conversation(
                id=f"c-{i}",
                source=Source.CLAUDE_CODE,
                source_path="/tmp/proj",
                created_at=_now(),
                updated_at=_now(),
                turn_count=5,
            )
            for i in range(3)
        ]
        assert store.insert_conversations(convs) == 3
        assert store.count("conversations") == 3


def test_insert_turn_with_block_roundtrip_json(tmp_path: Path) -> None:
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run(store)
        store.insert_conversations(
            [
                Conversation(
                    id="c-1",
                    source=Source.CLAUDE_CODE,
                    source_path="/tmp",
                    created_at=_now(),
                    updated_at=_now(),
                    turn_count=1,
                )
            ]
        )
        turn = Turn(
            id="t-1",
            conversation_id="c-1",
            index=0,
            role=Role.USER,
            content="hi",
            blocks=[TextBlock(text="hi")],
        )
        assert store.insert_turns([turn]) == 1
        rows = store.fetchall("SELECT blocks_json FROM turns WHERE id = ?", ("t-1",))
        assert len(rows) == 1
        assert '"type":"text"' in rows[0]["blocks_json"].replace(" ", "")


# ----- findings uniqueness --------------------------------------------


def _finding(
    behavior: str = "safe-redirection",
    conversation_id: str | None = "c-1",
    turn_ids: list[str] | None = None,
    finding_id: str = "f-1",
) -> Finding:
    turn_ids = turn_ids or ["t-1"]
    return Finding(
        id=finding_id,
        audit_run_id="run-1",
        conversation_id=conversation_id,
        turn_ids=turn_ids,
        turn_ids_hash=_turn_ids_hash(turn_ids),
        module=ModuleName.A_SPIRALBENCH,
        behavior=behavior,
        intensity=2,
        confidence=0.8,
        explanation="e",
        citation="c",
        detected_by=["claude-opus-4-7"],
        detected_at=_now(),
        prompt_version="v1",
        prompt_hash="h",
    )


def test_finding_insert_and_select(tmp_path: Path) -> None:
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run(store)
        store.insert_conversations(
            [
                Conversation(
                    id="c-1",
                    source=Source.CLAUDE_CODE,
                    source_path="/tmp",
                    created_at=_now(),
                    updated_at=_now(),
                    turn_count=1,
                )
            ]
        )
        store.insert_finding(_finding())
        assert store.count("findings") == 1


def test_finding_idempotency_key_collision(tmp_path: Path) -> None:
    """Duplicate (run, module, conv, turns, behavior) must raise IntegrityError."""
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run(store)
        store.insert_conversations(
            [
                Conversation(
                    id="c-1",
                    source=Source.CLAUDE_CODE,
                    source_path="/tmp",
                    created_at=_now(),
                    updated_at=_now(),
                    turn_count=1,
                )
            ]
        )
        store.insert_finding(_finding(finding_id="f-1"))
        with pytest.raises(sqlite3.IntegrityError):
            # Different PK id but same (audit_run_id, module, conversation_id,
            # turn_ids_hash, behavior) tuple → collides on the UNIQUE.
            store.insert_finding(_finding(finding_id="f-2"))


def test_finding_different_behavior_no_collision(tmp_path: Path) -> None:
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run(store)
        store.insert_conversations(
            [
                Conversation(
                    id="c-1",
                    source=Source.CLAUDE_CODE,
                    source_path="/tmp",
                    created_at=_now(),
                    updated_at=_now(),
                    turn_count=1,
                )
            ]
        )
        store.insert_finding(_finding(finding_id="f-1", behavior="safe-redirection"))
        store.insert_finding(_finding(finding_id="f-2", behavior="emotional-validation"))
        assert store.count("findings") == 2


# ----- CHECK constraint coverage --------------------------------------


def test_finding_confidence_out_of_range_rejected_by_db(tmp_path: Path) -> None:
    """Pydantic catches this at the model layer, but the DB is belt + suspenders.

    We bypass Pydantic with a raw INSERT to prove the CHECK fires.
    """
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run(store)
        conn = store.connect()
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            conn.execute(
                """
                INSERT INTO findings (
                    id, audit_run_id, turn_ids_hash, module, behavior,
                    intensity, confidence, explanation, citation,
                    detected_by_json, detected_at, prompt_version, prompt_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "f-bad",
                    "run-1",
                    "h",
                    "A",
                    "b",
                    1,
                    1.5,  # > 1.0 — CHECK constraint rejects
                    "e",
                    "c",
                    '["claude-opus-4-7"]',
                    _now().isoformat(),
                    "v1",
                    "h",
                ),
            )


def test_finding_empty_detected_by_rejected_by_db(tmp_path: Path) -> None:
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run(store)
        conn = store.connect()
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            conn.execute(
                """
                INSERT INTO findings (
                    id, audit_run_id, turn_ids_hash, module, behavior,
                    intensity, confidence, explanation, citation,
                    detected_by_json, detected_at, prompt_version, prompt_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "f-bad2",
                    "run-1",
                    "h",
                    "A",
                    "b",
                    1,
                    0.5,
                    "e",
                    "c",
                    "[]",  # empty detected_by → CHECK rejects
                    _now().isoformat(),
                    "v1",
                    "h",
                ),
            )


# ----- module_progress writers (L1) -----------------------------------


def test_mark_module_started_inserts_running_row(tmp_path: Path) -> None:
    """First call for a (run, module) pair lands a row with
    ``status='running'`` and a populated ``started_at``."""
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run(store)
        store.mark_module_started("run-1", ModuleName.A_SPIRALBENCH)

        rows = store.fetchall(
            "SELECT module, status, started_at, completed_at, error_message "
            "FROM module_progress WHERE audit_run_id = ?",
            ("run-1",),
        )
    assert len(rows) == 1
    assert rows[0]["module"] == "A"
    assert rows[0]["status"] == "running"
    assert rows[0]["started_at"] is not None
    assert rows[0]["completed_at"] is None
    assert rows[0]["error_message"] is None


def test_mark_module_started_resets_terminal_state_on_retry(tmp_path: Path) -> None:
    """A retry — start, finish, start again — must leave the row as
    ``running`` again with ``completed_at`` and ``error_message``
    cleared. Otherwise resume tooling sees stale terminal state.
    """
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run(store)
        store.mark_module_started("run-1", ModuleName.A_SPIRALBENCH)
        store.mark_module_finished(
            "run-1",
            ModuleName.A_SPIRALBENCH,
            status="failed",
            error_message="first attempt blew up",
        )
        store.mark_module_started("run-1", ModuleName.A_SPIRALBENCH)

        rows = store.fetchall(
            "SELECT status, completed_at, error_message FROM module_progress "
            "WHERE audit_run_id = ? AND module = ?",
            ("run-1", "A"),
        )
    assert len(rows) == 1
    assert rows[0]["status"] == "running"
    assert rows[0]["completed_at"] is None
    assert rows[0]["error_message"] is None


@pytest.mark.parametrize("status", ["completed", "failed", "skipped"])
def test_mark_module_finished_accepts_every_terminal_status(tmp_path: Path, status: str) -> None:
    """All three terminal statuses round-trip: status + completed_at +
    error_message land in the row. Catches a typo in the SQL CHECK
    constraint or the Python literal alias."""
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run(store)
        store.mark_module_started("run-1", ModuleName.B_SHARMA)
        store.mark_module_finished(
            "run-1",
            ModuleName.B_SHARMA,
            status=status,  # type: ignore[arg-type]
            error_message=f"reason for {status}",
        )

        row = store.fetchall(
            "SELECT status, completed_at, error_message FROM module_progress "
            "WHERE audit_run_id = ? AND module = ?",
            ("run-1", "B"),
        )[0]
    assert row["status"] == status
    assert row["completed_at"] is not None
    assert row["error_message"] == f"reason for {status}"


@pytest.mark.parametrize("non_terminal", ["pending", "running"])
def test_mark_module_finished_rejects_non_terminal_status(
    tmp_path: Path, non_terminal: str
) -> None:
    """Calling ``mark_module_finished`` with ``running`` or ``pending``
    is a programmer error — surface it loudly rather than letting
    the row land in an inconsistent state."""
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run(store)
        with pytest.raises(ValueError, match="terminal status"):
            store.mark_module_finished(
                "run-1",
                ModuleName.C_SYCEVAL,
                status=non_terminal,  # type: ignore[arg-type]
            )


def test_mark_module_finished_upserts_when_no_started_row_exists(tmp_path: Path) -> None:
    """Short-circuit modules (D opt-out, H no provider, no client)
    skip ``mark_module_started`` entirely. ``mark_module_finished``
    must still land a row so resume / triage tooling can see what
    happened."""
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run(store)
        store.mark_module_finished(
            "run-1",
            ModuleName.D_PERSPECTIVE,
            status="skipped",
            error_message="opt-in flag absent",
        )

        rows = store.fetchall(
            "SELECT module, status, started_at, completed_at, error_message "
            "FROM module_progress WHERE audit_run_id = ?",
            ("run-1",),
        )
    assert len(rows) == 1
    assert rows[0]["module"] == "D"
    assert rows[0]["status"] == "skipped"
    # Same skip-decision timestamp on both sides — duration of a
    # skipped module is undefined-but-reported-as-zero.
    assert rows[0]["started_at"] == rows[0]["completed_at"]
    assert rows[0]["error_message"] == "opt-in flag absent"


def test_fetch_module_progress_returns_typed_models(tmp_path: Path) -> None:
    """Verify rehydration produces ``ModuleProgress`` instances with
    the right fields and ``datetime`` values, not raw row dicts."""
    from lucid.schemas import ModuleProgress

    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run(store)
        store.mark_module_started("run-1", ModuleName.A_SPIRALBENCH)
        store.mark_module_finished("run-1", ModuleName.A_SPIRALBENCH, status="completed")
        store.mark_module_finished(
            "run-1",
            ModuleName.D_PERSPECTIVE,
            status="skipped",
            error_message="opt-in absent",
        )

        progress = store.fetch_module_progress("run-1")
    assert all(isinstance(p, ModuleProgress) for p in progress)
    # Module-name ascending order.
    assert [p.module for p in progress] == [
        ModuleName.A_SPIRALBENCH,
        ModuleName.D_PERSPECTIVE,
    ]
    a_row = progress[0]
    assert a_row.status == "completed"
    assert isinstance(a_row.started_at, datetime)
    assert isinstance(a_row.completed_at, datetime)
    d_row = progress[1]
    assert d_row.status == "skipped"
    assert d_row.error_message == "opt-in absent"


# ----- report_sections (synthesis-layer prose) ------------------------


def test_report_sections_table_exists(tmp_path: Path) -> None:
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        rows = store.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='report_sections'"
        )
        assert len(rows) == 1


def test_report_sections_schema_columns(tmp_path: Path) -> None:
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        rows = store.fetchall("PRAGMA table_info(report_sections)")
        columns = {row["name"] for row in rows}
        expected = {
            "id",
            "audit_run_id",
            "section_id",
            "markdown",
            "cited_finding_ids_json",
            "cited_turn_ids_json",
            "insufficient_evidence",
            "decline_reason",
            "created_at",
        }
        assert expected.issubset(columns)


def test_report_sections_unique_key(tmp_path: Path) -> None:
    """(audit_run_id, section_id) is unique — re-run overwrites, does not dup."""
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run(store, run_id="run-ut1")
        conn = store.connect()
        conn.execute(
            "INSERT INTO report_sections(id, audit_run_id, section_id, markdown, "
            "cited_finding_ids_json, cited_turn_ids_json, insufficient_evidence, "
            "decline_reason, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "rs-1",
                "run-ut1",
                "exec_summary",
                "x",
                "[]",
                "[]",
                0,
                None,
                "2026-04-24T10:00:00+00:00",
            ),
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO report_sections(id, audit_run_id, section_id, markdown, "
                "cited_finding_ids_json, cited_turn_ids_json, insufficient_evidence, "
                "decline_reason, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    "rs-2",
                    "run-ut1",
                    "exec_summary",
                    "y",
                    "[]",
                    "[]",
                    0,
                    None,
                    "2026-04-24T10:01:00+00:00",
                ),
            )
            conn.commit()


def test_report_sections_check_insufficient_requires_empty_citations(tmp_path):
    """insufficient_evidence=1 with populated citation JSON must be rejected."""
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        conn = store.connect()
        _seed_audit_run(store, run_id="run-inv1")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO report_sections(id, audit_run_id, section_id, markdown, "
                "cited_finding_ids_json, cited_turn_ids_json, insufficient_evidence, "
                "decline_reason, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                ("rs-bad", "run-inv1", "exec_summary", "",
                 '["f1"]', "[]", 1, "declined", "2026-04-24T10:00:00+00:00"),
            )
            conn.commit()


def test_report_sections_check_insufficient_requires_decline_reason(tmp_path):
    """insufficient_evidence=1 without a decline_reason must be rejected."""
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        conn = store.connect()
        _seed_audit_run(store, run_id="run-inv2")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO report_sections(id, audit_run_id, section_id, markdown, "
                "cited_finding_ids_json, cited_turn_ids_json, insufficient_evidence, "
                "decline_reason, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                ("rs-bad2", "run-inv2", "exec_summary", "",
                 "[]", "[]", 1, None, "2026-04-24T10:00:00+00:00"),
            )
            conn.commit()


def test_report_sections_check_populated_rejects_decline_reason(tmp_path):
    """insufficient_evidence=0 with a non-null decline_reason must be rejected."""
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        conn = store.connect()
        _seed_audit_run(store, run_id="run-inv3")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO report_sections(id, audit_run_id, section_id, markdown, "
                "cited_finding_ids_json, cited_turn_ids_json, insufficient_evidence, "
                "decline_reason, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                ("rs-bad3", "run-inv3", "exec_summary", "populated",
                 "[]", "[]", 0, "should not be here", "2026-04-24T10:00:00+00:00"),
            )
            conn.commit()


def test_report_sections_check_section_id_length_cap(tmp_path):
    """section_id longer than 64 characters must be rejected."""
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        conn = store.connect()
        _seed_audit_run(store, run_id="run-inv4")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO report_sections(id, audit_run_id, section_id, markdown, "
                "cited_finding_ids_json, cited_turn_ids_json, insufficient_evidence, "
                "decline_reason, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                ("rs-bad4", "run-inv4", "x" * 65, "populated",
                 "[]", "[]", 0, None, "2026-04-24T10:00:00+00:00"),
            )
            conn.commit()


def test_upsert_report_section_roundtrip(tmp_path):
    """Round-trip: upsert a section, fetch, assert equality on all fields."""
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run(store, run_id="run-ut2")
        section = ReportSection(
            audit_run_id="run-ut2",
            section_id="exec_summary",
            markdown="One paragraph with [F:f001].",
            cited_finding_ids=["f001"],
            cited_turn_ids=[],
            insufficient_evidence=False,
            decline_reason=None,
            created_at=datetime(2026, 4, 24, 10, 0, tzinfo=UTC),
        )
        store.upsert_report_section(section)
        rows = store.fetch_report_sections_for_run("run-ut2")
        assert len(rows) == 1
        assert rows[0] == section


def test_upsert_report_section_is_idempotent(tmp_path):
    """Re-running synthesis replaces the row rather than duplicating."""
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run(store, run_id="run-ut3")
        section_v1 = ReportSection(
            audit_run_id="run-ut3",
            section_id="exec_summary",
            markdown="version one",
            cited_finding_ids=["f1"],
            cited_turn_ids=[],
            insufficient_evidence=False,
            decline_reason=None,
            created_at=datetime(2026, 4, 24, 10, 0, tzinfo=UTC),
        )
        store.upsert_report_section(section_v1)

        section_v2 = ReportSection(
            audit_run_id="run-ut3",
            section_id="exec_summary",  # same section_id — should upsert
            markdown="version two (overwrites)",
            cited_finding_ids=["f2"],
            cited_turn_ids=["t1"],
            insufficient_evidence=False,
            decline_reason=None,
            created_at=datetime(2026, 4, 24, 11, 0, tzinfo=UTC),
        )
        store.upsert_report_section(section_v2)

        rows = store.fetch_report_sections_for_run("run-ut3")
        assert len(rows) == 1, "Upsert must not duplicate"
        assert rows[0].markdown == "version two (overwrites)"
        assert rows[0].cited_finding_ids == ["f2"]
        assert rows[0].cited_turn_ids == ["t1"]


def test_upsert_insufficient_evidence_section_roundtrip(tmp_path):
    """Declined sections round-trip with empty fields + decline_reason preserved."""
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run(store, run_id="run-ut4")
        section = ReportSection(
            audit_run_id="run-ut4",
            section_id="top_3_actions",
            markdown="",
            cited_finding_ids=[],
            cited_turn_ids=[],
            insufficient_evidence=True,
            decline_reason="fewer than 5 qualifying findings",
            created_at=datetime(2026, 4, 24, 10, 0, tzinfo=UTC),
        )
        store.upsert_report_section(section)
        rows = store.fetch_report_sections_for_run("run-ut4")
        assert len(rows) == 1
        assert rows[0].insufficient_evidence is True
        assert rows[0].decline_reason == "fewer than 5 qualifying findings"
        assert rows[0].markdown == ""
        assert rows[0].cited_finding_ids == []


def test_fetch_report_sections_ordering(tmp_path):
    """fetch_report_sections_for_run returns rows ordered by section_id ASC."""
    db = tmp_path / "lucid.sqlite3"
    initialize_db(db)
    with CorpusStore(db) as store:
        _seed_audit_run(store, run_id="run-ut5")
        for section_id in ["module_b_narrative", "exec_summary", "top_3_actions"]:
            store.upsert_report_section(ReportSection(
                audit_run_id="run-ut5",
                section_id=section_id,
                markdown=f"prose for {section_id}",
                cited_finding_ids=[],
                cited_turn_ids=[],
                insufficient_evidence=False,
                decline_reason=None,
                created_at=datetime.now(tz=UTC),
            ))
        rows = store.fetch_report_sections_for_run("run-ut5")
        assert [r.section_id for r in rows] == [
            "exec_summary", "module_b_narrative", "top_3_actions",
        ]
