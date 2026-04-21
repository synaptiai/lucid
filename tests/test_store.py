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
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
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
