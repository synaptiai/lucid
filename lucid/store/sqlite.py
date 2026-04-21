"""CorpusStore — thin wrapper around sqlite3 / aiosqlite.

Sync helpers are the hot path for ingest (called from a ProcessPoolExecutor).
Async helpers land here for the orchestrator's custom-tool handlers in
Phase 5. The shapes are stable now so Phase 5 just fills in method bodies.

JSON columns: Pydantic models serialize to JSON strings with
`model_dump_json()` and come back through `model_validate_json()`. For
simple dicts / lists we use `orjson` directly — marginally faster and the
dep is already pinned.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

import orjson

from lucid.schemas import (
    AuditRun,
    Conversation,
    Finding,
    MemoryClaim,
    MemoryFile,
    Project,
    Turn,
)
from lucid.store.init import connect


class CorpusStore:
    """Synchronous handle to the corpus DB. Used by ingest + tests.

    Async reads for the orchestrator are added in Phase 5 via a sibling
    `AsyncCorpusStore` backed by aiosqlite.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve()
        self._conn: sqlite3.Connection | None = None

    # ----- connection lifecycle --------------------------------------

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = connect(self.path)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> CorpusStore:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ----- audit runs ------------------------------------------------

    def insert_audit_run(self, run: AuditRun) -> None:
        self.connect().execute(
            """
            INSERT INTO audit_runs (
                id, sources_json, source_paths_json, started_at, completed_at,
                corpus_stats_json, token_usage_json, sampling_config_json, status,
                corpus_fingerprint, prompt_versions_json, schema_version,
                skipped_modules_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.id,
                orjson.dumps([s.value for s in run.sources]).decode(),
                orjson.dumps({s.value: p for s, p in run.source_paths.items()}).decode(),
                run.started_at.isoformat(),
                run.completed_at.isoformat() if run.completed_at else None,
                run.corpus_stats.model_dump_json(),
                run.token_usage.model_dump_json(),
                run.sampling_config.model_dump_json(),
                run.status,
                run.corpus_fingerprint,
                orjson.dumps({m.value: v for m, v in run.prompt_versions.items()}).decode(),
                run.schema_version,
                orjson.dumps([m.value for m in run.skipped_modules]).decode(),
            ),
        )
        self._commit()

    # ----- conversations + turns ------------------------------------

    def insert_conversations(self, convs: Iterable[Conversation]) -> int:
        rows = [
            (
                c.id,
                c.source.value,
                c.source_path,
                c.created_at.isoformat(),
                c.updated_at.isoformat(),
                c.model,
                c.title,
                c.summary,
                c.turn_count,
                c.project_slug,
                orjson.dumps(c.metadata).decode(),
            )
            for c in convs
        ]
        self.connect().executemany(
            """
            INSERT INTO conversations (
                id, source, source_path, created_at, updated_at, model, title,
                summary, turn_count, project_slug, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._commit()
        return len(rows)

    def insert_turns(self, turns: Iterable[Turn]) -> int:
        rows = [
            (
                t.id,
                t.conversation_id,
                t.index,
                t.role.value,
                t.content,
                # list-of-discriminated-unions -> JSON via model_dump list comp
                orjson.dumps([b.model_dump(mode="json") for b in t.blocks]).decode(),
                t.timestamp.isoformat() if t.timestamp else None,
                t.parent_message_uuid,
                t.token_count,
            )
            for t in turns
        ]
        self.connect().executemany(
            """
            INSERT INTO turns (
                id, conversation_id, turn_index, role, content, blocks_json,
                timestamp, parent_message_uuid, token_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._commit()
        return len(rows)

    # ----- findings --------------------------------------------------

    def insert_finding(self, finding: Finding) -> None:
        self.connect().execute(
            """
            INSERT INTO findings (
                id, audit_run_id, conversation_id, turn_ids_json, turn_ids_hash,
                module, behavior, intensity, confidence, confidence_alpha,
                confidence_beta, quote_user, quote_assistant,
                evidence_quotes_json, explanation, citation, detected_by_json,
                detected_at, prompt_version, prompt_hash, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                finding.id,
                finding.audit_run_id,
                finding.conversation_id,
                orjson.dumps(finding.turn_ids).decode(),
                finding.turn_ids_hash,
                finding.module.value,
                finding.behavior,
                finding.intensity,
                finding.confidence,
                finding.confidence_alpha,
                finding.confidence_beta,
                finding.quote_user,
                finding.quote_assistant,
                orjson.dumps(finding.evidence_quotes).decode(),
                finding.explanation,
                finding.citation,
                orjson.dumps(finding.detected_by).decode(),
                finding.detected_at.isoformat(),
                finding.prompt_version,
                finding.prompt_hash,
                orjson.dumps(finding.metadata).decode(),
            ),
        )
        self._commit()

    # ----- projects + memory ----------------------------------------

    def insert_project(self, project: Project) -> None:
        self.connect().execute(
            """
            INSERT INTO projects (
                uuid, name, description, prompt_template, created_at,
                updated_at, doc_count, doc_char_total
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project.uuid,
                project.name,
                project.description,
                project.prompt_template,
                project.created_at.isoformat(),
                project.updated_at.isoformat(),
                project.doc_count,
                project.doc_char_total,
            ),
        )
        self._commit()

    def insert_memory_file(self, mf: MemoryFile) -> None:
        self.connect().execute(
            """
            INSERT INTO memory_files (
                account_uuid, conversations_memory, project_memories_json
            ) VALUES (?, ?, ?)
            """,
            (
                mf.account_uuid,
                mf.conversations_memory,
                orjson.dumps(mf.project_memories).decode(),
            ),
        )
        for claim in mf.extracted_claims:
            self.insert_memory_claim(claim, account_uuid=mf.account_uuid)
        self._commit()

    def insert_memory_claim(self, claim: MemoryClaim, *, account_uuid: str) -> None:
        self.connect().execute(
            """
            INSERT INTO memory_claims (id, account_uuid, source, claim_text, category)
            VALUES (?, ?, ?, ?, ?)
            """,
            (claim.id, account_uuid, claim.source, claim.claim_text, claim.category),
        )

    # ----- counters used by tests -----------------------------------

    def count(self, table: str) -> int:
        # Whitelist to avoid `table` becoming an injection vector.
        allowed = {
            "audit_runs",
            "conversations",
            "turns",
            "findings",
            "projects",
            "memory_files",
            "memory_claims",
            "embeddings",
            "module_progress",
        }
        if table not in allowed:
            raise ValueError(f"count(): refusing unknown table {table!r}")
        cursor = self.connect().execute(f"SELECT COUNT(*) FROM {table}")
        (n,) = cursor.fetchone()
        return int(n)

    def fetchall(self, sql: str, params: Sequence[object] = ()) -> list[sqlite3.Row]:
        cursor = self.connect().execute(sql, params)
        return cursor.fetchall()

    # ----- internals -------------------------------------------------

    def _commit(self) -> None:
        assert self._conn is not None
        self._conn.commit()
