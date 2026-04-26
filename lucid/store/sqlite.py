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
import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import orjson

from lucid.schemas import (
    AuditRun,
    Conversation,
    CorpusStats,
    Finding,
    MemoryClaim,
    MemoryFile,
    ModuleName,
    ModuleProgress,
    ModuleProgressStatus,
    Project,
    ReportSection,
    SamplingConfigRecord,
    Source,
    SynthesisBlock,
    TokenUsage,
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

    def fetch_audit_run(self, run_id: str) -> AuditRun | None:
        """Load one :class:`AuditRun` row by id, or ``None`` if absent.

        The CLI calls this after :func:`lucid.run.run_audit` to feed the
        audit run into :func:`lucid.report.write_report`. Keeping the
        rehydration here (rather than in ``run.py``) means there is one
        canonical round-trip for the audit-run columns — including the
        JSON fields that roundtrip through Pydantic.
        """
        rows = self.fetchall(
            "SELECT id, sources_json, source_paths_json, started_at, completed_at, "
            "corpus_stats_json, token_usage_json, sampling_config_json, status, "
            "corpus_fingerprint, prompt_versions_json, schema_version, "
            "skipped_modules_json FROM audit_runs WHERE id = ?",
            (run_id,),
        )
        if not rows:
            return None
        row = rows[0]
        sources = [Source(s) for s in orjson.loads(row["sources_json"])]
        source_paths = {Source(s): p for s, p in orjson.loads(row["source_paths_json"]).items()}
        prompt_versions = {
            ModuleName(m): v for m, v in orjson.loads(row["prompt_versions_json"]).items()
        }
        skipped_modules = [ModuleName(m) for m in orjson.loads(row["skipped_modules_json"])]
        return AuditRun(
            id=row["id"],
            sources=sources,
            source_paths=source_paths,
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=(
                datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
            ),
            corpus_stats=CorpusStats.model_validate_json(row["corpus_stats_json"]),
            token_usage=TokenUsage.model_validate_json(row["token_usage_json"]),
            sampling_config=SamplingConfigRecord.model_validate_json(row["sampling_config_json"]),
            status=row["status"],
            corpus_fingerprint=row["corpus_fingerprint"],
            prompt_versions=prompt_versions,
            schema_version=int(row["schema_version"]),
            skipped_modules=skipped_modules,
        )

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

    def fetch_findings_for_run(self, audit_run_id: str) -> list[Finding]:
        """Load every persisted :class:`Finding` for an audit run.

        Mirrors the rehydration done inside the orchestrator's
        ``_load_findings`` (which is scoped per-conversation for Module
        C's second pass). The CLI calls this once after a session
        completes to feed the report generator.
        """
        rows = self.fetchall(
            "SELECT id, audit_run_id, conversation_id, turn_ids_json, turn_ids_hash, "
            "module, behavior, intensity, confidence, confidence_alpha, "
            "confidence_beta, quote_user, quote_assistant, evidence_quotes_json, "
            "explanation, citation, detected_by_json, detected_at, prompt_version, "
            "prompt_hash, metadata_json FROM findings WHERE audit_run_id = ? "
            "ORDER BY detected_at ASC",
            (audit_run_id,),
        )
        out: list[Finding] = []
        for row in rows:
            out.append(
                Finding(
                    id=row["id"],
                    audit_run_id=row["audit_run_id"],
                    conversation_id=row["conversation_id"],
                    turn_ids=orjson.loads(row["turn_ids_json"]) if row["turn_ids_json"] else [],
                    turn_ids_hash=row["turn_ids_hash"],
                    module=ModuleName(row["module"]),
                    behavior=row["behavior"],
                    intensity=row["intensity"],
                    confidence=row["confidence"],
                    confidence_alpha=row["confidence_alpha"],
                    confidence_beta=row["confidence_beta"],
                    quote_user=row["quote_user"],
                    quote_assistant=row["quote_assistant"],
                    evidence_quotes=(
                        orjson.loads(row["evidence_quotes_json"])
                        if row["evidence_quotes_json"]
                        else []
                    ),
                    explanation=row["explanation"],
                    citation=row["citation"],
                    detected_by=orjson.loads(row["detected_by_json"]),
                    detected_at=datetime.fromisoformat(row["detected_at"]),
                    prompt_version=row["prompt_version"],
                    prompt_hash=row["prompt_hash"],
                    metadata=orjson.loads(row["metadata_json"]) if row["metadata_json"] else {},
                )
            )
        return out

    # ----- module_progress (per-module lifecycle for resume) --------

    def mark_module_started(self, audit_run_id: str, module: ModuleName) -> None:
        """Record that a module has started running for ``audit_run_id``.

        Idempotent across re-invocations: the row is upserted and the
        ``completed_at`` / ``error_message`` columns are cleared so a
        retry presents as ``running`` again rather than carrying over
        the previous attempt's terminal state.
        """
        now = datetime.now(tz=UTC).isoformat()
        self.connect().execute(
            """
            INSERT INTO module_progress (
                audit_run_id, module, status, started_at, completed_at, error_message
            ) VALUES (?, ?, 'running', ?, NULL, NULL)
            ON CONFLICT (audit_run_id, module) DO UPDATE SET
                status = 'running',
                started_at = excluded.started_at,
                completed_at = NULL,
                error_message = NULL
            """,
            (audit_run_id, module.value, now),
        )
        self._commit()

    def mark_module_finished(
        self,
        audit_run_id: str,
        module: ModuleName,
        *,
        status: ModuleProgressStatus,
        error_message: str | None = None,
    ) -> None:
        """Set a module's terminal status (``completed``/``failed``/``skipped``).

        Upserts so that short-circuit modules (those that never go
        through :meth:`mark_module_started` — e.g. Module D without
        ``--include-module-d``) leave a row anyway. ``started_at`` and
        ``completed_at`` carry the same skip-decision timestamp on
        the upsert path.
        """
        if status in {"running", "pending"}:
            raise ValueError(f"mark_module_finished requires a terminal status, got {status!r}")
        now = datetime.now(tz=UTC).isoformat()
        self.connect().execute(
            """
            INSERT INTO module_progress (
                audit_run_id, module, status, started_at, completed_at, error_message
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (audit_run_id, module) DO UPDATE SET
                status = excluded.status,
                completed_at = excluded.completed_at,
                error_message = excluded.error_message
            """,
            (audit_run_id, module.value, status, now, now, error_message),
        )
        self._commit()

    def fetch_module_progress(self, audit_run_id: str) -> list[ModuleProgress]:
        """Return every ``module_progress`` row for ``audit_run_id``.

        Order is module-name ascending so the report and resume logic
        see a stable sequence regardless of insertion order.
        """
        rows = self.fetchall(
            "SELECT audit_run_id, module, status, started_at, completed_at, "
            "error_message FROM module_progress WHERE audit_run_id = ? "
            "ORDER BY module ASC",
            (audit_run_id,),
        )
        out: list[ModuleProgress] = []
        for row in rows:
            out.append(
                ModuleProgress(
                    audit_run_id=row["audit_run_id"],
                    module=ModuleName(row["module"]),
                    status=row["status"],
                    started_at=(
                        datetime.fromisoformat(row["started_at"]) if row["started_at"] else None
                    ),
                    completed_at=(
                        datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
                    ),
                    error_message=row["error_message"],
                )
            )
        return out

    # ----- report sections (synthesis-layer prose) ------------------

    def upsert_report_section(self, section: ReportSection) -> None:
        """Insert or update one :class:`ReportSection` row.

        Uses ``ON CONFLICT(audit_run_id, section_id) DO UPDATE SET ...``
        so re-running synthesis against the same audit replaces the row
        in place (matches the UNIQUE (audit_run_id, section_id) semantics
        in ``store/schema.sql``). The row ``id`` is minted on first insert
        and preserved across updates — the UPDATE only touches the data
        columns, never ``id``.

        ``blocks`` + ``citation_confidence`` are the Sonnet post-processor
        output (Phase 5.6). On first insert they are typically empty /
        NULL — the section ships as Opus markdown — and then a second
        upsert after post-processing fills them in via ON CONFLICT,
        preserving the row id.
        """
        now_iso = section.created_at.isoformat()
        blocks_json = orjson.dumps([b.model_dump(mode="json") for b in section.blocks]).decode()
        self.connect().execute(
            """
            INSERT INTO report_sections (
                id, audit_run_id, section_id, markdown,
                cited_finding_ids_json, cited_turn_ids_json,
                insufficient_evidence, decline_reason, created_at,
                blocks_json, citation_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (audit_run_id, section_id) DO UPDATE SET
                markdown = excluded.markdown,
                cited_finding_ids_json = excluded.cited_finding_ids_json,
                cited_turn_ids_json = excluded.cited_turn_ids_json,
                insufficient_evidence = excluded.insufficient_evidence,
                decline_reason = excluded.decline_reason,
                created_at = excluded.created_at,
                blocks_json = excluded.blocks_json,
                citation_confidence = excluded.citation_confidence
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
                now_iso,
                blocks_json,
                section.citation_confidence,
            ),
        )
        self._commit()

    def fetch_report_sections_for_run(self, audit_run_id: str) -> list[ReportSection]:
        """Load every :class:`ReportSection` for ``audit_run_id``.

        Rows are ordered by ``section_id`` ASC so downstream renderers
        see a stable, alphabetical sequence regardless of insertion
        order. SQLite returns ``insufficient_evidence`` as an ``int``
        (0/1); we cast to ``bool`` on the way out to match the Pydantic
        field type. ``blocks_json`` is deserialized back through
        :class:`SynthesisBlock` so callers see structured models, not raw
        JSON; ``citation_confidence`` round-trips as ``float | None``.
        """
        rows = self.fetchall(
            "SELECT id, audit_run_id, section_id, markdown, "
            "cited_finding_ids_json, cited_turn_ids_json, "
            "insufficient_evidence, decline_reason, created_at, "
            "blocks_json, citation_confidence "
            "FROM report_sections WHERE audit_run_id = ? "
            "ORDER BY section_id ASC",
            (audit_run_id,),
        )
        out: list[ReportSection] = []
        for row in rows:
            raw_blocks = orjson.loads(row["blocks_json"]) if row["blocks_json"] else []
            blocks = [SynthesisBlock.model_validate(b) for b in raw_blocks]
            citation_confidence = row["citation_confidence"]
            out.append(
                ReportSection(
                    audit_run_id=row["audit_run_id"],
                    section_id=row["section_id"],
                    markdown=row["markdown"],
                    cited_finding_ids=orjson.loads(row["cited_finding_ids_json"]),
                    cited_turn_ids=orjson.loads(row["cited_turn_ids_json"]),
                    insufficient_evidence=bool(row["insufficient_evidence"]),
                    decline_reason=row["decline_reason"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    blocks=blocks,
                    citation_confidence=(
                        float(citation_confidence) if citation_confidence is not None else None
                    ),
                )
            )
        return out

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

    def fetch_all_projects(self) -> list[Project]:
        """Load every ingested :class:`Project` row.

        Module H uses the name + UUID to fuzzy-match audit conversations
        back onto projects (conversations don't carry ``project_uuid``
        directly in Claude.ai exports; see ``module_h_memory.py``).
        """
        rows = self.fetchall(
            "SELECT uuid, name, description, prompt_template, created_at, "
            "updated_at, doc_count, doc_char_total FROM projects",
        )
        out: list[Project] = []
        for row in rows:
            out.append(
                Project(
                    uuid=row["uuid"],
                    name=row["name"],
                    description=row["description"],
                    prompt_template=row["prompt_template"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    doc_count=int(row["doc_count"]),
                    doc_char_total=int(row["doc_char_total"]),
                )
            )
        return out

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

    def fetch_memory_files(self) -> list[MemoryFile]:
        """Load every ingested :class:`MemoryFile`.

        Memory-file rows carry ``conversations_memory`` (top-level) plus
        ``project_memories_json`` (dict keyed by project UUID). We
        rehydrate without attaching ``extracted_claims`` — that's the
        ingest-time snapshot and Module H doesn't trust it as the claim
        set (H.1 re-extracts with its own prompt).
        """
        rows = self.fetchall(
            "SELECT account_uuid, conversations_memory, project_memories_json "
            "FROM memory_files ORDER BY account_uuid"
        )
        out: list[MemoryFile] = []
        for row in rows:
            project_memories = (
                orjson.loads(row["project_memories_json"]) if row["project_memories_json"] else {}
            )
            out.append(
                MemoryFile(
                    account_uuid=row["account_uuid"],
                    conversations_memory=row["conversations_memory"],
                    project_memories=project_memories,
                )
            )
        return out

    # ----- embeddings (Module H) ------------------------------------

    def insert_embedding(
        self,
        *,
        id: str,
        conversation_id: str | None,
        turn_id: str | None,
        chunk_text: str,
        vector_blob: bytes,
        dim: int,
        model: str,
    ) -> None:
        """Insert one embedding row.

        ``vector_blob`` is the raw bytes of a numpy float32 array of
        length ``dim``. Callers construct it with
        ``vec.astype(np.float32).tobytes()``; readers recover via
        ``np.frombuffer(blob, dtype=np.float32)``. Keeping the blob
        opaque at the DB layer preserves cross-platform portability.
        """
        self.connect().execute(
            """
            INSERT OR REPLACE INTO embeddings (
                id, conversation_id, turn_id, chunk_text,
                vector_blob, dim, model, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                id,
                conversation_id,
                turn_id,
                chunk_text,
                vector_blob,
                dim,
                model,
                datetime.now(tz=UTC).isoformat(),
            ),
        )
        self._commit()

    def insert_embeddings(
        self,
        rows: Iterable[tuple[str, str | None, str | None, str, bytes, int, str]],
    ) -> int:
        """Bulk insert embedding rows. Each tuple is
        ``(id, conversation_id, turn_id, chunk_text, vector_blob, dim, model)``.

        Uses ``INSERT OR REPLACE`` so re-embedding a chunk overwrites
        the stale entry rather than raising a primary-key conflict.
        This is the resume path: if a prior audit crashed mid-embedding,
        the next run re-embeds whatever didn't persist and overwrites
        nothing valuable.
        """
        now = datetime.now(tz=UTC).isoformat()
        materialized = [(*row, now) for row in rows]
        self.connect().executemany(
            """
            INSERT OR REPLACE INTO embeddings (
                id, conversation_id, turn_id, chunk_text,
                vector_blob, dim, model, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            materialized,
        )
        self._commit()
        return len(materialized)

    def fetch_embeddings_for_conversations(
        self,
        conversation_ids: Sequence[str],
        *,
        model: str | None = None,
    ) -> list[sqlite3.Row]:
        """Fetch persisted embedding rows for the given conversations.

        Returns rows with columns ``id, conversation_id, turn_id,
        chunk_text, vector_blob, dim, model``. Callers rebuild the numpy
        matrix by stacking ``np.frombuffer(row["vector_blob"],
        np.float32)`` per row.
        """
        if not conversation_ids:
            return []
        placeholders = ",".join("?" for _ in conversation_ids)
        params: list[object] = list(conversation_ids)
        model_clause = ""
        if model is not None:
            model_clause = " AND model = ?"
            params.append(model)
        return self.fetchall(
            f"SELECT id, conversation_id, turn_id, chunk_text, vector_blob, "
            f"dim, model FROM embeddings WHERE conversation_id IN "
            f"({placeholders}){model_clause}",
            params,
        )

    def fetch_embedding_ids(
        self,
        ids: Sequence[str],
        *,
        model: str | None = None,
    ) -> set[str]:
        """Return the subset of ``ids`` that are already present in the
        embeddings table — cache-hit lookup before embedding a batch."""
        if not ids:
            return set()
        placeholders = ",".join("?" for _ in ids)
        params: list[object] = list(ids)
        model_clause = ""
        if model is not None:
            model_clause = " AND model = ?"
            params.append(model)
        rows = self.fetchall(
            f"SELECT id FROM embeddings WHERE id IN ({placeholders}){model_clause}",
            params,
        )
        return {r["id"] for r in rows}

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
