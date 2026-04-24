-- Lucid storage schema v1.
--
-- Design notes:
--   * Foreign keys must be enabled per-connection via `PRAGMA foreign_keys = ON;`
--     (the library does this in its connection factory). The schema declares
--     FK constraints anyway so a future tool that reads the DB sees them.
--   * `metadata`, `blocks`, and similar flexible structures are JSON strings
--     (TEXT) so the Pydantic discriminated unions round-trip losslessly.
--   * CHECK constraints enforce the Finding provenance invariants documented
--     in CLAUDE.md §Findings provenance. Inserts missing a required field
--     fail at the DB level regardless of what the application did.
--   * `findings.turn_ids_hash` is sha256(sorted(turn_ids)) and is part of
--     the idempotency key so re-running a module over the same turns does
--     not double-count.
--   * `module_progress` is the resume contract: an audit that crashes mid-run
--     restarts from the last successfully-completed module here.
--   * `embeddings` stores Voyage outputs as a BLOB of float32s; Module H
--     loads the whole matrix into numpy at audit start.

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------------
-- Audit-level tables
-- ------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS audit_runs (
    id                    TEXT PRIMARY KEY,
    sources_json          TEXT NOT NULL,   -- JSON array of Source enum values
    source_paths_json     TEXT NOT NULL,   -- JSON dict[Source, str]
    started_at            TEXT NOT NULL,   -- ISO 8601 UTC
    completed_at          TEXT,            -- ISO 8601 UTC, NULL while running
    corpus_stats_json     TEXT NOT NULL,
    token_usage_json      TEXT NOT NULL,
    sampling_config_json  TEXT NOT NULL,
    status                TEXT NOT NULL CHECK (
        status IN ('running', 'completed', 'failed', 'partial', 'aborted_pre_spend')
    ),
    corpus_fingerprint    TEXT NOT NULL,
    prompt_versions_json  TEXT NOT NULL,
    schema_version        INTEGER NOT NULL CHECK (schema_version >= 1),
    skipped_modules_json  TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_audit_runs_status ON audit_runs(status);


-- ------------------------------------------------------------------
-- Corpus tables
-- ------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS conversations (
    id             TEXT PRIMARY KEY,
    source         TEXT NOT NULL CHECK (source IN ('claude-code', 'claude-ai')),
    source_path    TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    model          TEXT,
    title          TEXT,
    summary        TEXT,
    turn_count     INTEGER NOT NULL CHECK (turn_count >= 0),
    project_slug   TEXT,
    metadata_json  TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_conversations_source ON conversations(source);
CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at);
CREATE INDEX IF NOT EXISTS idx_conversations_project ON conversations(project_slug);


CREATE TABLE IF NOT EXISTS turns (
    id                    TEXT PRIMARY KEY,
    conversation_id       TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn_index            INTEGER NOT NULL CHECK (turn_index >= 0),
    role                  TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content               TEXT NOT NULL,
    blocks_json           TEXT NOT NULL DEFAULT '[]',
    timestamp             TEXT,
    parent_message_uuid   TEXT,
    token_count           INTEGER CHECK (token_count IS NULL OR token_count >= 0),
    UNIQUE (conversation_id, turn_index)
);

CREATE INDEX IF NOT EXISTS idx_turns_conversation ON turns(conversation_id);
CREATE INDEX IF NOT EXISTS idx_turns_parent ON turns(parent_message_uuid);


CREATE TABLE IF NOT EXISTS projects (
    uuid             TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    description      TEXT,
    prompt_template  TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    doc_count        INTEGER NOT NULL CHECK (doc_count >= 0),
    doc_char_total   INTEGER NOT NULL CHECK (doc_char_total >= 0)
);


CREATE TABLE IF NOT EXISTS memory_files (
    account_uuid            TEXT PRIMARY KEY,
    conversations_memory    TEXT,
    project_memories_json   TEXT NOT NULL DEFAULT '{}'
);


CREATE TABLE IF NOT EXISTS memory_claims (
    id            TEXT PRIMARY KEY,
    account_uuid  TEXT NOT NULL REFERENCES memory_files(account_uuid) ON DELETE CASCADE,
    source        TEXT NOT NULL,   -- 'conversations_memory' or 'project_memories.<uuid>'
    claim_text    TEXT NOT NULL,
    category      TEXT
);

CREATE INDEX IF NOT EXISTS idx_memory_claims_account ON memory_claims(account_uuid);


-- ------------------------------------------------------------------
-- Findings + progress
-- ------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS findings (
    id                TEXT PRIMARY KEY,
    audit_run_id      TEXT NOT NULL REFERENCES audit_runs(id) ON DELETE CASCADE,
    conversation_id   TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    turn_ids_json     TEXT NOT NULL DEFAULT '[]',
    turn_ids_hash     TEXT NOT NULL,
    module            TEXT NOT NULL CHECK (
        module IN ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')
    ),
    behavior          TEXT NOT NULL CHECK (length(behavior) > 0),
    intensity         INTEGER CHECK (intensity IS NULL OR (intensity BETWEEN 1 AND 3)),
    confidence        REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    confidence_alpha  REAL CHECK (confidence_alpha IS NULL OR confidence_alpha > 0),
    confidence_beta   REAL CHECK (confidence_beta IS NULL OR confidence_beta > 0),
    quote_user        TEXT,
    quote_assistant   TEXT,
    evidence_quotes_json TEXT NOT NULL DEFAULT '[]',
    explanation       TEXT NOT NULL CHECK (length(explanation) > 0),
    citation          TEXT NOT NULL CHECK (length(citation) > 0),
    detected_by_json  TEXT NOT NULL CHECK (detected_by_json != '[]'),
    detected_at       TEXT NOT NULL,
    prompt_version    TEXT NOT NULL CHECK (length(prompt_version) > 0),
    prompt_hash       TEXT NOT NULL CHECK (length(prompt_hash) > 0),
    metadata_json     TEXT NOT NULL DEFAULT '{}',

    -- Idempotency: same (run, module, conversation, turns, behavior) must
    -- collapse to one row even if the module re-runs.
    UNIQUE (audit_run_id, module, conversation_id, turn_ids_hash, behavior)
);

CREATE INDEX IF NOT EXISTS idx_findings_run ON findings(audit_run_id);
CREATE INDEX IF NOT EXISTS idx_findings_module ON findings(module);
CREATE INDEX IF NOT EXISTS idx_findings_conv ON findings(conversation_id);


CREATE TABLE IF NOT EXISTS module_progress (
    audit_run_id  TEXT NOT NULL REFERENCES audit_runs(id) ON DELETE CASCADE,
    module        TEXT NOT NULL CHECK (
        module IN ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')
    ),
    status        TEXT NOT NULL CHECK (
        status IN ('pending', 'running', 'completed', 'failed', 'skipped')
    ),
    started_at    TEXT,
    completed_at  TEXT,
    error_message TEXT,
    PRIMARY KEY (audit_run_id, module)
);


-- ------------------------------------------------------------------
-- Voyage embeddings cache (Module H)
-- ------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS embeddings (
    id                 TEXT PRIMARY KEY,         -- sha256 of the chunk text
    conversation_id    TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    turn_id            TEXT REFERENCES turns(id) ON DELETE CASCADE,
    chunk_text         TEXT NOT NULL,
    vector_blob        BLOB NOT NULL,            -- float32[dim]
    dim                INTEGER NOT NULL CHECK (dim > 0),
    model              TEXT NOT NULL,            -- 'voyage-3-large' etc.
    created_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embeddings_conv ON embeddings(conversation_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_turn ON embeddings(turn_id);


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
    section_id               TEXT NOT NULL CHECK (length(section_id) BETWEEN 1 AND 64),
    markdown                 TEXT NOT NULL DEFAULT '',
    cited_finding_ids_json   TEXT NOT NULL DEFAULT '[]',
    cited_turn_ids_json      TEXT NOT NULL DEFAULT '[]',
    insufficient_evidence    INTEGER NOT NULL DEFAULT 0 CHECK (insufficient_evidence IN (0, 1)),
    decline_reason           TEXT,
    created_at               TEXT NOT NULL,
    blocks_json              TEXT NOT NULL DEFAULT '[]',
    citation_confidence      REAL CHECK (
        citation_confidence IS NULL
        OR (citation_confidence >= 0.0 AND citation_confidence <= 1.0)
    ),
    CHECK (
        (insufficient_evidence = 1
         AND length(markdown) = 0
         AND cited_finding_ids_json = '[]'
         AND cited_turn_ids_json = '[]'
         AND decline_reason IS NOT NULL
         AND length(decline_reason) > 0)
        OR
        (insufficient_evidence = 0
         AND length(markdown) > 0
         AND decline_reason IS NULL)
    ),
    UNIQUE (audit_run_id, section_id)
);
