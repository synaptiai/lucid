-- v1 → v2: add report_sections table for synthesis-layer agent prose.
--
-- Additive — no data migration. The table mirrors what schema.sql
-- contains for fresh installs; keep the two in sync if either side
-- evolves. See lucid/schemas.py::ReportSection for the Python-side
-- invariant; the CHECK below enforces it at the DB layer per the
-- two-layer discipline documented in CLAUDE.md.

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
