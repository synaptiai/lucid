-- Phase 5.6a: Sonnet post-processor output on report_sections.
--
-- Adds blocks_json (SynthesisBlock list as JSON, default empty array)
-- and citation_confidence (0.0-1.0 or NULL for "not yet post-processed").
-- SQLite ALTER TABLE ADD COLUMN supports DEFAULT only for scalar defaults,
-- so blocks_json defaults to the empty-array sentinel '[]' matching the
-- schema.sql contract. Existing rows get the default backfilled.
--
-- Keep this file in sync with schema.sql's report_sections table.

ALTER TABLE report_sections ADD COLUMN blocks_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE report_sections ADD COLUMN citation_confidence REAL CHECK (
    citation_confidence IS NULL
    OR (citation_confidence >= 0.0 AND citation_confidence <= 1.0)
);
