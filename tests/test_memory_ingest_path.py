"""Coverage for the CLI's memory-file persistence (B4).

The plan's B4 gap: ``parse_memory_file`` and ``insert_memory_file``
both existed before Phase 7 but no caller wired them together. Module
H's ``fetch_memory_files()`` therefore returned an empty list even
when a real ``memories.json`` was present, and Module H produced no
findings.

These tests exercise :func:`lucid.cli._persist_memories_for_sources`
end-to-end against the committed demo corpus and confirm:

- a ``memories.json`` from a Claude.ai source path lands in
  ``memory_files``;
- Claude Code source paths are silently skipped (no memories file);
- re-running on the same DB does not double-insert (idempotent).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lucid.cli import _persist_memories_for_sources
from lucid.schemas import Source
from lucid.store.sqlite import CorpusStore

DEMO_CORPUS_PATH = Path(__file__).resolve().parent.parent / "demo" / "corpus"


def test_demo_corpus_has_memories_file() -> None:
    """Sanity check: the committed demo corpus has the memories.json
    we rely on. If this fails, fix the corpus before debugging the
    helper."""
    assert (DEMO_CORPUS_PATH / "memories.json").is_file()


def test_persist_memories_for_claude_ai_source_inserts_one_row(tmp_path: Path) -> None:
    """The Claude.ai source path's memories.json is read and persisted.

    Maps directly to B4: confirms the call chain
    ``parse_memory_file`` → ``CorpusStore.insert_memory_file`` runs
    and exactly one row lands.
    """
    persisted = _persist_memories_for_sources(
        tmp_path / ".lucid",
        {Source.CLAUDE_AI: DEMO_CORPUS_PATH},
    )
    assert persisted == 1

    db_path = tmp_path / ".lucid" / "lucid.sqlite3"
    with CorpusStore(db_path) as store:
        rows = store.fetchall("SELECT account_uuid, conversations_memory FROM memory_files")
    assert len(rows) == 1
    assert rows[0]["account_uuid"] == "demo-acct-1"
    # The fixture's conversations_memory is the well-known demo string.
    assert "backend engineer" in (rows[0]["conversations_memory"] or "")


def test_persist_memories_skips_claude_code_source(tmp_path: Path) -> None:
    """Claude Code paths have no memories file; the helper should
    silently no-op rather than raising or mis-reading another file."""
    persisted = _persist_memories_for_sources(
        tmp_path / ".lucid",
        {Source.CLAUDE_CODE: DEMO_CORPUS_PATH},  # path content is irrelevant
    )
    assert persisted == 0
    db_path = tmp_path / ".lucid" / "lucid.sqlite3"
    with CorpusStore(db_path) as store:
        rows = store.fetchall("SELECT * FROM memory_files")
    assert rows == []


def test_persist_memories_is_idempotent(tmp_path: Path) -> None:
    """Re-running the helper on the same DB does not double-insert.

    The helper checks the existing ``account_uuid`` set before
    inserting; without that guard, sqlite3 would raise on the second
    call (PRIMARY KEY conflict on ``memory_files.account_uuid``).
    """
    paths = {Source.CLAUDE_AI: DEMO_CORPUS_PATH}
    first = _persist_memories_for_sources(tmp_path / ".lucid", paths)
    second = _persist_memories_for_sources(tmp_path / ".lucid", paths)
    assert first == 1
    assert second == 0
    db_path = tmp_path / ".lucid" / "lucid.sqlite3"
    with CorpusStore(db_path) as store:
        (count,) = store.connect().execute("SELECT COUNT(*) FROM memory_files").fetchone()
    assert count == 1


def test_persist_memories_missing_file_no_ops(tmp_path: Path) -> None:
    """A Claude.ai path without a memories.json is silently skipped.

    Some real exports omit the file (older vintages, partial exports);
    that should not crash the audit.
    """
    empty_export = tmp_path / "empty-export"
    empty_export.mkdir()
    persisted = _persist_memories_for_sources(
        tmp_path / ".lucid",
        {Source.CLAUDE_AI: empty_export},
    )
    assert persisted == 0


def test_persist_memories_warns_on_malformed_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A malformed memories.json is skipped with a visible warning,
    not silently. Users running against a broken export should see
    the failure, not discover an empty Module H section in the report
    and wonder why."""
    bad_export = tmp_path / "bad-export"
    bad_export.mkdir()
    # account_uuid is required; a dict missing it trips the parser.
    (bad_export / "memories.json").write_text(json.dumps({"not_what_you_expect": True}))
    persisted = _persist_memories_for_sources(
        tmp_path / ".lucid",
        {Source.CLAUDE_AI: bad_export},
    )
    assert persisted == 0
    # Strip newlines (not whitespace) before substring search: rich
    # wraps long paths to terminal width, which on narrow CI runners
    # splits ``memories.json`` mid-token (e.g. ``memori\nes.json``).
    # ``replace("\n", "")`` rejoins the token; collapsing whitespace
    # would leave a stray space mid-filename.
    out_unwrapped = capsys.readouterr().out.replace("\n", "")
    assert "Skipping" in out_unwrapped and "memories.json" in out_unwrapped
