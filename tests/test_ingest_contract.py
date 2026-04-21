"""Cross-adapter contract tests: security + canary + fingerprint determinism."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from lucid.ingest.base import (
    MAX_DISCOVER_FILES,
    IngestError,
    assert_file_size,
    assert_not_symlink_escape,
    fingerprint_corpus,
    safe_resolve_path,
)
from lucid.ingest.claude_ai import parse_memory_file
from lucid.ingest.claude_code import ClaudeCodeAdapter
from lucid.logging import configure_logging

CANARY = "LUCID_CANARY_SENTINEL_XYZ123"
FIXTURE_CLAUDE_AI = Path(__file__).parent / "fixtures" / "claude_ai"


# ----- path safety ----------------------------------------------------


def test_safe_resolve_path_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        safe_resolve_path(tmp_path / "nope")


def test_assert_not_symlink_escape_rejects_outside(tmp_path: Path) -> None:
    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path.parent / "lucid-fixture-outside"
    outside.mkdir(exist_ok=True)
    outside_file = outside / "target.txt"
    outside_file.write_text("payload")
    try:
        link = inside / "link.txt"
        link.symlink_to(outside_file)
        with pytest.raises(IngestError):
            assert_not_symlink_escape(link, inside.resolve())
    finally:
        outside_file.unlink(missing_ok=True)
        if outside.exists():
            outside.rmdir()


def test_assert_not_symlink_escape_allows_inside(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    child = root / "child.txt"
    child.write_text("ok")
    # Should not raise.
    assert_not_symlink_escape(child, root.resolve())


def test_assert_file_size_rejects_oversize(tmp_path: Path) -> None:
    f = tmp_path / "big.bin"
    with f.open("wb") as fh:
        fh.seek(1024 * 1024)
        fh.write(b"x")
    with pytest.raises(IngestError, match="exceeds cap"):
        assert_file_size(f, 1024)


# ----- canary sentinel (memory redaction) -----------------------------


def test_memory_content_never_reaches_debug_log(caplog: pytest.LogCaptureFixture) -> None:
    """Parsing memories.json at DEBUG level must not surface memory content.

    The sentinel lives inside the memory text; if any future code path
    logs `mf.conversations_memory` or equivalent, this test starts failing.
    """
    configure_logging("DEBUG")
    with caplog.at_level(logging.DEBUG):
        mf = parse_memory_file(FIXTURE_CLAUDE_AI / "memories.json")
    # Sanity: the sentinel IS in the parsed structure.
    assert mf.conversations_memory is not None
    assert CANARY in mf.conversations_memory
    # Invariant: it is NOT in any captured log record.
    for record in caplog.records:
        rendered = record.getMessage()
        assert CANARY not in rendered, (
            f"Canary leaked into log record: logger={record.name!r} level={record.levelname!r}"
        )


# ----- fingerprint determinism ----------------------------------------


def test_fingerprint_corpus_same_inputs_same_output() -> None:
    items = [("conv-1", "abc"), ("conv-2", "def")]
    assert fingerprint_corpus(items) == fingerprint_corpus(items)


def test_fingerprint_corpus_order_independent() -> None:
    items_a = [("conv-1", "abc"), ("conv-2", "def")]
    items_b = [("conv-2", "def"), ("conv-1", "abc")]
    assert fingerprint_corpus(items_a) == fingerprint_corpus(items_b)


def test_fingerprint_corpus_changes_when_content_changes() -> None:
    base = [("conv-1", "abc"), ("conv-2", "def")]
    edited = [("conv-1", "abc2"), ("conv-2", "def")]
    assert fingerprint_corpus(base) != fingerprint_corpus(edited)


def test_fingerprint_corpus_changes_when_conv_added() -> None:
    base = [("conv-1", "abc")]
    extended = [("conv-1", "abc"), ("conv-2", "def")]
    assert fingerprint_corpus(base) != fingerprint_corpus(extended)


# ----- discover cap ----------------------------------------------------


def test_claude_code_discover_enforces_file_cap(tmp_path: Path) -> None:
    adapter = ClaudeCodeAdapter(max_workers=1)
    # We won't actually create MAX_DISCOVER_FILES files; just test the constant
    # is a sane upper bound (positive, large, int).
    assert isinstance(MAX_DISCOVER_FILES, int)
    assert MAX_DISCOVER_FILES > 10_000
    # Empty tmp dir: discover returns empty list.
    (tmp_path / "proj").mkdir()
    assert adapter.discover(tmp_path) == []
