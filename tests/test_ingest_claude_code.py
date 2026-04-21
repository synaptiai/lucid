"""Claude Code JSONL adapter tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from lucid.ingest.base import IngestError
from lucid.ingest.claude_code import ClaudeCodeAdapter, parse_session_file
from lucid.schemas import Role, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "claude_code"
SESSION_FILE = FIXTURE_ROOT / "a-test-project" / "session-001.jsonl"


def test_parse_session_returns_one_conversation() -> None:
    parsed = parse_session_file(SESSION_FILE)
    assert parsed is not None
    assert parsed.conversation.source.value == "claude-code"
    assert parsed.conversation.id == "sess-001"
    assert parsed.conversation.project_slug == "a-test-project"
    assert parsed.conversation.turn_count == 4
    # 'summary' records are skipped.
    assert len(parsed.turns) == 4


def test_parse_session_block_types_all_present() -> None:
    parsed = parse_session_file(SESSION_FILE)
    assert parsed is not None
    all_blocks = [b for t in parsed.turns for b in t.blocks]
    assert any(isinstance(b, TextBlock) for b in all_blocks)
    assert any(isinstance(b, ThinkingBlock) for b in all_blocks)
    assert any(isinstance(b, ToolUseBlock) for b in all_blocks)
    assert any(isinstance(b, ToolResultBlock) for b in all_blocks)


def test_parse_session_role_mapping() -> None:
    parsed = parse_session_file(SESSION_FILE)
    assert parsed is not None
    roles = [t.role for t in parsed.turns]
    assert roles == [Role.USER, Role.ASSISTANT, Role.USER, Role.ASSISTANT]


def test_parse_session_parent_chain() -> None:
    parsed = parse_session_file(SESSION_FILE)
    assert parsed is not None
    chain = [t.parent_message_uuid for t in parsed.turns]
    assert chain == [None, "turn-1", "turn-2", "turn-3"]


def test_parse_session_plaintext_rendering() -> None:
    parsed = parse_session_file(SESSION_FILE)
    assert parsed is not None
    turn1 = parsed.turns[0]
    assert "list the files" in turn1.content.lower()


def test_parse_session_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.jsonl"
    f.write_text("")
    assert parse_session_file(f) is None


def test_parse_session_summary_only_file(tmp_path: Path) -> None:
    f = tmp_path / "summary_only.jsonl"
    f.write_text('{"type": "summary", "summary": "noop", "sessionId": "s"}\n')
    assert parse_session_file(f) is None


def test_parse_session_oversize_rejected(tmp_path: Path) -> None:
    # Write a file that declares size past the cap without actually allocating it.
    f = tmp_path / "huge.jsonl"
    # seek + write a single byte to inflate apparent size beyond MAX_SESSION_FILE_SIZE.
    with f.open("wb") as fh:
        fh.seek(60 * 1024 * 1024)
        fh.write(b"x")
    with pytest.raises(IngestError, match="exceeds cap"):
        parse_session_file(f)


def test_parse_session_malformed_line_is_skipped(tmp_path: Path) -> None:
    f = tmp_path / "mixed.jsonl"
    f.write_text(
        '{"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]},'
        ' "timestamp": "2026-04-10T12:00:00Z", "sessionId": "s", "uuid": "u1", "parentUuid": null}\n'
        "not json at all\n"
        '{"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "hey"}]},'
        ' "timestamp": "2026-04-10T12:00:01Z", "sessionId": "s", "uuid": "u2", "parentUuid": "u1"}\n'
    )
    parsed = parse_session_file(f)
    assert parsed is not None
    assert parsed.conversation.turn_count == 2


# ----- adapter-level discover -----------------------------------------


def test_discover_respects_root_boundary(tmp_path: Path) -> None:
    # Put a valid jsonl inside tmp_path, and another outside linked via symlink.
    outside_dir = tmp_path.parent / "outside-fixture"
    outside_dir.mkdir(exist_ok=True)
    outside = outside_dir / "escape.jsonl"
    outside.write_text('{"type": "summary", "sessionId": "x"}\n')
    try:
        inside_dir = tmp_path / "proj"
        inside_dir.mkdir()
        # Good file stays inside.
        good = inside_dir / "good.jsonl"
        good.write_text('{"type": "summary", "sessionId": "g"}\n')
        # Symlink that points outside root.
        link = inside_dir / "bad.jsonl"
        link.symlink_to(outside)

        adapter = ClaudeCodeAdapter(max_workers=1)
        with pytest.raises(IngestError, match="resolves outside"):
            adapter.discover(tmp_path)
    finally:
        outside.unlink(missing_ok=True)
        if outside_dir.exists():
            outside_dir.rmdir()


def test_parse_all_serial_mode(tmp_path: Path) -> None:
    # Copy the fixture into a fresh tmp tree so parse_all walks it deterministically.
    dest = tmp_path / "proj"
    dest.mkdir()
    target = dest / "s.jsonl"
    target.write_text(SESSION_FILE.read_text())

    adapter = ClaudeCodeAdapter(max_workers=1)
    results = adapter.parse_all(tmp_path)
    assert len(results) == 1
    fp = adapter.fingerprint(results)
    assert isinstance(fp, str) and len(fp) == 64  # sha256 hex


def test_fingerprint_stable_same_parse(tmp_path: Path) -> None:
    dest = tmp_path / "proj"
    dest.mkdir()
    target = dest / "s.jsonl"
    target.write_text(SESSION_FILE.read_text())

    adapter = ClaudeCodeAdapter(max_workers=1)
    fp1 = adapter.fingerprint(adapter.parse_all(tmp_path))
    fp2 = adapter.fingerprint(adapter.parse_all(tmp_path))
    assert fp1 == fp2
