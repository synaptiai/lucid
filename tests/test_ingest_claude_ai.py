"""Claude.ai export adapter tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from lucid.ingest.base import IngestError
from lucid.ingest.claude_ai import (
    ClaudeAiAdapter,
    iter_conversations,
    parse_memory_file,
    parse_projects,
)
from lucid.schemas import Role, TextBlock, ToolResultBlock, ToolUseBlock

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "claude_ai"


def test_adapter_discover_requires_all_files(tmp_path: Path) -> None:
    adapter = ClaudeAiAdapter()
    # Empty dir -> missing files.
    with pytest.raises(IngestError, match="missing required file"):
        adapter.discover(tmp_path)


def test_iter_conversations_yields_both_fixtures() -> None:
    results = list(iter_conversations(FIXTURE_ROOT / "conversations.json"))
    assert len(results) == 2
    assert {p.conversation.id for p in results} == {"conv-aaa", "conv-bbb"}


def test_claude_ai_branch_reconstruction_preserves_parent_links() -> None:
    results = list(iter_conversations(FIXTURE_ROOT / "conversations.json"))
    branched = next(p for p in results if p.conversation.id == "conv-bbb")
    # Two regeneration children share a parent_message_uuid.
    children = [t for t in branched.turns if t.parent_message_uuid == "m-4b"]
    assert len(children) == 2
    assert {t.id for t in children} == {"m-5-regenA", "m-5-regenB"}


def test_claude_ai_mcp_metadata_preserved() -> None:
    results = list(iter_conversations(FIXTURE_ROOT / "conversations.json"))
    branched = next(p for p in results if p.conversation.id == "conv-bbb")
    tool_use_blocks = [b for t in branched.turns for b in t.blocks if isinstance(b, ToolUseBlock)]
    assert len(tool_use_blocks) == 1
    tu = tool_use_blocks[0]
    assert tu.is_mcp_app is True
    assert tu.mcp_integration == "Deployer"
    assert tu.mcp_server_url == "https://deploy.example.com/mcp"


def test_claude_ai_roles_mapped_from_sender() -> None:
    results = list(iter_conversations(FIXTURE_ROOT / "conversations.json"))
    single = next(p for p in results if p.conversation.id == "conv-aaa")
    assert [t.role for t in single.turns] == [Role.USER, Role.ASSISTANT]


def test_claude_ai_tool_result_coerces_content() -> None:
    results = list(iter_conversations(FIXTURE_ROOT / "conversations.json"))
    branched = next(p for p in results if p.conversation.id == "conv-bbb")
    tr = next(b for t in branched.turns for b in t.blocks if isinstance(b, ToolResultBlock))
    assert tr.content == "healthy"


def test_claude_ai_text_fallback_when_content_empty() -> None:
    """If `content` is empty but `text` is set, emit a synthetic TextBlock."""
    # Provided via a small inline fixture.
    import json

    tmp_file = FIXTURE_ROOT.parent / "_inline_empty_content.json"
    tmp_file.write_text(
        json.dumps(
            [
                {
                    "uuid": "conv-empty",
                    "name": "t",
                    "created_at": "2026-03-01T00:00:00+00:00",
                    "updated_at": "2026-03-01T00:00:00+00:00",
                    "chat_messages": [
                        {
                            "uuid": "m1",
                            "sender": "human",
                            "text": "plain text fallback",
                            "content": [],
                            "created_at": "2026-03-01T00:00:00+00:00",
                            "updated_at": "2026-03-01T00:00:00+00:00",
                            "parent_message_uuid": None,
                        }
                    ],
                }
            ]
        )
    )
    try:
        results = list(iter_conversations(tmp_file))
        assert len(results) == 1
        blocks = results[0].turns[0].blocks
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert blocks[0].text == "plain text fallback"
    finally:
        tmp_file.unlink()


def test_parse_projects_roundtrip() -> None:
    projects = parse_projects(FIXTURE_ROOT / "projects.json")
    assert len(projects) == 1
    p = projects[0]
    assert p.uuid == "proj-fixture-1"
    assert p.doc_count == 1
    assert p.doc_char_total == len("Keep answers tight and practical.")


def test_parse_memory_file_extracts_both_sections() -> None:
    mf = parse_memory_file(FIXTURE_ROOT / "memories.json")
    assert mf.account_uuid == "acct-1"
    assert mf.conversations_memory is not None
    assert "LUCID_CANARY_SENTINEL_XYZ123" in mf.conversations_memory
    assert "proj-fixture-1" in mf.project_memories


def test_adapter_parse_all_yields_both_fixtures() -> None:
    adapter = ClaudeAiAdapter()
    results = adapter.parse_all(FIXTURE_ROOT)
    assert len(results) == 2
    fp = adapter.fingerprint(results)
    assert isinstance(fp, str) and len(fp) == 64
