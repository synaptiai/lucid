"""Claude.ai export adapter.

The export is an unzipped directory with four files:
`conversations.json`, `projects.json`, `memories.json`, `users.json`.
`conversations.json` can be several hundred MB, so it is parsed with
`ijson.items(f, 'item')` — incremental, bounded memory. The other three
are small and parsed with stdlib `json`.

Important traits (BUILD_GUIDE §3.2, confirmed against a real export):
- No `model` field anywhere. Module G (attribution) infers it later.
- No `project_uuid` field on conversations. Projects are their own file.
- `parent_message_uuid` enables branch detection. Root value is
  `00000000-0000-4000-8000-000000000000`. All messages are emitted as
  Turns in wall-clock order; branch selection is a downstream concern.
- `tool_result.content` can be a string OR a list of content blocks —
  both shapes are coerced to a string via the same helper Claude Code uses.
- MCP-integration metadata on tool_use blocks is preserved into the
  `ToolUseBlock` schema fields (`mcp_integration`, `mcp_server_url`,
  `is_mcp_app`).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import ijson

from lucid.ingest._rendering import plaintext_of_blocks as _plaintext_of
from lucid.ingest.base import (
    MAX_BLOCKS_PER_TURN,
    MAX_EXPORT_FILE_SIZE,
    MAX_TEXT_LEN_PER_BLOCK,
    MAX_TURNS_PER_CONVERSATION,
    IngestAdapter,
    IngestError,
    ParsedConversation,
    assert_file_size,
    content_hash_for,
    fingerprint_corpus,
    safe_resolve_path,
)
from lucid.ingest.claude_code import _coerce_tool_result_content, _parse_timestamp
from lucid.schemas import (
    ContentBlock,
    Conversation,
    MemoryFile,
    Project,
    Role,
    Source,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Turn,
)

_LOGGER = logging.getLogger(__name__)

_EXPECTED_FILES = ("conversations.json", "projects.json", "memories.json")


def _require_export_layout(root: Path) -> None:
    """Reject export directories that don't have the expected top-level files."""
    missing = [name for name in _EXPECTED_FILES if not (root / name).is_file()]
    if missing:
        raise IngestError(
            f"{root}: export directory missing required file(s): {', '.join(missing)}"
        )


def _parse_claude_ai_block(raw: dict[str, Any]) -> ContentBlock | None:
    """Translate a Claude.ai export block dict to a schema variant."""
    block_type = raw.get("type")
    if not isinstance(block_type, str):
        return None

    if block_type == "text":
        text = raw.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        if len(text) > MAX_TEXT_LEN_PER_BLOCK:
            raise IngestError(f"text block exceeds {MAX_TEXT_LEN_PER_BLOCK:,} chars")
        return TextBlock(text=text)

    if block_type == "thinking":
        thinking = raw.get("thinking", "")
        if not isinstance(thinking, str):
            thinking = str(thinking)
        if len(thinking) > MAX_TEXT_LEN_PER_BLOCK:
            raise IngestError(f"thinking block exceeds {MAX_TEXT_LEN_PER_BLOCK:,} chars")
        signature = raw.get("signature")
        return ThinkingBlock(
            thinking=thinking,
            signature=signature if isinstance(signature, str) else None,
        )

    if block_type == "tool_use":
        # Claude.ai uses `name` / `input` / `id` just like Claude Code, plus
        # optional MCP-integration hints.
        name = raw.get("name")
        if not isinstance(name, str):
            return None
        tool_input = raw.get("input", {})
        if not isinstance(tool_input, dict):
            tool_input = {}
        return ToolUseBlock(
            tool_name=name,
            tool_input=tool_input,
            tool_use_id=raw.get("id") if isinstance(raw.get("id"), str) else None,
            mcp_integration=(
                raw.get("integration_name")
                if isinstance(raw.get("integration_name"), str)
                else None
            ),
            mcp_server_url=(
                raw.get("mcp_server_url") if isinstance(raw.get("mcp_server_url"), str) else None
            ),
            is_mcp_app=raw.get("is_mcp_app") if isinstance(raw.get("is_mcp_app"), bool) else None,
        )

    if block_type == "tool_result":
        tool_use_id = raw.get("tool_use_id")
        if not isinstance(tool_use_id, str):
            return None
        content = _coerce_tool_result_content(raw.get("content", ""))
        if len(content) > MAX_TEXT_LEN_PER_BLOCK:
            raise IngestError(f"tool_result content exceeds {MAX_TEXT_LEN_PER_BLOCK:,} chars")
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=content,
            is_error=bool(raw.get("is_error", False)),
        )

    _LOGGER.warning("Claude.ai: unknown block type %r", block_type)
    return None


def _role_from_sender(sender: object) -> Role | None:
    if sender == "human":
        return Role.USER
    if sender == "assistant":
        return Role.ASSISTANT
    return None


def _parse_one_conversation(raw: dict[str, Any]) -> ParsedConversation | None:
    """Transform a single Claude.ai conversation object into schema rows."""
    conv_uuid = raw.get("uuid")
    if not isinstance(conv_uuid, str) or not conv_uuid:
        _LOGGER.warning("skipping conversation with missing uuid")
        return None

    created_at = _parse_timestamp(raw.get("created_at"))
    if created_at is None:
        _LOGGER.warning("conversation %s has no created_at; skipping", conv_uuid)
        return None
    updated_at = _parse_timestamp(raw.get("updated_at")) or created_at

    chat_messages = raw.get("chat_messages") or []
    if not isinstance(chat_messages, list):
        _LOGGER.warning("conversation %s: chat_messages not a list; skipping", conv_uuid)
        return None

    if len(chat_messages) > MAX_TURNS_PER_CONVERSATION:
        raise IngestError(f"conversation {conv_uuid}: {len(chat_messages)} messages exceeds cap")

    turns: list[Turn] = []
    for idx, msg in enumerate(chat_messages):
        if not isinstance(msg, dict):
            continue
        role = _role_from_sender(msg.get("sender"))
        if role is None:
            continue

        raw_blocks = msg.get("content") or []
        if not isinstance(raw_blocks, list):
            raw_blocks = []
        if len(raw_blocks) > MAX_BLOCKS_PER_TURN:
            raise IngestError(
                f"conversation {conv_uuid} turn {idx}: {len(raw_blocks)} blocks exceeds cap"
            )

        blocks: list[ContentBlock] = []
        for rb in raw_blocks:
            if not isinstance(rb, dict):
                continue
            parsed = _parse_claude_ai_block(rb)
            if parsed is not None:
                blocks.append(parsed)

        # If `content` was empty, fall back to the plaintext `text` field so
        # tests and Module A still see something.
        if not blocks:
            text = msg.get("text")
            if isinstance(text, str) and text:
                blocks.append(TextBlock(text=text))

        turns.append(
            Turn(
                id=str(msg.get("uuid") or f"{conv_uuid}-{idx}"),
                conversation_id=conv_uuid,
                index=idx,
                role=role,
                content=_plaintext_of(blocks) or (msg.get("text") or ""),
                blocks=blocks,
                timestamp=_parse_timestamp(msg.get("created_at")),
                parent_message_uuid=(
                    str(msg["parent_message_uuid"])
                    if isinstance(msg.get("parent_message_uuid"), str)
                    else None
                ),
            )
        )

    if not turns:
        return None

    conversation = Conversation(
        id=conv_uuid,
        source=Source.CLAUDE_AI,
        source_path="conversations.json",
        created_at=created_at,
        updated_at=updated_at,
        title=(raw.get("name") if isinstance(raw.get("name"), str) else None),
        summary=(raw.get("summary") if isinstance(raw.get("summary"), str) else None),
        turn_count=len(turns),
    )
    return ParsedConversation(conversation=conversation, turns=turns)


def iter_conversations(path: Path) -> Iterator[ParsedConversation]:
    """Stream parsed conversations from a `conversations.json` file."""
    assert_file_size(path, MAX_EXPORT_FILE_SIZE)
    with path.open("rb") as f:
        for raw in ijson.items(f, "item"):
            if not isinstance(raw, dict):
                continue
            parsed = _parse_one_conversation(raw)
            if parsed is not None:
                yield parsed


def parse_projects(path: Path) -> list[Project]:
    """Parse a `projects.json` file into a list of `Project` rows."""
    assert_file_size(path, MAX_EXPORT_FILE_SIZE)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise IngestError(f"{path}: expected a top-level array")
    projects: list[Project] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            projects.append(
                Project(
                    uuid=str(entry["uuid"]),
                    name=str(entry.get("name") or entry["uuid"]),
                    description=(
                        entry.get("description")
                        if isinstance(entry.get("description"), str)
                        else None
                    ),
                    prompt_template=(
                        entry.get("prompt_template")
                        if isinstance(entry.get("prompt_template"), str)
                        else None
                    ),
                    created_at=_parse_timestamp(entry.get("created_at"))
                    or datetime.fromtimestamp(path.stat().st_mtime).astimezone(),
                    updated_at=_parse_timestamp(entry.get("updated_at"))
                    or datetime.fromtimestamp(path.stat().st_mtime).astimezone(),
                    doc_count=len(entry.get("docs") or []),
                    doc_char_total=sum(
                        len(doc.get("content") or "")
                        for doc in (entry.get("docs") or [])
                        if isinstance(doc, dict)
                    ),
                )
            )
        except KeyError as err:
            _LOGGER.warning("projects entry missing %s; skipping", err)
    return projects


def parse_memory_file(path: Path) -> MemoryFile:
    """Parse a `memories.json` file.

    The file is a JSON *list* containing a single memory object, per
    BUILD_GUIDE §3.2. Some exports wrap it as a bare object instead, so
    both shapes are handled.
    """
    assert_file_size(path, MAX_EXPORT_FILE_SIZE)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        if not raw:
            raise IngestError(f"{path}: memories.json is empty list")
        raw = raw[0]
    if not isinstance(raw, dict):
        raise IngestError(f"{path}: expected a dict or list-of-dict")

    account_uuid = raw.get("account_uuid")
    if not isinstance(account_uuid, str) or not account_uuid:
        raise IngestError(f"{path}: account_uuid missing")

    conversations_memory = raw.get("conversations_memory")
    if conversations_memory is not None and not isinstance(conversations_memory, str):
        conversations_memory = str(conversations_memory)

    project_memories_raw = raw.get("project_memories") or {}
    project_memories: dict[str, str] = {}
    if isinstance(project_memories_raw, dict):
        for k, v in project_memories_raw.items():
            if isinstance(k, str) and isinstance(v, str):
                project_memories[k] = v

    return MemoryFile(
        account_uuid=account_uuid,
        conversations_memory=conversations_memory,
        project_memories=project_memories,
        extracted_claims=[],  # Module H fills these on its own pass
    )


class ClaudeAiAdapter(IngestAdapter):
    """Adapter for Claude.ai export directories."""

    name = "claude-ai"

    # ----- discover ---------------------------------------------------

    def discover(self, root: Path) -> list[Path]:
        root = safe_resolve_path(root)
        if not root.is_dir():
            raise IngestError(f"{root}: not a directory")
        _require_export_layout(root)
        # `conversations.json` is the authoritative source; the others are
        # surfaced via `parse_projects` / `parse_memory_file` helpers.
        return [root / "conversations.json"]

    # ----- parse ------------------------------------------------------

    def parse_one(self, path: Path) -> list[ParsedConversation]:
        return list(iter_conversations(path))

    def parse_all(self, root: Path) -> list[ParsedConversation]:
        root = safe_resolve_path(root)
        _require_export_layout(root)
        return list(iter_conversations(root / "conversations.json"))

    # ----- fingerprint ------------------------------------------------

    def fingerprint(self, parsed: list[ParsedConversation]) -> str:
        items: Iterable[tuple[str, str]] = (
            (p.conversation.id, content_hash_for(p.conversation, p.turns)) for p in parsed
        )
        return fingerprint_corpus(items)
