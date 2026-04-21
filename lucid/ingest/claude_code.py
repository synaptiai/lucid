"""Claude Code JSONL adapter.

Claude Code writes one JSONL line per turn under
`~/.claude/projects/<project-slug>/<session-id>.jsonl`. Each line is the
full message envelope:

    {
        "type": "user" | "assistant" | "system" | "summary",
        "message": {"role": ..., "content": [<block>, ...]},
        "timestamp": "...", "sessionId": "...", "parentUuid": "...",
        "uuid": "...", "cwd": "...", "version": "..."
    }

One session file == one Conversation. Block types map 1-1 to the Pydantic
content-block union in `lucid/schemas.py`, with a handful of field-name
translations (Claude Code `id` -> schema `tool_use_id`, `name` ->
`tool_name`, `input` -> `tool_input`).

Parallel parse uses ProcessPoolExecutor. `parse_one` is a module-level
function (callable by pickled workers) so the 9,840-file corpus fans out
cleanly. `parse_all` streams file completions and aggregates at the end.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson

from lucid.ingest.base import (
    MAX_BLOCKS_PER_TURN,
    MAX_DISCOVER_FILES,
    MAX_SESSION_FILE_SIZE,
    MAX_TEXT_LEN_PER_BLOCK,
    MAX_TURNS_PER_CONVERSATION,
    IngestAdapter,
    IngestError,
    ParsedConversation,
    assert_file_size,
    assert_not_symlink_escape,
    content_hash_for,
    fingerprint_corpus,
    safe_resolve_path,
)
from lucid.schemas import (
    ContentBlock,
    Conversation,
    Role,
    Source,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Turn,
)

_LOGGER = logging.getLogger(__name__)


_ROLE_BY_TYPE = {
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
    "system": Role.SYSTEM,
}

# Record types that appear in real Claude Code JSONL but don't represent a
# conversation turn. Silently skipped; no warning.
_SKIP_RECORD_TYPES = frozenset({"summary", "progress", "hook", "compaction"})


def _coerce_tool_result_content(value: object) -> str:
    """Claude Code usually writes `content` as a string, but defensively
    handles the list-of-blocks shape seen in Claude.ai exports."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    pieces.append(text)
                    continue
            pieces.append(str(item))
        return "\n".join(pieces)
    return str(value)


def _parse_block(raw: dict[str, Any]) -> ContentBlock | None:
    """Translate a raw JSONL block dict into one of the schema variants.

    Returns None (and warns) for unknown block types. Cap-violating blocks
    raise IngestError.
    """
    block_type = raw.get("type")
    if not isinstance(block_type, str):
        _LOGGER.warning("skipping block with non-string type: %r", block_type)
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
        name = raw.get("name")
        if not isinstance(name, str):
            _LOGGER.warning("tool_use block missing name; skipping")
            return None
        tool_input = raw.get("input", {})
        if not isinstance(tool_input, dict):
            tool_input = {}
        tool_use_id = raw.get("id")
        return ToolUseBlock(
            tool_name=name,
            tool_input=tool_input,
            tool_use_id=tool_use_id if isinstance(tool_use_id, str) else None,
        )

    if block_type == "tool_result":
        tool_use_id = raw.get("tool_use_id")
        if not isinstance(tool_use_id, str):
            _LOGGER.warning("tool_result block missing tool_use_id; skipping")
            return None
        content = _coerce_tool_result_content(raw.get("content", ""))
        if len(content) > MAX_TEXT_LEN_PER_BLOCK:
            raise IngestError(f"tool_result content exceeds {MAX_TEXT_LEN_PER_BLOCK:,} chars")
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=content,
            is_error=bool(raw.get("is_error", False)),
        )

    # Image blocks are common in real exports (screenshots from vision tooling)
    # but Lucid doesn't operate on pixels; skip silently at DEBUG level.
    _LOGGER.debug("unknown block type %r; skipping", block_type)
    return None


def _plaintext_of(blocks: list[ContentBlock]) -> str:
    """Turn.content is a plaintext rendering of the text blocks only."""
    pieces = [b.text for b in blocks if isinstance(b, TextBlock)]
    return "\n".join(pieces)


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    # Python's fromisoformat accepts "Z" suffix from 3.11 onwards.
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _LOGGER.warning("unparseable timestamp %r; leaving as None", value)
        return None


def parse_session_file(path: Path) -> ParsedConversation | None:
    """Parse a single Claude Code JSONL session.

    Returns None if the file is empty or contains nothing but summary
    lines (an "opened and exited" session). Returns a ParsedConversation
    otherwise.

    This function is module-level so ProcessPoolExecutor can pickle it.
    """
    assert_file_size(path, MAX_SESSION_FILE_SIZE)

    turns: list[Turn] = []
    session_id: str | None = None
    project_slug = path.parent.name
    version: str | None = None
    cwd: str | None = None

    with path.open("rb") as f:
        for lineno, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                rec = orjson.loads(raw_line)
            except orjson.JSONDecodeError:
                _LOGGER.warning("%s:%d: malformed JSON; skipping line", path, lineno)
                continue
            if not isinstance(rec, dict):
                _LOGGER.warning("%s:%d: top-level not an object; skipping", path, lineno)
                continue

            rec_type = rec.get("type")
            if rec_type in _SKIP_RECORD_TYPES:
                continue

            session_id = session_id or rec.get("sessionId")
            version = version or rec.get("version")
            cwd = cwd or rec.get("cwd")

            role = _ROLE_BY_TYPE.get(rec_type) if isinstance(rec_type, str) else None
            if role is None:
                _LOGGER.debug("%s:%d: unknown record type %r; skipping", path, lineno, rec_type)
                continue

            message = rec.get("message")
            if not isinstance(message, dict):
                _LOGGER.debug("%s:%d: missing message object; skipping", path, lineno)
                continue

            raw_blocks = message.get("content", [])
            if not isinstance(raw_blocks, list):
                # Claude Code occasionally emits `content: "..."` for short user turns.
                raw_blocks = [{"type": "text", "text": str(raw_blocks)}]
            if len(raw_blocks) > MAX_BLOCKS_PER_TURN:
                raise IngestError(
                    f"{path}:{lineno}: {len(raw_blocks)} blocks exceeds cap {MAX_BLOCKS_PER_TURN}"
                )

            blocks: list[ContentBlock] = []
            for rb in raw_blocks:
                if not isinstance(rb, dict):
                    continue
                parsed = _parse_block(rb)
                if parsed is not None:
                    blocks.append(parsed)

            turn = Turn(
                id=str(rec.get("uuid") or f"{path.stem}-{lineno}"),
                conversation_id=str(session_id or path.stem),
                index=len(turns),
                role=role,
                content=_plaintext_of(blocks),
                blocks=blocks,
                timestamp=_parse_timestamp(rec.get("timestamp")),
                parent_message_uuid=(
                    str(rec["parentUuid"]) if isinstance(rec.get("parentUuid"), str) else None
                ),
            )
            turns.append(turn)

            if len(turns) > MAX_TURNS_PER_CONVERSATION:
                raise IngestError(f"{path}: more than {MAX_TURNS_PER_CONVERSATION} turns; aborting")

    if not turns:
        return None

    conv_id = session_id or path.stem
    # Rewrite turn.conversation_id now that we know the final ID.
    turns = [t.model_copy(update={"conversation_id": conv_id}) for t in turns]

    timestamps = [t.timestamp for t in turns if t.timestamp is not None]
    created_at = (
        min(timestamps) if timestamps else datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    )
    updated_at = max(timestamps) if timestamps else created_at

    metadata: dict[str, object] = {}
    if version is not None:
        metadata["claude_code_version"] = version
    if cwd is not None:
        metadata["cwd"] = cwd

    conversation = Conversation(
        id=conv_id,
        source=Source.CLAUDE_CODE,
        source_path=str(path),
        created_at=created_at,
        updated_at=updated_at,
        turn_count=len(turns),
        project_slug=project_slug,
        metadata=metadata,
    )
    return ParsedConversation(conversation=conversation, turns=turns)


class ClaudeCodeAdapter(IngestAdapter):
    """Adapter for `~/.claude/projects/<slug>/<session>.jsonl` corpora."""

    name = "claude-code"

    def __init__(self, *, max_workers: int | None = None) -> None:
        self.max_workers = max_workers or (os.cpu_count() or 4)

    # ----- discover -----------------------------------------------

    def discover(self, root: Path) -> list[Path]:
        root = safe_resolve_path(root)
        if not root.is_dir():
            raise IngestError(f"{root}: not a directory")
        files: list[Path] = []
        for child in root.rglob("*.jsonl"):
            # Reject symlinks that resolve outside the declared root.
            assert_not_symlink_escape(child, root)
            files.append(child)
            if len(files) > MAX_DISCOVER_FILES:
                raise IngestError(f"{root}: more than {MAX_DISCOVER_FILES} .jsonl files; refusing")
        return files

    # ----- parse --------------------------------------------------

    def parse_one(self, path: Path) -> list[ParsedConversation]:
        result = parse_session_file(path)
        return [result] if result is not None else []

    def parse_all(self, root: Path) -> list[ParsedConversation]:
        root = safe_resolve_path(root)
        paths = self.discover(root)
        if not paths:
            return []
        results: list[ParsedConversation] = []
        # Serial fallback when max_workers == 1 (makes tests deterministic).
        if self.max_workers == 1:
            for p in paths:
                results.extend(self.parse_one(p))
            return results
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(parse_session_file, p): p for p in paths}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    parsed = future.result()
                except IngestError as err:
                    _LOGGER.warning("skipping %s: %s", path, err)
                    continue
                if parsed is not None:
                    results.append(parsed)
        return results

    # ----- fingerprint --------------------------------------------

    def fingerprint(self, parsed: list[ParsedConversation]) -> str:
        items: Iterable[tuple[str, str]] = (
            (p.conversation.id, content_hash_for(p.conversation, p.turns)) for p in parsed
        )
        return fingerprint_corpus(items)
