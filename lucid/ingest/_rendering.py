"""Canonical block → plaintext rendering shared by every ingest adapter.

Classifier modules read ``turn.content`` — never ``turn.blocks``. The
content string is built by :func:`plaintext_of_blocks` which walks a
block list and emits:

- ``TextBlock.text`` verbatim
- ``[tool: Name(summary)]`` placeholder per :class:`ToolUseBlock`, where
  ``summary`` is an explicit allow-list extraction of a single safe
  input field (e.g. Read's ``file_path``, Bash's ``command``)
- nothing for :class:`ThinkingBlock` or :class:`ToolResultBlock`

Tool results never feed into the content string: they may contain
secrets, command output, scraped pages, or personal data, and
surfacing them to classifiers would violate both the user's privacy
expectations and the Spiral-Bench / Sharma rubrics' definitions of
"assistant utterance".

Tool *use* blocks do feed in as compact placeholders because they are
the assistant's externally-visible actions — hiding them would leave
classifiers blind to the ~37% of turns in a typical Claude Code
corpus that contain only tool calls.
"""

from __future__ import annotations

from lucid.schemas import ContentBlock, TextBlock, ToolUseBlock

__all__ = ["plaintext_of_blocks", "tool_placeholder"]

# Known Claude-tooling names whose first argument is user-visible
# metadata (a file path, a command, a pattern). The summary extraction
# pulls only that one argument — never arbitrary input fields — so the
# placeholder is safe to include in ``turn.content`` without leaking
# credentials or private data that may live in other input slots.
_TOOL_SUMMARY_FIELDS: dict[str, tuple[str, ...]] = {
    "Read": ("file_path",),
    "Write": ("file_path",),
    "Edit": ("file_path",),
    "NotebookEdit": ("notebook_path", "file_path"),
    "Bash": ("command",),
    "bash_tool": ("command",),
    "Grep": ("pattern",),
    "Glob": ("pattern",),
    "view": ("path", "file_path"),
    "str_replace_editor": ("path",),
    "Task": ("description", "prompt"),
    "Agent": ("description", "prompt"),
    "WebFetch": ("url",),
    "WebSearch": ("query",),
}

_TOOL_SUMMARY_MAX_CHARS = 40


def tool_placeholder(block: ToolUseBlock) -> str:
    """Return a one-line ``[tool: Name(summary)]`` placeholder.

    Unknown tools render with no parameter — safer than guessing,
    because arbitrary tool inputs may carry credentials or private
    data.
    """
    name = block.tool_name
    fields = _TOOL_SUMMARY_FIELDS.get(name)
    if not fields:
        return f"[tool: {name}]"
    summary: str | None = None
    for key in fields:
        value = block.tool_input.get(key)
        if isinstance(value, str) and value.strip():
            summary = value.strip()
            break
    if summary is None:
        return f"[tool: {name}]"
    if len(summary) > _TOOL_SUMMARY_MAX_CHARS:
        summary = summary[: _TOOL_SUMMARY_MAX_CHARS - 1].rstrip() + "…"
    return f"[tool: {name}({summary})]"


def plaintext_of_blocks(blocks: list[ContentBlock]) -> str:
    """Canonical ``turn.content`` renderer — see module docstring."""
    pieces: list[str] = []
    for b in blocks:
        if isinstance(b, TextBlock):
            pieces.append(b.text)
        elif isinstance(b, ToolUseBlock):
            pieces.append(tool_placeholder(b))
        # ThinkingBlock and ToolResultBlock intentionally dropped.
    return "\n".join(pieces)
