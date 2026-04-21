"""Shared ingest contract, security constants, and helpers.

Every adapter (Claude Code JSONL, Claude.ai export) goes through this base.
Security constants are module-level (not adapter-level) so any new adapter
inherits the same caps without re-declaring them. Violations raise
`IngestError` — callers should surface the message verbatim to the user but
must NOT pass the offending payload to a log sink tagged for user content.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from lucid.schemas import Conversation, Turn

# ──────────────────────────────────────────────────────────────────────────
# Security constants
# ──────────────────────────────────────────────────────────────────────────
#
# Values are deliberate over-estimates: a real Claude.ai export tops out
# around 20 MB and a Claude Code session around 5 MB. These caps protect
# against accidentally pointing Lucid at a hostile or truncated file.

MAX_TURNS_PER_CONVERSATION = 10_000
MAX_BLOCKS_PER_TURN = 1_000
MAX_TEXT_LEN_PER_BLOCK = 10_000_000  # 10 MB
MAX_SESSION_FILE_SIZE = 50 * 1024 * 1024  # 50 MB per Claude Code JSONL
MAX_EXPORT_FILE_SIZE = 500 * 1024 * 1024  # 500 MB per Claude.ai export file
IJSON_RECURSION_CAP = 32  # reject pathologically-nested JSON
MAX_DISCOVER_FILES = 200_000  # upper bound on files returned by discover()


class IngestError(Exception):
    """Raised when an ingest input is malformed, too large, or hostile.

    The message is safe to surface to the user; it never contains corpus
    content, only metadata (path, size, counts).
    """


# ──────────────────────────────────────────────────────────────────────────
# Path safety
# ──────────────────────────────────────────────────────────────────────────


def safe_resolve_path(user_path: Path | str) -> Path:
    """Resolve `user_path` to an absolute canonical Path.

    Uses `strict=True` so a missing target raises `FileNotFoundError`
    instead of silently returning a path that doesn't exist. Does NOT by
    itself check symlink escape — the caller must pair this with
    `assert_not_symlink_escape()` for each discovered child.
    """
    return Path(user_path).expanduser().resolve(strict=True)


def assert_not_symlink_escape(child: Path, root: Path) -> None:
    """Reject `child` if it resolves outside `root`.

    Both arguments should already be absolute. We `resolve(strict=False)`
    on `child` to follow any symlink chain; if the target is outside
    `root`, raise `IngestError`.
    """
    resolved = child.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as err:
        raise IngestError(
            f"Refusing to ingest {child}: resolves outside source root {root}"
        ) from err


def assert_file_size(path: Path, cap_bytes: int) -> None:
    """Raise IngestError if `path` exceeds `cap_bytes`."""
    size = path.stat().st_size
    if size > cap_bytes:
        raise IngestError(f"{path} is {size:,} bytes; exceeds cap of {cap_bytes:,} bytes")


# ──────────────────────────────────────────────────────────────────────────
# Fingerprint
# ──────────────────────────────────────────────────────────────────────────


def fingerprint_corpus(items: Iterable[tuple[str, str]]) -> str:
    """Return a stable sha256 over `(conversation_id, content_hash)` pairs.

    `fingerprint_corpus` satisfies the plan's contract: identical parses
    yield identical fingerprints; adding a conversation changes the
    fingerprint; internal content edits of existing conversations change
    it only if the per-conversation `content_hash` changes (which is the
    adapter's responsibility to compute appropriately).
    """
    h = hashlib.sha256()
    for conv_id, content_hash in sorted(items):
        h.update(f"{conv_id}:{content_hash}\n".encode())
    return h.hexdigest()


def content_hash_for(conv: Conversation, turns: list[Turn]) -> str:
    """Per-conversation content hash used as an input to the corpus fingerprint."""
    h = hashlib.sha256()
    h.update(conv.id.encode())
    h.update(conv.updated_at.isoformat().encode())
    for t in turns:
        h.update(t.id.encode())
        h.update(t.content.encode())
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────
# Adapter base
# ──────────────────────────────────────────────────────────────────────────


class ParsedConversation:
    """Return type of `parse_one()` — a Conversation plus its ordered Turns."""

    __slots__ = ("conversation", "turns")

    def __init__(self, conversation: Conversation, turns: list[Turn]) -> None:
        self.conversation = conversation
        self.turns = turns


class IngestAdapter(ABC):
    """Contract every source-adapter must satisfy.

    The three methods are deliberately sync — parallelism is the caller's
    concern (a `ProcessPoolExecutor` for Claude Code, serial streaming
    for Claude.ai). Making the adapter sync keeps `parse_one` trivially
    picklable for subprocess workers.
    """

    #: stable identifier used in logs + config
    name: str

    @abstractmethod
    def discover(self, root: Path) -> list[Path]:
        """Return the list of on-disk files this adapter would parse."""

    @abstractmethod
    def parse_one(self, path: Path) -> list[ParsedConversation]:
        """Parse a single file. Returns one or many conversations.

        Claude Code yields exactly one conversation per session file; the
        Claude.ai adapter yields many conversations from a single
        conversations.json. Callers should iterate without assumption.
        """

    def parse_all(self, root: Path) -> list[ParsedConversation]:
        """Default implementation: sequentially parse every discovered file.

        Adapters that benefit from parallelism should override this (the
        Claude Code adapter does via `ProcessPoolExecutor`).
        """
        root = safe_resolve_path(root)
        results: list[ParsedConversation] = []
        for p in self.discover(root):
            assert_not_symlink_escape(p, root)
            results.extend(self.parse_one(p))
        return results

    @abstractmethod
    def fingerprint(self, parsed: list[ParsedConversation]) -> str:
        """Return the corpus fingerprint for these conversations."""
