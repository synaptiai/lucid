"""Ingest adapters for the two supported corpus sources."""

from lucid.ingest.base import (
    IJSON_RECURSION_CAP,
    MAX_BLOCKS_PER_TURN,
    MAX_EXPORT_FILE_SIZE,
    MAX_SESSION_FILE_SIZE,
    MAX_TEXT_LEN_PER_BLOCK,
    MAX_TURNS_PER_CONVERSATION,
    IngestAdapter,
    IngestError,
    fingerprint_corpus,
    safe_resolve_path,
)

__all__ = [
    "IJSON_RECURSION_CAP",
    "MAX_BLOCKS_PER_TURN",
    "MAX_EXPORT_FILE_SIZE",
    "MAX_SESSION_FILE_SIZE",
    "MAX_TEXT_LEN_PER_BLOCK",
    "MAX_TURNS_PER_CONVERSATION",
    "IngestAdapter",
    "IngestError",
    "fingerprint_corpus",
    "safe_resolve_path",
]
