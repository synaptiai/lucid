"""Memory file handling — re-exports plus the claim-extraction stub.

The heavy claim extraction (an Opus 4.7 call) lives in `lucid/modules/module_h_memory.py`
and runs during the audit, not at ingest time. This module exists so the
ingest layer has a single import path for memories handling even when
Module H is opted out.
"""

from lucid.ingest.claude_ai import parse_memory_file
from lucid.schemas import MemoryClaim, MemoryFile

__all__ = ["MemoryClaim", "MemoryFile", "parse_memory_file"]
