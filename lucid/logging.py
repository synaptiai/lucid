"""Logging configuration.

`SafeFormatter` drops any LogRecord whose extras flag `contains_user_content`.
Corpus content (Claude.ai export bodies, memory text, etc.) must never reach a
log handler; the flag is the belt-and-suspenders to the ordinary discipline of
not passing that content into log messages in the first place.
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler


class SafeFormatter(logging.Formatter):
    """Formatter that refuses to format records tagged as containing user content."""

    SENTINEL_ATTR = "contains_user_content"

    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, self.SENTINEL_ATTR, False):
            # Return a neutral line; do not include any part of record.msg or args.
            return f"[{record.levelname}] <record dropped: contains_user_content=True>"
        return super().format(record)


def configure_logging(level: str = "INFO") -> None:
    """Install a Rich console handler with SafeFormatter at the given level."""
    handler = RichHandler(
        show_time=True,
        show_level=True,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
    )
    handler.setFormatter(SafeFormatter(fmt="%(message)s"))

    root = logging.getLogger()
    # Remove any default handlers so we don't double-log.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level.upper())
