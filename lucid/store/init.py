"""Idempotent DB initialization.

`initialize_db(path)` opens (or creates) a SQLite file at `path` and applies
`schema.sql` only if `PRAGMA user_version == 0`. Subsequent calls are no-ops
— the function is safe to invoke at every audit startup without guarding on
file existence. When schema changes later (Phase 9+), bump `SCHEMA_VERSION`
and add a matching branch here.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

_SCHEMA_SQL_PATH = Path(__file__).with_name("schema.sql")


def _read_schema_sql() -> str:
    return _SCHEMA_SQL_PATH.read_text(encoding="utf-8")


def initialize_db(path: Path | str) -> Path:
    """Apply `schema.sql` to the DB at `path` if its user_version is 0.

    Returns the resolved `Path` so callers can keep it for later use.
    Raises `RuntimeError` if the DB is at a user_version > SCHEMA_VERSION
    (the user is running an older Lucid against a newer-format DB).
    """
    db_path = Path(path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        cursor = conn.execute("PRAGMA user_version;")
        (current_version,) = cursor.fetchone()

        if current_version == 0:
            conn.executescript(_read_schema_sql())
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
            conn.commit()
        elif current_version == SCHEMA_VERSION:
            pass  # already current
        elif current_version > SCHEMA_VERSION:
            raise RuntimeError(
                f"DB at {db_path} has user_version={current_version}, "
                f"but this Lucid ships schema v{SCHEMA_VERSION}. "
                "Upgrade Lucid or use a fresh DB."
            )
        else:
            # current_version < SCHEMA_VERSION: a real migration would go here.
            raise RuntimeError(
                f"DB at {db_path} is at schema v{current_version}, "
                f"but v{SCHEMA_VERSION} is required. No migration path is "
                "defined yet (this is a hackathon build). Use a fresh DB."
            )
    finally:
        conn.close()

    return db_path


def connect(path: Path | str) -> sqlite3.Connection:
    """Open a connection with `foreign_keys = ON` and Row factory set."""
    conn = sqlite3.connect(Path(path).expanduser().resolve())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn
