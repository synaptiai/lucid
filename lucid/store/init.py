"""Idempotent DB initialization + migration application.

:func:`initialize_db` opens (or creates) a SQLite file at ``path``,
brings it to :data:`SCHEMA_VERSION`, and is safe to invoke at every
audit startup. Three on-disk states are recognised:

- ``user_version == 0`` (fresh DB) — apply :file:`schema.sql`,
  set ``user_version = SCHEMA_VERSION``.
- ``0 < user_version < SCHEMA_VERSION`` (older DB) — replay each
  migration ``m_(current+1).sql`` … ``m_SCHEMA_VERSION.sql`` from
  :data:`MIGRATIONS_DIR`, advancing ``user_version`` after each one.
- ``user_version == SCHEMA_VERSION`` — no-op.
- ``user_version > SCHEMA_VERSION`` — refuse and ask the user to
  upgrade Lucid.

Migration files live under ``lucid/store/migrations/m_<NNN>.sql``;
see the README in that directory for the authoring contract. Tests
exercise the migration framework with an injected
``migrations_dir``; production callers use the default.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

_SCHEMA_SQL_PATH = Path(__file__).with_name("schema.sql")
MIGRATIONS_DIR: Path = Path(__file__).parent / "migrations"
_MIGRATION_FILENAME_RE = re.compile(r"^m_(\d+)\.sql$")


def _read_schema_sql() -> str:
    return _SCHEMA_SQL_PATH.read_text(encoding="utf-8")


def _load_migrations(migrations_dir: Path) -> dict[int, str]:
    """Discover migration files in ``migrations_dir`` and load their SQL.

    Files matching ``m_<NNN>.sql`` are mapped to ``{version: sql}``;
    everything else (README.md, .DS_Store, partial drafts) is ignored.
    Returning a dict means callers see "no migration for v3" as a
    plain ``KeyError`` instead of a silent skip.
    """
    if not migrations_dir.is_dir():
        return {}
    out: dict[int, str] = {}
    for entry in sorted(migrations_dir.iterdir()):
        if not entry.is_file():
            continue
        match = _MIGRATION_FILENAME_RE.match(entry.name)
        if match is None:
            continue
        version = int(match.group(1))
        out[version] = entry.read_text(encoding="utf-8")
    return out


def initialize_db(
    path: Path | str,
    *,
    migrations_dir: Path | None = None,
) -> Path:
    """Bring the SQLite file at ``path`` to :data:`SCHEMA_VERSION`.

    ``migrations_dir`` defaults to :data:`MIGRATIONS_DIR`; tests inject
    a temporary directory to exercise the upgrade path against
    fixture migrations without bumping the production schema version.

    Returns the resolved path so callers can hold onto it.
    """
    db_path = Path(path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_migrations_dir = migrations_dir if migrations_dir is not None else MIGRATIONS_DIR

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
            pass  # already current — nothing to do
        elif current_version > SCHEMA_VERSION:
            raise RuntimeError(
                f"DB at {db_path} has user_version={current_version}, "
                f"but this Lucid ships schema v{SCHEMA_VERSION}. "
                "Upgrade Lucid or use a fresh DB."
            )
        else:
            _apply_migrations(
                conn,
                db_path=db_path,
                from_version=current_version,
                to_version=SCHEMA_VERSION,
                migrations_dir=resolved_migrations_dir,
            )
    finally:
        conn.close()

    return db_path


def _apply_migrations(
    conn: sqlite3.Connection,
    *,
    db_path: Path,
    from_version: int,
    to_version: int,
    migrations_dir: Path,
) -> None:
    """Replay each migration from ``from_version + 1`` to ``to_version``.

    Each migration runs in its own transaction (one
    ``executescript`` call) and bumps ``user_version`` on success.
    A missing migration file is a hard error — silently skipping
    would leave the DB in an undefined intermediate state.
    """
    migrations = _load_migrations(migrations_dir)
    for target_version in range(from_version + 1, to_version + 1):
        sql = migrations.get(target_version)
        if sql is None:
            raise RuntimeError(
                f"DB at {db_path} is at schema v{from_version} but no "
                f"migration was found for v{target_version} in {migrations_dir}. "
                "This Lucid build is incomplete; reinstall or use a fresh DB."
            )
        conn.executescript(sql)
        conn.execute(f"PRAGMA user_version = {target_version};")
        conn.commit()


def connect(path: Path | str) -> sqlite3.Connection:
    """Open a connection with `foreign_keys = ON` and Row factory set."""
    conn = sqlite3.connect(Path(path).expanduser().resolve())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn
