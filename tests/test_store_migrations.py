"""L2 — migration framework coverage for ``lucid.store.init``.

Validates that the framework — which has zero shipped migration files
today, since :data:`SCHEMA_VERSION` is still ``1`` — actually upgrades
an older DB when one is added in the future. We use ``monkeypatch`` to
move ``SCHEMA_VERSION`` and inject a temporary migrations directory so
the test can exercise the upgrade path without bumping the production
schema.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lucid.store import SCHEMA_VERSION, initialize_db
from lucid.store.init import _load_migrations

# ──────────────────────────────────────────────────────────────────────────
# _load_migrations
# ──────────────────────────────────────────────────────────────────────────


def test_load_migrations_picks_up_only_numbered_migration_files(tmp_path: Path) -> None:
    """README files, .DS_Store, partial drafts must be ignored."""
    (tmp_path / "m_002.sql").write_text("CREATE TABLE m2 (id INTEGER);")
    (tmp_path / "m_003.sql").write_text("CREATE TABLE m3 (id INTEGER);")
    # Decoy files the production directory is allowed to carry:
    (tmp_path / "README.md").write_text("docs")
    (tmp_path / ".DS_Store").write_text("garbage")
    (tmp_path / "draft_005.sql").write_text("not yet")  # missing m_ prefix
    (tmp_path / "m_007.sql.bak").write_text("backup")  # wrong extension

    loaded = _load_migrations(tmp_path)
    assert set(loaded.keys()) == {2, 3}
    assert "CREATE TABLE m2" in loaded[2]
    assert "CREATE TABLE m3" in loaded[3]


def test_load_migrations_returns_empty_dict_when_directory_missing(tmp_path: Path) -> None:
    """A missing migrations dir is not an error — fresh installs always
    have an empty dir until the first migration ships."""
    missing = tmp_path / "does-not-exist"
    assert _load_migrations(missing) == {}


# ──────────────────────────────────────────────────────────────────────────
# initialize_db — fresh + no-op + future paths (existing coverage in
# test_store.py asserts these against the production SCHEMA_VERSION; the
# tests below additionally cover the migration replay branch).
# ──────────────────────────────────────────────────────────────────────────


def _seed_v1_db(tmp_path: Path) -> Path:
    """Bring a DB to ``user_version == 1`` using the production schema.sql."""
    db = tmp_path / "old.sqlite3"
    initialize_db(db)
    return db


def test_initialize_db_applies_single_migration_for_old_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Old DB at v1 + simulated SCHEMA_VERSION=2 with one migration =
    migration applied, ``user_version`` bumped to 2."""
    db = _seed_v1_db(tmp_path)
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "m_002.sql").write_text(
        "CREATE TABLE migration_marker_v2 (id INTEGER PRIMARY KEY);"
    )

    monkeypatch.setattr("lucid.store.init.SCHEMA_VERSION", 2)
    initialize_db(db, migrations_dir=migrations)

    conn = sqlite3.connect(db)
    try:
        (uv,) = conn.execute("PRAGMA user_version;").fetchone()
        assert uv == 2
        marker = conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'migration_marker_v2'"
        ).fetchone()
        assert marker is not None
    finally:
        conn.close()


def test_initialize_db_applies_migrations_in_ascending_version_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A two-step upgrade replays both migrations and lands at the
    final version. The marker table from each migration must exist."""
    db = _seed_v1_db(tmp_path)
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    # Intentionally write v3 first to confirm ordering depends on
    # parsed integers, not directory iteration order.
    (migrations / "m_003.sql").write_text(
        "CREATE TABLE migration_marker_v3 (id INTEGER PRIMARY KEY);"
    )
    (migrations / "m_002.sql").write_text(
        "CREATE TABLE migration_marker_v2 (id INTEGER PRIMARY KEY);"
    )

    monkeypatch.setattr("lucid.store.init.SCHEMA_VERSION", 3)
    initialize_db(db, migrations_dir=migrations)

    conn = sqlite3.connect(db)
    try:
        (uv,) = conn.execute("PRAGMA user_version;").fetchone()
        assert uv == 3
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE 'migration_marker_%'"
            ).fetchall()
        }
        assert names == {"migration_marker_v2", "migration_marker_v3"}
    finally:
        conn.close()


def test_initialize_db_raises_when_a_required_migration_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1 → v3 with only m_002.sql present must fail at v3 — silently
    skipping would leave the DB in an undefined intermediate state."""
    db = _seed_v1_db(tmp_path)
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "m_002.sql").write_text(
        "CREATE TABLE migration_marker_v2 (id INTEGER PRIMARY KEY);"
    )
    # Missing: m_003.sql

    monkeypatch.setattr("lucid.store.init.SCHEMA_VERSION", 3)
    with pytest.raises(RuntimeError, match="no migration was found for v3"):
        initialize_db(db, migrations_dir=migrations)


def test_initialize_db_propagates_sql_error_inside_a_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A migration that raises a SQL error must surface to the
    caller. The DB state may be partial, but never silently inconsistent."""
    db = _seed_v1_db(tmp_path)
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "m_002.sql").write_text("THIS IS NOT VALID SQL;")

    monkeypatch.setattr("lucid.store.init.SCHEMA_VERSION", 2)
    with pytest.raises(sqlite3.Error):
        initialize_db(db, migrations_dir=migrations)


def test_initialize_db_does_not_call_migrations_on_fresh_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh installs apply schema.sql in one shot and never replay
    migrations. A migrations dir holding a poison-pill migration must
    therefore not run on a brand-new DB."""
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "m_002.sql").write_text("THIS WOULD BLOW UP IF RUN;")

    db = tmp_path / "fresh.sqlite3"
    monkeypatch.setattr("lucid.store.init.SCHEMA_VERSION", 2)
    # Schema applies; user_version jumps straight to 2 (the new
    # SCHEMA_VERSION); the bad migration is skipped because fresh DBs
    # don't replay.
    initialize_db(db, migrations_dir=migrations)

    conn = sqlite3.connect(db)
    try:
        (uv,) = conn.execute("PRAGMA user_version;").fetchone()
        assert uv == 2
    finally:
        conn.close()


def test_initialize_db_at_current_schema_version_is_a_noop(tmp_path: Path) -> None:
    """Repeated calls on a DB already at SCHEMA_VERSION must not raise
    or rewrite. The existing test_store.py covers a thinner version of
    this; we re-assert the contract here in the migration-focused
    suite so it stays nailed down even if test_store.py grows."""
    db = _seed_v1_db(tmp_path)
    initialize_db(db)
    initialize_db(db)
    conn = sqlite3.connect(db)
    try:
        (uv,) = conn.execute("PRAGMA user_version;").fetchone()
        assert uv == SCHEMA_VERSION
    finally:
        conn.close()
