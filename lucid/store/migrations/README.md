# Lucid SQLite migrations

Files in this directory upgrade an existing Lucid DB from one schema
version to the next. They are applied in numeric order by
:func:`lucid.store.init.initialize_db` whenever the on-disk
``PRAGMA user_version`` is **lower** than
:data:`lucid.store.init.SCHEMA_VERSION`.

## File naming

```
m_<NNN>.sql       e.g. m_002.sql, m_003.sql
```

`<NNN>` is the **target** schema version: ``m_002.sql`` upgrades a
v1 DB to v2. The fresh-install path uses ``schema.sql`` directly and
sets ``user_version`` to :data:`SCHEMA_VERSION` in one shot — fresh
DBs never replay migrations.

## Authoring

1. Edit ``lucid/store/schema.sql`` to reflect the new shape (the
   snapshot fresh installs apply).
2. Bump ``SCHEMA_VERSION`` in ``lucid/store/init.py`` to the new
   number.
3. Add ``m_<new_version>.sql`` here with the SQL needed to upgrade
   *from the previous version* to the new one.
4. Add a test under ``tests/test_store_migrations.py`` that seeds a
   DB at the previous version and asserts the migration runs cleanly.

Each migration runs in its own transaction (one
``conn.executescript`` call). Use ``CREATE TABLE IF NOT EXISTS`` /
``ALTER TABLE`` carefully — SQLite's ``ALTER TABLE`` is restricted
(no ``DROP COLUMN`` before SQLite 3.35) so prefer additive changes.
