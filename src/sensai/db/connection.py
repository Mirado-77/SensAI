"""SQLite connection and forward-only `.sql` migrations."""

import re
import sqlite3
import time
from pathlib import Path

from sensai.core.errors import MigrationError

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_MIGRATION_NAME = re.compile(r"^(\d{3})_([a-z0-9]+)_([a-z0-9_]+)\.sql$")


def connect(path: Path | str) -> sqlite3.Connection:
    """Open the database, create its parent directory and apply the pending migrations."""
    if str(path) != ":memory:":
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    migrate(connection)
    return connection


def migrate(connection: sqlite3.Connection, migrations_dir: Path = MIGRATIONS_DIR) -> list[int]:
    """Apply the migrations not yet recorded in `schema_version`; return their versions.

    Files are named `NNN_<feature-id>_<description>.sql`. Each one runs in a transaction that
    is rolled back if it fails.
    """
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_version ("
        "version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at REAL NOT NULL)"
    )
    connection.commit()
    applied = {row[0] for row in connection.execute("SELECT version FROM schema_version")}
    done = []
    for version, name, sql in _load(migrations_dir):
        if version in applied:
            continue
        try:
            connection.executescript(f"BEGIN;\n{sql}")
            connection.execute(
                "INSERT INTO schema_version (version, name, applied_at) VALUES (?, ?, ?)",
                (version, name, time.time()),
            )
            connection.commit()
        except sqlite3.Error as exc:
            connection.rollback()
            raise MigrationError(f"Échec de la migration {name} : {exc}") from exc
        done.append(version)
    return done


def _load(migrations_dir: Path) -> list[tuple[int, str, str]]:
    migrations: dict[int, tuple[str, str]] = {}
    for path in sorted(migrations_dir.glob("*.sql")):
        match = _MIGRATION_NAME.match(path.name)
        if match is None:
            raise MigrationError(f"Nom de migration invalide : {path.name}")
        version = int(match.group(1))
        if version in migrations:
            raise MigrationError(f"Deux migrations portent le numéro {version:03d}")
        migrations[version] = (path.stem, path.read_text(encoding="utf-8"))
    return [(version, name, sql) for version, (name, sql) in sorted(migrations.items())]
