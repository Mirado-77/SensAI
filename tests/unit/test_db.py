import sqlite3

import pytest

from sensai.core.errors import MigrationError
from sensai.db.connection import connect, migrate

EXPECTED_TABLES = {
    "sessions",
    "messages",
    "runs",
    "tool_calls",
    "events",
    "profiles",
    "schema_version",
}


def tables(connection):
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    return {row[0] for row in rows}


def versions(connection):
    return [row[0] for row in connection.execute("SELECT version FROM schema_version")]


def test_connect_applies_the_initial_migration():
    connection = connect(":memory:")
    assert EXPECTED_TABLES <= tables(connection)
    row = connection.execute("SELECT version, name FROM schema_version").fetchone()
    assert (row[0], row[1]) == (1, "001_b1_initial_schema")


def test_migrate_is_idempotent():
    connection = connect(":memory:")
    assert migrate(connection) == []
    assert versions(connection) == [1, 2]


def test_connect_creates_the_parent_directory_and_keeps_the_data(tmp_path):
    path = tmp_path / "nested" / "sensai.db"
    first = connect(path)
    first.execute("INSERT INTO profiles (id, data_json) VALUES ('dev', '{}')")
    first.commit()
    first.close()
    second = connect(path)
    assert second.execute("SELECT id FROM profiles").fetchone()[0] == "dev"
    assert versions(second) == [1, 2]


def test_new_migrations_are_applied_in_order(tmp_path):
    (tmp_path / "001_t1_first.sql").write_text("CREATE TABLE a (x INTEGER);")
    (tmp_path / "002_t1_second.sql").write_text("CREATE TABLE b (x INTEGER);")
    connection = sqlite3.connect(":memory:")
    assert migrate(connection, tmp_path) == [1, 2]
    (tmp_path / "003_t2_third.sql").write_text("CREATE TABLE c (x INTEGER);")
    assert migrate(connection, tmp_path) == [3]
    assert {"a", "b", "c"} <= tables(connection)


def test_failing_migration_is_rolled_back_and_not_recorded(tmp_path):
    (tmp_path / "001_t1_ok.sql").write_text("CREATE TABLE a (x INTEGER);")
    (tmp_path / "002_t1_broken.sql").write_text(
        "CREATE TABLE b (x INTEGER); INSERT INTO missing VALUES (1);"
    )
    connection = sqlite3.connect(":memory:")
    with pytest.raises(MigrationError, match="002_t1_broken"):
        migrate(connection, tmp_path)
    assert "b" not in tables(connection)
    assert versions(connection) == [1]


def test_invalid_file_name_is_rejected(tmp_path):
    (tmp_path / "init.sql").write_text("CREATE TABLE a (x INTEGER);")
    with pytest.raises(MigrationError, match="invalide"):
        migrate(sqlite3.connect(":memory:"), tmp_path)


def test_duplicate_version_is_rejected(tmp_path):
    (tmp_path / "001_a_one.sql").write_text("CREATE TABLE a (x INTEGER);")
    (tmp_path / "001_b_two.sql").write_text("CREATE TABLE b (x INTEGER);")
    with pytest.raises(MigrationError, match="001"):
        migrate(sqlite3.connect(":memory:"), tmp_path)


def test_foreign_keys_are_enforced():
    connection = connect(":memory:")
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO messages (id, session_id, role, content, created_at)"
            " VALUES ('m', 'unknown', 'user', 'x', 0)"
        )
