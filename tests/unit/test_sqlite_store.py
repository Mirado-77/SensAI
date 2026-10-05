"""SqliteSessionStore behaviors outside the shared SessionStore contract."""

from sensai.core.types import Message, ToolCall
from sensai.db.connection import connect
from sensai.memory.sqlite_store import SqliteSessionStore


def test_sessions_and_messages_survive_reopening_the_database(tmp_path):
    path = tmp_path / "sensai.db"
    first = SqliteSessionStore(connect(path))
    session_id = first.create_session()
    first_id = first.append(session_id, Message("user", "hello"), None)
    first.append(session_id, Message("assistant", "hi"), first_id)

    second = SqliteSessionStore(connect(path))
    assert [m.content for m in second.history(session_id)] == ["hello", "hi"]


def test_a_fork_survives_reopening_the_database(tmp_path):
    path = tmp_path / "sensai.db"
    store = SqliteSessionStore(connect(path))
    session_id = store.create_session()
    first_id = store.append(session_id, Message("user", "hello"), None)
    store.append(session_id, Message("assistant", "hi"), first_id)
    fork_id = store.fork(session_id, first_id)

    reopened = SqliteSessionStore(connect(path))
    assert [m.content for m in reopened.history(fork_id)] == ["hello"]
    assert [m.content for m in reopened.history(session_id)] == ["hello", "hi"]


def test_nested_tool_call_arguments_round_trip():
    store = SqliteSessionStore(connect(":memory:"))
    session_id = store.create_session()
    args = {"path": "é.py", "options": {"lines": [1, 2], "strict": True}}
    message = Message("assistant", "", tool_calls=(ToolCall("read_file", args),))
    store.append(session_id, message, None)
    assert store.history(session_id)[0].tool_calls == (ToolCall("read_file", args),)


def test_add_profile_replaces_an_existing_profile():
    store = SqliteSessionStore(connect(":memory:"))
    store.add_profile("dev", {"language": "python"})
    store.add_profile("dev", {"language": "rust"})
    assert store.profile("dev") == {"language": "rust"}


def test_the_fork_keeps_the_profile_of_its_origin():
    connection = connect(":memory:")
    store = SqliteSessionStore(connection)
    session_id = store.create_session(profile_id="dev")
    message_id = store.append(session_id, Message("user", "hello"), None)
    fork_id = store.fork(session_id, message_id)
    row = connection.execute("SELECT profile_id FROM sessions WHERE id = ?", (fork_id,)).fetchone()
    assert row[0] == "dev"
