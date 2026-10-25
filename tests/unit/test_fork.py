import pytest
from tests.unit.test_cli import chat_setup

from sensai.cli.chat import LAST_SESSION, run_chat
from sensai.core.fakes import FakeModelGateway
from sensai.core.types import Chunk, Message
from sensai.db.connection import connect
from sensai.memory.sqlite_store import SqliteSessionStore

pytestmark = pytest.mark.feature("X2")


def roles(call):
    return [(m.role, m.content) for m in call.messages]


async def test_history_numbers_the_messages_and_marks_the_interrupted_one(tmp_path):
    config, _, _, _ = chat_setup(tmp_path)
    store = SqliteSessionStore(connect(config.storage.db_path))
    session_id = store.create_session()
    parent = store.append(session_id, Message("user", "explain the bug"), None)
    store.append(session_id, Message("assistant", "The bug is", status="interrupted"), parent)

    _, terminal, out, _ = chat_setup(tmp_path, "/history")
    await run_chat(config, terminal, FakeModelGateway(), resume=LAST_SESSION)
    assert "1 user      explain the bug\n" in out.getvalue()
    assert "2 assistant The bug is (interrupted)\n" in out.getvalue()


async def test_fork_continues_from_a_message_and_leaves_the_original_session(tmp_path):
    config, terminal, out, _ = chat_setup(tmp_path, "a", "b", "/fork 2", "c")
    gateway = FakeModelGateway([Chunk("one")], [Chunk("two")], [Chunk("three")])
    assert await run_chat(config, terminal, gateway) == 0
    assert "Forked session" in out.getvalue()
    assert roles(gateway.calls[2]) == [("user", "a"), ("assistant", "one"), ("user", "c")]

    store = SqliteSessionStore(connect(config.storage.db_path))
    contents = [[m.content for m in store.history(row["id"])] for row in store.list_sessions(5)]
    assert contents == [["a", "one", "c", "three"], ["a", "one", "b", "two"]]


@pytest.mark.parametrize("command", ["/fork", "/fork x", "/fork 0", "/fork 3"])
async def test_an_invalid_fork_shows_the_help_and_stays_in_the_session(tmp_path, command):
    config, terminal, _, err = chat_setup(tmp_path, "a", command, "b")
    gateway = FakeModelGateway([Chunk("one")], [Chunk("two")])
    await run_chat(config, terminal, gateway)
    assert "/fork <n>" in err.getvalue()
    assert roles(gateway.calls[1])[:2] == [("user", "a"), ("assistant", "one")]
