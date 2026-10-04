from dataclasses import replace

import pytest

from sensai.core.contracts import SessionStore
from sensai.core.types import Message, ToolCall


def conversation(store, count=3):
    """Create a session of `count` alternating user/assistant messages: (session, ids)."""
    session_id = store.create_session()
    ids: list[str] = []
    parent = None
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        parent = store.append(session_id, Message(role, f"m{i}"), parent)
        ids.append(parent)
    return session_id, ids


def contents(messages):
    return [message.content for message in messages]


def test_satisfies_the_protocol(harness):
    assert isinstance(harness.store, SessionStore)


def test_create_session_returns_distinct_ids(harness):
    ids = {harness.store.create_session() for _ in range(3)}
    assert len(ids) == 3
    assert all(ids)


def test_empty_session_has_empty_history(harness):
    assert harness.store.history(harness.store.create_session()) == []


def test_history_is_ordered_from_root_to_head(harness):
    session_id, ids = conversation(harness.store)
    history = harness.store.history(session_id)
    assert contents(history) == ["m0", "m1", "m2"]
    assert [message.id for message in history] == ids


def test_append_ignores_the_given_id(harness):
    store = harness.store
    session_id = store.create_session()
    message_id = store.append(session_id, Message("user", "hi", id="forced"), None)
    assert message_id != "forced"
    assert store.history(session_id)[0].id == message_id


def test_all_message_fields_are_preserved(harness):
    store = harness.store
    session_id = store.create_session()
    partial = Message(
        "assistant",
        "partial answer",
        status="interrupted",
        tool_calls=(ToolCall("read_file", {"path": "a.py"}),),
    )
    result = Message("tool", "content", tool_name="read_file")
    first = store.append(session_id, partial, None)
    store.append(session_id, result, first)
    stored = store.history(session_id)
    assert [replace(message, id=None) for message in stored] == [partial, result]


def test_history_with_head_id_returns_the_prefix(harness):
    session_id, ids = conversation(harness.store)
    assert contents(harness.store.history(session_id, head_id=ids[1])) == ["m0", "m1"]


def test_append_under_an_older_message_creates_a_branch(harness):
    store = harness.store
    session_id, ids = conversation(store)
    store.append(session_id, Message("assistant", "other"), ids[0])
    assert contents(store.history(session_id)) == ["m0", "other"]
    assert contents(store.history(session_id, head_id=ids[2])) == ["m0", "m1", "m2"]


def test_fork_shares_history_then_diverges(harness):
    store = harness.store
    session_id, ids = conversation(store)
    fork_id = store.fork(session_id, ids[1])
    assert fork_id != session_id
    assert contents(store.history(fork_id)) == ["m0", "m1"]

    store.append(fork_id, Message("user", "in the fork"), ids[1])
    store.append(session_id, Message("user", "in the original"), ids[2])
    assert contents(store.history(fork_id)) == ["m0", "m1", "in the fork"]
    assert contents(store.history(session_id)) == ["m0", "m1", "m2", "in the original"]


def test_fork_of_a_fork(harness):
    store = harness.store
    session_id, ids = conversation(store)
    first = store.fork(session_id, ids[1])
    new_id = store.append(first, Message("user", "next"), ids[1])
    second = store.fork(first, new_id)
    assert contents(store.history(second)) == ["m0", "m1", "next"]
    third = store.fork(first, ids[0])
    assert contents(store.history(third)) == ["m0"]


def test_messages_never_change(harness):
    store = harness.store
    session_id, ids = conversation(store)
    before = store.history(session_id)
    store.append(session_id, Message("user", "other"), ids[0])
    assert store.history(session_id, head_id=ids[2]) == before


def test_unknown_session(harness):
    store = harness.store
    message = Message("user", "x")
    with pytest.raises(KeyError):
        store.append("unknown", message, None)
    with pytest.raises(KeyError):
        store.history("unknown")
    with pytest.raises(KeyError):
        store.fork("unknown", "m")


def test_unknown_message(harness):
    store = harness.store
    session_id, _ = conversation(store)
    with pytest.raises(KeyError):
        store.append(session_id, Message("user", "x"), "unknown")
    with pytest.raises(KeyError):
        store.history(session_id, head_id="unknown")
    with pytest.raises(KeyError):
        store.fork(session_id, "unknown")


def test_message_from_another_session_is_rejected(harness):
    store = harness.store
    session_id, _ = conversation(store)
    _, other_ids = conversation(store)
    with pytest.raises(KeyError):
        store.history(session_id, head_id=other_ids[0])
    with pytest.raises(KeyError):
        store.fork(session_id, other_ids[0])


def test_profile_returns_the_data(harness):
    harness.seed_profile("dev", {"language": "python", "tags": ["a"]})
    assert harness.store.profile("dev") == {"language": "python", "tags": ["a"]}


def test_mutating_the_returned_profile_does_not_change_the_store(harness):
    harness.seed_profile("dev", {"tags": ["a"]})
    harness.store.profile("dev")["tags"].append("b")
    assert harness.store.profile("dev") == {"tags": ["a"]}


def test_unknown_profile(harness):
    with pytest.raises(KeyError):
        harness.store.profile("unknown")
