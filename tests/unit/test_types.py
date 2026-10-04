import dataclasses

import pytest

from sensai.core.types import Message, ToolResult, new_id


def test_message_defaults():
    message = Message("user", "hello")
    assert message.id is None
    assert message.status == "complete"
    assert message.tool_calls == ()
    assert message.tool_name is None


def test_message_is_immutable():
    message = Message("user", "hello")
    with pytest.raises(dataclasses.FrozenInstanceError):
        message.content = "other"  # type: ignore[misc]


def test_tool_result_defaults():
    result = ToolResult("ok")
    assert result.is_error is False
    assert result.untrusted is False


def test_new_id_returns_distinct_ids():
    assert len({new_id() for _ in range(100)}) == 100
