"""Fake behaviors that the runtime tests will rely on (outside the contract)."""

import pytest

from sensai.core.errors import ModelConnectionError
from sensai.core.events import RUN_STARTED, make_event
from sensai.core.fakes import (
    CollectingEventSink,
    FakeModelGateway,
    FakeTool,
    FakeUserChannel,
)
from sensai.core.types import (
    ApprovalRequest,
    Chunk,
    Done,
    Message,
    ToolContext,
    ToolResult,
    ToolSpec,
)

MESSAGES = [Message("user", "Hello")]
REQUEST = ApprovalRequest("r1", "write_file", "write", "diff", {})
CTX = ToolContext("r1", "s1")


async def collect(gateway, **kwargs):
    return [event async for event in gateway.stream_chat(MESSAGES, **kwargs)]


async def test_gateway_consumes_one_script_per_call():
    gateway = FakeModelGateway([Chunk("one")], [Chunk("two")])
    assert await collect(gateway) == [Chunk("one"), Done()]
    assert await collect(gateway) == [Chunk("two"), Done()]


async def test_gateway_records_calls():
    gateway = FakeModelGateway([])
    spec = ToolSpec("t", "tool", {"type": "object"})
    await collect(gateway, tools=[spec], schema={"type": "object"})
    [call] = gateway.calls
    assert call.messages == MESSAGES
    assert call.tools == [spec]
    assert call.schema == {"type": "object"}


async def test_gateway_reports_an_exhausted_script():
    gateway = FakeModelGateway([])
    await collect(gateway)
    with pytest.raises(AssertionError, match="plus de script"):
        await collect(gateway)


def test_gateway_rejects_a_script_containing_done():
    with pytest.raises(ValueError, match="Done"):
        FakeModelGateway([Chunk("a"), Done()])


async def test_gateway_raises_the_given_error_on_every_call():
    gateway = FakeModelGateway(error=ModelConnectionError("dropped"))
    for _ in range(2):
        with pytest.raises(ModelConnectionError):
            await collect(gateway)
    assert len(gateway.calls) == 2


async def test_tool_records_calls_and_returns_its_result():
    result = ToolResult("content", untrusted=True)
    tool = FakeTool(result=result)
    assert await tool.run({"path": "a.py"}, CTX) == result
    assert tool.calls == [({"path": "a.py"}, CTX)]


async def test_tool_raises_the_given_error():
    tool = FakeTool(error=RuntimeError("failure"))
    with pytest.raises(RuntimeError, match="failure"):
        await tool.run({}, CTX)
    assert len(tool.calls) == 1


async def test_channel_answers_in_order_then_by_default():
    channel = FakeUserChannel(approvals=[False, True], default_approval=False)
    assert [await channel.confirm(REQUEST) for _ in range(3)] == [False, True, False]
    assert channel.confirm_requests == [REQUEST] * 3


async def test_channel_records_questions():
    channel = FakeUserChannel(answers=[{"choice": "a"}])
    assert await channel.ask({"q": 1}) == {"choice": "a"}
    assert await channel.ask({"q": 2}) == {}
    assert channel.questions == [{"q": 1}, {"q": 2}]


def test_sink_exposes_received_types():
    sink = CollectingEventSink()
    sink.emit(make_event(RUN_STARTED, "r1"))
    assert sink.types() == ["run.started"]
