import asyncio

import pytest

from sensai.core.contracts import ModelGateway
from sensai.core.errors import ModelConnectionError, ModelError, ModelNotFoundError
from sensai.core.types import Chunk, Done, Message, ToolCall, ToolSpec, Usage

MESSAGES = [Message("user", "Hello")]


async def collect(gateway, **kwargs):
    return [event async for event in gateway.stream_chat(MESSAGES, **kwargs)]


def test_satisfies_the_protocol(make_gateway):
    assert isinstance(make_gateway(), ModelGateway)


async def test_replays_events_in_order(make_gateway):
    script = [Chunk("Hel"), Chunk("lo"), Usage(5, 2)]
    events = await collect(make_gateway(script))
    assert events[:-1] == script


async def test_stream_ends_with_a_single_done(make_gateway):
    events = await collect(make_gateway([Chunk("a"), Usage(1, 1)]))
    assert events[-1] == Done()
    assert sum(isinstance(event, Done) for event in events) == 1


async def test_empty_stream_contains_done(make_gateway):
    assert await collect(make_gateway([])) == [Done()]


async def test_forwards_a_tool_call(make_gateway):
    call = ToolCall("read_file", {"path": "a.py"})
    spec = ToolSpec("read_file", "Read a file.", {"type": "object", "properties": {}})
    events = await collect(make_gateway([call]), tools=[spec], schema={"type": "object"})
    assert call in events


@pytest.mark.parametrize("error", [ModelConnectionError, ModelNotFoundError])
async def test_failures_raise_a_model_error(make_gateway, error):
    with pytest.raises(ModelError) as raised:
        await collect(make_gateway(error=error))
    assert isinstance(raised.value, error)


async def test_cancelling_the_task_stops_the_stream(make_gateway):
    gateway = make_gateway([Chunk(str(i)) for i in range(1000)])
    received: list[object] = []

    async def consume() -> None:
        async for event in gateway.stream_chat(MESSAGES):
            received.append(event)

    task = asyncio.create_task(consume())
    while not received and not task.done():
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert 0 < len(received) < 1000
