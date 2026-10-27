import asyncio
import json

import httpx
import pytest
import respx

from sensai.core.errors import ModelConnectionError, ModelError, ModelNotFoundError
from sensai.core.types import Chunk, Done, Message, ToolCall, ToolSpec, Usage
from sensai.model.ollama import OllamaGateway

pytestmark = pytest.mark.feature("B1")

MESSAGES = [Message("user", "Hello")]


@pytest.fixture
def router(ollama_host):
    with respx.mock(base_url=ollama_host, assert_all_called=False) as router:
        yield router


@pytest.fixture
def gateway(ollama_host):
    return OllamaGateway(ollama_host, "test-model", timeout_s=5)


def reply(router, *lines, status=200):
    return router.post("/api/chat").mock(
        return_value=httpx.Response(status, content=b"".join(lines))
    )


async def collect(gateway, messages=MESSAGES, **kwargs):
    return [event async for event in gateway.stream_chat(messages, **kwargs)]


def sent_body(route):
    return json.loads(route.calls.last.request.content)


async def test_streams_chunks_then_usage_then_done(router, gateway, ollama_lines):
    reply(router, *ollama_lines([Chunk("Hel"), Chunk("lo"), Usage(5, 2)]))
    assert await collect(gateway) == [Chunk("Hel"), Chunk("lo"), Usage(5, 2), Done()]


async def test_usage_is_omitted_when_ollama_sends_no_counts(router, gateway, ollama_lines):
    reply(router, *ollama_lines([Chunk("a")]))
    assert await collect(gateway) == [Chunk("a"), Done()]


async def test_parses_a_tool_call(router, gateway, ollama_lines):
    call = ToolCall("read_file", {"path": "a.py"})
    reply(router, *ollama_lines([call]))
    assert await collect(gateway) == [call, Done()]


async def test_ignores_thinking_tokens(router, gateway, ollama_lines):
    thinking = (
        b'{"message": {"role": "assistant", "content": "", "thinking": "hmm"}, "done": false}\n'
    )
    reply(router, thinking, *ollama_lines([Chunk("answer")]))
    assert await collect(gateway) == [Chunk("answer"), Done()]


async def test_sends_model_messages_and_stream_flag(router, gateway, ollama_lines):
    route = reply(router, *ollama_lines([]))
    await collect(gateway)
    assert sent_body(route) == {
        "model": "test-model",
        "stream": True,
        "think": False,
        "messages": [{"role": "user", "content": "Hello"}],
    }


async def test_thinking_can_be_enabled_per_gateway(router, ollama_host, ollama_lines):
    route = reply(router, *ollama_lines([]))
    gateway = OllamaGateway(ollama_host, "test-model", timeout_s=5, think=True)
    await collect(gateway)
    assert sent_body(route)["think"] is True


async def test_sends_tool_calls_and_tool_results(router, gateway, ollama_lines):
    messages = [
        Message("user", "Read a.py"),
        Message("assistant", "", tool_calls=(ToolCall("read_file", {"path": "a.py"}),)),
        Message("tool", "print(1)", tool_name="read_file"),
    ]
    route = reply(router, *ollama_lines([]))
    await collect(gateway, messages)
    assert sent_body(route)["messages"] == [
        {"role": "user", "content": "Read a.py"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}],
        },
        {"role": "tool", "content": "print(1)", "tool_name": "read_file"},
    ]


async def test_sends_tools_and_schema(router, gateway, ollama_lines):
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}}
    spec = ToolSpec("read_file", "Read a file.", parameters)
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    route = reply(router, *ollama_lines([]))
    await collect(gateway, tools=[spec], schema=schema)
    body = sent_body(route)
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": parameters,
            },
        }
    ]
    assert body["format"] == schema


async def test_host_with_a_trailing_slash_is_accepted(router, ollama_host, ollama_lines):
    reply(router, *ollama_lines([Chunk("a")]))
    gateway = OllamaGateway(ollama_host + "/", "test-model", timeout_s=5)
    assert await collect(gateway) == [Chunk("a"), Done()]


async def test_error_line_in_the_stream(router, gateway):
    reply(router, b'{"error": "boom"}\n')
    with pytest.raises(ModelError, match="boom"):
        await collect(gateway)


async def test_model_not_found_error_line(router, gateway):
    reply(router, b'{"error": "model \'x\' not found"}\n')
    with pytest.raises(ModelNotFoundError):
        await collect(gateway)


async def test_http_404_model_not_found(router, gateway):
    reply(router, b'{"error": "model \'test-model\' not found"}', status=404)
    with pytest.raises(ModelNotFoundError, match="test-model"):
        await collect(gateway)


async def test_http_404_with_another_message_is_a_plain_model_error(router, gateway):
    reply(router, b'{"error": "page missing"}', status=404)
    with pytest.raises(ModelError) as raised:
        await collect(gateway)
    assert not isinstance(raised.value, ModelNotFoundError)


async def test_http_500(router, gateway):
    reply(router, b'{"error": "out of memory"}', status=500)
    with pytest.raises(ModelError, match="500.*out of memory"):
        await collect(gateway)


async def test_http_error_with_a_plain_text_body(router, gateway):
    reply(router, b"oops", status=502)
    with pytest.raises(ModelError, match="oops"):
        await collect(gateway)


async def test_connection_refused(router, gateway):
    router.post("/api/chat").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(ModelConnectionError, match="injoignable"):
        await collect(gateway)


async def test_timeout(router, gateway):
    router.post("/api/chat").mock(side_effect=httpx.ReadTimeout(""))
    with pytest.raises(ModelConnectionError, match="ReadTimeout"):
        await collect(gateway)


async def test_stream_ending_without_done(router, gateway, ollama_lines):
    reply(router, *ollama_lines([Chunk("a")])[:-1])
    with pytest.raises(ModelConnectionError, match="interrompue"):
        await collect(gateway)


async def test_invalid_json_line(router, gateway):
    reply(router, b"not json\n")
    with pytest.raises(ModelError, match="invalide"):
        await collect(gateway)


async def test_malformed_tool_call(router, gateway):
    reply(router, b'{"message": {"tool_calls": [{"function": {}}]}, "done": false}\n')
    with pytest.raises(ModelError, match="invalide"):
        await collect(gateway)


async def test_cancelling_closes_the_response_stream(router, gateway, ollama_lines, slow_body):
    body, probe = slow_body(ollama_lines([Chunk(str(i)) for i in range(1000)]))
    router.post("/api/chat").mock(return_value=httpx.Response(200, content=body))
    received = []

    async def consume():
        async for event in gateway.stream_chat(MESSAGES):
            received.append(event)

    task = asyncio.create_task(consume())
    while not received and not task.done():
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert probe.closed
