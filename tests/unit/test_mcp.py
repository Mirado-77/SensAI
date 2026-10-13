import asyncio
import json
import sys
from pathlib import Path

import pytest

from sensai.core.config import McpConfig, McpServerConfig
from sensai.core.types import ToolContext
from sensai.tools.mcp.adapter import McpTool, result_text, tool_name
from sensai.tools.mcp.client import McpClient, McpError
from sensai.tools.mcp.servers import McpServers
from sensai.tools.registry import ToolRegistry

pytestmark = pytest.mark.feature("T1")

FAKE_SERVER = Path(__file__).resolve().parents[1] / "fixtures" / "mcp_fake_server.py"
CTX = ToolContext("run", "session")


def fake_client(mode="normal", timeout_s=5.0):
    return McpClient("fake", sys.executable, [str(FAKE_SERVER), mode], timeout_s=timeout_s)


def server_config(name="fake", mode="normal", **fields):
    return McpServerConfig(
        name=name, command=sys.executable, args=[str(FAKE_SERVER), mode], **fields
    )


@pytest.fixture
async def client():
    client = fake_client()
    await client.start()
    yield client
    await client.aclose()


async def test_handshake_reads_the_server_info(client):
    assert client.server_info == {"name": "fake", "version": "1"}


async def test_list_tools_follows_pagination(client):
    tools = await client.list_tools()
    assert [tool["name"] for tool in tools] == ["echo", "fail", "readonly"]


async def test_call_tool_returns_the_result(client):
    result = await client.call_tool("echo", {"text": "hi"})
    assert result == {"content": [{"type": "text", "text": '{"text": "hi"}'}]}


async def test_concurrent_calls_are_matched_by_id(client):
    results = await asyncio.gather(*(client.call_tool("echo", {"text": str(i)}) for i in range(5)))
    assert [result_text(result) for result in results] == [
        json.dumps({"text": str(i)}) for i in range(5)
    ]


async def test_a_jsonrpc_error_becomes_an_mcp_error(client):
    with pytest.raises(McpError, match="Method not found: resources/list"):
        await client.request("resources/list", {})


async def test_a_server_ping_is_answered():
    client = fake_client("ping_first")
    await client.start()
    try:
        assert len(await client.list_tools()) == 3
    finally:
        await client.aclose()


async def test_a_missing_command_fails_to_start():
    client = McpClient("ghost", "/nonexistent/sensai-mcp-server")
    with pytest.raises(McpError, match="impossible à lancer"):
        await client.start()


async def test_an_initialize_error_fails_to_start_and_stops_the_process():
    client = fake_client("init_error")
    with pytest.raises(McpError, match="init refused"):
        await client.start()
    with pytest.raises(McpError, match="arrêté"):
        await client.list_tools()


async def test_a_crashing_server_fails_the_pending_call():
    client = fake_client("crash_on_call")
    await client.start()
    try:
        with pytest.raises(McpError, match="arrêté de façon inattendue"):
            await client.call_tool("echo", {"text": "x"})
        with pytest.raises(McpError, match="arrêté"):
            await client.list_tools()
    finally:
        await client.aclose()


async def test_a_silent_server_times_out():
    client = fake_client("hang_on_call", timeout_s=0.3)
    await client.start()
    try:
        with pytest.raises(McpError, match="pas de réponse à tools/call"):
            await client.call_tool("echo", {"text": "x"})
    finally:
        await client.aclose()


async def test_cancelling_a_call_leaves_the_client_usable():
    client = fake_client("normal")
    await client.start()
    try:
        task = asyncio.ensure_future(client.call_tool("echo", {"text": "x"}))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert result_text(await client.call_tool("echo", {"text": "y"})) == '{"text": "y"}'
    finally:
        await client.aclose()


async def test_aclose_is_idempotent(client):
    await client.aclose()
    await client.aclose()


async def test_mcp_tool_runs_through_the_client_and_marks_output_untrusted(client):
    [spec] = [tool for tool in await client.list_tools() if tool["name"] == "echo"]
    tool = McpTool("fake", spec, client, "exec")
    assert (tool.name, tool.risk, tool.parameters) == ("fake__echo", "exec", spec["inputSchema"])
    result = await tool.run({"text": "hi"}, CTX)
    assert (result.content, result.is_error, result.untrusted) == ('{"text": "hi"}', False, True)


async def test_mcp_tool_reports_server_errors_as_error_results(client):
    tool = McpTool("fake", {"name": "fail"}, client, "exec")
    result = await tool.run({}, CTX)
    assert (result.content, result.is_error) == ("it failed", True)


async def test_mcp_tool_turns_a_dead_server_into_an_error_result():
    client = fake_client("crash_on_call")
    await client.start()
    try:
        tool = McpTool("fake", {"name": "echo"}, client, "exec")
        result = await tool.run({"text": "x"}, CTX)
        assert result.is_error
        assert "arrêté" in result.content
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    ("setting", "annotations", "risk"),
    [
        ("auto", {"readOnlyHint": True}, "read"),
        ("auto", {"readOnlyHint": False}, "exec"),
        ("auto", None, "exec"),
        ("write", {"readOnlyHint": True}, "write"),
        ("read", None, "read"),
    ],
)
def test_risk_comes_from_the_config_or_the_read_only_hint(setting, annotations, risk):
    spec = {"name": "t", "annotations": annotations}
    assert McpTool("s", spec, None, setting).risk == risk


def test_missing_or_partial_input_schema_becomes_an_object_schema():
    assert McpTool("s", {"name": "t"}, None, "exec").parameters == {
        "type": "object",
        "properties": {},
    }
    partial = {"name": "t", "inputSchema": {"properties": {"a": {"type": "string"}}}}
    assert McpTool("s", partial, None, "exec").parameters["type"] == "object"


def test_tool_names_are_prefixed_sanitized_and_bounded():
    assert tool_name("git", "git_log") == "git__git_log"
    assert tool_name("my.server", "do it") == "my_server__do_it"
    assert len(tool_name("s", "x" * 100)) == 64


def test_result_text_handles_every_content_type():
    result = {
        "content": [
            {"type": "text", "text": "a"},
            {"type": "image", "data": "...", "mimeType": "image/png"},
            {"type": "resource", "resource": {"uri": "file:///x", "text": "b"}},
            {"type": "resource", "resource": {"uri": "file:///y", "blob": "..."}},
            {"type": "resource_link", "uri": "file:///z"},
        ]
    }
    assert result_text(result).splitlines() == [
        "a",
        "[image content omitted]",
        "b",
        "[resource file:///y omitted]",
        "[resource link file:///z]",
    ]
    assert result_text({"content": [], "structuredContent": {"n": 1}}) == '{"n": 1}'


async def test_servers_start_register_and_stop():
    config = McpConfig(timeout_s=5, servers=[server_config(risk="auto")])
    async with McpServers(config) as servers:
        names = [tool.name for tool in servers.tools]
        assert names == ["fake__echo", "fake__fail", "fake__readonly"]
        assert [tool.risk for tool in servers.tools] == ["exec", "exec", "read"]
        registry = ToolRegistry(servers.tools)
        assert len(registry) == 3
        assert servers.warnings == []
        client = servers.tools[0]._caller
    with pytest.raises(McpError, match="arrêté"):
        await client.list_tools()


async def test_a_failing_server_is_skipped_with_a_warning():
    config = McpConfig(
        timeout_s=5,
        servers=[
            server_config("broken", "init_error"),
            server_config("good"),
            server_config("off", enabled=False),
        ],
    )
    async with McpServers(config) as servers:
        assert {tool.name.split("__")[0] for tool in servers.tools} == {"good"}
        assert len(servers.warnings) == 1
        assert "init refused" in servers.warnings[0]
