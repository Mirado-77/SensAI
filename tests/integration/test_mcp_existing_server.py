"""T1 against an existing MCP server, the official filesystem server run with npx.

Run with `make test-integration`. Skipped when npx is missing. The first run downloads the
package from npm.
"""

import shutil

import pytest

from sensai.core.config import McpConfig, McpServerConfig
from sensai.core.types import ToolContext
from sensai.tools.mcp.servers import McpServers
from sensai.tools.registry import ToolRegistry

pytestmark = [pytest.mark.integration, pytest.mark.feature("T1")]

PACKAGE = "@modelcontextprotocol/server-filesystem@2026.8.31"


@pytest.fixture
def filesystem_server(tmp_path):
    if shutil.which("npx") is None:
        pytest.skip("npx is not installed")
    (tmp_path / "notes.txt").write_text("hello from the demo repo\n", encoding="utf-8")
    server = McpServerConfig(
        name="fs", command="npx", args=["-y", PACKAGE, str(tmp_path)], risk="auto"
    )
    return McpConfig(timeout_s=120, servers=[server]), tmp_path


async def test_our_client_lists_and_calls_the_tools_of_an_existing_server(filesystem_server):
    config, root = filesystem_server
    async with McpServers(config) as servers:
        assert servers.warnings == []
        registry = ToolRegistry(servers.tools)
        tool = registry.get("fs__list_directory")
        assert tool is not None
        assert tool.risk == "read"
        result = await tool.run({"path": str(root)}, ToolContext("r", "s"))
    assert not result.is_error
    assert result.untrusted
    assert "notes.txt" in result.content


async def test_the_existing_server_refuses_paths_outside_its_root(filesystem_server, tmp_path):
    config, root = filesystem_server
    async with McpServers(config) as servers:
        tool = ToolRegistry(servers.tools).get("fs__list_directory")
        result = await tool.run({"path": "/etc"}, ToolContext("r", "s"))
    assert result.is_error
