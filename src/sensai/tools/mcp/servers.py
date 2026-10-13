"""Lifecycle of the configured MCP servers (T1): start them, expose their tools, stop them."""

import asyncio
import contextlib
from collections.abc import Callable

from sensai.core.config import McpConfig, McpServerConfig
from sensai.tools.mcp.adapter import McpTool
from sensai.tools.mcp.client import McpClient, McpError

ClientFactory = Callable[[McpServerConfig, float], McpClient]


def _client(server: McpServerConfig, timeout_s: float) -> McpClient:
    return McpClient(server.name, server.command, server.args, server.env, server.cwd, timeout_s)


class McpServers:
    """Async context manager. A server that fails to start is skipped and listed in `warnings`."""

    def __init__(self, config: McpConfig, make_client: ClientFactory = _client) -> None:
        self._config = config
        self._make_client = make_client
        self._clients: list[McpClient] = []
        self.tools: list[McpTool] = []
        self.warnings: list[str] = []

    async def __aenter__(self) -> "McpServers":
        servers = [server for server in self._config.servers if server.enabled]
        results = await asyncio.gather(*(self._start(server) for server in servers))
        for tools in results:
            self.tools.extend(tools)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        clients, self._clients = self._clients, []
        await asyncio.gather(*(client.aclose() for client in clients))

    async def _start(self, server: McpServerConfig) -> list[McpTool]:
        client = self._make_client(server, self._config.timeout_s)
        try:
            await client.start()
            specs = await client.list_tools()
        except McpError as exc:
            self.warnings.append(str(exc))
            with contextlib.suppress(McpError):
                await client.aclose()
            return []
        self._clients.append(client)
        return [McpTool(server.name, spec, client, server.risk) for spec in specs]
