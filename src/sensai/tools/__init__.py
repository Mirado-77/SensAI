"""Tool registry and tools (files, sandbox, MCP)."""

from sensai.core.config import Config
from sensai.core.contracts import Tool
from sensai.tools.filesystem.edit_file import EditFileTool
from sensai.tools.filesystem.guard import PathGuard
from sensai.tools.filesystem.list_dir import ListDirTool
from sensai.tools.filesystem.read_file import ReadFileTool
from sensai.tools.filesystem.write_file import WriteFileTool
from sensai.tools.sandbox.docker import docker_unavailable_reason
from sensai.tools.sandbox.run_command import RunCommandTool
from sensai.tools.web.searxng import WebSearchTool


def native_tools(config: Config) -> list[Tool]:
    files = config.files
    if not files.allowed_roots:
        return []
    guard = PathGuard(files.allowed_roots)
    return [
        ReadFileTool(guard, files.max_read_bytes),
        ListDirTool(guard, files.max_list_entries),
        WriteFileTool(guard, files.max_write_bytes, files.max_diff_lines),
        EditFileTool(guard, files.max_write_bytes, files.max_diff_lines),
    ]


async def sandbox_tools(config: Config) -> tuple[list[Tool], str | None]:
    """Return the sandbox tool (T2), or no tool and the reason why it is disabled."""
    sandbox = config.sandbox
    if not sandbox.enabled or not config.files.allowed_roots:
        return [], None
    reason = await docker_unavailable_reason(sandbox.image)
    if reason is not None:
        return [], reason
    return [RunCommandTool(PathGuard(config.files.allowed_roots), sandbox)], None


def web_tools(config: Config) -> list[Tool]:
    """Return the web search tool (T3) when it is enabled in `[web]`."""
    if not config.web.enabled:
        return []
    return [WebSearchTool(config.web)]
