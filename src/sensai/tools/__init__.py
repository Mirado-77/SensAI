"""Tool registry and tools (files, sandbox, MCP)."""

from sensai.core.config import Config
from sensai.core.contracts import Tool
from sensai.tools.filesystem.edit_file import EditFileTool
from sensai.tools.filesystem.guard import PathGuard
from sensai.tools.filesystem.list_dir import ListDirTool
from sensai.tools.filesystem.read_file import ReadFileTool
from sensai.tools.filesystem.write_file import WriteFileTool


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
