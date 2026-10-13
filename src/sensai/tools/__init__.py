"""Tool registry and tools (files, sandbox, MCP)."""

from sensai.core.config import Config
from sensai.tools.filesystem.edit_file import EditFileTool
from sensai.tools.filesystem.guard import PathGuard
from sensai.tools.filesystem.list_dir import ListDirTool
from sensai.tools.filesystem.read_file import ReadFileTool
from sensai.tools.filesystem.write_file import WriteFileTool
from sensai.tools.registry import ToolRegistry


def build_registry(config: Config) -> ToolRegistry:
    if not config.files.allowed_roots:
        return ToolRegistry()
    guard = PathGuard(config.files.allowed_roots)
    return ToolRegistry(
        [
            ReadFileTool(guard, config.files.max_read_bytes),
            ListDirTool(guard, config.files.max_list_entries),
            WriteFileTool(guard, config.files.max_write_bytes),
            EditFileTool(guard, config.files.max_write_bytes),
        ]
    )
