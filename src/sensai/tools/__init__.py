"""Tool registry and tools (files, sandbox, code search, MCP)."""

import sqlite3

from sensai.core.config import Config
from sensai.core.contracts import Tool
from sensai.model.ollama import OllamaEmbedder
from sensai.rag.index import CodeIndex
from sensai.tools.filesystem.edit_file import EditFileTool
from sensai.tools.filesystem.guard import PathGuard
from sensai.tools.filesystem.list_dir import ListDirTool
from sensai.tools.filesystem.read_file import ReadFileTool
from sensai.tools.filesystem.write_file import WriteFileTool
from sensai.tools.sandbox.docker import docker_unavailable_reason
from sensai.tools.sandbox.run_command import RunCommandTool
from sensai.tools.search_code import SearchCodeTool


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


def code_index(config: Config, connection: sqlite3.Connection) -> CodeIndex:
    rag, model = config.rag, config.model
    embedder = OllamaEmbedder(model.host, rag.embed_model, model.timeout_s)
    return CodeIndex(connection, embedder, rag.chunk_lines, rag.overlap)


def search_tools(config: Config, connection: sqlite3.Connection) -> list[Tool]:
    """The search_code tool (R1), only once `sensai index` has filled the index."""
    if not config.rag.enabled or not config.files.allowed_roots:
        return []
    index = code_index(config, connection)
    if not index.count():
        return []
    return [SearchCodeTool(index, PathGuard(config.files.allowed_roots), config.rag.top_k)]


async def sandbox_tools(config: Config) -> tuple[list[Tool], str | None]:
    """Return the sandbox tool (T2), or no tool and the reason why it is disabled."""
    sandbox = config.sandbox
    if not sandbox.enabled or not config.files.allowed_roots:
        return [], None
    reason = await docker_unavailable_reason(sandbox.image)
    if reason is not None:
        return [], reason
    return [RunCommandTool(PathGuard(config.files.allowed_roots), sandbox)], None
