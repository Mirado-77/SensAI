import asyncio
from pathlib import Path
from typing import Any

from sensai.core.types import Risk, ToolContext, ToolResult
from sensai.tools.filesystem.guard import PathDeniedError, PathGuard


class ListDirTool:
    """A tool that gives the model the ability to list directories."""

    name = "list_dir"
    description = (
        "List all directories inside a root."
        " Call it without arguments to list the project root, then explore subfolders."
    )
    risk: Risk = "read"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Path of the folder to list. Relative paths start from the first allowed root."
                    "Omit it to list the first allowed root."
                ),
            }
        },
    }

    def __init__(self, guard: PathGuard, max_list_entries: int) -> None:
        self._guard = guard
        self._max_list_entries = max_list_entries

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = args.get("path")
        if path is None:
            path = "."
        if not isinstance(path, str) or not path.strip():
            return ToolResult('Arugment "path" manquant ou invalide.', is_error=True)

        try:
            safe_path = self._guard.resolve(path, write=False)
        except PathDeniedError as error:
            return ToolResult(str(error), is_error=True)
        return await asyncio.to_thread(self._list, safe_path)

    def _list(self, path: Path) -> ToolResult:
        if not path.is_dir():
            return ToolResult(f"Dossier introuvable : {path}", is_error=True)
        try:
            entries = sorted(path.iterdir(), key=lambda entry: entry.name)
        except OSError as error:
            return ToolResult(f"Lecture impossible : {path} ({error})", is_error=True)
        return ToolResult(self._format(entries), untrusted=True)

    def _format(self, entries: list[Path]) -> str:
        if not entries:
            return "(empty directory)"
        names = [entry.name + "/" if entry.is_dir() else entry.name for entry in entries]
        lines = names[: self._max_list_entries]
        hidden = len(names) - len(lines)
        if hidden:
            lines.append(f"[truncated: {hidden} more entries]")
        return "\n".join(lines)
