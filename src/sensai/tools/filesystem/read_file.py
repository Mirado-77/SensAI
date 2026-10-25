import asyncio
from pathlib import Path
from typing import Any

from sensai.core.types import Risk, ToolContext, ToolResult
from sensai.tools.filesystem.guard import PathDeniedError, PathGuard


class ReadFileTool:
    """A tool that gives the model the ability to read files"""

    name = "read_file"
    description = (
        "Read a text file inside the allowed roots and return its content (truncated if too large)."
    )
    risk: Risk = "read"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Path of the file to read. Relative paths start from the first allowed root."
                ),
            }
        },
        "required": ["path"],
    }

    def __init__(
        self,
        guard: PathGuard,
        max_read_bytes: int,
    ) -> None:
        self._guard = guard
        self._max_bytes = max_read_bytes

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = args.get("path")
        if not isinstance(path, str) or not path.strip():
            return ToolResult('Argument "path" manquant ou invalide.', is_error=True)

        try:
            safe_path = self._guard.resolve(path, write=False)
        except PathDeniedError as error:
            return ToolResult(str(error), is_error=True)
        return await asyncio.to_thread(self._read, safe_path)

    def _read(self, path: Path) -> ToolResult:
        if not path.is_file():
            return ToolResult(f"Fichier introuvable : {path}", is_error=True)
        try:
            with path.open("rb") as file:
                data = file.read(self._max_bytes + 1)
        except OSError as error:
            return ToolResult(f"Lecture impossible : {path} ({error})", is_error=True)
        if b"\x00" in data:
            return ToolResult(f"Fichier binaire non lisible : {path}", is_error=True)
        return ToolResult(self._decode(data), untrusted=True)

    def _decode(self, data: bytes) -> str:
        text = data[: self._max_bytes].decode("utf-8", errors="replace")
        if len(data) > self._max_bytes:
            text += f"\n[truncated: file is larger than {self._max_bytes} bytes]"
        return text
