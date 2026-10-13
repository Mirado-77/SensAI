import asyncio
from pathlib import Path
from typing import Any

from sensai.core.types import Risk, ToolContext, ToolResult
from sensai.safety.untrusted import contains_untrusted_marker
from sensai.tools.filesystem.atomic import atomic_write
from sensai.tools.filesystem.guard import PathDeniedError, PathGuard


class WriteFileTool:
    """A tool that gives the model the ability to write content inside a file"""

    name = "write_file"
    description = (
        "Write content to a file."
        " Call it with path, and content."
        " Be careful this replaces all content of the file !"
    )
    risk: Risk = "write"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": ("Path of the file you want to write")},
            "content": {"type": "string", "description": ("New content of the file")},
        },
        "required": ["path", "content"],
    }

    def __init__(
        self,
        guard: PathGuard,
        max_write_bytes: int,
    ) -> None:
        self._guard = guard
        self._max_bytes = max_write_bytes

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = args.get("path")
        content = args.get("content")
        if not isinstance(path, str) or not path.strip():
            return ToolResult('Argument "path" manquant ou invalide.', is_error=True)
        if not isinstance(content, str):
            return ToolResult('Argument "content" manquant ou invalide.', is_error=True)
        if contains_untrusted_marker(content):
            return ToolResult(
                "Le contenu contient le marqueur de contenu externe (<untrusted_content>) : "
                "réécris le contenu sans ce marqueur ni la phrase d'avertissement qui le précède.",
                is_error=True,
            )
        size = len(content.encode("utf-8"))
        if size > self._max_bytes:
            return ToolResult(
                f"Contenu trop volumineux : {size} octets (maximum {self._max_bytes}).",
                is_error=True,
            )

        try:
            safe_path = self._guard.resolve(path, write=True)
        except PathDeniedError as error:
            return ToolResult(str(error), is_error=True)
        return await asyncio.to_thread(self._write, safe_path, content)

    def _write(self, path: Path, content: str) -> ToolResult:
        if path.is_dir():
            return ToolResult(f'"{path}" est un dossier.', is_error=True)
        existed = path.exists()
        try:
            atomic_write(path, content)
        except OSError as error:
            return ToolResult(f"Écriture impossible : {path} ({error})", is_error=True)
        return ToolResult(f"Fichier {'modifié' if existed else 'créé'} : {path}")
