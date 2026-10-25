import asyncio
from pathlib import Path
from typing import Any

from sensai.core.types import Risk, ToolContext, ToolResult
from sensai.safety.diff import unified_diff
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
        max_diff_lines: int,
    ) -> None:
        self._guard = guard
        self._max_bytes = max_write_bytes
        self._max_diff_lines = max_diff_lines

    def _check_arguments(self, args: dict[str, Any]) -> ToolResult | None:
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
        return None

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        error = self._check_arguments(args)
        if error is not None:
            return error
        try:
            safe_path = self._guard.resolve(args["path"], write=True)
        except PathDeniedError as exc:
            return ToolResult(str(exc), is_error=True)
        return await asyncio.to_thread(self._write, safe_path, args["content"])

    def _write(self, path: Path, content: str) -> ToolResult:
        if path.is_dir():
            return ToolResult(f'"{path}" est un dossier.', is_error=True)
        existed = path.exists()
        try:
            atomic_write(path, content)
        except OSError as error:
            return ToolResult(f"Écriture impossible : {path} ({error})", is_error=True)
        return ToolResult(f"Fichier {'modifié' if existed else 'créé'} : {path}")

    def _read_current(self, path: Path) -> str | None:
        if not path.exists():
            return None
        if path.stat().st_size > self._max_bytes:
            raise OSError("Fichier trop volumineux pour l'aperçu")
        return path.read_text(encoding="utf-8")

    async def preview(self, args: dict[str, Any]) -> str | None:
        if self._check_arguments(args) is not None:
            return None
        try:
            safe_path = self._guard.resolve(args["path"], write=True)
        except PathDeniedError:
            return None
        try:
            old = await asyncio.to_thread(self._read_current, safe_path)
        except (OSError, UnicodeDecodeError):
            return None
        return unified_diff(args["path"], old, args["content"], self._max_diff_lines)
