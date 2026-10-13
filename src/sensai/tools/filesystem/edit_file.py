import asyncio
from pathlib import Path
from typing import Any

from sensai.core.types import Risk, ToolContext, ToolResult
from sensai.safety.untrusted import contains_untrusted_marker
from sensai.tools.filesystem.atomic import atomic_write
from sensai.tools.filesystem.guard import PathDeniedError, PathGuard


def _invalid_argument(name: str) -> ToolResult:
    return ToolResult(f'Argument "{name}" manquant ou invalide.', is_error=True)


class EditFileTool:
    """A tool that gives the model the ability to write content inside a file
    without rewriting all the file content"""

    name = "edit_file"
    description = (
        "Write content to a file."
        " Call it with path, old_text and new_text"
        " It will replace the old_text with the new_text"
        " Prefere this tool for file editing"
    )
    risk: Risk = "write"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": ("Path to the file you want to change text")},
            "old_text": {
                "type": "string",
                "description": ("Exact text to replace with enough lines in order to be unique"),
            },
            "new_text": {
                "type": "string",
                "description": (
                    "Replacement text, if empty it will delete the old_text from the file"
                ),
            },
        },
        "required": ["path", "old_text", "new_text"],
    }

    def __init__(self, guard: PathGuard, max_write_bytes: int) -> None:
        self._guard = guard
        self._max_bytes = max_write_bytes

    def _check_arguments(self, args: dict[str, Any]) -> ToolResult | None:
        path = args.get("path")
        old_text = args.get("old_text")
        new_text = args.get("new_text")
        if not isinstance(path, str) or not path.strip():
            return _invalid_argument("path")
        if not isinstance(old_text, str) or not old_text:
            return _invalid_argument("old_text")
        if not isinstance(new_text, str):
            return _invalid_argument("new_text")
        if old_text == new_text:
            return ToolResult('"old_text" et "new_text" doivent être différents.', is_error=True)
        if contains_untrusted_marker(new_text):
            return ToolResult(
                "Le contenu contient le marqueur de contenu externe (<untrusted_content>) : "
                "réécris le contenu sans ce marqueur ni la phrase d'avertissement qui le précède.",
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
        return await asyncio.to_thread(self._edit, safe_path, args["old_text"], args["new_text"])

    def _edit(self, path: Path, old_text: str, new_text: str) -> ToolResult:
        text = self._load(path)
        if isinstance(text, ToolResult):
            return text
        new_content = self._replace_once(path, text, old_text, new_text)
        if isinstance(new_content, ToolResult):
            return new_content
        try:
            atomic_write(path, new_content)
        except OSError as error:
            return ToolResult(f"Écriture impossible : {path} ({error})", is_error=True)
        return ToolResult(f"Fichier modifié : {path}")

    def _load(self, path: Path) -> str | ToolResult:
        if not path.is_file():
            message = f"Fichier introuvable : {path}. Utilise write_file pour créer un fichier."
            return ToolResult(message, is_error=True)
        if path.stat().st_size > self._max_bytes:
            return ToolResult(f"Fichier trop volumineux pour être modifié : {path}", is_error=True)
        try:
            return path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return ToolResult(f"Lecture impossible : {path}", is_error=True)

    def _replace_once(
        self, path: Path, text: str, old_text: str, new_text: str
    ) -> str | ToolResult:
        count = text.count(old_text)
        if count == 0:
            return ToolResult(
                f"Texte à remplacer introuvable dans {path} : relis le fichier et recopie "
                "le texte exactement (espaces et indentation compris).",
                is_error=True,
            )
        if count > 1:
            return ToolResult(
                f"Texte ambigu : {count} occurrences dans {path} : ajoute des lignes de "
                "contexte pour qu'il soit unique.",
                is_error=True,
            )
        new_content = text.replace(old_text, new_text, 1)
        if len(new_content.encode("utf-8")) > self._max_bytes:
            message = "Contenu trop volumineux : le fichier n'a pas été modifié."
            return ToolResult(message, is_error=True)
        return new_content
