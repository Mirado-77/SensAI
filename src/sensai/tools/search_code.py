from typing import Any

from sensai.core.errors import ModelError
from sensai.core.types import Risk, ToolContext, ToolResult
from sensai.rag.index import CodeIndex, SearchOptions
from sensai.tools.filesystem.guard import PathDeniedError, PathGuard


class SearchCodeTool:
    """Semantic and keyword search over the indexed codebase (R1, R2)."""

    name = "search_code"
    description = (
        "Search the indexed codebase and return the most relevant excerpts with their path and "
        "line numbers. Use it to find where something is implemented, then read_file."
    )
    risk: Risk = "read"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look for, in natural language or with identifiers.",
            },
            "path": {
                "type": "string",
                "description": "Optional: only search files whose path contains this text.",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        index: CodeIndex,
        guard: PathGuard,
        top_k: int,
        *,
        candidates: int = 20,
        min_score: float = 0.55,
        max_chars: int = 4000,
    ) -> None:
        self._index = index
        self._guard = guard
        self._top_k = top_k
        self._candidates = candidates
        self._min_score = min_score
        self._max_chars = max_chars

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult('Argument "query" manquant ou invalide.', is_error=True)
        path = args.get("path")
        if path is not None and not isinstance(path, str):
            return ToolResult('Argument "path" invalide.', is_error=True)
        options = SearchOptions(self._top_k, self._candidates, self._min_score, path or None)
        try:
            hits = await self._index.search(query, options)
        except ModelError as error:
            return ToolResult(f"Recherche impossible : {error}", is_error=True)
        excerpts = []
        used = 0
        for hit in hits:
            try:
                # The index may hold files of a root that is no longer allowed.
                self._guard.resolve(hit.path, write=False)
            except PathDeniedError:
                continue
            header = f"{hit.path}:{hit.start_line}-{hit.end_line} (score {hit.score:.2f})"
            block = f"{header}\n{hit.text}"
            room = self._max_chars - used
            if len(block) > room:
                # The budget keeps the context small: cut the last excerpt, then stop.
                excerpts.append(block[:room] + "\n[extrait tronqué]")
                break
            excerpts.append(block)
            used += len(block)
        if not excerpts:
            return ToolResult(
                "Aucun résultat assez pertinent. Reformuler la requête, ou lancer `sensai index` "
                "si l'index est vide."
            )
        return ToolResult("\n\n".join(excerpts), untrusted=True)
