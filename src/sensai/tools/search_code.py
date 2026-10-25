from typing import Any

from sensai.core.errors import ModelError
from sensai.core.types import Risk, ToolContext, ToolResult
from sensai.rag.index import CodeIndex
from sensai.tools.filesystem.guard import PathDeniedError, PathGuard


class SearchCodeTool:
    """Semantic search over the indexed codebase (R1)."""

    name = "search_code"
    description = (
        "Search the indexed codebase by meaning and return the most relevant excerpts with their "
        "path and line numbers. Use it to find where something is implemented, then read_file."
    )
    risk: Risk = "read"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look for, in natural language or with identifiers.",
            }
        },
        "required": ["query"],
    }

    def __init__(self, index: CodeIndex, guard: PathGuard, top_k: int) -> None:
        self._index = index
        self._guard = guard
        self._top_k = top_k

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult('Argument "query" manquant ou invalide.', is_error=True)
        try:
            hits = await self._index.search(query, self._top_k)
        except ModelError as error:
            return ToolResult(f"Recherche impossible : {error}", is_error=True)
        excerpts = []
        for hit in hits:
            try:
                # The index may hold files of a root that is no longer allowed.
                self._guard.resolve(hit.path, write=False)
            except PathDeniedError:
                continue
            header = f"{hit.path}:{hit.start_line}-{hit.end_line} (score {hit.score:.2f})"
            excerpts.append(f"{header}\n{hit.text}")
        if not excerpts:
            return ToolResult("Aucun résultat. L'index est peut-être vide : lancer `sensai index`.")
        return ToolResult("\n\n".join(excerpts), untrusted=True)
