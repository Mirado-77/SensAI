from typing import Any
from urllib.parse import urlsplit

import httpx

from sensai.core.config import WebConfig
from sensai.core.types import Risk, ToolContext, ToolResult

MIN_QUERY_LENGTH = 2


class WebSearchTool:
    """Search the web through a local SearXNG instance (T3). Results are external content."""

    name = "web_search"
    description = (
        "Search the web and return titles, URLs and snippets. Use it for library documentation "
        "and recent information. Cite the URLs you use in your answer. Set allowed_domains to "
        "limit results to official documentation sites."
    )
    risk: Risk = "read"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query. Include the library name and the current year.",
            },
            "allowed_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Keep only results from these domains (e.g. docs.python.org).",
            },
            "blocked_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Remove results from these domains.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, config: WebConfig) -> None:
        self._config = config

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = args.get("query")
        if not isinstance(query, str) or len(query.strip()) < MIN_QUERY_LENGTH:
            return ToolResult('Argument "query" manquant ou trop court.', is_error=True)
        allowed = _domains(args.get("allowed_domains"))
        blocked = _domains(args.get("blocked_domains"))
        if allowed and blocked:
            return ToolResult(
                "allowed_domains et blocked_domains ne peuvent pas être utilisés ensemble.",
                is_error=True,
            )
        try:
            hits = await self._search(query.strip())
        except httpx.HTTPStatusError as error:
            return ToolResult(
                f"SearXNG a refusé la requête (HTTP {error.response.status_code}) : vérifie que "
                "le format json est activé dans settings.yml.",
                is_error=True,
            )
        except httpx.HTTPError as error:
            return ToolResult(_unreachable(self._config.base_url, error), is_error=True)
        except ValueError:
            return ToolResult(
                "Réponse inattendue de SearXNG : le format JSON est-il activé dans settings.yml ?",
                is_error=True,
            )
        hits = _filter(hits, allowed, blocked)[: self._config.max_results]
        if not hits:
            return ToolResult(f"Aucun résultat pour : {query.strip()}", untrusted=True)
        return ToolResult(self._format(query.strip(), hits), untrusted=True)

    async def _search(self, query: str) -> list[dict[str, str]]:
        async with httpx.AsyncClient(timeout=self._config.timeout_s) as client:
            response = await client.get(
                f"{self._config.base_url.rstrip('/')}/search",
                params={"q": query, "format": "json"},
            )
        response.raise_for_status()
        results = response.json().get("results", [])
        return [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "snippet": str(item.get("content", "")),
            }
            for item in results
            if item.get("url")
        ]

    def _format(self, query: str, hits: list[dict[str, str]]) -> str:
        lines = [f"Résultats pour : {query}"]
        for number, hit in enumerate(hits, start=1):
            lines.append(f"{number}. {hit['title']}\n   {hit['url']}\n   {hit['snippet']}")
        text = "\n".join(lines)
        limit = self._config.max_result_chars
        return text if len(text) <= limit else f"{text[:limit]}\n[truncated]"


def _domains(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip().lower().removeprefix("www.") for item in value if str(item).strip()]


def _filter(
    hits: list[dict[str, str]], allowed: list[str], blocked: list[str]
) -> list[dict[str, str]]:
    kept = []
    for hit in hits:
        host = (urlsplit(hit["url"]).hostname or "").removeprefix("www.")
        if allowed and not any(_matches(host, domain) for domain in allowed):
            continue
        if any(_matches(host, domain) for domain in blocked):
            continue
        kept.append(hit)
    return kept


def _matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


def _unreachable(base_url: str, error: httpx.HTTPError) -> str:
    return (
        f"Recherche web impossible : SearXNG ne répond pas à {base_url} ({error}). "
        "Lance-le avec `make searxng`."
    )
