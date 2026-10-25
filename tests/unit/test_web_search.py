"""T3: web search through a local SearXNG instance, with no network access in the tests."""

import httpx
import pytest
import respx

from sensai.core.config import Config, WebConfig, load_config
from sensai.core.types import ToolContext
from sensai.tools import web_tools
from sensai.tools.web.searxng import WebSearchTool

pytestmark = pytest.mark.feature("T3")

BASE = "http://localhost:8080"
CONTEXT = ToolContext("run", "session")


def web_config(**overrides) -> WebConfig:
    values = {
        "enabled": True,
        "base_url": BASE,
        "max_results": 5,
        "timeout_s": 5,
        "max_result_chars": 6000,
        **overrides,
    }
    return WebConfig(**values)


def hit(url: str, title: str = "Title", content: str = "Snippet") -> dict:
    return {"url": url, "title": title, "content": content}


async def search(args: dict, config: WebConfig | None = None):
    return await WebSearchTool(config or web_config()).run(args, CONTEXT)


@respx.mock
async def test_results_are_formatted_and_marked_untrusted():
    route = respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(
            200,
            json={"results": [hit("https://docs.python.org/3/library/re.html", "re", "Regex")]},
        )
    )
    result = await search({"query": "python re module"})

    assert route.calls.last.request.url.params["q"] == "python re module"
    assert route.calls.last.request.url.params["format"] == "json"
    assert not result.is_error
    assert result.untrusted
    assert "1. re" in result.content
    assert "https://docs.python.org/3/library/re.html" in result.content
    assert "Regex" in result.content


@respx.mock
async def test_allowed_domains_keep_only_matching_hosts():
    respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    hit("https://docs.python.org/3/a.html", "official"),
                    hit("https://blog.example.com/b.html", "blog"),
                    hit("https://www.docs.python.org/c.html", "www"),
                ]
            },
        )
    )
    result = await search({"query": "python", "allowed_domains": ["docs.python.org"]})

    assert "official" in result.content
    assert "www" in result.content
    assert "blog" not in result.content


@respx.mock
async def test_blocked_domains_remove_matching_hosts_and_subdomains():
    respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    hit("https://spam.example.com/a", "spam"),
                    hit("https://example.com/b", "root"),
                    hit("https://ok.org/c", "kept"),
                ]
            },
        )
    )
    result = await search({"query": "python", "blocked_domains": ["example.com"]})

    assert "kept" in result.content
    assert "spam" not in result.content
    assert "root" not in result.content


async def test_allowed_and_blocked_domains_cannot_be_used_together():
    result = await search(
        {"query": "python", "allowed_domains": ["a.org"], "blocked_domains": ["b.org"]}
    )
    assert result.is_error
    assert "ne peuvent pas être utilisés ensemble" in result.content


async def test_a_too_short_query_is_refused_without_a_request():
    result = await search({"query": " a "})
    assert result.is_error


@respx.mock
async def test_no_result_is_reported_clearly():
    respx.get(f"{BASE}/search").mock(return_value=httpx.Response(200, json={"results": []}))
    result = await search({"query": "nothing here"})
    assert not result.is_error
    assert result.content == "Aucun résultat pour : nothing here"


@respx.mock
async def test_refused_request_points_to_the_json_format():
    respx.get(f"{BASE}/search").mock(return_value=httpx.Response(403))
    result = await search({"query": "python"})
    assert result.is_error
    assert "HTTP 403" in result.content
    assert "settings.yml" in result.content


@respx.mock
async def test_unreachable_instance_says_how_to_start_it():
    respx.get(f"{BASE}/search").mock(side_effect=httpx.ConnectError("refused"))
    result = await search({"query": "python"})
    assert result.is_error
    assert "make searxng" in result.content


@respx.mock
async def test_non_json_answer_is_reported():
    respx.get(f"{BASE}/search").mock(return_value=httpx.Response(200, text="<html>"))
    result = await search({"query": "python"})
    assert result.is_error
    assert "settings.yml" in result.content


@respx.mock
async def test_results_are_capped_by_count_and_size():
    respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(
            200,
            json={"results": [hit(f"https://x.org/{i}", f"t{i}", "s" * 500) for i in range(10)]},
        )
    )
    capped = await search({"query": "python"}, web_config(max_results=2))
    assert "t1" in capped.content and "t2" not in capped.content

    short = await search({"query": "python"}, web_config(max_result_chars=200))
    assert short.content.endswith("[truncated]")
    assert len(short.content) <= 200 + len("\n[truncated]")


def test_web_tools_offers_the_tool_only_when_enabled():
    config = load_config(overrides={"web": {"enabled": False}})
    assert web_tools(config) == []

    config: Config = load_config()
    tools = web_tools(config)
    assert [tool.name for tool in tools] == ["web_search"]
    assert tools[0].risk == "read"
    assert tools[0].parameters["required"] == ["query"]
