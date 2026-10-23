import json
import re

import httpx
import pytest
import respx
from rich.console import Console

from sensai.cli.index import run_index
from sensai.cli.terminal import Terminal
from sensai.core.config import AllowedRoot, load_config
from sensai.core.errors import ConfigError, ModelConnectionError, ModelNotFoundError
from sensai.core.types import ToolContext
from sensai.db.connection import connect
from sensai.model.ollama import OllamaEmbedder
from sensai.rag.index import CodeIndex, IndexReport, chunk_lines
from sensai.tools import search_tools
from sensai.tools.filesystem.guard import PathGuard
from sensai.tools.search_code import SearchCodeTool

pytestmark = pytest.mark.feature("R1")

CTX = ToolContext(run_id="r1", session_id="s1")


class WordEmbedder:
    """Bag of words hashed into 64 dimensions: texts sharing words are close."""

    model = "fake-embed"

    def __init__(self):
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        vectors = []
        for text in texts:
            vector = [0.0] * 64
            for word in re.findall(r"[a-z]+", text.lower()):
                vector[sum(map(ord, word)) % 64] += 1.0
            vectors.append(vector)
        return vectors


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "storage.py").write_text("def save_tasks(path):\n    write json file\n")
    (root / "pkg" / "stats.py").write_text("def average_duration(tasks):\n    compute mean\n")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("save tasks json file")
    (root / "image.bin").write_bytes(b"\x00\x01save")
    return root


@pytest.fixture
def index():
    return CodeIndex(connect(":memory:"), WordEmbedder(), chunk_size=40, overlap=10)


async def indexed(index, root):
    report = IndexReport()
    await index.index_root(root, 65536, report)
    return report


def test_chunk_lines_overlap_and_cover_every_line():
    text = "\n".join(f"line {n}" for n in range(1, 11))
    chunks = chunk_lines(text, size=4, overlap=1)
    assert [(start, end) for start, end, _ in chunks] == [(1, 4), (4, 7), (7, 10)]
    assert chunks[0][2] == "line 1\nline 2\nline 3\nline 4"
    assert chunk_lines("", 4, 1) == []
    assert chunk_lines("\n\n  \n", 4, 1) == []


async def test_search_finds_the_relevant_file_and_skips_hidden_and_binary(index, repo):
    report = await indexed(index, repo)
    assert (report.indexed, report.chunks) == (2, 2)
    hits = await index.search("save to a json file", top_k=5)
    assert [hit.path for hit in hits] == [
        str(repo / "pkg" / "storage.py"),
        str(repo / "pkg" / "stats.py"),
    ]
    assert (hits[0].start_line, hits[0].end_line) == (1, 2)
    assert hits[0].score > hits[1].score


async def test_reindex_skips_unchanged_files_and_drops_deleted_ones(index, repo):
    await indexed(index, repo)
    calls = index._embedder.calls
    (repo / "pkg" / "stats.py").unlink()
    (repo / "pkg" / "storage.py").write_text("def load_tasks(path):\n    read json\n")
    report = await indexed(index, repo)
    assert (report.indexed, report.unchanged, report.removed) == (1, 0, 1)
    assert index._embedder.calls == calls + 1
    report = await indexed(index, repo)
    assert (report.indexed, report.unchanged) == (0, 1)
    assert index.count() == 1


async def test_tool_returns_untrusted_excerpts_with_path_and_lines(index, repo):
    await indexed(index, repo)
    guard = PathGuard([AllowedRoot(path=repo, mode="read")])
    result = await SearchCodeTool(index, guard, top_k=1).run({"query": "save json"}, CTX)
    assert result.untrusted and not result.is_error
    assert result.content.startswith(f"{repo / 'pkg' / 'storage.py'}:1-2 (score ")
    assert "def save_tasks(path):" in result.content


async def test_tool_hides_files_outside_the_allowed_roots(index, repo, tmp_path):
    await indexed(index, repo)
    other = tmp_path / "other"
    other.mkdir()
    guard = PathGuard([AllowedRoot(path=other, mode="read")])
    result = await SearchCodeTool(index, guard, top_k=5).run({"query": "save json"}, CTX)
    assert "Aucun résultat" in result.content
    assert "save_tasks" not in result.content


async def test_tool_rejects_an_empty_query(index):
    guard = PathGuard([])
    result = await SearchCodeTool(index, guard, top_k=5).run({"query": " "}, CTX)
    assert result.is_error


async def test_index_command_then_chat_gets_the_tool(repo, tmp_path):
    db = tmp_path / "sensai.db"
    overrides = {
        "storage": {"db_path": str(db)},
        "files": {"allowed_roots": [{"path": str(repo), "mode": "read"}]},
        "rag": {"embed_model": WordEmbedder.model},
    }
    config = load_config(overrides=overrides)
    connection = connect(db)
    assert search_tools(config, connection) == []
    terminal = Terminal(Console(record=True), Console(record=True))
    assert await run_index(config, terminal, [], embedder=WordEmbedder()) == 0
    assert "2 files indexed (2 chunks)" in terminal.console.export_text()
    assert [tool.name for tool in search_tools(config, connection)] == ["search_code"]
    assert await run_index(config, terminal, [str(tmp_path)], embedder=WordEmbedder()) == 2


def test_overlap_must_be_smaller_than_chunk_lines():
    with pytest.raises(ConfigError, match="overlap"):
        load_config(overrides={"rag": {"chunk_lines": 10, "overlap": 10}})


async def test_embedder_posts_to_api_embed(ollama_host):
    embedder = OllamaEmbedder(ollama_host, "nomic-embed-text", timeout_s=5)
    with respx.mock(base_url=ollama_host) as router:
        route = router.post("/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
        )
        assert await embedder.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert json.loads(route.calls.last.request.content) == {
        "model": "nomic-embed-text",
        "input": ["a", "b"],
    }


async def test_embedder_maps_ollama_errors(ollama_host):
    embedder = OllamaEmbedder(ollama_host, "missing", timeout_s=5)
    with respx.mock(base_url=ollama_host) as router:
        router.post("/api/embed").mock(
            return_value=httpx.Response(404, json={"error": 'model "missing" not found'})
        )
        with pytest.raises(ModelNotFoundError):
            await embedder.embed(["a"])
        router.post("/api/embed").mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ModelConnectionError):
            await embedder.embed(["a"])
