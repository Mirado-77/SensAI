import pytest

from sensai.core.config import AllowedRoot
from sensai.core.types import ToolContext
from sensai.db.connection import connect
from sensai.rag.index import CodeIndex, IndexReport, SearchOptions, bm25_scores, tokenize
from sensai.tools.filesystem.guard import PathGuard
from sensai.tools.search_code import SearchCodeTool

pytestmark = pytest.mark.feature("R2")

CTX = ToolContext(run_id="r2", session_id="s2")


def vector_for(text: str) -> list[float]:
    """Hand-made vectors: 'alpha' and 'retry' point the same way; 'gamma' is far from them."""
    if text in ("alpha", "retry") or "alpha" in text:
        return [1.0, 0.0]
    if "gamma" in text:
        return [0.3, 0.95]
    if "delta" in text:
        return [0.0, 1.0]
    return [0.0, 0.0]


class TableEmbedder:
    model = "table"

    async def embed(self, texts):
        return [vector_for(text) for text in texts]


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("alpha beta\n")
    (root / "b.py").write_text("gamma retry\n")
    (root / "c.py").write_text("delta\n")
    return root


@pytest.fixture
def index():
    return CodeIndex(connect(":memory:"), TableEmbedder(), chunk_size=40, overlap=10)


async def indexed(index, root):
    await index.index_root(root, max_bytes=10_000, report=IndexReport())


def options(**changes):
    values = {"top_k": 5, "candidates": 20, "min_score": 0.5}
    values.update(changes)
    return SearchOptions(**values)


def test_tokenize_keeps_identifiers_and_lowercases():
    assert tokenize("Save_Tasks(path)") == ["save_tasks", "path"]


def test_bm25_scores_only_the_documents_with_the_terms():
    scores = bm25_scores(["retry"], [["gamma", "retry"], ["alpha"]])
    assert scores[0] > 0
    assert scores[1] == 0.0
    assert bm25_scores(["retry"], []) == []
    assert bm25_scores([], [["alpha"]]) == [0.0]


async def test_relevance_cut_drops_chunks_under_min_score(index, root):
    await indexed(index, root)
    hits = await index.search("alpha", options())
    assert [hit.path for hit in hits] == [str(root / "a.py")]


async def test_zero_min_score_keeps_every_chunk(index, root):
    await indexed(index, root)
    hits = await index.search("alpha", options(min_score=0.0))
    assert len(hits) == 3


async def test_keyword_match_enters_the_pool_when_vectors_rank_it_low(index, root):
    # With a pool of one, the vectors alone would keep a.py; the BM25 rank of b.py adds it.
    await indexed(index, root)
    hits = await index.search("retry", options(candidates=1, min_score=0.0))
    assert [hit.path for hit in hits] == [str(root / "b.py")]


async def test_path_filter_keeps_only_matching_paths(index, root):
    await indexed(index, root)
    hits = await index.search("alpha", options(min_score=0.0, path="B.PY"))
    assert [hit.path for hit in hits] == [str(root / "b.py")]
    assert await index.search("alpha", options(min_score=0.0, path="missing")) == []


async def test_tool_budget_cuts_the_last_excerpt(index, root):
    (root / "long.py").write_text("alpha " * 300 + "\n")
    await indexed(index, root)
    guard = PathGuard([AllowedRoot(path=root, mode="read")])
    tool = SearchCodeTool(index, guard, top_k=5, max_chars=60)
    result = await tool.run({"query": "alpha"}, CTX)
    assert "[extrait tronqué]" in result.content
    assert len(result.content) <= 60 + len("\n[extrait tronqué]")


async def test_tool_passes_the_path_argument(index, root):
    await indexed(index, root)
    guard = PathGuard([AllowedRoot(path=root, mode="read")])
    tool = SearchCodeTool(index, guard, top_k=5, min_score=0.0)
    result = await tool.run({"query": "alpha", "path": "c.py"}, CTX)
    assert str(root / "c.py") in result.content
    assert str(root / "a.py") not in result.content


async def test_tool_rejects_a_path_that_is_not_text(index, root):
    guard = PathGuard([AllowedRoot(path=root, mode="read")])
    result = await SearchCodeTool(index, guard, top_k=5).run({"query": "alpha", "path": 3}, CTX)
    assert result.is_error


async def test_tool_says_when_nothing_is_relevant(index, root):
    await indexed(index, root)
    guard = PathGuard([AllowedRoot(path=root, mode="read")])
    # "beta" has no vector of its own: every chunk scores 0 and none passes the cut.
    result = await SearchCodeTool(index, guard, top_k=5).run({"query": "beta"}, CTX)
    assert "Aucun résultat assez pertinent" in result.content


class RecordingEmbedder:
    model = "recording"

    def __init__(self):
        self.texts = []

    async def embed(self, texts):
        self.texts += texts
        return [[1.0, 0.0] for _ in texts]


async def test_prefixes_are_added_to_chunks_and_to_the_query(root):
    embedder = RecordingEmbedder()
    index = CodeIndex(
        connect(":memory:"), embedder, 40, 10, document_prefix="doc: ", query_prefix="q: "
    )
    await index.index_root(root, max_bytes=10_000, report=IndexReport())
    assert len(embedder.texts) == 3
    assert all(text.startswith("doc: ") for text in embedder.texts)
    await index.search("alpha", options(min_score=0.0))
    assert embedder.texts[-1] == "q: alpha"


async def test_changing_the_document_prefix_reembeds_unchanged_files(root):
    connection = connect(":memory:")
    first = IndexReport()
    await CodeIndex(connection, TableEmbedder(), 40, 10, document_prefix="v1 ").index_root(
        root, max_bytes=10_000, report=first
    )
    second = IndexReport()
    await CodeIndex(connection, TableEmbedder(), 40, 10, document_prefix="v2 ").index_root(
        root, max_bytes=10_000, report=second
    )
    assert (first.indexed, second.indexed, second.unchanged) == (3, 3, 0)
