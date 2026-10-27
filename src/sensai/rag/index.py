"""Index of the codebase: chunks of lines, their embeddings, and a cosine search over them."""

import hashlib
import math
import os
import re
import sqlite3
from array import array
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# ponytail: fixed skip list, read .gitignore if indexing picks up too much
SKIPPED_DIRS = {"node_modules", "__pycache__", "venv", "build", "dist", "site-packages"}
_BATCH = 32  # texts per /api/embed request
_WORD = re.compile(r"[a-z0-9_]+")
_RRF_K = 60  # reciprocal rank fusion constant, the usual default
_BM25_K1 = 1.5
_BM25_B = 0.75


class Embedder(Protocol):
    model: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class Hit:
    path: str
    start_line: int
    end_line: int
    text: str
    score: float


@dataclass
class IndexReport:
    indexed: int = 0
    chunks: int = 0
    unchanged: int = 0
    removed: int = 0


@dataclass(frozen=True)
class SearchOptions:
    top_k: int  # excerpts returned at most
    candidates: int  # pool kept from the fusion, before the relevance cut
    min_score: float  # cosine similarity under which a chunk is not relevant
    path: str | None = None  # keep only the chunks whose path contains this text


def chunk_lines(text: str, size: int, overlap: int) -> list[tuple[int, int, str]]:
    """Windows of `size` lines overlapping by `overlap`, as (first line, last line, text)."""
    lines = text.splitlines()
    chunks = []
    step = size - overlap
    for start in range(0, len(lines), step):
        window = lines[start : start + size]
        if any(line.strip() for line in window):
            chunks.append((start + 1, start + len(window), "\n".join(window)))
        if start + size >= len(lines):
            break
    return chunks


def iter_files(root: Path) -> Iterator[Path]:
    """Files under `root`, without hidden or build/dependency directories."""
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if not d.startswith(".") and d not in SKIPPED_DIRS)
        for name in sorted(files):
            if not name.startswith("."):
                yield Path(directory) / name


def _read_text(path: Path, max_bytes: int) -> str | None:
    """Text of a file, or None if it is binary, too large or unreadable."""
    try:
        if path.stat().st_size > max_bytes:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    return data.decode("utf-8", errors="replace")


def _normalized(vector: list[float]) -> array:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return array("f", (x / norm for x in vector))


def tokenize(text: str) -> list[str]:
    """Lowercase words and identifiers: the keyword side of the search."""
    return _WORD.findall(text.lower())


def bm25_scores(query_terms: list[str], documents: list[list[str]]) -> list[float]:
    """Okapi BM25 score of each document for the query terms (0 when nothing matches)."""
    count = len(documents)
    terms = set(query_terms)
    if not count or not terms:
        return [0.0] * count
    average = sum(len(doc) for doc in documents) / count or 1.0
    document_freq: Counter[str] = Counter()
    for doc in documents:
        document_freq.update(terms & set(doc))
    scores = []
    for doc in documents:
        frequencies = Counter(doc)
        score = 0.0
        for term in terms & frequencies.keys():
            freq = frequencies[term]
            idf = math.log(1 + (count - document_freq[term] + 0.5) / (document_freq[term] + 0.5))
            length_norm = 1 - _BM25_B + _BM25_B * len(doc) / average
            score += idf * freq * (_BM25_K1 + 1) / (freq + _BM25_K1 * length_norm)
        scores.append(score)
    return scores


def _ranks(scores: list[float]) -> list[int]:
    """Position of each score in the descending order, 0 for the best."""
    ranks = [0] * len(scores)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    for position, i in enumerate(order):
        ranks[i] = position
    return ranks


class CodeIndex:
    def __init__(
        self,
        connection: sqlite3.Connection,
        embedder: Embedder,
        chunk_size: int,
        overlap: int,
        *,
        document_prefix: str = "",
        query_prefix: str = "",
    ) -> None:
        self._db = connection
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._overlap = overlap
        # Some embedding models (nomic-embed-text) expect a task prefix on each side.
        self._document_prefix = document_prefix
        self._query_prefix = query_prefix

    def count(self) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) FROM code_chunks WHERE model = ?", (self._embedder.model,)
        ).fetchone()
        return row[0]

    async def index_root(self, root: Path, max_bytes: int, report: IndexReport) -> None:
        """Index the files under `root`: re-embed changed files, drop the ones that are gone."""
        seen = set()
        for path in iter_files(root):
            text = _read_text(path, max_bytes)
            if text is None:
                continue
            seen.add(str(path))
            await self._index_file(path, text, report)
        for (stored,) in self._db.execute("SELECT DISTINCT path FROM code_chunks").fetchall():
            if stored not in seen and Path(stored).is_relative_to(root):
                self._db.execute("DELETE FROM code_chunks WHERE path = ?", (stored,))
                report.removed += 1
        self._db.commit()

    async def _index_file(self, path: Path, text: str, report: IndexReport) -> None:
        # The prefix is part of the hash: changing it re-embeds every file.
        sha = hashlib.sha256((self._document_prefix + text).encode()).hexdigest()
        model = self._embedder.model
        stored = self._db.execute(
            "SELECT 1 FROM code_chunks WHERE path = ? AND sha = ? AND model = ? LIMIT 1",
            (str(path), sha, model),
        ).fetchone()
        if stored is not None:
            report.unchanged += 1
            return
        chunks = chunk_lines(text, self._chunk_size, self._overlap)
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), _BATCH):
            # The path is embedded with the lines: a query naming a file or module finds it.
            batch = [
                f"{self._document_prefix}{path.name}\n{chunk}"
                for _, _, chunk in chunks[i : i + _BATCH]
            ]
            vectors += await self._embedder.embed(batch)
        self._db.execute("DELETE FROM code_chunks WHERE path = ?", (str(path),))
        self._db.executemany(
            "INSERT INTO code_chunks (path, sha, model, start_line, end_line, text, embedding)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (str(path), sha, model, start, end, chunk, _normalized(vector).tobytes())
                for (start, end, chunk), vector in zip(chunks, vectors, strict=True)
            ],
        )
        self._db.commit()
        report.indexed += 1
        report.chunks += len(chunks)

    def _rows(self, path: str | None) -> list[tuple[str, int, int, str, bytes]]:
        sql = "SELECT path, start_line, end_line, text, embedding FROM code_chunks WHERE model = ?"
        params: list[str] = [self._embedder.model]
        if path:
            sql += " AND instr(lower(path), lower(?)) > 0"
            params.append(path)
        return self._db.execute(sql, params).fetchall()

    async def search(self, query: str, options: SearchOptions) -> list[Hit]:
        """Chunks relevant to `query`: vectors and keywords fused, cut by relevance, then reranked.

        1. Fusion: each chunk gets a rank by cosine similarity and a rank by BM25, and the two
           ranks are combined (reciprocal rank fusion). The best `candidates` are kept.
        2. Relevance cut: a candidate whose cosine similarity is under `min_score` is dropped.
        3. Rerank: the survivors are sorted again on cosine and BM25, both scaled to 0..1.
        """
        query_vector = _normalized((await self._embedder.embed([self._query_prefix + query]))[0])
        rows = self._rows(options.path)
        # ponytail: brute-force scan in Python, fine up to ~10k chunks; sqlite-vec or numpy beyond
        vector_scores = []
        for *_, blob in rows:
            vector = array("f")
            vector.frombytes(blob)
            vector_scores.append(sum(a * b for a, b in zip(query_vector, vector, strict=True)))
        keyword_scores = bm25_scores(tokenize(query), [tokenize(row[3]) for row in rows])

        vector_ranks = _ranks(vector_scores)
        keyword_ranks = _ranks(keyword_scores)
        fused = [
            1 / (_RRF_K + vector_ranks[i])
            + (1 / (_RRF_K + keyword_ranks[i]) if keyword_scores[i] > 0 else 0.0)
            for i in range(len(rows))
        ]
        pool = sorted(range(len(rows)), key=lambda i: fused[i], reverse=True)[: options.candidates]
        relevant = [i for i in pool if vector_scores[i] >= options.min_score]

        best_keyword = max((keyword_scores[i] for i in relevant), default=0.0) or 1.0
        relevant.sort(
            key=lambda i: 0.5 * vector_scores[i] + 0.5 * keyword_scores[i] / best_keyword,
            reverse=True,
        )
        hits = []
        for i in relevant[: options.top_k]:
            path, start, end, text, _ = rows[i]
            hits.append(Hit(path, start, end, text, vector_scores[i]))
        return hits
