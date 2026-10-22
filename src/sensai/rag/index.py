"""Index of the codebase: chunks of lines, their embeddings, and a cosine search over them."""

import hashlib
import math
import os
import sqlite3
from array import array
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# ponytail: fixed skip list, read .gitignore if indexing picks up too much
SKIPPED_DIRS = {"node_modules", "__pycache__", "venv", "build", "dist", "site-packages"}
_BATCH = 32  # texts per /api/embed request


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


class CodeIndex:
    def __init__(
        self, connection: sqlite3.Connection, embedder: Embedder, chunk_size: int, overlap: int
    ) -> None:
        self._db = connection
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._overlap = overlap

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
        sha = hashlib.sha256(text.encode()).hexdigest()
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
            batch = [f"{path.name}\n{chunk}" for _, _, chunk in chunks[i : i + _BATCH]]
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

    async def search(self, query: str, top_k: int) -> list[Hit]:
        """The `top_k` chunks closest to `query` (cosine similarity), best first."""
        query_vector = _normalized((await self._embedder.embed([query]))[0])
        rows = self._db.execute(
            "SELECT path, start_line, end_line, text, embedding FROM code_chunks WHERE model = ?",
            (self._embedder.model,),
        )
        # ponytail: brute-force scan in Python, fine up to ~10k chunks; sqlite-vec or numpy beyond
        hits = []
        for path, start, end, text, blob in rows:
            vector = array("f")
            vector.frombytes(blob)
            score = sum(a * b for a, b in zip(query_vector, vector, strict=True))
            hits.append(Hit(path, start, end, text, score))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]
