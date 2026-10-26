"""Durable project memory (M3): facts and the relations between them, saved in sqlite3.

The model drives it through the memory tools (create, search, update, delete). At every turn the
facts closest to the message are recalled into the context, so they survive across sessions.
"""

import sqlite3
import time
from dataclasses import dataclass
from typing import Literal

from sensai.core.fewshots import keywords
from sensai.core.types import new_id
from sensai.safety.untrusted import wrap_untrusted

Kind = Literal["convention", "decision", "fact", "entity"]
KINDS: tuple[Kind, ...] = ("convention", "decision", "fact", "entity")


@dataclass(frozen=True)
class Fact:
    id: str
    kind: Kind
    subject: str
    content: str
    updated_at: float

    def line(self) -> str:
        return f"[{self.kind}] {self.id[:8]} {self.subject} : {self.content}"


@dataclass(frozen=True)
class Relation:
    id: str
    subject: str
    relation: str
    object: str

    def line(self) -> str:
        return f"[relation] {self.id[:8]} {self.subject} -> {self.relation} -> {self.object}"


Record = Fact | Relation


class FactStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection

    def save(self, kind: Kind, subject: str, content: str) -> str:
        fact_id, now = new_id(), time.time()
        with self._db:
            self._db.execute(
                "INSERT INTO facts (id, kind, subject, content, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (fact_id, kind, subject, content, now, now),
            )
        return fact_id

    def relate(self, subject: str, relation: str, object_: str) -> str:
        """Link two subjects; the same link saved twice keeps its first id."""
        with self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO relations (id, subject, relation, object, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (new_id(), subject, relation, object_, time.time()),
            )
        row = self._db.execute(
            "SELECT id FROM relations WHERE subject = ? AND relation = ? AND object = ?",
            (subject, relation, object_),
        ).fetchone()
        return row["id"]

    def update(self, fact_id: str, content: str) -> None:
        with self._db:
            self._db.execute(
                "UPDATE facts SET content = ?, updated_at = ? WHERE id = ?",
                (content, time.time(), fact_id),
            )

    def delete(self, record_id: str) -> None:
        with self._db:
            self._db.execute("DELETE FROM facts WHERE id = ?", (record_id,))
            self._db.execute("DELETE FROM relations WHERE id = ?", (record_id,))

    def find(self, prefix: str) -> list[Record]:
        """Records whose id starts with `prefix` (the 8 characters shown are enough)."""
        return [record for record in self.records() if record.id.startswith(prefix)]

    def records(self) -> list[Record]:
        """Every fact (most recently updated first), then every relation."""
        facts = self._db.execute(
            "SELECT id, kind, subject, content, updated_at FROM facts ORDER BY updated_at DESC"
        ).fetchall()
        relations = self._db.execute(
            "SELECT id, subject, relation, object FROM relations ORDER BY created_at DESC"
        ).fetchall()
        return [Fact(**dict(row)) for row in facts] + [Relation(**dict(row)) for row in relations]

    def search(self, text: str, limit: int, kind: str | None = None) -> list[Record]:
        """The records sharing the most words with `text`, best first; all of them for an empty
        `text`. shortcut: word overlap over every record, fine for one project's memory; switch to
        SQL full-text search (FTS5) if it grows to thousands of records."""
        records = [r for r in self.records() if kind is None or getattr(r, "kind", None) == kind]
        words = keywords(text)
        if not words:
            return records[:limit]
        scored = [(len(words & keywords(_text_of(record))), record) for record in records]
        ranked = sorted((pair for pair in scored if pair[0]), key=lambda pair: -pair[0])
        return [record for _, record in ranked[:limit]]


def _text_of(record: Record) -> str:
    if isinstance(record, Fact):
        return f"{record.subject} {record.content}"
    return f"{record.subject} {record.relation} {record.object}"


RECALL_HEADER = (
    "Project memory saved in earlier sessions. Use it when it is relevant and keep it up to date"
    " with the memory tools."
)


def recall_note(store: FactStore, text: str, limit: int) -> str | None:
    """The conventions (project-wide) and the other records closest to `text`, or None."""
    conventions = [r for r in store.records() if getattr(r, "kind", None) == "convention"]
    closest = (
        [r for r in store.search(text, limit) if r not in conventions] if keywords(text) else []
    )
    records = (conventions + closest)[:limit]
    if not records:
        return None
    return f"{RECALL_HEADER}\n" + wrap_untrusted("\n".join(record.line() for record in records))
