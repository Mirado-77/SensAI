"""State document of a session (M4): sections patched one at a time by the model, in sqlite3."""

import sqlite3
import time
from typing import Literal

from sensai.core.types import new_id
from sensai.safety.untrusted import wrap_untrusted

Op = Literal["replace", "append"]
OPS: tuple[Op, ...] = ("replace", "append")
NOTE_HEADER = (
    "State document of this session: goal, decisions, what is done and what is left. Keep it up"
    " to date with artifact_patch, one section at a time, when a decision is taken or a task ends."
)
EMPTY_SECTION = "_(vide)_"


def patched(before: str, op: Op, content: str) -> str:
    if op == "replace":
        return content.strip()
    if op == "append":
        if not before or not content.strip():
            return before or content.strip()
        else:
            return before.rstrip() + "\n" + content.strip()
    raise ValueError(f"Opération inconnue : {op}")


class ArtifactStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection

    def sections(self, session_id: str) -> dict[str, str]:
        rows = self._db.execute(
            "SELECT section, content FROM artifact_sections WHERE session_id = ?", (session_id,)
        ).fetchall()
        return {row["section"]: row["content"] for row in rows}

    def apply(
        self, session_id: str, run_id: str | None, section: str, op: Op, content: str
    ) -> tuple[str, str]:
        before = self.sections(session_id).get(section, "")
        after = patched(before, op, content)
        if after == before:
            return (before, after)
        now = time.time()
        with self._db:
            self._db.execute(
                "INSERT INTO artifact_sections (session_id, section, content, updated_at)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT (session_id, section)"
                " DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at",
                (session_id, section, after, now),
            )
            self._db.execute(
                "INSERT INTO artifact_patches"
                " (id, session_id, run_id, section, op, before, after, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (new_id(), session_id, run_id, section, op, before, after, now),
            )
        return (before, after)

    def render(self, session_id: str, order: list[str]) -> str:
        """The document in Markdown: the sections of `order`, then any other one still saved."""
        sections = self.sections(session_id)
        names = order + [name for name in sections if name not in order]
        return "\n\n".join(f"## {name}\n{sections.get(name) or EMPTY_SECTION}" for name in names)

    def copy(self, from_session: str, to_session: str) -> None:
        """Give a forked session the document of its source; the patch history stays behind."""
        with self._db:
            self._db.execute(
                "INSERT INTO artifact_sections (session_id, section, content, updated_at)"
                " SELECT ?, section, content, ? FROM artifact_sections WHERE session_id = ?",
                (to_session, time.time(), from_session),
            )


def artifact_note(store: ArtifactStore, session_id: str, order: list[str]) -> str | None:
    """What the runtime adds to the context at each turn, or None while the document is empty."""
    if not any(store.sections(session_id).values()):
        return None
    return f"{NOTE_HEADER}\n" + wrap_untrusted(store.render(session_id, order))
