"""SessionStore backed by sqlite3."""

import json
import sqlite3
import time
from typing import Any

from sensai.core.types import Message, ToolCall, new_id

_PATH_SQL = """
WITH RECURSIVE path(id, parent_id, role, content, status, tool_calls_json, tool_name, depth) AS (
    SELECT id, parent_id, role, content, status, tool_calls_json, tool_name, 0
    FROM messages WHERE id = ?
    UNION ALL
    SELECT m.id, m.parent_id, m.role, m.content, m.status, m.tool_calls_json, m.tool_name,
           path.depth + 1
    FROM messages m JOIN path ON m.id = path.parent_id
)
SELECT id, role, content, status, tool_calls_json, tool_name FROM path ORDER BY depth DESC
"""

_IN_TREE_SQL = """
WITH RECURSIVE tips(id) AS (
    SELECT id FROM messages WHERE session_id = :session
    UNION
    SELECT head_message_id FROM sessions WHERE id = :session AND head_message_id IS NOT NULL
),
tree(id) AS (
    SELECT id FROM tips
    UNION
    SELECT m.parent_id FROM messages m JOIN tree ON m.id = tree.id WHERE m.parent_id IS NOT NULL
)
SELECT 1 FROM tree WHERE id = :message
"""


_INSERT_MESSAGE = (
    "INSERT INTO messages (id, session_id, parent_id, role, content, status,"
    " tool_calls_json, tool_name, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

_SET_HEAD = "UPDATE sessions SET head_message_id = ? WHERE id = ?"


class SqliteSessionStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection

    def add_profile(self, profile_id: str, data: dict[str, Any]) -> None:
        """Not part of the contract: create or replace a profile."""
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO profiles (id, data_json) VALUES (?, ?)",
                (profile_id, json.dumps(data)),
            )

    def create_session(self, profile_id: str | None = None) -> str:
        session_id = new_id()
        with self._db:
            self._db.execute(
                "INSERT INTO sessions (id, created_at, profile_id) VALUES (?, ?, ?)",
                (session_id, time.time(), profile_id),
            )
        return session_id

    def append(self, session_id: str, msg: Message, parent_id: str | None) -> str:
        self._require_session(session_id)
        if parent_id is not None:
            self._require_message(parent_id)
        message_id = new_id()
        row = _message_row(message_id, session_id, parent_id, msg)
        with self._db:
            self._db.execute(_INSERT_MESSAGE, row)
            self._db.execute(_SET_HEAD, (message_id, session_id))
        return message_id

    def history(self, session_id: str, head_id: str | None = None) -> list[Message]:
        self._require_session(session_id)
        if head_id is None:
            row = self._db.execute(
                "SELECT head_message_id FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            head_id = row[0]
        elif not self._in_tree(session_id, head_id):
            raise KeyError(f"Message hors de la session {session_id} : {head_id}")
        if head_id is None:
            return []
        return [_to_message(row) for row in self._db.execute(_PATH_SQL, (head_id,))]

    def fork(self, session_id: str, from_message_id: str) -> str:
        self._require_session(session_id)
        if not self._in_tree(session_id, from_message_id):
            raise KeyError(f"Message hors de la session {session_id} : {from_message_id}")
        fork_id = new_id()
        with self._db:
            self._db.execute(
                "INSERT INTO sessions (id, created_at, profile_id, head_message_id)"
                " SELECT ?, ?, profile_id, ? FROM sessions WHERE id = ?",
                (fork_id, time.time(), from_message_id, session_id),
            )
        return fork_id

    def profile(self, profile_id: str) -> dict[str, Any]:
        row = self._db.execute(
            "SELECT data_json FROM profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Profil inconnu : {profile_id}")
        return json.loads(row[0])

    def _exists(self, table: str, row_id: str) -> bool:
        row = self._db.execute(f"SELECT 1 FROM {table} WHERE id = ?", (row_id,))
        return row.fetchone() is not None

    def _require_session(self, session_id: str) -> None:
        if not self._exists("sessions", session_id):
            raise KeyError(f"Session inconnue : {session_id}")

    def _require_message(self, message_id: str) -> None:
        if not self._exists("messages", message_id):
            raise KeyError(f"Message inconnu : {message_id}")

    def _in_tree(self, session_id: str, message_id: str) -> bool:
        row = self._db.execute(_IN_TREE_SQL, {"session": session_id, "message": message_id})
        return row.fetchone() is not None


def _message_row(message_id: str, session_id: str, parent_id: str | None, msg: Message) -> tuple:
    tool_calls = [{"name": call.name, "args": call.args} for call in msg.tool_calls]
    return (
        message_id,
        session_id,
        parent_id,
        msg.role,
        msg.content,
        msg.status,
        json.dumps(tool_calls) if tool_calls else None,
        msg.tool_name,
        time.time(),
    )


def _to_message(row: sqlite3.Row) -> Message:
    tool_calls = tuple(
        ToolCall(call["name"], call["args"]) for call in json.loads(row["tool_calls_json"] or "[]")
    )
    return Message(
        role=row["role"],
        content=row["content"],
        id=row["id"],
        status=row["status"],
        tool_calls=tool_calls,
        tool_name=row["tool_name"],
    )
