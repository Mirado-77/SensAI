"""Event journal: an EventSink that records every event in sqlite.

Besides the `events` table, it keeps `runs` and `tool_calls` up to date from the events. Run
status: running, completed, cancelled, failed. Tool call status: requested, awaiting_approval,
approved, refused, running, ok, error, failed, denied.
"""

import json
import sqlite3

from sensai.core.events import (
    APPROVAL_REQUESTED,
    APPROVAL_RESOLVED,
    MODEL_TOOL_CALL,
    MODEL_USAGE,
    RUN_CANCELLED,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_STARTED,
    TOOL_FAILED,
    TOOL_FINISHED,
    TOOL_STARTED,
    Event,
)

_RUN_END_STATUS = {RUN_COMPLETED: "completed", RUN_CANCELLED: "cancelled", RUN_FAILED: "failed"}


class SqliteEventJournal:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection

    def emit(self, event: Event) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO events (run_id, ts, type, payload_json) VALUES (?, ?, ?, ?)",
                (event.run_id, event.ts, event.type, _dumps(event.payload)),
            )
            self._update_tables(event)

    def _update_tables(self, event: Event) -> None:
        payload = event.payload
        call_id = payload.get("call_id")
        if event.type == RUN_STARTED:
            self._db.execute(
                "INSERT INTO runs (id, session_id, model, status, started_at)"
                " VALUES (?, ?, ?, 'running', ?)",
                (event.run_id, payload["session_id"], payload.get("model"), event.ts),
            )
        elif event.type == MODEL_USAGE:
            self._db.execute(
                "UPDATE runs SET prompt_tokens = prompt_tokens + ?,"
                " completion_tokens = completion_tokens + ? WHERE id = ?",
                (payload["prompt_tokens"], payload["completion_tokens"], event.run_id),
            )
        elif event.type in _RUN_END_STATUS:
            self._db.execute(
                "UPDATE runs SET status = ?, ended_at = ? WHERE id = ?",
                (_RUN_END_STATUS[event.type], event.ts, event.run_id),
            )
        elif event.type == MODEL_TOOL_CALL:
            self._db.execute(
                "INSERT INTO tool_calls (id, run_id, tool, args_json, status)"
                " VALUES (?, ?, ?, ?, 'requested')",
                (call_id, event.run_id, payload["name"], _dumps(payload["args"])),
            )
        elif event.type == APPROVAL_REQUESTED:
            self._update_tool_call(call_id, status="awaiting_approval", risk=payload["risk"])
        elif event.type == APPROVAL_RESOLVED:
            approved = bool(payload["approved"])
            status = "approved" if approved else "refused"
            self._update_tool_call(call_id, status=status, approved=int(approved))
        elif event.type == TOOL_STARTED:
            self._update_tool_call(call_id, status="running", risk=payload["risk"])
        elif event.type == TOOL_FINISHED:
            result = {"content": payload["content"], "is_error": payload["is_error"]}
            self._update_tool_call(
                call_id,
                status="error" if payload["is_error"] else "ok",
                result_json=_dumps(result),
                duration_ms=payload["duration_ms"],
            )
        elif event.type == TOOL_FAILED:
            status = "denied" if payload["error"] == "denied" else "failed"
            self._update_tool_call(
                call_id, status=status, result_json=_dumps({"error": payload["error"]})
            )

    def _update_tool_call(self, call_id: str, **columns: object) -> None:
        assignments = ", ".join(f"{column} = ?" for column in columns)
        self._db.execute(
            f"UPDATE tool_calls SET {assignments} WHERE id = ?", (*columns.values(), call_id)
        )


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
