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
        self._handlers = {
            RUN_STARTED: self._run_started,
            MODEL_USAGE: self._add_usage,
            RUN_COMPLETED: self._run_ended,
            RUN_CANCELLED: self._run_ended,
            RUN_FAILED: self._run_ended,
            MODEL_TOOL_CALL: self._tool_requested,
            APPROVAL_REQUESTED: self._approval_requested,
            APPROVAL_RESOLVED: self._approval_resolved,
            TOOL_STARTED: self._tool_started,
            TOOL_FINISHED: self._tool_finished,
            TOOL_FAILED: self._tool_failed,
        }

    def emit(self, event: Event) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO events (run_id, ts, type, payload_json) VALUES (?, ?, ?, ?)",
                (event.run_id, event.ts, event.type, _dumps(event.payload)),
            )
            handler = self._handlers.get(event.type)
            if handler is not None:
                handler(event)

    def _run_started(self, event: Event) -> None:
        self._db.execute(
            "INSERT INTO runs (id, session_id, model, status, started_at)"
            " VALUES (?, ?, ?, 'running', ?)",
            (event.run_id, event.payload["session_id"], event.payload.get("model"), event.ts),
        )

    def _add_usage(self, event: Event) -> None:
        self._db.execute(
            "UPDATE runs SET prompt_tokens = prompt_tokens + ?,"
            " completion_tokens = completion_tokens + ? WHERE id = ?",
            (event.payload["prompt_tokens"], event.payload["completion_tokens"], event.run_id),
        )

    def _run_ended(self, event: Event) -> None:
        self._db.execute(
            "UPDATE runs SET status = ?, ended_at = ? WHERE id = ?",
            (_RUN_END_STATUS[event.type], event.ts, event.run_id),
        )

    def _tool_requested(self, event: Event) -> None:
        self._db.execute(
            "INSERT INTO tool_calls (id, run_id, tool, args_json, status)"
            " VALUES (?, ?, ?, ?, 'requested')",
            (
                event.payload["call_id"],
                event.run_id,
                event.payload["name"],
                _dumps(event.payload["args"]),
            ),
        )

    def _approval_requested(self, event: Event) -> None:
        self._update_tool_call(event, status="awaiting_approval", risk=event.payload["risk"])

    def _approval_resolved(self, event: Event) -> None:
        approved = bool(event.payload["approved"])
        status = "approved" if approved else "refused"
        self._update_tool_call(event, status=status, approved=int(approved))

    def _tool_started(self, event: Event) -> None:
        self._update_tool_call(event, status="running", risk=event.payload["risk"])

    def _tool_finished(self, event: Event) -> None:
        payload = event.payload
        result = {"content": payload["content"], "is_error": payload["is_error"]}
        self._update_tool_call(
            event,
            status="error" if payload["is_error"] else "ok",
            result_json=_dumps(result),
            duration_ms=payload["duration_ms"],
        )

    def _tool_failed(self, event: Event) -> None:
        error = event.payload["error"]
        status = "denied" if error == "denied" else "failed"
        self._update_tool_call(event, status=status, result_json=_dumps({"error": error}))

    def _update_tool_call(self, event: Event, **columns: object) -> None:
        assignments = ", ".join(f"{column} = ?" for column in columns)
        self._db.execute(
            f"UPDATE tool_calls SET {assignments} WHERE id = ?",
            (*columns.values(), event.payload["call_id"]),
        )


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
