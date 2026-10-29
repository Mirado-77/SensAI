"""Run metrics (EV3): runs, tokens, latency, tools and errors, read from the event journal.

Everything comes from the tables the journal already fills (`runs`, `tool_calls`, `events`), so
no column is added. `since` (a Unix timestamp) keeps only the runs started from that time.
"""

import json
import math
import sqlite3
from typing import Any

_OVERVIEW = """
SELECT
    COUNT(*) AS runs,
    COALESCE(SUM(status = 'completed'), 0) AS completed,
    COALESCE(SUM(status = 'cancelled'), 0) AS cancelled,
    COALESCE(SUM(status = 'failed'), 0) AS failed,
    COALESCE(SUM(status = 'running'), 0) AS running,
    COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
    COALESCE(SUM(completion_tokens), 0) AS completion_tokens
FROM runs
WHERE started_at >= ?
"""

# Latency is measured on completed runs only: a failed run (model not found) ends at once and an
# interrupted one ends when the user decides, so either would skew the figures.
_DURATIONS = """
SELECT ended_at - started_at AS seconds
FROM runs
WHERE started_at >= ? AND status = 'completed'
"""

# model.chunk events are journaled one by one, so the first one dates the first token.
_FIRST_TOKENS = """
SELECT MIN(e.ts) - r.started_at AS seconds
FROM runs r
JOIN events e ON e.run_id = r.id AND e.type = 'model.chunk'
WHERE r.started_at >= ? AND r.status = 'completed'
GROUP BY r.id
"""

_TOOLS = """
SELECT
    t.tool AS tool,
    COUNT(*) AS calls,
    SUM(t.status = 'ok') AS ok,
    SUM(t.status IN ('error', 'failed')) AS errors,
    SUM(t.status IN ('refused', 'denied')) AS refused,
    AVG(t.duration_ms) AS avg_ms
FROM tool_calls t
JOIN runs r ON r.id = t.run_id
WHERE r.started_at >= ?
GROUP BY t.tool
ORDER BY calls DESC, tool
"""

_GUARDS = """
SELECT type, COUNT(*) AS count
FROM events
WHERE type IN ('guard.blocked', 'guard.redacted') AND ts >= ?
GROUP BY type
"""

# A refused tool call (tool.failed with "denied") is the user's choice, not an error.
_ERRORS = """
SELECT ts, type, payload_json
FROM events
WHERE ts >= ? AND (
    type = 'run.failed'
    OR (type = 'tool.failed' AND json_extract(payload_json, '$.error') != 'denied')
)
ORDER BY ts DESC, id DESC
LIMIT ?
"""


def run_stats(
    connection: sqlite3.Connection, since: float | None = None, errors: int = 5
) -> dict[str, Any]:
    """Overview, latency, tools, guards and the `errors` most recent errors, newest first."""
    start = since or 0.0
    overview = dict(connection.execute(_OVERVIEW, (start,)).fetchone())
    total = overview["prompt_tokens"] + overview["completion_tokens"]
    overview["avg_tokens"] = total / overview["runs"] if overview["runs"] else None
    guards = {row["type"]: row["count"] for row in connection.execute(_GUARDS, (start,))}
    return {
        "since": since,
        "overview": overview,
        "latency": {
            "turn": _summary(_column(connection, _DURATIONS, start)),
            "first_token": _summary(_column(connection, _FIRST_TOKENS, start)),
        },
        "tools": [dict(row) for row in connection.execute(_TOOLS, (start,))],
        "guards": {
            "blocked": guards.get("guard.blocked", 0),
            "redacted": guards.get("guard.redacted", 0),
        },
        "errors": [_error(row) for row in connection.execute(_ERRORS, (start, errors))],
    }


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of non-empty `values`: the smallest value covering `fraction`."""
    ordered = sorted(values)
    return ordered[max(math.ceil(fraction * len(ordered)), 1) - 1]


def _column(connection: sqlite3.Connection, query: str, start: float) -> list[float]:
    return [row["seconds"] for row in connection.execute(query, (start,))]


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "avg": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "avg": sum(values) / len(values),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def _error(row: sqlite3.Row) -> dict[str, Any]:
    payload = json.loads(row["payload_json"])
    if row["type"] == "run.failed":
        kind, message = payload.get("error_type", "error"), payload.get("error", "")
    else:
        kind, message = payload.get("tool", "tool"), payload.get("error", "")
    return {"ts": row["ts"], "type": row["type"], "kind": kind, "message": message}
