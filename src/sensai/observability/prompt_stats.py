"""Comparison of the system prompt versions (A4), read from the event journal."""

import sqlite3
from typing import Any

# The version comes from the run.started payload, so no column is added to `runs`.
_STATS = """
SELECT
    COALESCE(json_extract(e.payload_json, '$.prompt_version'), '(none)') AS version,
    COUNT(*) AS runs,
    SUM(r.status = 'completed') AS completed,
    SUM(r.status = 'failed') AS failed,
    AVG(r.prompt_tokens + r.completion_tokens) AS avg_tokens,
    AVG(r.ended_at - r.started_at) AS avg_seconds,
    COALESCE(SUM(t.calls), 0) AS tool_calls,
    COALESCE(SUM(t.errors), 0) AS tool_errors
FROM runs r
JOIN events e ON e.run_id = r.id AND e.type = 'run.started'
LEFT JOIN (
    SELECT run_id, COUNT(*) AS calls, SUM(status IN ('error', 'failed')) AS errors
    FROM tool_calls GROUP BY run_id
) t ON t.run_id = r.id
GROUP BY version
ORDER BY MIN(r.started_at)
"""


def prompt_stats(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """One row per prompt version, oldest first."""
    return [dict(row) for row in connection.execute(_STATS)]
