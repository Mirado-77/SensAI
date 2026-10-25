"""Scheduled tasks (X5) and the allowed commands, saved in sqlite3."""

import json
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass

from sensai.core.types import new_id

ACTIVE = "active"
CANCELLED = "cancelled"


@dataclass(frozen=True)
class ScheduledTask:
    id: str
    argv: tuple[str, ...]
    interval_minutes: int
    status: str
    created_at: float
    last_run_at: float | None
    last_status: str | None


class ScheduleStore:
    """Allowed commands are the ones of the configuration plus the ones approved in the chat."""

    def __init__(self, connection: sqlite3.Connection, configured: Iterable[Iterable[str]]) -> None:
        self._db = connection
        self._configured = {tuple(command) for command in configured}

    def create_task(self, argv: tuple[str, ...], interval_minutes: int) -> str:
        task_id = new_id()
        with self._db:
            self._db.execute(
                "INSERT INTO scheduled_tasks (id, argv_json, interval_minutes, status, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (task_id, _dumps(argv), interval_minutes, ACTIVE, time.time()),
            )
        return task_id

    def active_tasks(self) -> list[ScheduledTask]:
        return self._tasks("WHERE status = ? ORDER BY created_at", (ACTIVE,))

    def find_active(self, prefix: str) -> list[ScheduledTask]:
        return self._tasks(
            "WHERE status = ? AND id LIKE ? ORDER BY created_at", (ACTIVE, prefix + "%")
        )

    def get(self, task_id: str) -> ScheduledTask | None:
        tasks = self._tasks("WHERE id = ?", (task_id,))
        return tasks[0] if tasks else None

    def cancel(self, task_id: str) -> None:
        with self._db:
            self._db.execute(
                "UPDATE scheduled_tasks SET status = ? WHERE id = ?", (CANCELLED, task_id)
            )

    def record_run(self, task_id: str, status: str) -> None:
        with self._db:
            self._db.execute(
                "UPDATE scheduled_tasks SET last_run_at = ?, last_status = ? WHERE id = ?",
                (time.time(), status, task_id),
            )

    def is_allowed(self, argv: tuple[str, ...]) -> bool:
        return argv in self._configured or argv in self._approved()

    def allow(self, argv: tuple[str, ...]) -> None:
        with self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO allowed_commands (argv_json, created_at) VALUES (?, ?)",
                (_dumps(argv), time.time()),
            )

    def _approved(self) -> set[tuple[str, ...]]:
        rows = self._db.execute("SELECT argv_json FROM allowed_commands").fetchall()
        return {tuple(json.loads(row["argv_json"])) for row in rows}

    def _tasks(self, where: str, params: tuple[str, ...]) -> list[ScheduledTask]:
        rows = self._db.execute(
            "SELECT id, argv_json, interval_minutes, status, created_at, last_run_at, last_status"
            f" FROM scheduled_tasks {where}",
            params,
        ).fetchall()
        return [
            ScheduledTask(
                id=row["id"],
                argv=tuple(json.loads(row["argv_json"])),
                interval_minutes=row["interval_minutes"],
                status=row["status"],
                created_at=row["created_at"],
                last_run_at=row["last_run_at"],
                last_status=row["last_status"],
            )
            for row in rows
        ]


def _dumps(argv: tuple[str, ...]) -> str:
    return json.dumps(list(argv), ensure_ascii=False)
