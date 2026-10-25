"""Runs the scheduled tasks at their interval (X5), in the process started by `sensai schedule`.

The chat saves the tasks; this process reads them every `sync_interval_s`, so a task created in
the chat runs without restarting anything. A task runs its command through the sandbox tool
(T2), and only if the command is still allowed at that moment.

Event payloads (all events also carry `run_id`):

- run.started: session_id ("scheduler"), model, task_id, command
- run.completed, run.failed: task_id, command, output (the end of the output, truncated)
"""

import asyncio
import shlex
from datetime import datetime, timedelta
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from sensai.core.contracts import EventSink, Tool
from sensai.core.events import RUN_COMPLETED, RUN_FAILED, RUN_STARTED, make_event
from sensai.core.types import ToolContext, ToolResult, new_id
from sensai.schedule.store import ACTIVE, ScheduledTask, ScheduleStore

SCHEDULER_SESSION = "scheduler"
SYNC_JOB = "sync"
MAX_OUTPUT_CHARS = 2000
NOT_ALLOWED = "Commande absente de la liste autorisée"


class ScheduleRunner:
    def __init__(
        self,
        store: ScheduleStore,
        run_command: Tool,
        sink: EventSink,
        sync_interval_s: float,
    ) -> None:
        self._store = store
        self._run_command = run_command
        self._sink = sink
        self._sync_interval_s = sync_interval_s
        self._scheduler = AsyncIOScheduler()
        self._jobs: set[str] = set()

    async def run_forever(self) -> None:
        self._scheduler.start()
        self._sync()
        self._scheduler.add_job(
            self._sync, "interval", seconds=self._sync_interval_s, id=SYNC_JOB, max_instances=1
        )
        try:
            await asyncio.Event().wait()
        finally:
            self._scheduler.shutdown(wait=False)

    def _sync(self) -> None:
        tasks = {task.id: task for task in self._store.active_tasks()}
        for task_id in self._jobs - tasks.keys():
            self._scheduler.remove_job(task_id)
        for task in tasks.values():
            if task.id not in self._jobs:
                self._schedule(task)
        self._jobs = set(tasks)

    def _schedule(self, task: ScheduledTask) -> None:
        self._scheduler.add_job(
            self.run_task,
            "interval",
            minutes=task.interval_minutes,
            id=task.id,
            args=[task.id],
            next_run_time=_next_run(task),
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )

    async def run_task(self, task_id: str) -> None:
        """Run one task once and record the result; a task that was cancelled does not run."""
        task = self._store.get(task_id)
        if task is None or task.status != ACTIVE:
            return
        run_id = new_id()
        command = shlex.join(task.argv)
        self._emit(
            RUN_STARTED,
            run_id,
            session_id=SCHEDULER_SESSION,
            model=None,
            task_id=task_id,
            command=command,
        )
        if self._store.is_allowed(task.argv):
            result = await self._execute(task, run_id, command)
        else:
            result = ToolResult(f"{NOT_ALLOWED} : {command}", is_error=True)
        status = "failed" if result.is_error else "completed"
        self._store.record_run(task_id, status)
        self._emit(
            RUN_FAILED if result.is_error else RUN_COMPLETED,
            run_id,
            task_id=task_id,
            command=command,
            output=result.content[-MAX_OUTPUT_CHARS:],
        )

    async def _execute(self, task: ScheduledTask, run_id: str, command: str) -> ToolResult:
        args: dict[str, Any] = {
            "command": command,
            "description": f"Tâche planifiée {task.id[:8]}",
        }
        try:
            return await self._run_command.run(args, ToolContext(run_id, SCHEDULER_SESSION))
        except Exception as exc:  # a failing run must not stop the other tasks
            return ToolResult(str(exc), is_error=True)

    def _emit(self, type_: str, run_id: str, **payload: Any) -> None:
        self._sink.emit(make_event(type_, run_id, payload))


def _next_run(task: ScheduledTask) -> datetime:
    """The first run comes one interval after the last run, or after the creation of the task."""
    anchor = task.last_run_at if task.last_run_at is not None else task.created_at
    due = datetime.fromtimestamp(anchor).astimezone() + timedelta(minutes=task.interval_minutes)
    return max(due, datetime.now().astimezone())
