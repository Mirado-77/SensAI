"""Scheduled tasks (X5): tools for the chat, and the runner of `sensai schedule`."""

import sqlite3

from sensai.core.config import Config
from sensai.core.contracts import Tool
from sensai.schedule.store import ScheduleStore
from sensai.schedule.tools import (
    AllowCommandTool,
    CancelScheduledTaskTool,
    CreateScheduledTaskTool,
    ListScheduledTasksTool,
)


def schedule_tools(config: Config, connection: sqlite3.Connection) -> list[Tool]:
    """The task tools of the chat (X5), when the scheduler is enabled in `[scheduler]`."""
    if not config.scheduler.enabled:
        return []
    settings = config.scheduler
    store = ScheduleStore(connection, settings.allowed_commands)
    return [
        CreateScheduledTaskTool(store, settings.min_interval_minutes),
        ListScheduledTasksTool(store),
        CancelScheduledTaskTool(store),
        AllowCommandTool(store),
    ]
