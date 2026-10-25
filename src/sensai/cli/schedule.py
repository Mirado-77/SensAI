"""`sensai schedule`: runs the scheduled tasks until Ctrl-C (X5)."""

import sqlite3
from datetime import datetime

from rich.text import Text

from sensai.cli.terminal import Terminal
from sensai.core.bus import EventBus
from sensai.core.config import Config
from sensai.core.errors import SensaiError
from sensai.core.events import RUN_COMPLETED, RUN_FAILED, RUN_STARTED, Event
from sensai.db.connection import connect
from sensai.observability.journal import SqliteEventJournal
from sensai.schedule.runner import ScheduleRunner
from sensai.schedule.store import ScheduleStore
from sensai.tools import sandbox_tools

RUN_LABELS = {
    RUN_STARTED: ("lancée", "cyan"),
    RUN_COMPLETED: ("terminée", "green"),
    RUN_FAILED: ("échouée", "red"),
}


class TerminalLog:
    """Prints the start and the end of each scheduled run."""

    def __init__(self, terminal: Terminal) -> None:
        self._terminal = terminal

    def emit(self, event: Event) -> None:
        if event.type not in RUN_LABELS:
            return
        label, style = RUN_LABELS[event.type]
        stamp = datetime.fromtimestamp(event.ts).strftime("%H:%M:%S")
        line = Text(f"{stamp} tâche {event.payload['task_id'][:8]} {label} : ", style=style)
        line.append(event.payload["command"])
        self._terminal.console.print(line)


async def run_schedule(config: Config, terminal: Terminal) -> int:
    """Run the scheduled tasks until interrupted. Return the process exit code."""
    if not config.scheduler.enabled:
        message = "Erreur : le planificateur est désactivé dans [scheduler]."
        terminal.errors.print(Text(message, style="bold red"))
        return 2
    try:
        connection = connect(config.storage.db_path)
    except (OSError, sqlite3.Error, SensaiError) as exc:
        message = f"Erreur : base de données inutilisable : {exc}"
        terminal.errors.print(Text(message, style="bold red"))
        return 2
    try:
        return await _run_scheduler(config, terminal, connection)
    finally:
        connection.close()


async def _run_scheduler(config: Config, terminal: Terminal, connection: sqlite3.Connection) -> int:
    tools, reason = await sandbox_tools(config)
    if not tools:
        message = reason or "le sandbox est désactivé dans [sandbox]"
        terminal.errors.print(Text(f"Erreur : exécution impossible (T2) : {message}", "bold red"))
        return 2
    store = ScheduleStore(connection, config.scheduler.allowed_commands)
    sink = EventBus([SqliteEventJournal(connection), TerminalLog(terminal)])
    runner = ScheduleRunner(store, tools[0], sink, config.scheduler.sync_interval_s)
    terminal.console.print(Text("Planificateur démarré. Ctrl-C pour arrêter.", "dim"))
    await runner.run_forever()
    return 0
