"""Tools of X5: the model plans, lists and cancels scheduled tasks, and proposes allowed commands.

The runtime asks the user before a tool of risk "write" or "exec" runs, and shows the text of
`preview`. So the user approves a task once, when it is created, and approves a new allowed
command before it is added. The scheduler then runs only what was approved.
"""

import shlex
from datetime import datetime
from typing import Any

from sensai.core.types import Risk, ToolContext, ToolResult
from sensai.schedule.store import ScheduledTask, ScheduleStore

UNITS = {"minutes": 1, "hours": 60, "days": 24 * 60}
UNIT_LABELS = {"minutes": "minute", "hours": "heure", "days": "jour"}
NOT_ALLOWED = "Commande absente de la liste autorisée : {command}."
NOT_ALLOWED_HINT = "Dis-le à l'utilisateur et propose de l'ajouter avec allow_command."
NOT_ALLOWED_PREVIEW = "ATTENTION : commande absente de la liste autorisée."
INVALID_INTERVAL = (
    'Intervalle invalide : "every" doit être un entier positif et "unit" minutes, hours ou days.'
)
INVALID_COMMAND = 'Argument "command" invalide : une commande non vide, guillemets fermés.'


def describe_interval(minutes: int) -> str:
    for unit in ("days", "hours", "minutes"):
        size = UNITS[unit]
        if minutes % size == 0:
            count = minutes // size
            plural = "s" if count > 1 else ""
            return f"toutes les {count} {UNIT_LABELS[unit]}{plural}"
    raise ValueError(f"Intervalle non décomposable : {minutes}")


def parse_command(text: Any) -> tuple[str, ...] | None:
    if not isinstance(text, str):
        return None
    try:
        argv = tuple(shlex.split(text))
    except ValueError:
        return None
    return argv or None


def _not_allowed_result(argv: tuple[str, ...]) -> ToolResult:
    message = NOT_ALLOWED.format(command=shlex.join(argv))
    return ToolResult(f"{message} {NOT_ALLOWED_HINT}", is_error=True)


class CreateScheduledTaskTool:
    name = "create_scheduled_task"
    description = (
        "Schedule a command to run again at a fixed interval, for example the tests every 2 days."
        " The command must be in the allowed list. If it is not, tell the user and offer to add"
        " it with allow_command. The user approves the task once, when it is created."
    )
    risk: Risk = "exec"
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Command line, for example: pytest -q"},
            "every": {"type": "integer", "minimum": 1, "description": "Units between two runs"},
            "unit": {"type": "string", "enum": ["minutes", "hours", "days"]},
        },
        "required": ["command", "every", "unit"],
    }

    def __init__(self, store: ScheduleStore, min_interval_minutes: int) -> None:
        self._store = store
        self._min_interval = min_interval_minutes

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        request = self._request(args)
        if isinstance(request, ToolResult):
            return request
        argv, minutes = request
        if not self._store.is_allowed(argv):
            return _not_allowed_result(argv)
        task_id = self._store.create_task(argv, minutes)
        return ToolResult(
            f"Tâche {task_id[:8]} planifiée : {shlex.join(argv)}, {describe_interval(minutes)}."
        )

    async def preview(self, args: dict[str, Any]) -> str | None:
        request = self._request(args)
        if isinstance(request, ToolResult):
            return None
        argv, minutes = request
        lines = [f"Tâche planifiée {describe_interval(minutes)}", f"$ {shlex.join(argv)}"]
        if not self._store.is_allowed(argv):
            lines.append(NOT_ALLOWED_PREVIEW)
        return "\n".join(lines)

    def _request(self, args: dict[str, Any]) -> tuple[tuple[str, ...], int] | ToolResult:
        argv = parse_command(args.get("command"))
        if argv is None:
            return ToolResult(INVALID_COMMAND, is_error=True)
        every, unit = args.get("every"), args.get("unit")
        valid_every = isinstance(every, int) and not isinstance(every, bool) and every >= 1
        if not valid_every or unit not in UNITS:
            return ToolResult(INVALID_INTERVAL, is_error=True)
        minutes = every * UNITS[unit]
        if minutes < self._min_interval:
            return ToolResult(
                f"Intervalle trop court : au moins {self._min_interval} minutes.", is_error=True
            )
        return argv, minutes


class ListScheduledTasksTool:
    name = "list_scheduled_tasks"
    description = "List the active scheduled tasks, with their interval and their last run."
    risk: Risk = "read"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, store: ScheduleStore) -> None:
        self._store = store

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        tasks = self._store.active_tasks()
        if not tasks:
            return ToolResult("Aucune tâche planifiée.")
        return ToolResult("\n".join(_describe_task(task) for task in tasks))


class CancelScheduledTaskTool:
    name = "cancel_scheduled_task"
    description = "Cancel an active scheduled task, given the start of its identifier."
    risk: Risk = "write"
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "minLength": 1, "description": "Start of the identifier"}
        },
        "required": ["task_id"],
    }

    def __init__(self, store: ScheduleStore) -> None:
        self._store = store

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        task = self._only_match(args)
        if isinstance(task, ToolResult):
            return task
        self._store.cancel(task.id)
        return ToolResult(f"Tâche {task.id[:8]} annulée : {shlex.join(task.argv)}.")

    async def preview(self, args: dict[str, Any]) -> str | None:
        task = self._only_match(args)
        if isinstance(task, ToolResult):
            return None
        return f"Annuler la tâche {task.id[:8]}\n$ {shlex.join(task.argv)}"

    def _only_match(self, args: dict[str, Any]) -> ScheduledTask | ToolResult:
        prefix = args.get("task_id")
        matches = self._store.find_active(prefix) if isinstance(prefix, str) else []
        if len(matches) == 1:
            return matches[0]
        if not matches:
            return ToolResult(f"Aucune tâche active ne commence par {prefix}.", is_error=True)
        return ToolResult(
            f"Plusieurs tâches commencent par {prefix} : précise l'identifiant.", is_error=True
        )


class AllowCommandTool:
    name = "allow_command"
    description = (
        "Add a command to the allowed list, so that scheduled tasks can run it. Only call it when"
        " the user has asked for it. The user confirms the addition before it is saved."
    )
    risk: Risk = "write"
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "Command line to allow"}},
        "required": ["command"],
    }

    def __init__(self, store: ScheduleStore) -> None:
        self._store = store

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        argv = parse_command(args.get("command"))
        if argv is None:
            return ToolResult(INVALID_COMMAND, is_error=True)
        if self._store.is_allowed(argv):
            return ToolResult(f"Commande déjà autorisée : {shlex.join(argv)}.")
        self._store.allow(argv)
        return ToolResult(f"Commande ajoutée à la liste autorisée : {shlex.join(argv)}.")

    async def preview(self, args: dict[str, Any]) -> str | None:
        argv = parse_command(args.get("command"))
        if argv is None:
            return None
        return f"Ajouter à la liste des commandes autorisées\n$ {shlex.join(argv)}"


def _describe_task(task: ScheduledTask) -> str:
    last = "jamais lancée"
    if task.last_run_at is not None:
        stamp = datetime.fromtimestamp(task.last_run_at).strftime("%Y-%m-%d %H:%M")
        last = f"dernier lancement {stamp} ({task.last_status})"
    return (
        f"{task.id[:8]} {describe_interval(task.interval_minutes)} :"
        f" {shlex.join(task.argv)} ({last})"
    )
