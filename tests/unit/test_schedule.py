import time

import pytest

from sensai.core.config import SchedulerConfig
from sensai.core.events import RUN_COMPLETED, RUN_FAILED, RUN_STARTED, Event
from sensai.core.types import ToolContext, ToolResult
from sensai.db.connection import connect
from sensai.schedule import schedule_tools
from sensai.schedule.runner import ScheduleRunner, _next_run
from sensai.schedule.store import CANCELLED, ScheduleStore
from sensai.schedule.tools import (
    AllowCommandTool,
    CancelScheduledTaskTool,
    CreateScheduledTaskTool,
    ListScheduledTasksTool,
    describe_interval,
)

pytestmark = pytest.mark.feature("X5")

CTX = ToolContext(run_id="r1", session_id="s1")
CONFIGURED = [["pytest", "-q"]]


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        return [event.type for event in self.events]


class FakeRunCommand:
    name = "run_command"
    risk = "exec"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, result: ToolResult | None = None, error: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self._result = result or ToolResult("ok")
        self._error = error

    async def run(self, args, ctx):
        self.calls.append(args)
        if self._error is not None:
            raise self._error
        return self._result


@pytest.fixture
def connection():
    return connect(":memory:")


@pytest.fixture
def store(connection):
    return ScheduleStore(connection, CONFIGURED)


def test_describe_interval_uses_the_largest_unit():
    assert describe_interval(2 * 24 * 60) == "toutes les 2 jours"
    assert describe_interval(60) == "toutes les 1 heure"
    assert describe_interval(90) == "toutes les 90 minutes"


async def test_create_saves_an_allowed_command(store):
    tool = CreateScheduledTaskTool(store, min_interval_minutes=5)
    args = {"command": "pytest -q", "every": 2, "unit": "days"}

    result = await tool.run(args, CTX)

    assert not result.is_error
    [task] = store.active_tasks()
    assert task.argv == ("pytest", "-q")
    assert task.interval_minutes == 2 * 24 * 60


async def test_create_refuses_a_command_missing_from_the_list(store, connection):
    tool = CreateScheduledTaskTool(store, min_interval_minutes=5)
    args = {"command": "make test", "every": 2, "unit": "days"}

    result = await tool.run(args, CTX)

    assert result.is_error
    assert "absente de la liste autorisée" in result.content
    assert "allow_command" in result.content
    assert store.active_tasks() == []


async def test_preview_warns_when_the_command_is_not_allowed(store):
    tool = CreateScheduledTaskTool(store, min_interval_minutes=5)

    allowed = await tool.preview({"command": "pytest -q", "every": 1, "unit": "hours"})
    refused = await tool.preview({"command": "make test", "every": 1, "unit": "hours"})

    assert allowed == "Tâche planifiée toutes les 1 heure\n$ pytest -q"
    assert "ATTENTION" in refused


@pytest.mark.parametrize(
    "args",
    [
        {"command": "pytest -q", "every": 1, "unit": "weeks"},
        {"command": "pytest -q", "every": 0, "unit": "days"},
        {"command": "pytest -q", "every": True, "unit": "days"},
        {"command": "pytest -q", "every": 1},
        {"command": "pytest 'unclosed", "every": 1, "unit": "days"},
        {"command": "   ", "every": 1, "unit": "days"},
    ],
)
async def test_create_rejects_invalid_arguments(store, args):
    tool = CreateScheduledTaskTool(store, min_interval_minutes=5)

    result = await tool.run(args, CTX)

    assert result.is_error
    assert store.active_tasks() == []


async def test_create_rejects_an_interval_below_the_minimum(store):
    tool = CreateScheduledTaskTool(store, min_interval_minutes=5)

    result = await tool.run({"command": "pytest -q", "every": 4, "unit": "minutes"}, CTX)

    assert result.is_error
    assert "au moins 5 minutes" in result.content


async def test_allow_command_adds_the_command_to_the_list(store):
    allow = AllowCommandTool(store)
    create = CreateScheduledTaskTool(store, min_interval_minutes=5)
    assert not store.is_allowed(("ruff", "format", "--check", "."))

    await allow.run({"command": "ruff format --check ."}, CTX)
    result = await create.run({"command": "ruff format --check .", "every": 1, "unit": "days"}, CTX)

    assert not result.is_error
    assert store.is_allowed(("ruff", "format", "--check", "."))


async def test_allow_command_stores_a_new_command_only_once(store, connection):
    allow = AllowCommandTool(store)

    await allow.run({"command": "ruff check ."}, CTX)
    result = await allow.run({"command": "ruff check ."}, CTX)

    assert "déjà autorisée" in result.content
    count = connection.execute("SELECT COUNT(*) FROM allowed_commands").fetchone()[0]
    assert count == 1


async def test_list_then_cancel_by_prefix(store):
    create = CreateScheduledTaskTool(store, min_interval_minutes=5)
    listing = ListScheduledTasksTool(store)
    cancel = CancelScheduledTaskTool(store)
    await create.run({"command": "pytest -q", "every": 1, "unit": "hours"}, CTX)
    [task] = store.active_tasks()

    before = await listing.run({}, CTX)
    result = await cancel.run({"task_id": task.id[:6]}, CTX)
    after = await listing.run({}, CTX)

    assert task.id[:8] in before.content
    assert not result.is_error
    assert after.content == "Aucune tâche planifiée."
    assert store.get(task.id).status == CANCELLED


async def test_cancel_refuses_an_ambiguous_or_unknown_prefix(store, connection):
    create = CreateScheduledTaskTool(store, min_interval_minutes=5)
    cancel = CancelScheduledTaskTool(store)
    for _ in range(2):
        await create.run({"command": "pytest -q", "every": 1, "unit": "hours"}, CTX)
    connection.execute("UPDATE scheduled_tasks SET id = 'a1' WHERE rowid = 1")
    connection.execute("UPDATE scheduled_tasks SET id = 'a2' WHERE rowid = 2")

    ambiguous = await cancel.run({"task_id": "a"}, CTX)
    unknown = await cancel.run({"task_id": "zz"}, CTX)

    assert ambiguous.is_error and "Plusieurs tâches" in ambiguous.content
    assert unknown.is_error and "Aucune tâche active" in unknown.content
    assert len(store.active_tasks()) == 2


def test_configured_commands_need_no_approval(connection):
    store = ScheduleStore(connection, [["ruff", "check", "."]])

    assert store.is_allowed(("ruff", "check", "."))
    assert not store.is_allowed(("ruff", "check"))


def test_schedule_tools_are_built_from_the_config(connection):
    from sensai.core.config import load_config

    config = load_config()

    names = [tool.name for tool in schedule_tools(config, connection)]

    assert names == [
        "create_scheduled_task",
        "list_scheduled_tasks",
        "cancel_scheduled_task",
        "allow_command",
    ]


def test_scheduler_config_rejects_an_empty_command():
    with pytest.raises(ValueError):
        SchedulerConfig(
            enabled=True,
            min_interval_minutes=5,
            sync_interval_s=30,
            allowed_commands=[[]],
        )


async def test_runner_records_a_successful_run(store, connection):
    task_id = store.create_task(("pytest", "-q"), 60)
    tool = FakeRunCommand(ToolResult("3 passed"))
    sink = RecordingSink()
    runner = ScheduleRunner(store, tool, sink, sync_interval_s=30)

    await runner.run_task(task_id)

    assert tool.calls[0]["command"] == "pytest -q"
    assert sink.types() == [RUN_STARTED, RUN_COMPLETED]
    assert sink.events[1].payload["output"] == "3 passed"
    task = store.get(task_id)
    assert task.last_status == "completed"
    assert task.last_run_at is not None


async def test_runner_records_a_failing_command(store):
    task_id = store.create_task(("pytest", "-q"), 60)
    tool = FakeRunCommand(ToolResult("1 failed", is_error=True))
    sink = RecordingSink()

    await ScheduleRunner(store, tool, sink, sync_interval_s=30).run_task(task_id)

    assert sink.types() == [RUN_STARTED, RUN_FAILED]
    assert store.get(task_id).last_status == "failed"


async def test_runner_survives_a_crashing_tool(store):
    task_id = store.create_task(("pytest", "-q"), 60)
    tool = FakeRunCommand(error=RuntimeError("docker down"))
    sink = RecordingSink()

    await ScheduleRunner(store, tool, sink, sync_interval_s=30).run_task(task_id)

    assert sink.types() == [RUN_STARTED, RUN_FAILED]
    assert sink.events[1].payload["output"] == "docker down"


async def test_runner_refuses_a_command_removed_from_the_list(connection):
    store = ScheduleStore(connection, [])
    task_id = store.create_task(("make", "test"), 60)
    tool = FakeRunCommand()
    sink = RecordingSink()

    await ScheduleRunner(store, tool, sink, sync_interval_s=30).run_task(task_id)

    assert tool.calls == []
    assert sink.types() == [RUN_STARTED, RUN_FAILED]
    assert "absente de la liste autorisée" in sink.events[1].payload["output"]


async def test_runner_skips_a_cancelled_task(store):
    task_id = store.create_task(("pytest", "-q"), 60)
    store.cancel(task_id)
    tool = FakeRunCommand()
    sink = RecordingSink()

    await ScheduleRunner(store, tool, sink, sync_interval_s=30).run_task(task_id)

    assert tool.calls == []
    assert sink.events == []


def test_first_run_waits_one_interval_after_creation(store):
    store.create_task(("pytest", "-q"), 120)
    [task] = store.active_tasks()

    due = _next_run(task)

    assert due.timestamp() >= task.created_at + 120 * 60 - 1


def test_next_run_follows_the_last_run(store):
    task_id = store.create_task(("pytest", "-q"), 60)
    store.record_run(task_id, "completed")
    [task] = store.active_tasks()

    due = _next_run(task)

    assert due.timestamp() >= task.last_run_at + 60 * 60 - 1
    assert due.timestamp() <= time.time() + 60 * 60 + 1
