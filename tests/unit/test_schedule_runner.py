import pytest

from sensai.core.events import Event
from sensai.core.types import ToolResult
from sensai.db.connection import connect
from sensai.schedule.runner import ScheduleRunner
from sensai.schedule.store import ScheduleStore

pytestmark = pytest.mark.feature("X5")


class NullSink:
    def emit(self, event: Event) -> None:
        pass


class NullRunCommand:
    name = "run_command"
    risk = "exec"
    parameters = {"type": "object", "properties": {}}

    async def run(self, args, ctx):
        return ToolResult("ok")


@pytest.fixture
def store():
    return ScheduleStore(connect(":memory:"), [["pytest", "-q"]])


@pytest.fixture
def runner(store):
    return ScheduleRunner(store, NullRunCommand(), NullSink(), sync_interval_s=30)


async def test_sync_schedules_new_tasks_and_drops_cancelled_ones(store, runner):
    task_id = store.create_task(("pytest", "-q"), 60)
    runner._scheduler.start()
    try:
        runner._sync()
        assert runner._scheduler.get_job(task_id) is not None

        store.cancel(task_id)
        runner._sync()
        assert runner._scheduler.get_job(task_id) is None
    finally:
        runner._scheduler.shutdown(wait=False)


async def test_sync_picks_up_a_task_created_after_start(store, runner):
    runner._scheduler.start()
    try:
        runner._sync()
        task_id = store.create_task(("pytest", "-q"), 60)
        runner._sync()
        assert runner._scheduler.get_job(task_id) is not None
    finally:
        runner._scheduler.shutdown(wait=False)
