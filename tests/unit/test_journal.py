import asyncio
import json
from dataclasses import dataclass
from typing import Any

from sensai.core.bus import EventBus
from sensai.core.config import RuntimeConfig
from sensai.core.errors import ModelConnectionError
from sensai.core.events import MODEL_CHUNK, make_event
from sensai.core.fakes import CollectingEventSink, FakeModelGateway, FakeTool, FakeUserChannel
from sensai.core.runtime import AgentRuntime
from sensai.core.types import Chunk, ToolCall, ToolResult, Usage
from sensai.db.connection import connect
from sensai.memory.sqlite_store import SqliteSessionStore
from sensai.observability.journal import SqliteEventJournal
from sensai.safety.policy import ApprovalPolicy
from sensai.tools.registry import ToolRegistry

LIMITS = RuntimeConfig(max_steps=8, max_tool_calls=8, timeout_s=5, max_tokens=1000)
RULES = {"read": "allow", "write": "ask", "exec": "deny"}


@dataclass
class Setup:
    runtime: AgentRuntime
    db: Any
    sink: CollectingEventSink
    session_id: str


def build(*scripts, tools=(), approvals=(), error=None) -> Setup:
    db = connect(":memory:")
    store = SqliteSessionStore(db)
    sink = CollectingEventSink()
    runtime = AgentRuntime(
        gateway=FakeModelGateway(*scripts, error=error),
        store=store,
        sink=EventBus([SqliteEventJournal(db), sink]),
        channel=FakeUserChannel(approvals=approvals),
        registry=ToolRegistry(tools),
        policy=ApprovalPolicy(RULES),
        config=LIMITS,
        model_name="test-model",
    )
    return Setup(runtime, db, sink, store.create_session())


def rows(setup, sql, *params):
    return [dict(row) for row in setup.db.execute(sql, params)]


async def test_every_event_is_recorded_in_order_with_its_payload():
    s = build([Chunk("Hello"), Usage(5, 2)])
    await s.runtime.run_turn(s.session_id, "Hi")
    recorded = rows(s, "SELECT type, payload_json, run_id FROM events ORDER BY id")
    assert [row["type"] for row in recorded] == s.sink.types()
    chunk = next(row for row in recorded if row["type"] == MODEL_CHUNK)
    assert json.loads(chunk["payload_json"]) == {"text": "Hello"}
    assert len({row["run_id"] for row in recorded}) == 1


async def test_a_completed_run_is_recorded_with_its_tokens():
    s = build([Chunk("Hello"), Usage(5, 2)])
    result = await s.runtime.run_turn(s.session_id, "Hi")
    [run] = rows(s, "SELECT * FROM runs")
    assert run["id"] == result.run_id
    assert run["session_id"] == s.session_id
    assert (run["model"], run["status"]) == ("test-model", "completed")
    assert (run["prompt_tokens"], run["completion_tokens"]) == (5, 2)
    assert run["ended_at"] >= run["started_at"]


async def test_tokens_are_added_up_over_the_model_calls_of_a_run():
    tool = FakeTool(name="read_file")
    s = build([ToolCall("read_file", {}), Usage(3, 1)], [Chunk("ok"), Usage(10, 4)], tools=[tool])
    await s.runtime.run_turn(s.session_id, "Hi")
    [run] = rows(s, "SELECT prompt_tokens, completion_tokens FROM runs")
    assert (run["prompt_tokens"], run["completion_tokens"]) == (13, 5)


async def test_each_turn_is_a_separate_run():
    s = build([Chunk("one")], [Chunk("two")])
    await s.runtime.run_turn(s.session_id, "first")
    await s.runtime.run_turn(s.session_id, "second")
    assert len(rows(s, "SELECT id FROM runs")) == 2


async def test_a_failed_run_is_recorded_as_failed():
    s = build(error=ModelConnectionError("down"))
    await s.runtime.run_turn(s.session_id, "Hi")
    assert rows(s, "SELECT status FROM runs") == [{"status": "failed"}]


async def test_a_cancelled_run_is_recorded_as_cancelled():
    s = build([Chunk(str(i)) for i in range(1000)])
    task = asyncio.create_task(s.runtime.run_turn(s.session_id, "Count"))
    while MODEL_CHUNK not in s.sink.types():
        await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert rows(s, "SELECT status FROM runs") == [{"status": "cancelled"}]


async def test_a_tool_call_is_recorded_with_its_result_and_duration():
    tool = FakeTool(name="read_file", risk="read", result=ToolResult("print(1)"))
    s = build([ToolCall("read_file", {"path": "a.py"})], [Chunk("ok")], tools=[tool])
    await s.runtime.run_turn(s.session_id, "Hi")
    [call] = rows(s, "SELECT * FROM tool_calls")
    assert (call["tool"], call["risk"], call["status"], call["approved"]) == (
        "read_file",
        "read",
        "ok",
        None,
    )
    assert json.loads(call["args_json"]) == {"path": "a.py"}
    assert json.loads(call["result_json"]) == {"content": "print(1)", "is_error": False}
    assert call["duration_ms"] >= 0


async def test_an_approved_tool_call_is_recorded_as_approved():
    tool = FakeTool(name="write_file", risk="write")
    s = build([ToolCall("write_file", {})], [Chunk("ok")], tools=[tool], approvals=[True])
    await s.runtime.run_turn(s.session_id, "Hi")
    [call] = rows(s, "SELECT risk, approved, status FROM tool_calls")
    assert call == {"risk": "write", "approved": 1, "status": "ok"}


async def test_a_refused_tool_call_is_recorded_as_refused():
    tool = FakeTool(name="write_file", risk="write")
    s = build([ToolCall("write_file", {})], [Chunk("ok")], tools=[tool], approvals=[False])
    await s.runtime.run_turn(s.session_id, "Hi")
    [call] = rows(s, "SELECT approved, status FROM tool_calls")
    assert call == {"approved": 0, "status": "refused"}


async def test_a_denied_tool_call_is_recorded_as_denied():
    tool = FakeTool(name="run_cmd", risk="exec")
    s = build([ToolCall("run_cmd", {})], [Chunk("ok")], tools=[tool])
    await s.runtime.run_turn(s.session_id, "Hi")
    assert rows(s, "SELECT status FROM tool_calls") == [{"status": "denied"}]


async def test_a_failing_tool_call_is_recorded_as_failed():
    tool = FakeTool(name="read_file", error=RuntimeError("disk full"))
    s = build([ToolCall("read_file", {})], [Chunk("ok")], tools=[tool])
    await s.runtime.run_turn(s.session_id, "Hi")
    [call] = rows(s, "SELECT status, result_json FROM tool_calls")
    assert call["status"] == "failed"
    assert json.loads(call["result_json"]) == {"error": "RuntimeError: disk full"}


def test_a_payload_that_is_not_json_serializable_is_still_recorded():
    db = connect(":memory:")
    SqliteEventJournal(db).emit(make_event(MODEL_CHUNK, "r1", {"text": object()}))
    assert db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
