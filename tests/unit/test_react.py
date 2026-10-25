"""A1: bounded ReAct loop (plan, action, observation)."""

import json

import pytest
from tests.unit.test_cli import chat_setup
from tests.unit.test_runtime import READ_CALL, build, events_of, saved

from sensai.cli.chat import run_chat
from sensai.core.config import ReactConfig, load_config
from sensai.core.errors import ConfigError
from sensai.core.fakes import FakeModelGateway, FakeTool
from sensai.core.react import plan_schema
from sensai.core.schema import schema_errors
from sensai.core.types import Chunk, ToolResult, Usage

pytestmark = pytest.mark.feature("A1")

REACT = ReactConfig(enabled=True, max_plan_steps=3)
PLAN = {"goal": "Fix the failing test", "steps": ["Read a.py", "Answer"]}


def plan_script(plan=PLAN, usage=None):
    return [Chunk(json.dumps(plan)), *([usage] if usage else [])]


def react_setup(*scripts, react=REACT, **kwargs):
    return build(*scripts, react=react, **kwargs)


def read_tool(result=None):
    return FakeTool("read_file", result=result or ToolResult("print('a')"))


async def test_a1_plan_comes_first_then_action_and_observation_events():
    s = react_setup(plan_script(), [READ_CALL], [Chunk("Done")], tools=[read_tool()])
    result = await s.runtime.run_turn(s.session_id, "Fix it")
    assert result.status == "completed"
    plan_call, act_call, answer_call = s.gateway.calls
    assert plan_call.schema == plan_schema(3) and plan_call.tools is None
    assert "read_file" in plan_call.messages[-1].content
    assert "1. Read a.py\n2. Answer" in act_call.messages[-1].content
    assert act_call.tools is not None
    assert events_of(s, "react.plan")[0].payload == {"goal": PLAN["goal"], "steps": PLAN["steps"]}
    assert events_of(s, "react.action")[0].payload == {"step": 1, "tools": ["read_file"]}
    assert events_of(s, "react.observation")[0].payload == {
        "step": 1,
        "results": [{"tool": "read_file", "is_error": False}],
    }
    react_types = [t for t in s.sink.types() if t.startswith("react.")]
    assert react_types == ["react.plan", "react.action", "react.observation"]


async def test_a1_plan_is_not_saved_in_the_session():
    s = react_setup(plan_script(), [Chunk("Hello")])
    await s.runtime.run_turn(s.session_id, "Hi")
    assert saved(s) == [("user", "Hi"), ("assistant", "Hello")]


async def test_a1_failed_observation_is_reported_and_the_model_can_retry():
    failing = read_tool(ToolResult("Traceback: AssertionError", is_error=True))
    s = react_setup(plan_script(), [READ_CALL], [READ_CALL], [Chunk("Fixed")], tools=[failing])
    result = await s.runtime.run_turn(s.session_id, "Run the tests")
    assert result.answer == "Fixed"
    observations = [e.payload for e in events_of(s, "react.observation")]
    assert [o["step"] for o in observations] == [1, 2]
    assert observations[0]["results"] == [{"tool": "read_file", "is_error": True}]


async def test_a1_empty_plan_adds_no_instruction():
    s = react_setup(plan_script({"goal": "Say hello", "steps": []}), [Chunk("Hello")])
    await s.runtime.run_turn(s.session_id, "Hi")
    assert [m.role for m in s.gateway.calls[1].messages] == ["user"]
    assert events_of(s, "react.plan")[0].payload["steps"] == []


async def test_a1_invalid_plan_is_skipped_and_the_turn_goes_on():
    too_long = {"goal": "g", "steps": ["a", "b", "c", "d"]}
    s = react_setup(plan_script(too_long), [Chunk("not json")], [Chunk("Hello")])
    result = await s.runtime.run_turn(s.session_id, "Hi")
    assert (result.status, result.answer) == ("completed", "Hello")
    assert "structured.repaired" in s.sink.types()
    assert "structured.failed" in s.sink.types()
    assert events_of(s, "react.plan") == []


async def test_a1_plan_tokens_count_toward_the_turn_budget():
    s = react_setup(plan_script(usage=Usage(900, 200)), [Chunk("Hello")])
    result = await s.runtime.run_turn(s.session_id, "Hi")
    assert result.status == "failed"
    assert events_of(s, "run.failed")[0].payload["error_type"] == "RunLimitError"


async def test_a1_disabled_makes_no_planning_call():
    s = react_setup([Chunk("Hello")], react=ReactConfig(enabled=False, max_plan_steps=3))
    await s.runtime.run_turn(s.session_id, "Hi")
    assert len(s.gateway.calls) == 1
    assert not [t for t in s.sink.types() if t.startswith("react.")]


async def test_a1_steps_stay_bounded_by_max_steps():
    s = react_setup(plan_script(), [READ_CALL], [READ_CALL], tools=[read_tool()], max_steps=1)
    result = await s.runtime.run_turn(s.session_id, "Loop")
    assert result.status == "failed"
    assert len(events_of(s, "react.action")) == 1


def test_a1_plan_schema_bounds_the_plan():
    schema = plan_schema(2)
    assert schema_errors(schema, PLAN) == []
    assert schema_errors(schema, {"goal": "g", "steps": ["a", "b", "c"]})
    assert schema_errors(schema, {"goal": "g", "steps": ["x" * 161]})
    assert schema_errors(schema, {"goal": "g", "steps": [], "thoughts": "long"})


def test_a1_react_is_enabled_by_default():
    react = load_config().react
    assert react.enabled and react.max_plan_steps == 5


@pytest.mark.parametrize("value", [0, 11])
def test_a1_max_plan_steps_is_bounded(value):
    with pytest.raises(ConfigError):
        load_config(overrides={"react": {"max_plan_steps": value}})


async def test_a1_chat_shows_the_plan_and_each_step(tmp_path):
    react = {"enabled": True, "max_plan_steps": 5}
    config, terminal, out, err = chat_setup(tmp_path, "fix it", react=react)
    gateway = FakeModelGateway(plan_script(), [Chunk("Done")])
    assert await run_chat(config, terminal, gateway) == 0
    assert "plan> Fix the failing test" in out.getvalue()
    assert "1. Read a.py" in out.getvalue()
    assert "sensai> Done" in out.getvalue()
    assert err.getvalue() == ""
