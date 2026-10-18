import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from sensai.core.config import RuntimeConfig
from sensai.core.errors import ModelConnectionError
from sensai.core.fakes import (
    CollectingEventSink,
    FakeModelGateway,
    FakeTool,
    FakeUserChannel,
    InMemorySessionStore,
)
from sensai.core.runtime import AgentRuntime, RuntimeDeps
from sensai.core.types import Chunk, Done, ToolCall, ToolResult, ToolSpec, Usage
from sensai.safety.policy import ApprovalPolicy
from sensai.tools.registry import ToolRegistry

LIMITS = {"max_steps": 8, "max_tool_calls": 8, "timeout_s": 5, "max_tokens": 1000}
RULES = {"read": "allow", "write": "ask", "exec": "ask"}
READ_CALL = ToolCall("read_file", {"path": "a.py"})


@dataclass
class Setup:
    runtime: AgentRuntime
    gateway: Any
    store: InMemorySessionStore
    sink: CollectingEventSink
    channel: FakeUserChannel
    session_id: str


def build(
    *scripts,
    gateway=None,
    tools=(),
    rules=None,
    approvals=(),
    system_prompt=None,
    react=None,
    **limits,
) -> Setup:
    gateway = gateway if gateway is not None else FakeModelGateway(*scripts)
    store = InMemorySessionStore()
    sink = CollectingEventSink()
    channel = FakeUserChannel(approvals=approvals)
    deps = RuntimeDeps(
        gateway=gateway,
        store=store,
        sink=sink,
        channel=channel,
        registry=ToolRegistry(tools),
        policy=ApprovalPolicy({**RULES, **(rules or {})}),
    )
    config = RuntimeConfig(**{**LIMITS, **limits})
    runtime = AgentRuntime(deps, config, "test-model", system_prompt, react=react)
    return Setup(runtime, gateway, store, sink, channel, store.create_session())


def events_of(setup, type_):
    return [event for event in setup.sink.events if event.type == type_]


def saved(setup):
    return [(m.role, m.content) for m in setup.store.history(setup.session_id)]


async def wait_for_event(sink, type_):
    async def poll():
        while type_ not in sink.types():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=2)


class HangingTool:
    name = "hang"
    description = "Never returns."
    parameters = {"type": "object", "properties": {}}
    risk = "read"

    async def run(self, args, ctx):
        await asyncio.sleep(60)
        return ToolResult("never")


class SlowGateway:
    async def stream_chat(self, messages, tools=None, schema=None):
        await asyncio.sleep(60)
        yield Done()


class BrokenGateway:
    async def stream_chat(self, messages, tools=None, schema=None):
        raise RuntimeError("bug")
        yield Done()


async def test_text_turn_emits_events_and_saves_the_exchange():
    s = build([Chunk("Hel"), Chunk("lo"), Usage(5, 2)])
    result = await s.runtime.run_turn(s.session_id, "Hi")
    assert (result.status, result.answer, result.error) == ("completed", "Hello", None)
    assert s.sink.types() == [
        "run.started",
        "model.chunk",
        "model.chunk",
        "model.usage",
        "run.completed",
    ]
    assert saved(s) == [("user", "Hi"), ("assistant", "Hello")]


async def test_events_share_the_run_id_and_carry_their_payload():
    s = build([Chunk("Hello"), Usage(5, 2)])
    result = await s.runtime.run_turn(s.session_id, "Hi")
    assert {event.run_id for event in s.sink.events} == {result.run_id}
    assert events_of(s, "run.started")[0].payload == {
        "session_id": s.session_id,
        "model": "test-model",
    }
    assert events_of(s, "model.chunk")[0].payload == {"text": "Hello"}
    assert events_of(s, "model.usage")[0].payload == {"prompt_tokens": 5, "completion_tokens": 2}
    assert events_of(s, "run.completed")[0].payload == {
        "session_id": s.session_id,
        "steps": 1,
        "tool_calls": 0,
    }


async def test_previous_turns_are_sent_to_the_model():
    s = build([Chunk("one")], [Chunk("two")])
    await s.runtime.run_turn(s.session_id, "first")
    await s.runtime.run_turn(s.session_id, "second")
    sent = [(m.role, m.content) for m in s.gateway.calls[1].messages]
    assert sent == [("user", "first"), ("assistant", "one"), ("user", "second")]


async def test_system_prompt_is_sent_but_not_saved():
    s = build([Chunk("ok")], system_prompt="Be brief.")
    await s.runtime.run_turn(s.session_id, "Hi")
    assert [(m.role, m.content) for m in s.gateway.calls[0].messages] == [
        ("system", "Be brief."),
        ("user", "Hi"),
    ]
    assert saved(s) == [("user", "Hi"), ("assistant", "ok")]


async def test_tools_are_offered_to_the_model_only_when_registered():
    without = build([Chunk("ok")])
    await without.runtime.run_turn(without.session_id, "Hi")
    assert without.gateway.calls[0].tools is None

    tool = FakeTool(name="read_file", description="Read.")
    with_tool = build([Chunk("ok")], tools=[tool])
    await with_tool.runtime.run_turn(with_tool.session_id, "Hi")
    assert with_tool.gateway.calls[0].tools == [ToolSpec("read_file", "Read.", tool.parameters)]


async def test_a_read_tool_runs_without_approval_and_its_result_reaches_the_model():
    tool = FakeTool(name="read_file", risk="read", result=ToolResult("print(1)"))
    s = build(
        [READ_CALL, Usage(3, 1)],
        [Chunk("It prints 1."), Usage(10, 4)],
        tools=[tool],
    )
    result = await s.runtime.run_turn(s.session_id, "What does a.py do?")

    assert result.answer == "It prints 1."
    [(args, ctx)] = tool.calls
    assert args == {"path": "a.py"}
    assert (ctx.run_id, ctx.session_id) == (result.run_id, s.session_id)
    assert s.channel.confirm_requests == []
    assert s.sink.types() == [
        "run.started",
        "model.tool_call",
        "model.usage",
        "tool.started",
        "tool.finished",
        "model.chunk",
        "model.usage",
        "run.completed",
    ]
    second_call = s.gateway.calls[1].messages
    assert [m.role for m in second_call] == ["user", "assistant", "tool"]
    assert second_call[1].tool_calls == (READ_CALL,)
    assert (second_call[2].content, second_call[2].tool_name) == ("print(1)", "read_file")
    roles = [m.role for m in s.store.history(s.session_id)]
    assert roles == ["user", "assistant", "tool", "assistant"]


async def test_tool_events_share_the_call_id():
    tool = FakeTool(name="read_file")
    s = build([READ_CALL], [Chunk("done")], tools=[tool])
    await s.runtime.run_turn(s.session_id, "Hi")
    call_id = events_of(s, "model.tool_call")[0].payload["call_id"]
    assert events_of(s, "tool.started")[0].payload["call_id"] == call_id
    assert events_of(s, "tool.finished")[0].payload["call_id"] == call_id
    assert events_of(s, "tool.finished")[0].payload["duration_ms"] >= 0


async def test_a_write_tool_runs_after_the_user_approves():
    tool = FakeTool(name="write_file", risk="write")
    call = ToolCall("write_file", {"path": "a.py"})
    s = build([call], [Chunk("written")], tools=[tool], approvals=[True])
    await s.runtime.run_turn(s.session_id, "Write a.py")

    [request] = s.channel.confirm_requests
    assert (request.tool, request.risk, request.args) == ("write_file", "write", {"path": "a.py"})
    assert request.preview == 'write_file({"path": "a.py"})'
    assert len(tool.calls) == 1
    assert events_of(s, "approval.requested")[0].payload["preview"] == request.preview
    assert events_of(s, "approval.resolved")[0].payload["approved"] is True


async def test_a_refused_action_is_not_executed_and_the_model_is_told():
    tool = FakeTool(name="write_file", risk="write")
    call = ToolCall("write_file", {"path": "a.py"})
    s = build([call], [Chunk("understood")], tools=[tool], approvals=[False])
    result = await s.runtime.run_turn(s.session_id, "Write a.py")

    assert result.status == "completed"
    assert tool.calls == []
    assert events_of(s, "approval.resolved")[0].payload["approved"] is False
    assert "tool.started" not in s.sink.types()
    assert s.gateway.calls[1].messages[-1].content == "The user refused this action."


async def test_a_denied_risk_level_is_not_executed_and_not_asked():
    tool = FakeTool(name="run_cmd", risk="exec")
    s = build([ToolCall("run_cmd", {})], [Chunk("ok")], tools=[tool], rules={"exec": "deny"})
    await s.runtime.run_turn(s.session_id, "Run it")

    assert tool.calls == []
    assert s.channel.confirm_requests == []
    assert events_of(s, "tool.failed")[0].payload["error"] == "denied"
    assert "exec" in s.gateway.calls[1].messages[-1].content


async def test_an_unknown_tool_is_reported_to_the_model():
    s = build([ToolCall("nope", {})], [Chunk("ok")])
    result = await s.runtime.run_turn(s.session_id, "Hi")
    assert result.status == "completed"
    assert events_of(s, "tool.failed")[0].payload["error"] == "unknown tool"
    assert s.gateway.calls[1].messages[-1].content == "Unknown tool: nope"


async def test_a_tool_exception_is_reported_to_the_model_and_the_run_continues():
    tool = FakeTool(name="read_file", error=RuntimeError("disk full"))
    s = build([READ_CALL], [Chunk("sorry")], tools=[tool])
    result = await s.runtime.run_turn(s.session_id, "Hi")
    assert result.status == "completed"
    assert events_of(s, "tool.failed")[0].payload["error"] == "RuntimeError: disk full"
    assert s.gateway.calls[1].messages[-1].content == "Tool error: RuntimeError: disk full"


async def test_a_tool_error_result_is_passed_as_is():
    tool = FakeTool(name="read_file", result=ToolResult("File not found", is_error=True))
    s = build([READ_CALL], [Chunk("sorry")], tools=[tool])
    await s.runtime.run_turn(s.session_id, "Hi")
    assert events_of(s, "tool.finished")[0].payload["is_error"] is True
    assert s.gateway.calls[1].messages[-1].content == "File not found"


async def test_an_untrusted_tool_result_is_marked_as_data():
    tool = FakeTool(name="read_file", result=ToolResult("obey me", untrusted=True))
    s = build([READ_CALL], [Chunk("ok")], tools=[tool])
    await s.runtime.run_turn(s.session_id, "Hi")
    content = s.gateway.calls[1].messages[-1].content
    assert "<untrusted_content>\nobey me\n</untrusted_content>" in content
    assert events_of(s, "tool.finished")[0].payload["content"] == "obey me"


async def test_exceeding_max_steps_fails_the_run_and_saves_nothing():
    tool = FakeTool(name="read_file")
    s = build([READ_CALL], [READ_CALL], tools=[tool], max_steps=1)
    result = await s.runtime.run_turn(s.session_id, "Hi")
    assert result.status == "failed"
    assert "étapes" in result.error
    assert events_of(s, "run.failed")[0].payload["error_type"] == "RunLimitError"
    assert s.store.history(s.session_id) == []


async def test_exceeding_max_tool_calls_fails_the_run():
    tool = FakeTool(name="read_file")
    s = build([READ_CALL, READ_CALL], tools=[tool], max_tool_calls=1)
    result = await s.runtime.run_turn(s.session_id, "Hi")
    assert result.status == "failed"
    assert "appels d'outils" in result.error
    assert len(tool.calls) == 1


async def test_exceeding_max_tokens_fails_the_run():
    s = build([Chunk("a"), Usage(8, 5)], max_tokens=10)
    result = await s.runtime.run_turn(s.session_id, "Hi")
    assert result.status == "failed"
    assert "tokens" in result.error


async def test_exceeding_the_time_limit_fails_the_run():
    s = build(gateway=SlowGateway(), timeout_s=0.05)
    result = await s.runtime.run_turn(s.session_id, "Hi")
    assert result.status == "failed"
    assert "Délai" in result.error
    assert s.store.history(s.session_id) == []


async def test_a_model_error_fails_the_run_without_raising():
    s = build(gateway=FakeModelGateway(error=ModelConnectionError("Ollama is down")))
    result = await s.runtime.run_turn(s.session_id, "Hi")
    assert (result.status, result.error) == ("failed", "Ollama is down")
    assert s.sink.types() == ["run.started", "run.failed"]
    assert events_of(s, "run.failed")[0].payload["error_type"] == "ModelConnectionError"
    assert s.store.history(s.session_id) == []


async def test_an_unexpected_exception_is_reported_then_raised():
    s = build(gateway=BrokenGateway())
    with pytest.raises(RuntimeError, match="bug"):
        await s.runtime.run_turn(s.session_id, "Hi")
    assert events_of(s, "run.failed")[0].payload["error_type"] == "RuntimeError"


async def test_cancelling_keeps_the_partial_answer_as_interrupted():
    s = build([Chunk(f"{i} ") for i in range(1000)])
    task = asyncio.create_task(s.runtime.run_turn(s.session_id, "Count"))
    while len(events_of(s, "model.chunk")) < 3:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    streamed = "".join(event.payload["text"] for event in events_of(s, "model.chunk"))
    history = s.store.history(s.session_id)
    statuses = [(m.role, m.status) for m in history]
    assert statuses == [("user", "complete"), ("assistant", "interrupted")]
    assert history[1].content == streamed
    assert s.sink.types()[-1] == "run.cancelled"
    assert events_of(s, "run.cancelled")[0].payload["partial"] is True


async def test_cancelling_during_a_tool_drops_the_unfinished_iteration():
    s = build([ToolCall("hang", {})], tools=[HangingTool()])
    task = asyncio.create_task(s.runtime.run_turn(s.session_id, "Hi"))
    await wait_for_event(s.sink, "tool.started")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert saved(s) == [("user", "Hi")]
    assert events_of(s, "run.cancelled")[0].payload["partial"] is False


async def test_hooks_can_change_the_question_and_the_answer():
    s = build([Chunk("secret")])

    async def entry(ctx):
        ctx.user_text = ctx.user_text.upper()

    async def exit_(ctx):
        ctx.answer = ctx.answer.replace("secret", "[hidden]")

    s.runtime.entry_hooks.append(entry)
    s.runtime.exit_hooks.append(exit_)
    result = await s.runtime.run_turn(s.session_id, "hi")

    assert s.gateway.calls[0].messages[-1].content == "HI"
    assert result.answer == "[hidden]"
    assert saved(s) == [("user", "HI"), ("assistant", "[hidden]")]


async def test_hooks_run_in_registration_order():
    s = build([Chunk("ok")])
    order = []

    def hook(name):
        async def run(ctx):
            order.append(name)

        return run

    s.runtime.entry_hooks.extend([hook("entry 1"), hook("entry 2")])
    s.runtime.exit_hooks.extend([hook("exit 1"), hook("exit 2")])
    await s.runtime.run_turn(s.session_id, "hi")
    assert order == ["entry 1", "entry 2", "exit 1", "exit 2"]


PATH_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}


@pytest.mark.feature("T5")
async def test_invalid_tool_arguments_are_sent_back_to_the_model_without_running():
    tool = FakeTool(name="read_file", parameters=PATH_SCHEMA)
    bad_call = ToolCall("read_file", {"file": "a.py"})
    s = build([bad_call], [READ_CALL], [Chunk("done")], tools=[tool])
    result = await s.runtime.run_turn(s.session_id, "Read a.py")

    assert result.status == "completed"
    assert [args for args, _ in tool.calls] == [{"path": "a.py"}]
    failed = events_of(s, "tool.failed")[0].payload
    assert (failed["tool"], failed["error"]) == ("read_file", "invalid arguments")
    feedback = s.gateway.calls[1].messages[-1].content
    assert feedback.startswith("Invalid arguments for read_file:")
    assert "'path' is a required property" in feedback
    assert "Fix them and call the tool again." in feedback


@pytest.mark.feature("T5")
async def test_invalid_arguments_are_checked_before_asking_the_user():
    tool = FakeTool(name="write_file", risk="write", parameters=PATH_SCHEMA)
    s = build([ToolCall("write_file", {"path": 3})], [Chunk("ok")], tools=[tool])
    await s.runtime.run_turn(s.session_id, "Write")
    assert s.channel.confirm_requests == []
    assert tool.calls == []


A3 = pytest.mark.feature("A3")
WRITE_CALL = ToolCall("write_file", {"path": "a.py"})
GENERIC_PREVIEW = 'write_file({"path": "a.py"})'


class PreviewTool(FakeTool):
    def __init__(self, text="DIFF", error=None, risk="write"):
        super().__init__(name="write_file", risk=risk)
        self._text = text
        self._preview_error = error
        self.preview_calls = []

    async def preview(self, args):
        self.preview_calls.append(args)
        if self._preview_error is not None:
            raise self._preview_error
        return self._text


class HangingPreviewTool(PreviewTool):
    async def preview(self, args):
        self.preview_calls.append(args)
        await asyncio.Event().wait()


@A3
async def test_the_approval_request_carries_the_preview_of_the_tool():
    tool = PreviewTool("DIFF")
    s = build([WRITE_CALL], [Chunk("written")], tools=[tool], approvals=[True])
    await s.runtime.run_turn(s.session_id, "Write a.py")

    [request] = s.channel.confirm_requests
    assert request.preview == "DIFF"
    assert events_of(s, "approval.requested")[0].payload["preview"] == "DIFF"
    assert tool.preview_calls == [{"path": "a.py"}]
    assert len(tool.calls) == 1


@A3
@pytest.mark.parametrize("text", [None, ""])
async def test_an_empty_preview_falls_back_to_the_generic_one(text):
    tool = PreviewTool(text)
    s = build([WRITE_CALL], [Chunk("written")], tools=[tool], approvals=[True])
    await s.runtime.run_turn(s.session_id, "Write a.py")

    [request] = s.channel.confirm_requests
    assert request.preview == GENERIC_PREVIEW
    assert events_of(s, "approval.requested")[0].payload["preview"] == GENERIC_PREVIEW


@A3
async def test_a_failing_preview_falls_back_and_does_not_stop_the_turn():
    tool = PreviewTool(error=RuntimeError("boom"))
    s = build([WRITE_CALL], [Chunk("written")], tools=[tool], approvals=[True])
    result = await s.runtime.run_turn(s.session_id, "Write a.py")

    assert result.status == "completed"
    [request] = s.channel.confirm_requests
    assert request.preview == GENERIC_PREVIEW
    assert len(tool.calls) == 1
    assert "run.failed" not in s.sink.types()


@A3
async def test_a_tool_without_preview_keeps_the_generic_preview():
    tool = FakeTool(name="write_file", risk="write")
    s = build([WRITE_CALL], [Chunk("written")], tools=[tool], approvals=[True])
    await s.runtime.run_turn(s.session_id, "Write a.py")

    [request] = s.channel.confirm_requests
    assert request.preview == GENERIC_PREVIEW
    assert events_of(s, "approval.requested")[0].payload["preview"] == GENERIC_PREVIEW


@A3
async def test_the_preview_is_shown_even_when_the_user_refuses():
    tool = PreviewTool("DIFF")
    s = build([WRITE_CALL], [Chunk("understood")], tools=[tool], approvals=[False])
    await s.runtime.run_turn(s.session_id, "Write a.py")

    assert s.channel.confirm_requests[0].preview == "DIFF"
    assert events_of(s, "approval.resolved")[0].payload["approved"] is False
    assert tool.calls == []


@A3
async def test_the_preview_is_not_built_when_no_approval_is_needed():
    tool = PreviewTool("DIFF", risk="read")
    s = build([WRITE_CALL], [Chunk("ok")], tools=[tool])
    await s.runtime.run_turn(s.session_id, "Read a.py")

    assert tool.preview_calls == []
    assert "approval.requested" not in s.sink.types()
    assert len(tool.calls) == 1


@A3
async def test_the_preview_is_not_built_when_the_policy_denies_the_tool():
    tool = PreviewTool("DIFF")
    s = build([WRITE_CALL], [Chunk("ok")], tools=[tool], rules={"write": "deny"})
    await s.runtime.run_turn(s.session_id, "Write a.py")

    assert tool.preview_calls == []
    assert tool.calls == []


@A3
async def test_the_preview_is_not_built_when_the_arguments_are_invalid():
    tool = PreviewTool("DIFF")
    tool.parameters = PATH_SCHEMA
    s = build([ToolCall("write_file", {"path": 3})], [Chunk("ok")], tools=[tool])
    await s.runtime.run_turn(s.session_id, "Write")

    assert tool.preview_calls == []
    assert s.channel.confirm_requests == []


@A3
async def test_the_preview_is_built_once_per_call():
    tool = PreviewTool("DIFF")
    s = build([WRITE_CALL], [Chunk("written")], tools=[tool], approvals=[True])
    await s.runtime.run_turn(s.session_id, "Write a.py")

    assert len(tool.preview_calls) == 1


@A3
async def test_cancelling_during_the_preview_cancels_the_run():
    tool = HangingPreviewTool()
    s = build([WRITE_CALL], tools=[tool], approvals=[True])
    task = asyncio.create_task(s.runtime.run_turn(s.session_id, "Write a.py"))
    while not tool.preview_calls:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert s.sink.types()[-1] == "run.cancelled"
    assert "approval.requested" not in s.sink.types()
    assert tool.calls == []
