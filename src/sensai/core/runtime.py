"""AgentRuntime: one conversation turn, run as a state machine.

    RECEIVE -> ENTRY_HOOKS -> BUILD_CONTEXT -> MODEL_STREAM
        -> ANSWER | TOOL_CALL -> POLICY -> APPROVAL? -> EXECUTE -> MODEL_STREAM
    -> EXIT_HOOKS -> PERSIST -> completed | cancelled | failed

By default a turn makes a single model call. Features that need more (guardrails, summaries,
caches) plug into `entry_hooks` / `exit_hooks` without changing the loop.

Event payloads (all events also carry `run_id`):

- run.started: session_id, model
- model.chunk: text
- model.tool_call: call_id, name, args
- model.usage: prompt_tokens, completion_tokens
- approval.requested: call_id, tool, risk, preview
- approval.resolved: call_id, tool, approved
- tool.started: call_id, tool, risk, args
- tool.finished: call_id, tool, is_error, duration_ms, content
- tool.failed: call_id, tool, error
- run.completed: session_id, steps, tool_calls
- run.cancelled: session_id, partial
- run.failed: session_id, error_type, error
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from sensai.core.config import RuntimeConfig
from sensai.core.contracts import EventSink, ModelGateway, SessionStore, Tool, UserChannel
from sensai.core.errors import RunLimitError, SensaiError
from sensai.core.events import (
    APPROVAL_REQUESTED,
    APPROVAL_RESOLVED,
    MODEL_CHUNK,
    MODEL_TOOL_CALL,
    MODEL_USAGE,
    RUN_CANCELLED,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_STARTED,
    TOOL_FAILED,
    TOOL_FINISHED,
    TOOL_STARTED,
    make_event,
)
from sensai.core.types import (
    ApprovalRequest,
    Chunk,
    Message,
    ToolCall,
    ToolContext,
    ToolResult,
    ToolSpec,
    Usage,
    new_id,
)
from sensai.safety.policy import ApprovalPolicy, preview_call
from sensai.safety.untrusted import wrap_untrusted
from sensai.tools.registry import ToolRegistry

RunStatus = Literal["completed", "cancelled", "failed"]


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: RunStatus
    answer: str = ""
    error: str | None = None


@dataclass
class RunContext:
    """What hooks see and may change. `answer` is only set once the model has answered."""

    run_id: str
    session_id: str
    user_text: str
    messages: list[Message] = field(default_factory=list)
    answer: str = ""


Hook = Callable[[RunContext], Awaitable[None]]


@dataclass(frozen=True)
class RuntimeDeps:
    """The collaborators of the runtime, built by the caller."""

    gateway: ModelGateway
    store: SessionStore
    sink: EventSink
    channel: UserChannel
    registry: ToolRegistry
    policy: ApprovalPolicy


@dataclass
class _RunState:
    new_messages: list[Message] = field(default_factory=list)
    partial_text: str = ""
    steps: int = 0
    tool_calls: int = 0
    tokens: int = 0


class AgentRuntime:
    def __init__(
        self,
        deps: RuntimeDeps,
        config: RuntimeConfig,
        model_name: str,
        system_prompt: str | None = None,
    ) -> None:
        self._deps = deps
        self._config = config
        self._model_name = model_name
        self._system_prompt = system_prompt
        self.entry_hooks: list[Hook] = []
        self.exit_hooks: list[Hook] = []

    async def run_turn(self, session_id: str, user_text: str) -> RunResult:
        """Run one turn. Expected failures are returned; cancellation is re-raised.

        Completed and cancelled turns are saved to the store, failed ones are not.
        """
        run_id = new_id()
        state = _RunState()
        self._emit(RUN_STARTED, run_id, session_id=session_id, model=self._model_name)
        try:
            answer = await self._run_and_save(run_id, session_id, user_text, state)
        except asyncio.CancelledError:
            self._cancel(run_id, session_id, state)
            raise
        except asyncio.TimeoutError:
            return self._fail(run_id, session_id, self._timeout_error())
        except SensaiError as exc:
            return self._fail(run_id, session_id, exc)
        except Exception as exc:
            self._emit_failed(run_id, session_id, exc)
            raise
        return self._complete(run_id, session_id, state, answer)

    async def _run_and_save(
        self, run_id: str, session_id: str, user_text: str, state: _RunState
    ) -> str:
        turn = self._execute(run_id, session_id, user_text, state)
        answer = await asyncio.wait_for(turn, self._config.timeout_s)
        self._persist(session_id, state.new_messages)
        return answer

    async def _execute(self, run_id: str, session_id: str, user_text: str, state: _RunState) -> str:
        ctx = RunContext(run_id, session_id, user_text)
        await _run_hooks(self.entry_hooks, ctx)
        self._build_context(ctx, state)
        assistant = await self._model_loop(ctx, state)
        ctx.answer = assistant.content
        await _run_hooks(self.exit_hooks, ctx)
        state.new_messages.append(replace(assistant, content=ctx.answer))
        return ctx.answer

    def _build_context(self, ctx: RunContext, state: _RunState) -> None:
        prefix = [Message("system", self._system_prompt)] if self._system_prompt else []
        user_message = Message("user", ctx.user_text)
        history = self._deps.store.history(ctx.session_id)
        ctx.messages = [*prefix, *history, user_message]
        state.new_messages.append(user_message)

    async def _model_loop(self, ctx: RunContext, state: _RunState) -> Message:
        """Call the model until it answers without asking for a tool."""
        specs = self._deps.registry.specs() or None
        while True:
            self._count_step(state)
            assistant, call_ids = await self._stream_model(ctx, state, specs)
            if not assistant.tool_calls:
                return assistant
            results = await self._run_tools(ctx, state, assistant, call_ids)
            # An iteration is kept only once all its tool results exist.
            ctx.messages.extend([assistant, *results])
            state.new_messages.extend([assistant, *results])

    async def _stream_model(
        self, ctx: RunContext, state: _RunState, specs: list[ToolSpec] | None
    ) -> tuple[Message, list[str]]:
        calls: list[ToolCall] = []
        call_ids: list[str] = []
        state.partial_text = ""
        async for event in self._deps.gateway.stream_chat(ctx.messages, tools=specs):
            if isinstance(event, Chunk):
                self._on_chunk(ctx, state, event)
            elif isinstance(event, ToolCall):
                calls.append(event)
                call_ids.append(self._on_tool_call(ctx, event))
            elif isinstance(event, Usage):
                self._on_usage(ctx, state, event)
        text, state.partial_text = state.partial_text, ""
        return Message("assistant", text, tool_calls=tuple(calls)), call_ids

    def _on_chunk(self, ctx: RunContext, state: _RunState, chunk: Chunk) -> None:
        state.partial_text += chunk.text
        self._emit(MODEL_CHUNK, ctx.run_id, text=chunk.text)

    def _on_tool_call(self, ctx: RunContext, call: ToolCall) -> str:
        call_id = new_id()
        self._emit(MODEL_TOOL_CALL, ctx.run_id, call_id=call_id, name=call.name, args=call.args)
        return call_id

    def _on_usage(self, ctx: RunContext, state: _RunState, usage: Usage) -> None:
        state.tokens += usage.prompt_tokens + usage.completion_tokens
        self._emit(
            MODEL_USAGE,
            ctx.run_id,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        if state.tokens > self._config.max_tokens:
            raise _limit_error(self._config.max_tokens, "tokens")

    def _count_step(self, state: _RunState) -> None:
        state.steps += 1
        if state.steps > self._config.max_steps:
            raise _limit_error(self._config.max_steps, "étapes")

    def _count_tool_call(self, state: _RunState) -> None:
        state.tool_calls += 1
        if state.tool_calls > self._config.max_tool_calls:
            raise _limit_error(self._config.max_tool_calls, "appels d'outils")

    async def _run_tools(
        self, ctx: RunContext, state: _RunState, assistant: Message, call_ids: list[str]
    ) -> list[Message]:
        messages = []
        for call_id, call in zip(call_ids, assistant.tool_calls, strict=True):
            self._count_tool_call(state)
            messages.append(await self._run_tool(ctx, call_id, call))
        return messages

    async def _run_tool(self, ctx: RunContext, call_id: str, call: ToolCall) -> Message:
        tool = self._deps.registry.get(call.name)
        if tool is None:
            self._emit_tool(TOOL_FAILED, ctx, call_id, call.name, error="unknown tool")
            result = _error(f"Unknown tool: {call.name}")
        else:
            result = await self._run_allowed(ctx, call_id, tool, call)
        return _tool_message(call.name, result)

    async def _run_allowed(
        self, ctx: RunContext, call_id: str, tool: Tool, call: ToolCall
    ) -> ToolResult:
        refusal = await self._check_policy(ctx, call_id, tool, call)
        if refusal is not None:
            return refusal
        return await self._execute_tool(ctx, call_id, tool, call)

    async def _check_policy(
        self, ctx: RunContext, call_id: str, tool: Tool, call: ToolCall
    ) -> ToolResult | None:
        """Return an error result if the policy or the user stops the call, else None."""
        decision = self._deps.policy.decide(tool)
        if decision == "deny":
            self._emit_tool(TOOL_FAILED, ctx, call_id, tool.name, error="denied")
            return _error(f"The policy does not allow {tool.risk} tools.")
        if decision == "ask" and not await self._ask_user(ctx, call_id, tool, call):
            return _error("The user refused this action.")
        return None

    async def _ask_user(self, ctx: RunContext, call_id: str, tool: Tool, call: ToolCall) -> bool:
        preview = preview_call(tool.name, call.args)
        request = ApprovalRequest(ctx.run_id, tool.name, tool.risk, preview, call.args)
        self._emit_tool(
            APPROVAL_REQUESTED, ctx, call_id, tool.name, risk=tool.risk, preview=preview
        )
        approved = await self._deps.channel.confirm(request)
        self._emit_tool(APPROVAL_RESOLVED, ctx, call_id, tool.name, approved=approved)
        return approved

    async def _execute_tool(
        self, ctx: RunContext, call_id: str, tool: Tool, call: ToolCall
    ) -> ToolResult:
        self._emit_tool(TOOL_STARTED, ctx, call_id, tool.name, risk=tool.risk, args=call.args)
        started = time.perf_counter()
        try:
            result = await tool.run(call.args, ToolContext(ctx.run_id, ctx.session_id))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._emit_tool(TOOL_FAILED, ctx, call_id, tool.name, error=error)
            return _error(f"Tool error: {error}")
        duration_ms = (time.perf_counter() - started) * 1000
        details = {"is_error": result.is_error, "duration_ms": duration_ms}
        self._emit_tool(TOOL_FINISHED, ctx, call_id, tool.name, content=result.content, **details)
        return result

    def _persist(self, session_id: str, messages: list[Message]) -> None:
        history = self._deps.store.history(session_id)
        parent_id = history[-1].id if history else None
        for message in messages:
            parent_id = self._deps.store.append(session_id, message, parent_id)

    def _complete(self, run_id: str, session_id: str, state: _RunState, answer: str) -> RunResult:
        steps, tool_calls = state.steps, state.tool_calls
        self._emit(RUN_COMPLETED, run_id, session_id=session_id, steps=steps, tool_calls=tool_calls)
        return RunResult(run_id, "completed", answer)

    def _cancel(self, run_id: str, session_id: str, state: _RunState) -> None:
        messages = list(state.new_messages)
        if state.partial_text:
            messages.append(Message("assistant", state.partial_text, status="interrupted"))
        if messages:
            self._persist(session_id, messages)
        self._emit(RUN_CANCELLED, run_id, session_id=session_id, partial=bool(state.partial_text))

    def _fail(self, run_id: str, session_id: str, error: Exception) -> RunResult:
        self._emit_failed(run_id, session_id, error)
        return RunResult(run_id, "failed", error=str(error))

    def _emit_failed(self, run_id: str, session_id: str, error: Exception) -> None:
        name = type(error).__name__
        self._emit(RUN_FAILED, run_id, session_id=session_id, error_type=name, error=str(error))

    def _timeout_error(self) -> RunLimitError:
        return RunLimitError(f"Délai dépassé : le tour a dépassé {self._config.timeout_s} s.")

    def _emit_tool(
        self, type_: str, ctx: RunContext, call_id: str, tool_name: str, **payload: Any
    ) -> None:
        self._emit(type_, ctx.run_id, call_id=call_id, tool=tool_name, **payload)

    def _emit(self, type_: str, run_id: str, **payload: Any) -> None:
        self._deps.sink.emit(make_event(type_, run_id, payload))


async def _run_hooks(hooks: list[Hook], ctx: RunContext) -> None:
    for hook in hooks:
        await hook(ctx)


def _limit_error(maximum: int, unit: str) -> RunLimitError:
    return RunLimitError(f"Limite atteinte : {maximum} {unit} maximum par tour.")


def _error(message: str) -> ToolResult:
    return ToolResult(message, is_error=True)


def _tool_message(name: str, result: ToolResult) -> Message:
    content = wrap_untrusted(result.content) if result.untrusted else result.content
    return Message("tool", content, tool_name=name)
