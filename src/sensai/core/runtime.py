"""AgentRuntime: one conversation turn, run as a state machine.

    RECEIVE -> ENTRY_HOOKS -> BUILD_CONTEXT -> MODEL_STREAM
        -> ANSWER | TOOL_CALL -> POLICY -> APPROVAL? -> EXECUTE -> MODEL_STREAM
    -> EXIT_HOOKS -> PERSIST -> completed | cancelled | failed

By default a turn makes a single model call. Features that need more (guardrails, summaries,
caches) plug into `entry_hooks` / `exit_hooks` without changing the loop.

Event payloads (all events also carry `run_id`):

    run.started         session_id, model
    model.chunk         text
    model.tool_call     call_id, name, args
    model.usage         prompt_tokens, completion_tokens
    approval.requested  call_id, tool, risk, preview
    approval.resolved   call_id, approved
    tool.started        call_id, tool, risk, args
    tool.finished       call_id, tool, is_error, duration_ms, content
    tool.failed         call_id, tool, error
    run.completed       session_id, steps, tool_calls
    run.cancelled       session_id, partial
    run.failed          session_id, error_type, error
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from sensai.core.config import RuntimeConfig
from sensai.core.contracts import EventSink, ModelGateway, SessionStore, UserChannel
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
        *,
        gateway: ModelGateway,
        store: SessionStore,
        sink: EventSink,
        channel: UserChannel,
        registry: ToolRegistry,
        policy: ApprovalPolicy,
        config: RuntimeConfig,
        model_name: str,
        system_prompt: str | None = None,
    ) -> None:
        self._gateway = gateway
        self._store = store
        self._sink = sink
        self._channel = channel
        self._registry = registry
        self._policy = policy
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
            answer = await asyncio.wait_for(
                self._execute(run_id, session_id, user_text, state), self._config.timeout_s
            )
            self._persist(session_id, state.new_messages)
        except asyncio.CancelledError:
            self._cancel(run_id, session_id, state)
            raise
        except asyncio.TimeoutError:
            limit = f"Délai dépassé : le tour a dépassé {self._config.timeout_s} s."
            return self._fail(run_id, session_id, RunLimitError(limit))
        except SensaiError as exc:
            return self._fail(run_id, session_id, exc)
        except Exception as exc:
            self._emit_failed(run_id, session_id, exc)
            raise
        self._emit(
            RUN_COMPLETED,
            run_id,
            session_id=session_id,
            steps=state.steps,
            tool_calls=state.tool_calls,
        )
        return RunResult(run_id, "completed", answer)

    async def _execute(self, run_id: str, session_id: str, user_text: str, state: _RunState) -> str:
        ctx = RunContext(run_id, session_id, user_text)
        for hook in self.entry_hooks:
            await hook(ctx)

        prefix = [Message("system", self._system_prompt)] if self._system_prompt else []
        user_message = Message("user", ctx.user_text)
        ctx.messages = [*prefix, *self._store.history(session_id), user_message]
        state.new_messages.append(user_message)
        specs = self._registry.specs() or None

        while True:
            state.steps += 1
            if state.steps > self._config.max_steps:
                raise RunLimitError(
                    f"Limite atteinte : {self._config.max_steps} étapes maximum par tour."
                )
            assistant, call_ids = await self._stream_model(ctx, state, specs)
            if not assistant.tool_calls:
                break
            tool_messages = []
            for call_id, call in zip(call_ids, assistant.tool_calls, strict=True):
                state.tool_calls += 1
                if state.tool_calls > self._config.max_tool_calls:
                    raise RunLimitError(
                        f"Limite atteinte : {self._config.max_tool_calls} appels d'outils "
                        "maximum par tour."
                    )
                tool_messages.append(await self._run_tool(ctx, call_id, call))
            # An iteration is saved only once all its tool results exist.
            ctx.messages.extend([assistant, *tool_messages])
            state.new_messages.extend([assistant, *tool_messages])

        ctx.answer = assistant.content
        for hook in self.exit_hooks:
            await hook(ctx)
        state.new_messages.append(replace(assistant, content=ctx.answer))
        return ctx.answer

    async def _stream_model(
        self, ctx: RunContext, state: _RunState, specs: list[ToolSpec] | None
    ) -> tuple[Message, list[str]]:
        calls: list[ToolCall] = []
        call_ids: list[str] = []
        state.partial_text = ""
        async for event in self._gateway.stream_chat(ctx.messages, tools=specs):
            if isinstance(event, Chunk):
                state.partial_text += event.text
                self._emit(MODEL_CHUNK, ctx.run_id, text=event.text)
            elif isinstance(event, ToolCall):
                call_id = new_id()
                calls.append(event)
                call_ids.append(call_id)
                self._emit(
                    MODEL_TOOL_CALL, ctx.run_id, call_id=call_id, name=event.name, args=event.args
                )
            elif isinstance(event, Usage):
                state.tokens += event.prompt_tokens + event.completion_tokens
                self._emit(
                    MODEL_USAGE,
                    ctx.run_id,
                    prompt_tokens=event.prompt_tokens,
                    completion_tokens=event.completion_tokens,
                )
                if state.tokens > self._config.max_tokens:
                    raise RunLimitError(
                        f"Limite atteinte : {self._config.max_tokens} tokens maximum par tour."
                    )
        text, state.partial_text = state.partial_text, ""
        return Message("assistant", text, tool_calls=tuple(calls)), call_ids

    async def _run_tool(self, ctx: RunContext, call_id: str, call: ToolCall) -> Message:
        tool = self._registry.get(call.name)
        if tool is None:
            self._emit(
                TOOL_FAILED, ctx.run_id, call_id=call_id, tool=call.name, error="unknown tool"
            )
            return _tool_message(call.name, ToolResult(f"Unknown tool: {call.name}", is_error=True))

        decision = self._policy.decide(tool)
        if decision == "deny":
            self._emit(TOOL_FAILED, ctx.run_id, call_id=call_id, tool=tool.name, error="denied")
            denied = ToolResult(f"The policy does not allow {tool.risk} tools.", is_error=True)
            return _tool_message(tool.name, denied)
        if decision == "ask":
            request = ApprovalRequest(
                ctx.run_id, tool.name, tool.risk, preview_call(tool.name, call.args), call.args
            )
            self._emit(
                APPROVAL_REQUESTED,
                ctx.run_id,
                call_id=call_id,
                tool=tool.name,
                risk=tool.risk,
                preview=request.preview,
            )
            approved = await self._channel.confirm(request)
            self._emit(APPROVAL_RESOLVED, ctx.run_id, call_id=call_id, approved=approved)
            if not approved:
                refused = ToolResult("The user refused this action.", is_error=True)
                return _tool_message(tool.name, refused)

        self._emit(
            TOOL_STARTED,
            ctx.run_id,
            call_id=call_id,
            tool=tool.name,
            risk=tool.risk,
            args=call.args,
        )
        started = time.perf_counter()
        try:
            result = await tool.run(call.args, ToolContext(ctx.run_id, ctx.session_id))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._emit(TOOL_FAILED, ctx.run_id, call_id=call_id, tool=tool.name, error=error)
            return _tool_message(tool.name, ToolResult(f"Tool error: {error}", is_error=True))
        self._emit(
            TOOL_FINISHED,
            ctx.run_id,
            call_id=call_id,
            tool=tool.name,
            is_error=result.is_error,
            duration_ms=(time.perf_counter() - started) * 1000,
            content=result.content,
        )
        return _tool_message(tool.name, result)

    def _persist(self, session_id: str, messages: list[Message]) -> None:
        history = self._store.history(session_id)
        parent_id = history[-1].id if history else None
        for message in messages:
            parent_id = self._store.append(session_id, message, parent_id)

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
        self._emit(
            RUN_FAILED,
            run_id,
            session_id=session_id,
            error_type=type(error).__name__,
            error=str(error),
        )

    def _emit(self, type_: str, run_id: str, **payload: Any) -> None:
        self._sink.emit(make_event(type_, run_id, payload))


def _tool_message(name: str, result: ToolResult) -> Message:
    content = wrap_untrusted(result.content) if result.untrusted else result.content
    return Message("tool", content, tool_name=name)
