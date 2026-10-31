"""AgentRuntime: one conversation turn, run as a state machine.

    RECEIVE -> ENTRY_HOOKS -> BUILD_CONTEXT -> PLAN? -> MODEL_STREAM
        -> ANSWER | TOOL_CALL -> POLICY -> APPROVAL? -> EXECUTE -> MODEL_STREAM
    -> EXIT_HOOKS -> PERSIST -> completed | cancelled | failed

By default a turn makes a single model call; a reply with neither text nor tool call gets one
reminder (EMPTY_ANSWER_REMINDER), then is accepted as is. Features that need more (guardrails,
summaries, caches) plug into `entry_hooks` / `exit_hooks` without changing the loop. With `[react]`
enabled (A1), one structured planning call comes first and every action / observation of the
tool loop is an event (see `sensai.core.react`). With `[questions]` enabled (X4), the model
can call `ask_user`, handled here through the UserChannel (see `sensai.core.questions`).

Event payloads (all events also carry `run_id`):

- run.started: session_id, model, prompt_version (A4: `<name>@<sha>` or None), persona (A5 or
  None)
- prompt.examples: examples (A7: ids of the few-shot examples sent before the conversation)
- model.chunk: text
- model.tool_call: call_id, name, args
- model.usage: prompt_tokens, completion_tokens
- approval.requested: call_id, tool, risk, preview (the diff of the change when the tool provides
  one, else the call as JSON)
- approval.resolved: call_id, tool, approved
- tool.started: call_id, tool, risk, args
- tool.finished: call_id, tool, is_error, duration_ms, content
- tool.failed: call_id, tool, error ("unknown tool", "not allowed" for a risk level outside
  `tool_risks` (A5), "invalid arguments", "denied", "too many questions" or the exception)
- react.plan, react.action, react.observation: see `sensai.core.react`
- question.requested, question.answered: see `sensai.core.questions`
- guard.blocked: stage ("entry" or "tool"), reason, and tool for a tool output
- guard.redacted: stage ("answer" or "tool"), count, and tool for a tool output
- run.completed: session_id, steps, tool_calls
- run.cancelled: session_id, partial
- run.failed: session_id, error_type, error

`recall` (M3) adds the project memory closest to the message as a system message, not saved.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from sensai.core.config import QuestionsConfig, ReactConfig, RuntimeConfig
from sensai.core.contracts import EventSink, ModelGateway, SessionStore, Tool, UserChannel
from sensai.core.errors import GuardBlockedError, RunLimitError, SensaiError, StructuredOutputError
from sensai.core.events import (
    APPROVAL_REQUESTED,
    APPROVAL_RESOLVED,
    GUARD_BLOCKED,
    MODEL_CHUNK,
    MODEL_TOOL_CALL,
    MODEL_USAGE,
    PROMPT_EXAMPLES,
    QUESTION_ANSWERED,
    QUESTION_REQUESTED,
    REACT_ACTION,
    REACT_OBSERVATION,
    REACT_PLAN,
    RUN_CANCELLED,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_STARTED,
    TOOL_FAILED,
    TOOL_FINISHED,
    TOOL_STARTED,
    make_event,
)
from sensai.core.fewshots import FewShots
from sensai.core.questions import (
    ASK_SPEC,
    ASK_TOOL,
    QUESTION_SCHEMA,
    answer_result,
    limit_result,
)
from sensai.core.react import make_plan
from sensai.core.schema import schema_errors
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


EMPTY_ANSWER_REMINDER = "Your last reply was empty. Answer the user now, in text."


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
    notices: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


Hook = Callable[[RunContext], Awaitable[None]]
ResultHook = Callable[[RunContext, str, ToolResult], Awaitable[ToolResult]]


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
    questions: int = 0
    tokens: int = 0


class AgentRuntime:
    def __init__(
        self,
        deps: RuntimeDeps,
        config: RuntimeConfig,
        model_name: str,
        system_prompt: str | None = None,
        react: ReactConfig | None = None,
        max_repairs: int = 1,
        questions: QuestionsConfig | None = None,
    ) -> None:
        self._deps = deps
        self._config = config
        self._react = react if react is not None and react.enabled else None
        self._max_repairs = max_repairs
        self._questions = questions if questions is not None and questions.enabled else None
        self._model_name = model_name
        self.system_prompt = system_prompt
        self.prompt_version: str | None = None
        self.persona: str | None = None  # A5, recorded in run.started
        self.tool_risks: frozenset[str] | None = None  # A5: risk levels offered, None = all
        self.fewshots: FewShots | None = None
        self.recall: Callable[[str], str | None] | None = None  # M3 project memory
        self.entry_hooks: list[Hook] = []
        self.exit_hooks: list[Hook] = []
        self.result_hooks: list[ResultHook] = []

    async def run_turn(self, session_id: str, user_text: str) -> RunResult:
        """Run one turn. Expected failures are returned; cancellation is re-raised.

        Completed and cancelled turns are saved to the store, failed ones are not.
        """
        run_id = new_id()
        state = _RunState()
        self._emit(
            RUN_STARTED,
            run_id,
            session_id=session_id,
            model=self._model_name,
            prompt_version=self.prompt_version,
            persona=self.persona,
        )
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
        await self._run_hooks(self.entry_hooks, ctx)
        self._build_context(ctx, state)
        if self._react is not None:
            await self._plan(ctx, state)
        assistant = await self._model_loop(ctx, state)
        ctx.answer = assistant.content
        await self._run_hooks(self.exit_hooks, ctx)
        state.new_messages.append(replace(assistant, content=ctx.answer))
        return ctx.answer

    def _build_context(self, ctx: RunContext, state: _RunState) -> None:
        prefix = [Message("system", self.system_prompt)] if self.system_prompt else []
        memory = self.recall(ctx.user_text) if self.recall else None
        if memory:
            prefix.append(Message("system", memory))
        examples = self.fewshots.select(ctx.user_text) if self.fewshots else []
        if examples:
            self._emit(PROMPT_EXAMPLES, ctx.run_id, examples=[example.id for example in examples])
        prefix += [message for example in examples for message in example.messages()]
        user_message = Message("user", ctx.user_text)
        history = self._deps.store.history(ctx.session_id)
        ctx.messages = [*prefix, *history, user_message]
        state.new_messages.append(user_message)

    async def _plan(self, ctx: RunContext, state: _RunState) -> None:
        """Ask for a plan and give it to the model. A plan that never conforms is skipped."""
        assert self._react is not None
        names = [spec.name for spec in self._tool_specs()]
        try:
            plan = await make_plan(
                self._deps.gateway,
                ctx.messages,
                names,
                max_steps=self._react.max_plan_steps,
                max_repairs=self._max_repairs,
                on_event=lambda type_, payload: self._emit(type_, ctx.run_id, **payload),
            )
        except StructuredOutputError:
            return  # structured.failed is already logged; the turn goes on without a plan
        for usage in plan.usage:
            self._on_usage(ctx, state, usage)
        self._emit(REACT_PLAN, ctx.run_id, goal=plan.goal, steps=plan.steps)
        message = plan.message()
        if message is not None:
            ctx.messages.append(message)

    async def _model_loop(self, ctx: RunContext, state: _RunState) -> Message:
        """Call the model until it answers without asking for a tool."""
        specs = self._tool_specs() or None
        reminded = False
        while True:
            self._count_step(state)
            assistant, call_ids = await self._stream_model(ctx, state, specs)
            if not assistant.tool_calls:
                if assistant.content.strip() or reminded:
                    return assistant
                # Thinking models (qwen3.5) sometimes stop after their reasoning, with no text.
                reminded = True
                ctx.messages.append(Message("system", EMPTY_ANSWER_REMINDER))
                continue
            self._emit_react(REACT_ACTION, ctx, state, tools=[c.name for c in assistant.tool_calls])
            results = await self._run_tools(ctx, state, assistant, call_ids)
            observed = [{"tool": c.name, "is_error": r.is_error} for c, r in results]
            self._emit_react(REACT_OBSERVATION, ctx, state, results=observed)
            messages = [_tool_message(call.name, result) for call, result in results]
            ctx.messages.extend([assistant, *messages])
            state.new_messages.extend([assistant, *messages])

    def _emit_react(self, type_: str, ctx: RunContext, state: _RunState, **payload: Any) -> None:
        if self._react is not None:
            self._emit(type_, ctx.run_id, step=state.steps, **payload)

    def _tool_specs(self) -> list[ToolSpec]:
        specs = [
            ToolSpec(tool.name, tool.description, tool.parameters)
            for tool in self._deps.registry.tools()
            if self._risk_allowed(tool)
        ]
        return [*specs, ASK_SPEC] if self._questions is not None else specs

    def _risk_allowed(self, tool: Tool) -> bool:
        return self.tool_risks is None or tool.risk in self.tool_risks

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
    ) -> list[tuple[ToolCall, ToolResult]]:
        results = []
        for call_id, call in zip(call_ids, assistant.tool_calls, strict=True):
            self._count_tool_call(state)
            results.append((call, await self._run_tool(ctx, state, call_id, call)))
        return results

    async def _run_tool(
        self, ctx: RunContext, state: _RunState, call_id: str, call: ToolCall
    ) -> ToolResult:
        if call.name == ASK_TOOL and self._questions is not None:
            return await self._ask_question(ctx, state, call_id, call)
        tool = self._deps.registry.get(call.name)
        if tool is None:
            self._emit_tool(TOOL_FAILED, ctx, call_id, call.name, error="unknown tool")
            return _error(f"Unknown tool: {call.name}")
        if not self._risk_allowed(tool):
            self._emit_tool(TOOL_FAILED, ctx, call_id, tool.name, error="not allowed")
            return _error(f"{tool.name} is not available to the {self.persona} persona.")
        return await self._run_allowed(ctx, call_id, tool, call)

    async def _run_allowed(
        self, ctx: RunContext, call_id: str, tool: Tool, call: ToolCall
    ) -> ToolResult:
        errors = schema_errors(tool.parameters, call.args)
        if errors:
            self._emit_tool(TOOL_FAILED, ctx, call_id, tool.name, error="invalid arguments")
            return _error(_invalid_arguments(tool.name, errors))
        refusal = await self._check_policy(ctx, call_id, tool, call)
        if refusal is not None:
            return refusal
        return await self._execute_tool(ctx, call_id, tool, call)

    async def _ask_question(
        self, ctx: RunContext, state: _RunState, call_id: str, call: ToolCall
    ) -> ToolResult:
        assert self._questions is not None
        errors = schema_errors(QUESTION_SCHEMA, call.args)
        if errors:
            self._emit_tool(TOOL_FAILED, ctx, call_id, ASK_TOOL, error="invalid arguments")
            return _error(_invalid_arguments(ASK_TOOL, errors))
        state.questions += 1
        if state.questions > self._questions.max_per_turn:
            self._emit_tool(TOOL_FAILED, ctx, call_id, ASK_TOOL, error="too many questions")
            return limit_result(self._questions.max_per_turn)
        question = {"question": call.args["question"], "options": call.args.get("options", [])}
        self._emit(QUESTION_REQUESTED, ctx.run_id, call_id=call_id, **question)
        answer = await self._deps.channel.ask(question)
        self._emit(QUESTION_ANSWERED, ctx.run_id, call_id=call_id, answer=answer.get("answer", ""))
        return answer_result(answer)

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
        preview = await self._preview(tool, call.args)
        request = ApprovalRequest(ctx.run_id, tool.name, tool.risk, preview, call.args)
        self._emit_tool(
            APPROVAL_REQUESTED, ctx, call_id, tool.name, risk=tool.risk, preview=preview
        )
        approved = await self._deps.channel.confirm(request)
        self._emit_tool(APPROVAL_RESOLVED, ctx, call_id, tool.name, approved=approved)
        return approved

    async def _preview(self, tool: Tool, args: dict[str, Any]) -> str:
        build = getattr(tool, "preview", None)

        if build is not None:
            try:
                text = await build(args)
            except Exception:
                text = None
            if text:
                return text
        return preview_call(tool.name, args)

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
        try:
            result = await self._guard_result(ctx, tool.name, result)
        except GuardBlockedError as exc:
            self._emit_tool(TOOL_FAILED, ctx, call_id, tool.name, error="blocked")
            return _error(f"The output of {tool.name} was blocked: {exc.reason}.")
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

    async def _run_hooks(self, hooks: list[Hook], ctx: RunContext) -> None:
        for hook in hooks:
            try:
                await hook(ctx)
            except GuardBlockedError as exc:
                self._emit(GUARD_BLOCKED, ctx.run_id, stage=exc.stage, reason=exc.reason)
                raise
            finally:
                self._flush_notices(ctx)

    async def _guard_result(
        self, ctx: RunContext, tool_name: str, result: ToolResult
    ) -> ToolResult:
        for hook in self.result_hooks:
            try:
                result = await hook(ctx, tool_name, result)
            except GuardBlockedError as exc:
                self._emit(
                    GUARD_BLOCKED, ctx.run_id, stage=exc.stage, reason=exc.reason, tool=tool_name
                )
                raise
            finally:
                self._flush_notices(ctx)
        return result

    def _flush_notices(self, ctx: RunContext) -> None:
        for type_, payload in ctx.notices:
            self._emit(type_, ctx.run_id, **payload)
        ctx.notices.clear()


def _limit_error(maximum: int, unit: str) -> RunLimitError:
    return RunLimitError(f"Limite atteinte : {maximum} {unit} maximum par tour.")


def _error(message: str) -> ToolResult:
    return ToolResult(message, is_error=True)


def _invalid_arguments(tool_name: str, errors: list[str]) -> str:
    listed = "\n".join(f"- {error}" for error in errors)
    return f"Invalid arguments for {tool_name}:\n{listed}\nFix them and call the tool again."


def _tool_message(name: str, result: ToolResult) -> Message:
    content = wrap_untrusted(result.content) if result.untrusted else result.content
    return Message("tool", content, tool_name=name)
