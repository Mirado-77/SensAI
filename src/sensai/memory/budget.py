"""Context budget (M2): the context size is shown after each turn, and before it reaches
`[memory] compress_at` of the model window, the old turns are replaced by a summary.

The summary is saved as the new root of the session, followed by copies of the last
`keep_turns` turns; the session head moves to the last copy. The old messages stay in the
session's tree, so `/fork` and `sensai sessions` still see them. Tool outputs are never sent to
the summary call (only their tool name): they are untrusted and the summary becomes a system
message.
"""

import json
from dataclasses import asdict, replace

from sensai.core.config import MemoryConfig
from sensai.core.contracts import ModelGateway, SessionStore
from sensai.core.errors import ModelError
from sensai.core.events import MEMORY_BUDGET, MEMORY_COMPRESSED, MEMORY_SKIPPED, MODEL_USAGE
from sensai.core.runtime import AgentRuntime, Hook, RunContext
from sensai.core.types import Chunk, Message, Usage

SUMMARY_PREFIX = "Summary of the earlier conversation:\n"
SUMMARY_INSTRUCTION = (
    "Summarize this conversation for the assistant who will continue it. Keep the user's goals,"
    " the decisions taken, file names, facts learned and what is left to do. Plain text, at most"
    " 200 words, no preamble."
)
# About 4 characters per token: good enough for a threshold (Ollama /api/tokenize is exact).
CHARS_PER_TOKEN = 4
_MAX_MESSAGE_CHARS = 2000


def estimate_tokens(messages: list[Message]) -> int:
    return sum(len(m.content) // CHARS_PER_TOKEN + 4 for m in messages)


def tool_tokens(runtime: AgentRuntime) -> int:
    """The tool schemas sent with every call: about 2,000 tokens for the chat's tools."""
    specs = [asdict(spec) for spec in runtime.tool_specs()]
    return len(json.dumps(specs)) // CHARS_PER_TOKEN if specs else 0


def install_memory(
    runtime: AgentRuntime,
    config: MemoryConfig,
    context_tokens: int,
    store: SessionStore,
    gateway: ModelGateway,
) -> None:
    """Register the budget and the compression on `runtime`, unless `[memory] compress` is off."""
    if not config.compress:
        return
    counted: dict[str, int] = {}
    runtime.entry_hooks.append(
        compress_history(runtime, config, context_tokens, store, gateway, counted)
    )
    runtime.exit_hooks.append(report_budget(runtime, context_tokens, counted))


def report_budget(runtime: AgentRuntime, context_tokens: int, counted: dict[str, int]) -> Hook:
    """`used` is the prompt Ollama counted on the last call plus the answer; without a count (an
    answer from the cache, a model that reports no usage) it is estimated."""

    async def hook(ctx: RunContext) -> None:
        answer = estimate_tokens([Message("assistant", ctx.answer)])
        if ctx.prompt_tokens:
            used = ctx.prompt_tokens + answer
        else:
            used = estimate_tokens(ctx.messages) + tool_tokens(runtime) + answer
        counted[ctx.session_id] = used
        payload = {"used": used, "limit": context_tokens, "counted": bool(ctx.prompt_tokens)}
        ctx.notices.append((MEMORY_BUDGET, payload))

    return hook


def compress_history(
    runtime: AgentRuntime,
    config: MemoryConfig,
    context_tokens: int,
    store: SessionStore,
    gateway: ModelGateway,
    counted: dict[str, int],
) -> Hook:
    """The context of the coming call is the larger of the size counted at the end of the last
    turn and an estimate that includes the tool schemas, plus the new message."""

    async def hook(ctx: RunContext) -> None:
        history = store.history(ctx.session_id)
        prefix = [Message("system", runtime.system_prompt)] if runtime.system_prompt else []
        user = estimate_tokens([Message("user", ctx.user_text)])
        estimated = estimate_tokens([*prefix, *history]) + tool_tokens(runtime) + user
        before = max(estimated, counted.get(ctx.session_id, 0) + user)
        if before < config.compress_at * context_tokens:
            return
        users = [i for i, message in enumerate(history) if message.role == "user"]
        if len(users) <= config.keep_turns:
            return  # nothing old enough to summarize
        cut = users[-config.keep_turns]
        old, recent = history[:cut], history[cut:]
        try:
            summary = await _summarize(gateway, old, ctx)
        except ModelError as exc:
            # The summary only saves room: the turn goes on with the full history.
            ctx.notices.append((MEMORY_SKIPPED, {"reason": str(exc)}))
            return
        if not summary:
            return
        parent = store.append(ctx.session_id, Message("system", SUMMARY_PREFIX + summary), None)
        for message in recent:
            parent = store.append(ctx.session_id, replace(message, id=None), parent)
        after = before - estimate_tokens(old) + estimate_tokens([Message("system", summary)])
        counted[ctx.session_id] = after - user
        payload = {"summarized": len(old), "kept": len(recent), "before": before, "after": after}
        ctx.notices.append((MEMORY_COMPRESSED, payload))

    return hook


async def _summarize(gateway: ModelGateway, messages: list[Message], ctx: RunContext) -> str:
    transcript = "\n\n".join(_transcript_line(message) for message in messages)
    request = [Message("system", SUMMARY_INSTRUCTION), Message("user", transcript)]
    text = ""
    async for event in gateway.stream_chat(request):
        if isinstance(event, Chunk):
            text += event.text
        elif isinstance(event, Usage):
            usage = {
                "prompt_tokens": event.prompt_tokens,
                "completion_tokens": event.completion_tokens,
            }
            ctx.notices.append((MODEL_USAGE, usage))
    return text.strip()


def _transcript_line(message: Message) -> str:
    if message.role == "tool":
        return f"[tool {message.tool_name} returned a result]"
    content = message.content.removeprefix(SUMMARY_PREFIX)
    return f"{message.role}: {content[:_MAX_MESSAGE_CHARS]}"
