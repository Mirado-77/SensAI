"""Guards of a turn: injection check on the input and on tool outputs, redaction of secrets."""

from collections.abc import Mapping
from dataclasses import replace

from sensai.core.config import GuardsConfig
from sensai.core.errors import GuardBlockedError
from sensai.core.events import GUARD_REDACTED
from sensai.core.runtime import AgentRuntime, Hook, ResultHook, RunContext
from sensai.core.types import ToolResult
from sensai.safety.injection import is_injection
from sensai.safety.secrets import SecretRedactor

INJECTION_REASON = "injection de prompt détectée"


def install_guards(runtime: AgentRuntime, config: GuardsConfig, secrets: Mapping[str, str]) -> None:
    """Register the guards on `runtime`, unless disabled by `config`."""
    if not config.enabled:
        return
    redactor = SecretRedactor(secrets.values())
    runtime.entry_hooks.append(check_input)
    runtime.exit_hooks.append(redact_answer(redactor))
    runtime.result_hooks.append(guard_tool_output(redactor))


async def check_input(ctx: RunContext) -> None:
    if is_injection(ctx.user_text):
        raise GuardBlockedError("entry", INJECTION_REASON)


def redact_answer(redactor: SecretRedactor) -> Hook:
    async def hook(ctx: RunContext) -> None:
        ctx.answer, count = redactor.redact(ctx.answer)
        if count:
            ctx.notices.append((GUARD_REDACTED, {"stage": "answer", "count": count}))

    return hook


def guard_tool_output(redactor: SecretRedactor) -> ResultHook:
    async def hook(ctx: RunContext, tool_name: str, result: ToolResult) -> ToolResult:
        if is_injection(result.content):
            raise GuardBlockedError("tool", INJECTION_REASON)
        content, count = redactor.redact(result.content)
        if count:
            ctx.notices.append(
                (GUARD_REDACTED, {"stage": "tool", "tool": tool_name, "count": count})
            )
        return replace(result, content=content)

    return hook
