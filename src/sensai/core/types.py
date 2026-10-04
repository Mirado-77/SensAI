"""Core types shared across the code base. All of them are immutable."""

import uuid
from dataclasses import dataclass
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
MessageStatus = Literal["complete", "interrupted"]
Risk = Literal["read", "write", "exec"]


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class ToolCall:
    """Tool call requested by the model."""

    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Message:
    """Chat message.

    `id` is assigned by the SessionStore.
    `status` is "interrupted" for the partial text of a cancelled stream.
    """

    role: Role
    content: str
    id: str | None = None
    status: MessageStatus = "complete"
    tool_calls: tuple[ToolCall, ...] = ()
    tool_name: str | None = None


@dataclass(frozen=True)
class Chunk:
    text: str


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class Done:
    """Last event of every model stream."""


ModelEvent = Chunk | ToolCall | Usage | Done


@dataclass(frozen=True)
class ToolSpec:
    """Tool description as the model sees it. `parameters` is a JSON Schema."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolContext:
    run_id: str
    session_id: str


@dataclass(frozen=True)
class ToolResult:
    """Result of a tool run.

    `untrusted` marks external content (repository file, MCP tool output) that must never be
    treated as instructions.
    """

    content: str
    is_error: bool = False
    untrusted: bool = False


@dataclass(frozen=True)
class ApprovalRequest:
    """Approval request. `preview` is the diff for a write or the command for an execution."""

    run_id: str
    tool: str
    risk: Risk
    preview: str
    args: dict[str, Any]
