"""The four contracts frozen at bootstrap.

To change a contract: announce it on Discord, then the PR updates the fake
(`core/fakes.py`) and the contract test (`tests/contract/`).
"""

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from sensai.core.events import Event
from sensai.core.types import (
    ApprovalRequest,
    Message,
    ModelEvent,
    Risk,
    ToolContext,
    ToolResult,
    ToolSpec,
)


@runtime_checkable
class ModelGateway(Protocol):
    def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> AsyncIterator[ModelEvent]:
        """Stream the model's events for a conversation.

        - The stream always ends with exactly one `Done`.
        - An optional `Usage` comes before `Done`.
        - Failures are raised as a `ModelError`, never as another exception type.
        - Cancelling the task that consumes the stream must close the connection to the model.
        """
        ...


@runtime_checkable
class Tool(Protocol):
    """Native or MCP tool: same contract, same registry, same permissions and logs.

    `parameters` is a JSON Schema of type "object". An expected failure (missing file, failing
    command) is returned as `ToolResult(is_error=True)`. An exception escaping `run` is treated
    by the runtime as a tool failure (`tool.failed`).
    """

    name: str
    description: str
    parameters: dict[str, Any]
    risk: Risk

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult: ...


@runtime_checkable
class EventSink(Protocol):
    """Consumer of the event stream (CLI, API, logs).

    `emit` is synchronous and must not block: an asynchronous consumer queues the event
    (`asyncio.Queue.put_nowait`).
    """

    def emit(self, event: Event) -> None: ...


@runtime_checkable
class UserChannel(Protocol):
    async def confirm(self, request: ApprovalRequest) -> bool: ...

    async def ask(self, question: dict[str, Any]) -> dict[str, Any]:
        """Reserved for X4 (structured questions)."""
        ...


@runtime_checkable
class SessionStore(Protocol):
    """Immutable messages stored as a tree (`parent_id`).

    An unknown identifier (session, message, profile) raises `KeyError`.
    """

    def create_session(self, profile_id: str | None = None) -> str: ...

    def append(self, session_id: str, msg: Message, parent_id: str | None) -> str:
        """Add `msg` under `parent_id` (None for the first message) and return its id.

        The new message becomes the session head. `msg.id` is ignored: the store assigns the
        id. `parent_id` may be a message inherited through a fork.
        """
        ...

    def history(self, session_id: str, head_id: str | None = None) -> list[Message]:
        """Messages from the root to `head_id` (the session head by default)."""
        ...

    def fork(self, session_id: str, from_message_id: str) -> str:
        """Create a new session whose head is `from_message_id` and return its id.

        History up to that message is shared. Later appends to either session do not appear in
        the other. `from_message_id` must belong to the tree of `session_id`.
        """
        ...

    def profile(self, profile_id: str) -> dict[str, Any]: ...
