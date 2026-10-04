"""One fake per contract, for tests (and the demo's fallback recordings)."""

import asyncio
import copy
from collections import deque
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, replace
from typing import Any

from sensai.core.errors import ModelError
from sensai.core.events import Event
from sensai.core.types import (
    ApprovalRequest,
    Done,
    Message,
    ModelEvent,
    Risk,
    ToolContext,
    ToolResult,
    ToolSpec,
    new_id,
)


@dataclass(frozen=True)
class GatewayCall:
    messages: list[Message]
    tools: list[ToolSpec] | None
    schema: dict[str, Any] | None


class FakeModelGateway:
    """Replay recorded responses: one script (list of ModelEvent) per call.

    A script does not contain `Done`: the fake appends it to every stream, as the contract
    requires. With `error`, every call raises it on the first read. Received calls are recorded
    in `calls` (on the first read of the stream).
    """

    def __init__(self, *scripts: Sequence[ModelEvent], error: ModelError | None = None) -> None:
        if any(isinstance(event, Done) for script in scripts for event in script):
            raise ValueError("Un script ne doit pas contenir Done : le fake l'ajoute.")
        self._scripts = deque(scripts)
        self._error = error
        self.calls: list[GatewayCall] = []

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> AsyncIterator[ModelEvent]:
        self.calls.append(GatewayCall(list(messages), tools, schema))
        if self._error is not None:
            raise self._error
        if not self._scripts:
            raise AssertionError("FakeModelGateway : plus de script pour cet appel.")
        for event in self._scripts.popleft():
            await asyncio.sleep(0)  # yield control so the stream can be cancelled
            yield event
        yield Done()


class FakeTool:
    """Test tool: return `result`, or raise `error` if given. Calls are recorded in `calls`."""

    def __init__(
        self,
        name: str = "fake_tool",
        description: str = "Test tool.",
        parameters: dict[str, Any] | None = None,
        risk: Risk = "read",
        result: ToolResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = (
            parameters if parameters is not None else {"type": "object", "properties": {}}
        )
        self.risk = risk
        self._result = result if result is not None else ToolResult("ok")
        self._error = error
        self.calls: list[tuple[dict[str, Any], ToolContext]] = []

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.calls.append((args, ctx))
        if self._error is not None:
            raise self._error
        return self._result


class CollectingEventSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        return [event.type for event in self.events]


class FakeUserChannel:
    """Answer requests in the order of `approvals` / `answers`, then with the defaults."""

    def __init__(
        self,
        approvals: Sequence[bool] = (),
        answers: Sequence[dict[str, Any]] = (),
        default_approval: bool = True,
    ) -> None:
        self._approvals = deque(approvals)
        self._answers = deque(answers)
        self._default_approval = default_approval
        self.confirm_requests: list[ApprovalRequest] = []
        self.questions: list[dict[str, Any]] = []

    async def confirm(self, request: ApprovalRequest) -> bool:
        self.confirm_requests.append(request)
        return self._approvals.popleft() if self._approvals else self._default_approval

    async def ask(self, question: dict[str, Any]) -> dict[str, Any]:
        self.questions.append(question)
        return self._answers.popleft() if self._answers else {}


@dataclass(frozen=True)
class _Node:
    session_id: str
    parent_id: str | None
    message: Message


class InMemorySessionStore:
    def __init__(self) -> None:
        self._nodes: dict[str, _Node] = {}
        self._heads: dict[str, str | None] = {}
        self._session_profiles: dict[str, str | None] = {}
        self._profiles: dict[str, dict[str, Any]] = {}

    def add_profile(self, profile_id: str, data: dict[str, Any]) -> None:
        """Not part of the contract: lets tests prepare a profile."""
        self._profiles[profile_id] = copy.deepcopy(data)

    def create_session(self, profile_id: str | None = None) -> str:
        session_id = new_id()
        self._heads[session_id] = None
        self._session_profiles[session_id] = profile_id
        return session_id

    def append(self, session_id: str, msg: Message, parent_id: str | None) -> str:
        self._check_session(session_id)
        if parent_id is not None:
            self._node(parent_id)
        message_id = new_id()
        self._nodes[message_id] = _Node(session_id, parent_id, replace(msg, id=message_id))
        self._heads[session_id] = message_id
        return message_id

    def history(self, session_id: str, head_id: str | None = None) -> list[Message]:
        self._check_session(session_id)
        if head_id is None:
            head_id = self._heads[session_id]
        elif head_id not in self._tree_ids(session_id):
            raise KeyError(f"Message hors de la session {session_id} : {head_id}")
        path: list[Message] = []
        while head_id is not None:
            node = self._node(head_id)
            path.append(node.message)
            head_id = node.parent_id
        path.reverse()
        return path

    def fork(self, session_id: str, from_message_id: str) -> str:
        self._check_session(session_id)
        if from_message_id not in self._tree_ids(session_id):
            raise KeyError(f"Message hors de la session {session_id} : {from_message_id}")
        fork_id = new_id()
        self._heads[fork_id] = from_message_id
        self._session_profiles[fork_id] = self._session_profiles[session_id]
        return fork_id

    def profile(self, profile_id: str) -> dict[str, Any]:
        return copy.deepcopy(self._profiles[profile_id])

    def _check_session(self, session_id: str) -> None:
        if session_id not in self._heads:
            raise KeyError(f"Session inconnue : {session_id}")

    def _node(self, message_id: str) -> _Node:
        try:
            return self._nodes[message_id]
        except KeyError:
            raise KeyError(f"Message inconnu : {message_id}") from None

    def _tree_ids(self, session_id: str) -> set[str]:
        """Ids of every message in the session's tree, including those inherited by a fork."""
        tips = {mid for mid, node in self._nodes.items() if node.session_id == session_id}
        head = self._heads[session_id]
        if head is not None:
            tips.add(head)
        ids: set[str] = set()
        for tip in tips:
            current: str | None = tip
            while current is not None and current not in ids:
                ids.add(current)
                current = self._nodes[current].parent_id
        return ids
