"""Events of the shared stream (CLI, Web, logs, tests) and their vocabulary."""

import time
from dataclasses import dataclass
from typing import Any

RUN_STARTED = "run.started"
RUN_COMPLETED = "run.completed"
RUN_CANCELLED = "run.cancelled"
RUN_FAILED = "run.failed"
MODEL_CHUNK = "model.chunk"
MODEL_USAGE = "model.usage"
MODEL_TOOL_CALL = "model.tool_call"
TOOL_STARTED = "tool.started"
TOOL_FINISHED = "tool.finished"
TOOL_FAILED = "tool.failed"
APPROVAL_REQUESTED = "approval.requested"
APPROVAL_RESOLVED = "approval.resolved"
STRUCTURED_REPAIRED = "structured.repaired"
STRUCTURED_FAILED = "structured.failed"
REACT_PLAN = "react.plan"
REACT_ACTION = "react.action"
REACT_OBSERVATION = "react.observation"
QUESTION_REQUESTED = "question.requested"
QUESTION_ANSWERED = "question.answered"

EVENT_TYPES = frozenset(
    {
        RUN_STARTED,
        RUN_COMPLETED,
        RUN_CANCELLED,
        RUN_FAILED,
        MODEL_CHUNK,
        MODEL_USAGE,
        MODEL_TOOL_CALL,
        TOOL_STARTED,
        TOOL_FINISHED,
        TOOL_FAILED,
        APPROVAL_REQUESTED,
        APPROVAL_RESOLVED,
        STRUCTURED_REPAIRED,
        STRUCTURED_FAILED,
        REACT_PLAN,
        REACT_ACTION,
        REACT_OBSERVATION,
        QUESTION_REQUESTED,
        QUESTION_ANSWERED,
    }
)

MEMORY_COMPRESSED = "memory.compressed"
ARTIFACT_PATCHED = "artifact.patched"
GUARD_BLOCKED = "guard.blocked"

RESERVED_EVENT_TYPES = frozenset(
    {
        MEMORY_COMPRESSED,
        ARTIFACT_PATCHED,
        GUARD_BLOCKED,
    }
)

ALL_EVENT_TYPES = EVENT_TYPES | RESERVED_EVENT_TYPES


@dataclass(frozen=True)
class Event:
    type: str
    run_id: str
    ts: float
    payload: dict[str, Any]


def make_event(
    type_: str,
    run_id: str,
    payload: dict[str, Any] | None = None,
    *,
    ts: float | None = None,
) -> Event:
    """Create an Event, rejecting a type that is not in the vocabulary (typo, undeclared type)."""
    if type_ not in ALL_EVENT_TYPES:
        raise ValueError(f"Type d'événement inconnu : {type_!r}")
    return Event(type_, run_id, time.time() if ts is None else ts, dict(payload or {}))
