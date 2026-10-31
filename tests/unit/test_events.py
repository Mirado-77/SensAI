import dataclasses

import pytest

from sensai.core import events
from sensai.core.events import (
    ALL_EVENT_TYPES,
    EVENT_TYPES,
    GUARD_BLOCKED,
    RESERVED_EVENT_TYPES,
    RUN_STARTED,
    Event,
    make_event,
)


def test_make_event_fills_the_fields():
    event = make_event(RUN_STARTED, "r1", {"model": "qwen3:4b"}, ts=12.5)
    assert event == Event("run.started", "r1", 12.5, {"model": "qwen3:4b"})


def test_make_event_timestamps_by_default():
    assert make_event(RUN_STARTED, "r1").ts > 0


def test_make_event_copies_the_payload():
    payload = {"a": 1}
    event = make_event(RUN_STARTED, "r1", payload)
    payload["a"] = 2
    assert event.payload == {"a": 1}


def test_make_event_rejects_an_unknown_type():
    with pytest.raises(ValueError, match="inconnu"):
        make_event("run.startd", "r1")


def test_reserved_types_are_accepted():
    assert make_event(GUARD_BLOCKED, "r1").type == "guard.blocked"


def test_event_is_immutable():
    event = make_event(RUN_STARTED, "r1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.type = "run.failed"


def test_initial_vocabulary_is_complete():
    assert EVENT_TYPES == {
        "run.started",
        "run.completed",
        "run.cancelled",
        "run.failed",
        "model.chunk",
        "model.usage",
        "model.tool_call",
        "tool.started",
        "tool.finished",
        "tool.failed",
        "approval.requested",
        "approval.resolved",
        "structured.repaired",
        "structured.failed",
        "react.plan",
        "react.action",
        "react.observation",
        "guard.blocked",
        "guard.redacted",
        "prompt.examples",
        "agents.handoff",
        "agents.verdict",
        "cache.hit",
        "cache.stored",
        "question.requested",
        "question.answered",
        "artifact.patched",
        "memory.compressed",
        "memory.budget",
    }

    assert RESERVED_EVENT_TYPES == set()
    assert ALL_EVENT_TYPES == EVENT_TYPES | RESERVED_EVENT_TYPES


def test_every_type_constant_is_in_the_vocabulary():
    constants = {
        value for name, value in vars(events).items() if name.isupper() and isinstance(value, str)
    }
    assert constants == ALL_EVENT_TYPES
