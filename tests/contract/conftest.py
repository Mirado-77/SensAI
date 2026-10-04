"""Factories used by the contract tests.

Each fixture is parametrized by implementation. When a real implementation lands
(OllamaGateway, sqlite3 SessionStore...), add its name to `params` and its branch to the
factory: the contract tests then run against it unchanged.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from sensai.core.contracts import EventSink, ModelGateway, SessionStore, Tool, UserChannel
from sensai.core.errors import ModelError
from sensai.core.fakes import (
    CollectingEventSink,
    FakeModelGateway,
    FakeTool,
    FakeUserChannel,
    InMemorySessionStore,
)
from sensai.core.types import ModelEvent, Risk


@pytest.fixture(params=["fake"])
def make_gateway(request: pytest.FixtureRequest) -> Callable[..., ModelGateway]:
    """Build a ModelGateway that replays `events` (without Done), or fails with `error`."""

    def make(
        events: Sequence[ModelEvent] = (), error: type[ModelError] | None = None
    ) -> ModelGateway:
        return FakeModelGateway(list(events), error=error("simulated") if error else None)

    return make


@dataclass
class StoreHarness:
    store: SessionStore
    seed_profile: Callable[[str, dict[str, Any]], None]


@pytest.fixture(params=["memory"])
def harness(request: pytest.FixtureRequest) -> StoreHarness:
    store = InMemorySessionStore()
    return StoreHarness(store, store.add_profile)


@dataclass
class SinkHarness:
    sink: EventSink
    received: Callable[[], list[Any]]


@pytest.fixture(params=["collecting"])
def sink_harness(request: pytest.FixtureRequest) -> SinkHarness:
    sink = CollectingEventSink()
    return SinkHarness(sink, lambda: sink.events)


@pytest.fixture(params=["read", "write", "exec"])
def tool(request: pytest.FixtureRequest) -> Tool:
    risk: Risk = request.param
    return FakeTool(risk=risk)


@pytest.fixture(params=["fake"])
def channel(request: pytest.FixtureRequest) -> UserChannel:
    return FakeUserChannel()
