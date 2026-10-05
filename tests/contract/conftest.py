"""Factories used by the contract tests.

Each fixture is parametrized by implementation. When a real implementation lands
(OllamaGateway, sqlite3 SessionStore...), add its name to `params` and its branch to the
factory: the contract tests then run against it unchanged.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
import respx

from sensai.core.contracts import EventSink, ModelGateway, SessionStore, Tool, UserChannel
from sensai.core.errors import ModelConnectionError, ModelError, ModelNotFoundError
from sensai.core.fakes import (
    CollectingEventSink,
    FakeModelGateway,
    FakeTool,
    FakeUserChannel,
    InMemorySessionStore,
)
from sensai.core.types import ModelEvent, Risk
from sensai.db.connection import connect
from sensai.memory.sqlite_store import SqliteSessionStore
from sensai.model.ollama import OllamaGateway


@pytest.fixture(params=["fake", "ollama"])
def make_gateway(
    request: pytest.FixtureRequest, ollama_host, ollama_lines, slow_body
) -> Iterator[Callable[..., ModelGateway]]:
    """Build a ModelGateway that replays `events` (without Done), or fails with `error`.

    For "ollama", the HTTP server is simulated with respx. A trailing Usage in `events`
    becomes the token counts of Ollama's final line.
    """
    if request.param == "fake":

        def make_fake(
            events: Sequence[ModelEvent] = (), error: type[ModelError] | None = None
        ) -> ModelGateway:
            return FakeModelGateway(list(events), error=error("simulated") if error else None)

        yield make_fake
        return

    with respx.mock(base_url=ollama_host, assert_all_called=False) as router:

        def make_ollama(
            events: Sequence[ModelEvent] = (), error: type[ModelError] | None = None
        ) -> ModelGateway:
            route = router.post("/api/chat")
            if error is ModelConnectionError:
                route.mock(side_effect=httpx.ConnectError("connection refused"))
            elif error is ModelNotFoundError:
                body = {"error": "model 'test-model' not found"}
                route.mock(return_value=httpx.Response(404, json=body))
            else:
                lines = ollama_lines(events)
                route.mock(
                    side_effect=lambda request: httpx.Response(200, content=slow_body(lines)[0])
                )
            return OllamaGateway(ollama_host, "test-model", timeout_s=5)

        yield make_ollama


@dataclass
class StoreHarness:
    store: SessionStore
    seed_profile: Callable[[str, dict[str, Any]], None]


@pytest.fixture(params=["memory", "sqlite"])
def harness(request: pytest.FixtureRequest) -> StoreHarness:
    store: InMemorySessionStore | SqliteSessionStore
    if request.param == "memory":
        store = InMemorySessionStore()
    else:
        store = SqliteSessionStore(connect(":memory:"))
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
