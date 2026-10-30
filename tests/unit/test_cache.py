"""A6: semantic cache through the runtime hooks, with a fake embedder."""

import pytest
from tests.unit.test_cli import chat_setup

from sensai.cli.chat import run_chat
from sensai.core.config import CacheConfig, RuntimeConfig
from sensai.core.errors import ModelConnectionError
from sensai.core.fakes import (
    CollectingEventSink,
    FakeModelGateway,
    FakeTool,
    FakeUserChannel,
    InMemorySessionStore,
)
from sensai.core.runtime import AgentRuntime, RuntimeDeps
from sensai.core.types import Chunk, ToolCall
from sensai.db.connection import connect
from sensai.memory.cache import SemanticCache, identifiers, install_cache
from sensai.safety.policy import ApprovalPolicy
from sensai.tools.registry import ToolRegistry

pytestmark = pytest.mark.feature("A6")

QUESTION = "what does completion_rate return"
# Unit vectors: the paraphrase is 0.96 close to the question, the other topic 0, and the same
# question about count_done 0.96 as well, as real embeddings do.
VECTORS = {
    QUESTION: [1.0, 0.0, 0.0],
    "what is returned by completion_rate": [0.96, 0.28, 0.0],
    "what does count_done return": [0.96, 0.0, 0.28],
    "how are tasks saved": [0.0, 0.0, 1.0],
}


class FakeEmbedder:
    model = "fake-embed"

    def __init__(self, error=None):
        self.calls = []
        self._error = error

    async def embed(self, texts):
        self.calls.append(texts)
        if self._error:
            raise self._error
        return [VECTORS[text] for text in texts]


def build(*scripts, tools=(), embedder=None, ttl_hours=24.0):
    gateway = FakeModelGateway(*scripts)
    store = InMemorySessionStore()
    sink = CollectingEventSink()
    registry = ToolRegistry(tools)
    deps = RuntimeDeps(
        gateway=gateway,
        store=store,
        sink=sink,
        channel=FakeUserChannel(),
        registry=registry,
        policy=ApprovalPolicy({"read": "allow", "write": "allow", "exec": "allow"}),
    )
    config = RuntimeConfig(max_steps=8, max_tool_calls=8, timeout_s=5, max_tokens=1000)
    runtime = AgentRuntime(deps, config, "test-model")
    connection = connect(":memory:")
    cache = SemanticCache(
        connection,
        embedder or FakeEmbedder(),
        CacheConfig(enabled=True, min_score=0.9, ttl_hours=ttl_hours),
    )
    install_cache(runtime, cache, store, registry, "test-model")
    return runtime, gateway, store, sink, connection


def events_of(sink, type_):
    return [event.payload for event in sink.events if event.type == type_]


async def test_a_close_question_in_a_new_session_is_answered_from_the_cache():
    runtime, gateway, store, sink, _ = build([Chunk("A float between 0 and 1.")])
    await runtime.run_turn(store.create_session(), QUESTION)

    session = store.create_session()
    result = await runtime.run_turn(session, "what is returned by completion_rate")

    assert result.status == "completed"
    assert result.answer == "A float between 0 and 1."
    assert len(gateway.calls) == 1
    assert events_of(sink, "cache.stored") == [{"question": QUESTION}]
    [hit] = events_of(sink, "cache.hit")
    assert hit == {"score": 0.96, "question": QUESTION}
    assert events_of(sink, "model.chunk")[-1] == {"text": "A float between 0 and 1."}
    assert [m.content for m in store.history(session)] == [
        "what is returned by completion_rate",
        "A float between 0 and 1.",
    ]


async def test_an_unrelated_question_goes_to_the_model():
    runtime, gateway, store, sink, _ = build([Chunk("A float.")], [Chunk("In a JSON file.")])
    await runtime.run_turn(store.create_session(), QUESTION)
    result = await runtime.run_turn(store.create_session(), "how are tasks saved")

    assert result.answer == "In a JSON file."
    assert len(gateway.calls) == 2
    assert events_of(sink, "cache.hit") == []


async def test_a_question_about_another_identifier_goes_to_the_model():
    runtime, gateway, store, sink, _ = build([Chunk("A float.")], [Chunk("An int.")])
    await runtime.run_turn(store.create_session(), QUESTION)
    result = await runtime.run_turn(store.create_session(), "what does count_done return")

    assert result.answer == "An int."
    assert events_of(sink, "cache.hit") == []


@pytest.mark.parametrize(
    "text, names",
    [
        ("Que retourne completion_rate ?", {"completion_rate"}),
        ("Lis stats.py.", {"stats.py"}),
        ("explique taskman/stats.py et load_tasks()", {"taskman/stats.py", "load_tasks()"}),
        ("Comment sont sauvegardées les tâches ?", set()),
    ],
)
def test_identifiers(text, names):
    assert identifiers(text) == names


async def test_only_the_first_message_of_a_session_uses_the_cache():
    embedder = FakeEmbedder()
    runtime, gateway, store, sink, _ = build(
        [Chunk("In a JSON file.")], [Chunk("A float.")], embedder=embedder
    )
    session = store.create_session()
    await runtime.run_turn(session, "how are tasks saved")
    await runtime.run_turn(session, QUESTION)

    assert embedder.calls == [["how are tasks saved"]]
    assert [p["question"] for p in events_of(sink, "cache.stored")] == ["how are tasks saved"]


@pytest.mark.parametrize("risk, stored", [("read", True), ("write", False), ("exec", False)])
async def test_a_turn_that_changed_something_is_not_stored(risk, stored):
    tool = FakeTool(name="tool", risk=risk)
    runtime, _, store, sink, _ = build([ToolCall("tool", {})], [Chunk("Done.")], tools=[tool])
    await runtime.run_turn(store.create_session(), QUESTION)

    assert bool(events_of(sink, "cache.stored")) is stored


async def test_another_persona_does_not_get_the_cached_answer():
    runtime, gateway, store, _, _ = build([Chunk("A float.")], [Chunk("Reviewed.")])
    await runtime.run_turn(store.create_session(), QUESTION)
    runtime.persona = "reviewer"
    result = await runtime.run_turn(store.create_session(), QUESTION)

    assert result.answer == "Reviewed."
    assert len(gateway.calls) == 2


async def test_an_expired_entry_is_not_served():
    runtime, gateway, store, _, connection = build([Chunk("Old.")], [Chunk("New.")])
    await runtime.run_turn(store.create_session(), QUESTION)
    connection.execute("UPDATE semantic_cache SET created_at = created_at - 25 * 3600")
    result = await runtime.run_turn(store.create_session(), QUESTION)

    assert result.answer == "New."
    rows = connection.execute("SELECT answer FROM semantic_cache").fetchall()
    assert [row[0] for row in rows] == ["New."]


async def test_without_an_embedding_model_the_turn_runs_as_usual():
    embedder = FakeEmbedder(error=ModelConnectionError("Ollama est injoignable"))
    runtime, gateway, store, sink, _ = build([Chunk("A float.")], embedder=embedder)
    result = await runtime.run_turn(store.create_session(), QUESTION)

    assert (result.status, result.answer) == ("completed", "A float.")
    assert not [p for p in sink.types() if p.startswith("cache.")]


async def test_cache_clear_in_the_chat(tmp_path):
    cache = {"enabled": True, "min_score": 0.92, "ttl_hours": 24}
    config, terminal, out, err = chat_setup(tmp_path, "/cache", "/cache clear", cache=cache)

    assert await run_chat(config, terminal, FakeModelGateway()) == 0

    assert "Usage : /cache clear" in err.getvalue()
    assert "Cache vidé : 0 réponse(s) oubliée(s)." in out.getvalue()


async def test_cache_command_when_disabled(tmp_path):
    config, terminal, _, err = chat_setup(tmp_path, "/cache clear")

    assert await run_chat(config, terminal, FakeModelGateway()) == 0

    assert "Cache sémantique désactivé" in err.getvalue()
