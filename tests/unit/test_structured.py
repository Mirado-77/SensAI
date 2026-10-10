import asyncio

import jsonschema
import pytest
from pydantic import BaseModel

from sensai.core.errors import ModelConnectionError, StructuredOutputError
from sensai.core.fakes import FakeModelGateway
from sensai.core.types import Chunk, Message, Usage
from sensai.model.structured import generate_structured, schema_errors

pytestmark = pytest.mark.feature("T5")

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "tests": {"type": "integer"}},
    "required": ["name", "tests"],
}
QUESTION = [Message("user", "Describe the project.")]


class Summary(BaseModel):
    name: str
    tests: int


def collector():
    events = []
    return events, lambda type_, payload: events.append((type_, payload))


async def test_valid_answer_is_parsed_and_the_schema_is_sent():
    gateway = FakeModelGateway([Chunk('{"name": "demo", '), Chunk('"tests": 3}'), Usage(10, 5)])
    result = await generate_structured(gateway, QUESTION, SCHEMA)
    assert result.value == {"name": "demo", "tests": 3}
    assert result.repairs == 0
    assert result.usage == [Usage(10, 5)]
    call = gateway.calls[0]
    assert call.schema == SCHEMA
    assert call.tools is None
    assert call.messages[0].role == "system"
    assert '"required"' in call.messages[0].content
    assert call.messages[1:] == QUESTION


async def test_pydantic_model_target_returns_an_instance():
    gateway = FakeModelGateway([Chunk('{"name": "demo", "tests": 3}')])
    result = await generate_structured(gateway, QUESTION, Summary)
    assert result.value == Summary(name="demo", tests=3)
    assert gateway.calls[0].schema == Summary.model_json_schema()


async def test_invalid_answer_is_repaired_once():
    gateway = FakeModelGateway([Chunk('{"name": "demo"}')], [Chunk('{"name": "demo", "tests": 1}')])
    events, on_event = collector()
    result = await generate_structured(gateway, QUESTION, SCHEMA, on_event=on_event)
    assert result.value == {"name": "demo", "tests": 1}
    assert result.repairs == 1
    repair_call = gateway.calls[1]
    assert repair_call.messages[-2] == Message("assistant", '{"name": "demo"}')
    assert "'tests' is a required property" in repair_call.messages[-1].content
    missing = "(root): 'tests' is a required property"
    assert events == [("structured.repaired", {"attempt": 1, "errors": [missing]})]


async def test_malformed_json_is_reported_to_the_model():
    gateway = FakeModelGateway([Chunk("Sure! {name: demo")], [Chunk('{"name": "x", "tests": 0}')])
    result = await generate_structured(gateway, QUESTION, SCHEMA)
    assert result.repairs == 1
    assert "invalid JSON" in gateway.calls[1].messages[-1].content


async def test_second_failure_raises_and_emits_failed():
    gateway = FakeModelGateway([Chunk("{}")], [Chunk('{"name": 1, "tests": 2}')])
    events, on_event = collector()
    with pytest.raises(StructuredOutputError, match="name: 1 is not of type 'string'"):
        await generate_structured(gateway, QUESTION, SCHEMA, on_event=on_event)
    assert [type_ for type_, _ in events] == ["structured.repaired", "structured.failed"]
    assert events[-1][1]["attempts"] == 2
    assert len(gateway.calls) == 2


async def test_max_repairs_zero_makes_a_single_call():
    gateway = FakeModelGateway([Chunk("{}")])
    with pytest.raises(StructuredOutputError):
        await generate_structured(gateway, QUESTION, SCHEMA, max_repairs=0)
    assert len(gateway.calls) == 1


async def test_pydantic_errors_are_readable():
    gateway = FakeModelGateway([Chunk('{"name": "x", "tests": "many"}')])
    with pytest.raises(StructuredOutputError, match="tests: Input should be a valid integer"):
        await generate_structured(gateway, QUESTION, Summary, max_repairs=0)


async def test_invalid_schema_is_rejected_before_calling_the_model():
    gateway = FakeModelGateway()
    with pytest.raises(jsonschema.SchemaError):
        await generate_structured(gateway, QUESTION, {"type": "nope"})
    assert gateway.calls == []


async def test_model_errors_propagate():
    gateway = FakeModelGateway(error=ModelConnectionError("down"))
    with pytest.raises(ModelConnectionError):
        await generate_structured(gateway, QUESTION, SCHEMA)


async def test_cancellation_propagates():
    started = asyncio.Event()

    class Hanging:
        async def stream_chat(self, messages, tools=None, schema=None):
            started.set()
            await asyncio.sleep(10)
            yield Chunk("{}")

    task = asyncio.ensure_future(generate_structured(Hanging(), QUESTION, SCHEMA))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_schema_errors_are_capped_and_located():
    schema = {"type": "array", "items": {"type": "integer"}}
    errors = schema_errors(schema, ["a", "b", "c", "d", "e", "f", "g"])
    assert len(errors) == 5
    assert errors[0] == "0: 'a' is not of type 'integer'"
