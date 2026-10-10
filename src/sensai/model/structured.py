"""Structured output (T5): force the model to answer with JSON that conforms to a schema.

The schema goes to Ollama's `format` parameter, which constrains generation. The answer is then
validated (Pydantic for a model class, JSON Schema otherwise). An invalid answer gets
`max_repairs` repair attempts: the model sees its answer and the validation errors, and tries
again. After that, `StructuredOutputError` is raised.

Event payloads, sent through `on_event` (the caller adds `run_id`):

- structured.repaired: attempt, errors
- structured.failed: attempts, errors
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from sensai.core.contracts import ModelGateway
from sensai.core.errors import StructuredOutputError
from sensai.core.events import STRUCTURED_FAILED, STRUCTURED_REPAIRED
from sensai.core.schema import MAX_ERRORS_SHOWN, check_schema, schema_errors
from sensai.core.types import Chunk, Message, Usage

Target = type[BaseModel] | dict[str, Any]
OnEvent = Callable[[str, dict[str, Any]], None]

_SCHEMA_INSTRUCTION = (
    "Answer only with a JSON value that conforms to this JSON Schema. "
    "No prose, no code fence.\n{schema}"
)
_REPAIR_INSTRUCTION = (
    "Your previous answer does not conform to the JSON Schema:\n{errors}\n"
    "Answer again with the corrected JSON only."
)


@dataclass(frozen=True)
class StructuredResult:
    """`value` is an instance of the model class, or the parsed JSON for a dict schema."""

    value: Any
    raw: str
    repairs: int
    usage: list[Usage] = field(default_factory=list)


async def generate_structured(
    gateway: ModelGateway,
    messages: list[Message],
    target: Target,
    *,
    max_repairs: int = 1,
    on_event: OnEvent | None = None,
) -> StructuredResult:
    """Ask the model for JSON conforming to `target` and return the validated value."""
    validator = _Validator.for_target(target)
    conversation = [_schema_message(validator.schema), *messages]
    usage: list[Usage] = []
    errors: list[str] = []
    for attempt in range(max_repairs + 1):
        if attempt:
            _notify(on_event, STRUCTURED_REPAIRED, attempt=attempt, errors=errors)
        raw = await _collect(gateway, conversation, validator.schema, usage)
        value, errors = validator.validate(raw)
        if not errors:
            return StructuredResult(value, raw, attempt, usage)
        conversation += [Message("assistant", raw), _repair_message(errors)]
    _notify(on_event, STRUCTURED_FAILED, attempts=max_repairs + 1, errors=errors)
    raise StructuredOutputError(
        "Le modèle n'a pas produit de JSON conforme au schéma : " + "; ".join(errors)
    )


@dataclass(frozen=True)
class _Validator:
    schema: dict[str, Any]
    model: type[BaseModel] | None

    @classmethod
    def for_target(cls, target: Target) -> "_Validator":
        if isinstance(target, dict):
            check_schema(target)
            return cls(target, None)
        return cls(target.model_json_schema(), target)

    def validate(self, raw: str) -> tuple[Any, list[str]]:
        """Return the value and an empty list, or None and the validation errors."""
        if self.model is not None:
            return _validate_model(self.model, raw)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, [f"invalid JSON: {exc.msg} (position {exc.pos})"]
        errors = schema_errors(self.schema, value)
        return (None, errors) if errors else (value, [])


def _validate_model(model: type[BaseModel], raw: str) -> tuple[Any, list[str]]:
    try:
        return model.model_validate_json(raw), []
    except ValidationError as exc:
        errors = []
        for error in exc.errors()[:MAX_ERRORS_SHOWN]:
            path = "/".join(str(part) for part in error["loc"])
            errors.append(f"{path or '(root)'}: {error['msg']}")
        return None, errors


async def _collect(
    gateway: ModelGateway,
    messages: list[Message],
    schema: dict[str, Any],
    usage: list[Usage],
) -> str:
    parts: list[str] = []
    async for event in gateway.stream_chat(messages, schema=schema):
        if isinstance(event, Chunk):
            parts.append(event.text)
        elif isinstance(event, Usage):
            usage.append(event)
    return "".join(parts).strip()


def _schema_message(schema: dict[str, Any]) -> Message:
    text = json.dumps(schema, ensure_ascii=False)
    return Message("system", _SCHEMA_INSTRUCTION.format(schema=text))


def _repair_message(errors: list[str]) -> Message:
    listed = "\n".join(f"- {error}" for error in errors)
    return Message("user", _REPAIR_INSTRUCTION.format(errors=listed))


def _notify(on_event: OnEvent | None, type_: str, **payload: Any) -> None:
    if on_event is not None:
        on_event(type_, payload)
