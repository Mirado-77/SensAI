"""`sensai ask --schema FILE`: one question, answered with JSON that conforms to the schema (T5).

The JSON goes to stdout, so the command can feed a script. Messages go to stderr. The run is
journaled like a chat turn: a session holding the question and the answer, and its events.

Exit codes: 0 success, 1 model or schema-conformance failure, 2 unusable schema, config or
database.
"""

import dataclasses
import json
import sqlite3
from pathlib import Path
from typing import Any

import jsonschema
from pydantic import BaseModel
from rich.console import Console
from rich.text import Text

from sensai.cli.terminal import Terminal
from sensai.core.bus import EventBus
from sensai.core.config import Config
from sensai.core.contracts import EventSink, ModelGateway
from sensai.core.errors import SensaiError
from sensai.core.events import (
    MODEL_USAGE,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_STARTED,
    STRUCTURED_REPAIRED,
    Event,
    make_event,
)
from sensai.core.schema import check_schema
from sensai.core.types import Message, new_id
from sensai.db.connection import connect
from sensai.memory.sqlite_store import SqliteSessionStore
from sensai.model.ollama import OllamaGateway
from sensai.model.structured import StructuredResult, generate_structured
from sensai.observability.journal import SqliteEventJournal


class SchemaFileError(SensaiError):
    """The schema file is missing, is not JSON, or is not a valid JSON Schema."""


def load_schema(path: Path) -> dict[str, Any]:
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SchemaFileError(f"Schéma introuvable : {path}") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaFileError(f"Schéma illisible ({path}) : {exc}") from exc
    if not isinstance(schema, dict):
        raise SchemaFileError(f"Le schéma doit être un objet JSON : {path}")
    try:
        check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise SchemaFileError(f"Schéma JSON invalide ({path}) : {exc.message}") from exc
    return schema


async def run_ask(
    config: Config,
    terminal: Terminal,
    schema_path: Path,
    question: str,
    gateway: ModelGateway | None = None,
) -> int:
    if not question.strip():
        _error(terminal, "la question est vide.")
        return 2
    try:
        schema = load_schema(schema_path)
        connection = connect(config.storage.db_path)
    except (OSError, sqlite3.Error, SensaiError) as exc:
        _error(terminal, str(exc))
        return 2
    try:
        model = config.model
        gateway = gateway or OllamaGateway(model.host, model.name, model.timeout_s)
        sink = EventBus([SqliteEventJournal(connection), _RepairNotices(terminal)])
        store = SqliteSessionStore(connection)
        result = await _ask(config, gateway, sink, store, schema, question)
    finally:
        connection.close()
    if result is None:
        return 1
    output = json.dumps(_plain(result.value), ensure_ascii=False, indent=2)
    terminal.console.out(output, highlight=False)
    return 0


async def _ask(
    config: Config,
    gateway: ModelGateway,
    sink: EventSink,
    store: SqliteSessionStore,
    schema: dict[str, Any],
    question: str,
) -> StructuredResult | None:
    run_id = new_id()
    session_id = store.create_session()

    def emit(type_: str, payload: dict[str, Any]) -> None:
        sink.emit(make_event(type_, run_id, payload))

    emit(RUN_STARTED, {"session_id": session_id, "model": config.model.name})
    user = Message("user", question)
    try:
        result = await generate_structured(
            gateway, [user], schema, max_repairs=config.structured.max_repairs, on_event=emit
        )
    except SensaiError as exc:
        error = {"error_type": type(exc).__name__, "error": str(exc)}
        emit(RUN_FAILED, {"session_id": session_id, **error})
        return None
    for usage in result.usage:
        emit(MODEL_USAGE, dataclasses.asdict(usage))
    parent = store.append(session_id, user, None)
    store.append(session_id, Message("assistant", result.raw), parent)
    emit(RUN_COMPLETED, {"session_id": session_id, "steps": result.repairs + 1, "tool_calls": 0})
    return result


def _plain(value: Any) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


class _RepairNotices:
    """Tell the user on stderr when the answer had to be repaired or the run failed."""

    def __init__(self, terminal: Terminal) -> None:
        self._errors = terminal.errors

    def emit(self, event: Event) -> None:
        if event.type == STRUCTURED_REPAIRED:
            message = f"JSON non conforme, tentative de réparation {event.payload['attempt']}"
            self._errors.print(Text(message, style="dim"))
        elif event.type == RUN_FAILED:
            _print_error(self._errors, event.payload["error"])


def _error(terminal: Terminal, message: str) -> None:
    _print_error(terminal.errors, message)


def _print_error(console: Console, message: str) -> None:
    text = Text("Erreur : ", style="bold red")
    text.append(message)
    console.print(text)
