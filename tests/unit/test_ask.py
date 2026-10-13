import io
import json
import sqlite3

import pytest
from rich.console import Console

from sensai.cli.ask import load_schema, run_ask
from sensai.cli.main import build_parser
from sensai.cli.terminal import Terminal
from sensai.core.config import DEFAULT_CONFIG_PATH, load_config
from sensai.core.errors import ModelConnectionError
from sensai.core.fakes import FakeModelGateway
from sensai.core.types import Chunk, Usage

pytestmark = pytest.mark.feature("T5")

SCHEMA = {
    "type": "object",
    "properties": {"bug": {"type": "string"}, "file": {"type": "string"}},
    "required": ["bug", "file"],
}


@pytest.fixture
def setup(tmp_path):
    schema_path = tmp_path / "bug.json"
    schema_path.write_text(json.dumps(SCHEMA), encoding="utf-8")
    db_path = tmp_path / "sensai.db"
    config = load_config(overrides={"storage": {"db_path": str(db_path)}})
    out, err = io.StringIO(), io.StringIO()
    terminal = Terminal(Console(file=out, width=40), Console(file=err, width=200))
    return config, terminal, schema_path, db_path, out, err


def rows(db_path, query):
    connection = sqlite3.connect(db_path)
    try:
        return connection.execute(query).fetchall()
    finally:
        connection.close()


async def test_prints_the_validated_json_on_stdout_and_journals_the_run(setup):
    config, terminal, schema_path, db_path, out, err = setup
    answer = '{"bug": "off by one in the loop that sums the values", "file": "calc.py"}'
    gateway = FakeModelGateway([Chunk(answer), Usage(20, 8)])

    code = await run_ask(config, terminal, schema_path, "Find the bug", gateway)

    assert code == 0
    assert json.loads(out.getvalue()) == json.loads(answer)
    assert err.getvalue() == ""
    assert rows(db_path, "SELECT status, prompt_tokens, completion_tokens FROM runs") == [
        ("completed", 20, 8)
    ]
    assert rows(db_path, "SELECT role FROM messages ORDER BY created_at") == [
        ("user",),
        ("assistant",),
    ]


async def test_a_repair_is_announced_on_stderr_and_journaled(setup):
    config, terminal, schema_path, db_path, out, err = setup
    gateway = FakeModelGateway([Chunk('{"bug": "x"}')], [Chunk('{"bug": "x", "file": "a.py"}')])

    assert await run_ask(config, terminal, schema_path, "Find the bug", gateway) == 0

    assert "tentative de réparation 1" in err.getvalue()
    types = [t for (t,) in rows(db_path, "SELECT type FROM events ORDER BY id")]
    assert types == ["run.started", "structured.repaired", "run.completed"]


async def test_a_non_conforming_answer_fails_with_exit_code_1(setup):
    config, terminal, schema_path, db_path, out, err = setup
    gateway = FakeModelGateway([Chunk("{}")], [Chunk("{}")])

    assert await run_ask(config, terminal, schema_path, "Find the bug", gateway) == 1

    assert out.getvalue() == ""
    assert "pas produit de JSON conforme" in err.getvalue()
    assert rows(db_path, "SELECT status FROM runs") == [("failed",)]
    assert rows(db_path, "SELECT count(*) FROM messages") == [(0,)]


async def test_an_unreachable_model_fails_with_exit_code_1(setup):
    config, terminal, schema_path, _, out, err = setup
    gateway = FakeModelGateway(error=ModelConnectionError("Ollama est injoignable"))
    assert await run_ask(config, terminal, schema_path, "Find the bug", gateway) == 1
    assert "injoignable" in err.getvalue()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "Schéma introuvable"),
        ("{not json", "Schéma illisible"),
        ("[1, 2]", "doit être un objet JSON"),
        ('{"type": "nope"}', "Schéma JSON invalide"),
    ],
)
async def test_an_unusable_schema_exits_with_code_2_without_calling_the_model(
    setup, tmp_path, content, message
):
    config, terminal, _, _, out, err = setup
    path = tmp_path / "schema.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    gateway = FakeModelGateway()

    assert await run_ask(config, terminal, path, "Find the bug", gateway) == 2

    assert message in err.getvalue()
    assert gateway.calls == []


async def test_an_empty_question_exits_with_code_2(setup):
    config, terminal, schema_path, _, _, err = setup
    gateway = FakeModelGateway()
    assert await run_ask(config, terminal, schema_path, "   ", gateway) == 2
    assert "vide" in err.getvalue()


def test_the_parser_requires_a_schema_and_joins_the_question():
    args = build_parser().parse_args(["ask", "--schema", "s.json", "what", "is", "it"])
    assert (args.command, str(args.schema), args.question) == (
        "ask",
        "s.json",
        ["what", "is", "it"],
    )
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ask", "question"])


def test_the_example_schemas_are_valid():
    schemas = sorted((DEFAULT_CONFIG_PATH.parent / "schemas").glob("*.json"))
    assert schemas
    for path in schemas:
        load_schema(path)
