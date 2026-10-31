import json
import sqlite3

import pytest
from tests.unit.test_cli import chat_setup

from sensai.cli.chat import run_chat
from sensai.core.fakes import FakeModelGateway
from sensai.core.personas import load_persona
from sensai.core.types import Chunk

pytestmark = pytest.mark.feature("A2")

APPROVED = Chunk(json.dumps({"approved": True, "issues": []}))
CHANGES = Chunk(json.dumps({"approved": False, "issues": ["add a test for the empty list"]}))


def events(config, type_):
    connection = sqlite3.connect(config.storage.db_path)
    try:
        rows = connection.execute(
            "SELECT payload_json FROM events WHERE type = ? ORDER BY id", (type_,)
        ).fetchall()
    finally:
        connection.close()
    return [json.loads(row[0]) for row in rows]


def system_text(call):
    return "\n".join(m.content for m in call.messages if m.role == "system")


async def test_the_developer_then_the_reviewer_work_until_approval(tmp_path):
    config, terminal, out, _ = chat_setup(
        tmp_path, "/pair corrige completion_rate", prompts={"system": "", "persona": "dev"}
    )
    gateway = FakeModelGateway([Chunk("Corrigé.")], [Chunk("C'est juste.")], [APPROVED])

    assert await run_chat(config, terminal, gateway) == 0

    dev, reviewer, verdict = gateway.calls
    assert load_persona("dev").prompt in system_text(dev)
    assert load_persona("reviewer").prompt in system_text(reviewer)
    assert verdict.schema is not None and "C'est juste." in verdict.messages[-1].content
    assert "approuvé par reviewer (tour 1)" in out.getvalue()
    assert events(config, "agents.handoff") == [{"round": 1, "from": "dev", "to": "reviewer"}]
    assert events(config, "agents.verdict") == [{"round": 1, "approved": True, "issues": []}]
    personas = [p["persona"] for p in events(config, "run.started")]
    assert personas == ["dev", "reviewer"]


async def test_the_issues_go_back_to_the_developer(tmp_path):
    config, terminal, out, _ = chat_setup(tmp_path, "/pair corrige completion_rate")
    gateway = FakeModelGateway(
        [Chunk("Corrigé.")],
        [Chunk("Il manque un test.")],
        [CHANGES],
        [Chunk("Test ajouté.")],
        [Chunk("C'est bon.")],
        [APPROVED],
    )

    assert await run_chat(config, terminal, gateway) == 0

    second_request = gateway.calls[3].messages[-1].content
    assert "add a test for the empty list" in second_request
    assert "approuvé par reviewer (tour 2)" in out.getvalue()
    handoffs = [(h["from"], h["to"]) for h in events(config, "agents.handoff")]
    assert handoffs == [("dev", "reviewer"), ("reviewer", "dev"), ("dev", "reviewer")]


async def test_the_pairing_stops_after_max_rounds(tmp_path):
    config, terminal, out, _ = chat_setup(
        tmp_path, "/pair corrige", agents={"dev": "dev", "reviewer": "reviewer", "max_rounds": 1}
    )
    gateway = FakeModelGateway([Chunk("Corrigé.")], [Chunk("Non.")], [CHANGES])

    assert await run_chat(config, terminal, gateway) == 0

    assert len(gateway.calls) == 3
    assert "pas approuvé après 1 tours" in out.getvalue()


async def test_an_unreadable_verdict_is_never_an_approval(tmp_path):
    config, terminal, out, _ = chat_setup(
        tmp_path, "/pair corrige", agents={"dev": "dev", "reviewer": "reviewer", "max_rounds": 1}
    )
    gateway = FakeModelGateway(
        [Chunk("Corrigé.")], [Chunk("Hmm.")], [Chunk("pas du JSON")], [Chunk("toujours pas")]
    )

    assert await run_chat(config, terminal, gateway) == 0

    assert events(config, "agents.verdict")[0]["approved"] is False
    assert "approuvé par" not in out.getvalue()


async def test_the_user_persona_is_back_after_the_pairing(tmp_path):
    config, terminal, _, _ = chat_setup(
        tmp_path, "/pair corrige", "et maintenant ?", prompts={"system": "", "persona": "reviewer"}
    )
    gateway = FakeModelGateway(
        [Chunk("Corrigé.")], [Chunk("Bien.")], [APPROVED], [Chunk("Rien à ajouter.")]
    )

    assert await run_chat(config, terminal, gateway) == 0

    assert load_persona("reviewer").prompt in system_text(gateway.calls[3])
    assert load_persona("dev").prompt not in system_text(gateway.calls[3])


async def test_pair_without_a_task_shows_the_usage(tmp_path):
    config, terminal, _, err = chat_setup(tmp_path, "/pair")
    gateway = FakeModelGateway()

    assert await run_chat(config, terminal, gateway) == 0

    assert "Usage : /pair <tâche>" in err.getvalue()
    assert gateway.calls == []


async def test_an_unknown_persona_is_reported(tmp_path):
    config, terminal, _, err = chat_setup(
        tmp_path, "/pair corrige", agents={"dev": "nobody", "reviewer": "reviewer", "max_rounds": 1}
    )

    assert await run_chat(config, terminal, FakeModelGateway()) == 0

    assert "Persona introuvable : nobody" in err.getvalue()
