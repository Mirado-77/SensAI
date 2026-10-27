import pytest
from tests.unit.test_cli import chat_setup
from tests.unit.test_runtime import build, saved

from sensai.cli.chat import run_chat
from sensai.core.fakes import FakeModelGateway
from sensai.core.types import Chunk, ToolCall
from sensai.db.connection import connect
from sensai.memory.facts import Fact, FactStore, Relation, recall_note
from sensai.memory.tools import (
    MemoryDeleteTool,
    MemoryRelateTool,
    MemorySaveTool,
    MemorySearchTool,
    MemoryUpdateTool,
)
from sensai.safety.untrusted import contains_untrusted_marker

pytestmark = pytest.mark.feature("M3")


@pytest.fixture
def facts():
    return FactStore(connect(":memory:"))


def test_saved_facts_and_relations_are_listed_newest_fact_first(facts):
    first = facts.save("decision", "storage", "Les tâches restent en JSON.")
    second = facts.save("convention", "style", "Noms en anglais.")
    link = facts.relate("taskman/storage.py", "uses", "taskman/models.py")

    records = facts.records()

    assert [record.id for record in records] == [second, first, link]
    assert isinstance(records[0], Fact) and isinstance(records[2], Relation)
    assert facts.relate("taskman/storage.py", "uses", "taskman/models.py") == link


def test_update_and_delete_change_only_the_target(facts):
    kept = facts.save("fact", "tests", "pytest seulement.")
    changed = facts.save("decision", "storage", "JSON.")
    link = facts.relate("a", "uses", "b")

    facts.update(changed, "SQLite à partir de la v2.")
    facts.delete(link)

    assert {r.id: r.content for r in facts.records()} == {
        changed: "SQLite à partir de la v2.",
        kept: "pytest seulement.",
    }


def test_search_ranks_by_shared_words_and_filters_by_kind(facts):
    facts.save("convention", "style", "Docstrings en français.")
    storage = facts.save("decision", "stockage", "Les tâches restent dans un fichier JSON.")
    facts.save("fact", "tests", "Les tests tournent avec pytest.")

    assert [r.id for r in facts.search("où sont stockées les tâches JSON ?", 5)] == [storage]
    assert [r.kind for r in facts.search("", 5, kind="convention")] == ["convention"]
    assert facts.search("bonjour", 5) == []


def test_recall_sends_the_conventions_and_the_closest_records_as_data(facts):
    facts.save("convention", "style", "Noms en anglais.")
    facts.save("decision", "stockage", "Les tâches restent en JSON.")
    facts.save("fact", "ci", "La CI tourne sur GitHub Actions.")

    note = recall_note(facts, "Comment sont stockées les tâches ?", limit=5)

    assert "Noms en anglais." in note and "restent en JSON" in note
    assert "GitHub Actions" not in note
    assert contains_untrusted_marker(note)


def test_recall_is_none_for_an_empty_memory(facts):
    assert recall_note(facts, "Comment sont stockées les tâches ?", limit=5) is None


async def test_the_tools_save_search_update_and_delete(facts):
    saved_text = (
        await MemorySaveTool(facts).run(
            {"kind": "decision", "subject": "stockage", "content": "JSON."}, None
        )
    ).content
    fact_id = facts.records()[0].id
    assert fact_id[:8] in saved_text

    found = await MemorySearchTool(facts).run({"query": "stockage"}, None)
    assert found.untrusted and "JSON." in found.content

    await MemoryUpdateTool(facts).run({"id": fact_id[:8], "content": "SQLite."}, None)
    assert facts.records()[0].content == "SQLite."

    await MemoryDeleteTool(facts).run({"id": fact_id[:8]}, None)
    assert facts.records() == []


async def test_an_unknown_or_ambiguous_id_is_an_error(facts):
    facts.save("fact", "a", "x")
    facts.save("fact", "b", "y")
    unknown = await MemoryDeleteTool(facts).run({"id": "zzzzzzzz"}, None)
    ambiguous = await MemoryDeleteTool(facts).run({"id": ""}, None)
    assert unknown.is_error and "memory_search" in unknown.content
    assert ambiguous.is_error and len(facts.records()) == 2


async def test_a_relation_cannot_be_updated(facts):
    link = facts.relate("a", "uses", "b")
    result = await MemoryUpdateTool(facts).run({"id": link[:8], "content": "z"}, None)
    assert result.is_error


async def test_changes_need_approval_and_show_what_changes(facts):
    fact_id = facts.save("decision", "stockage", "JSON.")
    tools = [MemorySaveTool, MemoryRelateTool, MemoryUpdateTool, MemoryDeleteTool]
    assert {tool(facts).risk for tool in tools} == {"write"}
    assert MemorySearchTool(facts).risk == "read"
    preview = await MemoryUpdateTool(facts).preview({"id": fact_id[:8], "content": "SQLite."})
    assert preview == "Modifier la mémoire du projet\n- JSON.\n+ SQLite."


async def test_the_recall_follows_the_system_prompt_and_is_not_saved():
    s = build([Chunk("JSON.")], system_prompt="Be brief.")
    s.runtime.recall = lambda text: f"memory for {text}"
    await s.runtime.run_turn(s.session_id, "stockage ?")
    sent = [(m.role, m.content) for m in s.gateway.calls[0].messages]
    assert sent == [
        ("system", "Be brief."),
        ("system", "memory for stockage ?"),
        ("user", "stockage ?"),
    ]
    assert saved(s) == [("user", "stockage ?"), ("assistant", "JSON.")]


async def test_a_fact_saved_in_one_session_is_recalled_in_the_next(tmp_path):
    save = ToolCall(
        "memory_save", {"kind": "decision", "subject": "stockage", "content": "JSON, pas SQLite."}
    )
    memory = {"enabled": True, "recall": 5}
    config, terminal, out, _ = chat_setup(
        tmp_path, "Retiens : stockage en JSON.", "y", "/memory", memory=memory
    )
    first = FakeModelGateway([save], [Chunk("Retenu.")])
    assert await run_chat(config, terminal, first) == 0
    assert "[decision]" in out.getvalue() and "JSON, pas SQLite." in out.getvalue()

    config, terminal, _, _ = chat_setup(tmp_path, "Quel stockage ?", memory=memory)
    second = FakeModelGateway([Chunk("JSON.")])
    assert await run_chat(config, terminal, second) == 0
    recalled = [m.content for m in second.calls[0].messages if m.role == "system"]
    assert any("JSON, pas SQLite." in content for content in recalled)


async def test_memory_forget_deletes_a_record_from_the_chat(tmp_path):
    connection = connect(tmp_path / "sensai.db")
    fact_id = FactStore(connection).save("fact", "ci", "GitHub Actions.")
    connection.close()
    config, terminal, out, err = chat_setup(
        tmp_path,
        f"/memory forget {fact_id[:8]}",
        "/memory",
        "/memory oops",
        memory={"enabled": True},
    )
    assert await run_chat(config, terminal, FakeModelGateway()) == 0
    assert "Oublié" in out.getvalue() and "La mémoire du projet est vide." in out.getvalue()
    assert "Usage" in err.getvalue()


async def test_memory_command_when_memory_is_off(tmp_path):
    config, terminal, _, err = chat_setup(tmp_path, "/memory")
    assert await run_chat(config, terminal, FakeModelGateway()) == 0
    assert "désactivée" in err.getvalue()
