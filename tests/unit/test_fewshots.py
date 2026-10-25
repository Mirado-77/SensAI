import pytest
from tests.unit.test_cli import chat_setup
from tests.unit.test_runtime import build, events_of, saved

from sensai.cli.chat import run_chat
from sensai.core.errors import ConfigError
from sensai.core.fakes import FakeModelGateway
from sensai.core.fewshots import Example, FewShots, fewshots_from, load_examples
from sensai.core.types import Chunk, Message

pytestmark = pytest.mark.feature("A7")

DATES = Example(id="dates", question="Pourquoi le test de parse_date échoue ?", answer="Lu.")
USERS = Example(id="users", question="Que renvoie find_user sans utilisateur ?", answer="None.")
TESTS = Example(id="tests", question="Comment lancer les tests du projet ?", answer="make test")


def test_shipped_examples_load_with_unique_ids():
    examples = load_examples()
    assert len(examples) >= 4
    assert len({example.id for example in examples}) == len(examples)


def test_the_closest_examples_are_selected_best_first():
    fewshots = FewShots((USERS, TESTS, DATES), k=2)
    selected = fewshots.select("Pourquoi le test parse_date échoue dans le projet ?")
    assert [example.id for example in selected] == ["dates", "tests"]


def test_k_limits_the_selection_and_accents_do_not_matter():
    fewshots = FewShots((DATES, TESTS), k=1)
    assert [example.id for example in fewshots.select("pourquoi ECHOUE")] == ["dates"]


def test_an_example_sharing_no_word_is_never_sent():
    assert FewShots((DATES, USERS), k=2).select("bonjour") == []


def test_zero_examples_means_no_few_shots():
    assert fewshots_from(0) is None


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('id = "x"\nquestion = "q"\n', "Exemple invalide"),
        ("not toml =", "Exemple illisible"),
    ],
)
def test_a_broken_example_file_is_reported(tmp_path, content, message):
    (tmp_path / "bad.toml").write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_examples(tmp_path)


async def test_selected_examples_follow_the_system_prompt_and_are_not_saved():
    s = build([Chunk("ok")], system_prompt="Be brief.")
    s.runtime.fewshots = FewShots((DATES, USERS), k=2)
    await s.runtime.run_turn(s.session_id, "Que renvoie find_user ?")
    assert s.gateway.calls[0].messages == [
        Message("system", "Be brief."),
        Message("user", USERS.question),
        Message("assistant", USERS.answer),
        Message("user", "Que renvoie find_user ?"),
    ]
    assert events_of(s, "prompt.examples")[0].payload == {"examples": ["users"]}
    assert saved(s) == [("user", "Que renvoie find_user ?"), ("assistant", "ok")]


async def test_no_event_when_no_example_is_selected():
    s = build([Chunk("ok")])
    s.runtime.fewshots = FewShots((DATES,), k=2)
    await s.runtime.run_turn(s.session_id, "bonjour")
    assert events_of(s, "prompt.examples") == []


async def test_the_chat_sends_the_configured_examples(tmp_path):
    config, terminal, out, _ = chat_setup(
        tmp_path, "Que renvoie la fonction find_user ?", prompts={"system": "", "examples": 1}
    )
    gateway = FakeModelGateway([Chunk("None.")])
    assert await run_chat(config, terminal, gateway) == 0
    sent = gateway.calls[0].messages
    assert [m.role for m in sent] == ["user", "assistant", "user"]
    assert "find_user" in sent[0].content
    assert "+ 1 examples" in out.getvalue()
