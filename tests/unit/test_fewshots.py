import json

import pytest
from tests.unit.test_cli import chat_setup
from tests.unit.test_runtime import build, events_of, saved

from sensai.cli.chat import run_chat
from sensai.core.config import load_config
from sensai.core.errors import ConfigError
from sensai.core.fakes import FakeModelGateway
from sensai.core.fewshots import Example, FewShots, fewshots_from, load_examples
from sensai.core.types import Chunk, Message
from sensai.evals import ev1
from sensai.evals.compare import compare
from sensai.evals.compare import main as compare_main
from sensai.evals.quality_report import to_markdown
from sensai.evals.questions import QuestionCase

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


def report(variant, mean, scores):
    questions = [{"id": question, "score": score} for question, score in scores.items()]
    return {
        "model": "m",
        "variant": variant,
        "summary": {"mean_score": mean},
        "questions": questions,
    }


def test_compare_shows_the_mean_and_each_question_side_by_side():
    before = report("v2@a + 0 exemples", 2.5, {"q1": 1, "q2": 4})
    after = report("v2@a + 2 exemples", 4.0, {"q1": 4, "q3": 5})

    table = compare(before, after)

    assert "| Variante | v2@a + 0 exemples | v2@a + 2 exemples | |" in table
    assert "| Score moyen | 2.5 | 4 | +1.5 |" in table
    assert "| q1 | 1 | 4 | +3 |" in table
    assert "| q2 | 4 | - | - |" in table and "| q3 | - | 5 | - |" in table


def test_a_report_written_before_a7_is_read_as_zero_examples():
    old = {"prompt_version": "v1@b", "summary": {"mean_score": None}, "questions": []}
    assert "v1@b + 0 exemples" in compare(old, old)


def test_compare_command_prints_the_table_or_exits_2(tmp_path, capsys):
    good, bad = tmp_path / "a.json", tmp_path / "b.json"
    good.write_text(json.dumps(report("v", 3, {"q": 3})), encoding="utf-8")
    bad.write_text('{"model": "m"}', encoding="utf-8")

    assert compare_main([str(good), str(good)]) == 0
    assert "| q | 3 | 3 | +0 |" in capsys.readouterr().out
    assert compare_main([str(good), str(bad)]) == 2
    assert "pas un rapport EV1" in capsys.readouterr().err
    assert compare_main([str(good), str(tmp_path / "absent.json")]) == 2


async def test_ev1_records_the_variant_and_the_examples_sent():
    config = load_config(
        overrides={"react": {"enabled": False}, "prompts": {"system": "v2", "examples": 1}}
    )
    case = QuestionCase(id="q", question="Que renvoie la fonction find_user ?", reference="None")
    grade = Chunk(json.dumps({"verdict": "correct", "score": 5, "reason": "Juste."}))
    agent, judge = FakeModelGateway([Chunk("None.")]), FakeModelGateway([grade])

    results, info = await ev1.run([case], config, "j", agent, judge, index=False)

    assert results[0].answer.examples == ["valeur-renvoyee"]
    assert info.examples == 1 and info.variant.endswith("+ 1 exemples")
    assert "- Few-shots : 1 exemples au plus par question" in to_markdown(results, info)
