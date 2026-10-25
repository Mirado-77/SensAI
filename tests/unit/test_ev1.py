import json
from datetime import datetime
from pathlib import Path

import pytest

from sensai.core.config import load_config
from sensai.core.fakes import FakeModelGateway
from sensai.core.types import Chunk, ToolCall
from sensai.evals import ev1
from sensai.evals.cases import CaseError, load_cases
from sensai.evals.grader import NO_ANSWER, Grade, grade_answer
from sensai.evals.quality_report import GradedAnswer, RunInfo, summarize, to_markdown
from sensai.evals.questions import Answer, QuestionCase, answer_questions

pytestmark = pytest.mark.feature("EV1")

CORPUS = Path(__file__).resolve().parents[2] / "evals" / "questions"
CONFIG = load_config(
    overrides={"react": {"enabled": False, "max_plan_steps": 5}, "prompts": {"system": "v2"}}
)
CASE = QuestionCase(
    id="ev1-test",
    question="Que renvoie load_tasks sans fichier ?",
    reference="Une liste vide.",
    criteria=["répond liste vide"],
)


def grade_json(verdict="correct", score=5, reason="Conforme à la référence."):
    return Chunk(json.dumps({"verdict": verdict, "score": score, "reason": reason}))


@pytest.fixture
def source(tmp_path) -> Path:
    repo = tmp_path / "demo"
    repo.mkdir()
    (repo / "storage.py").write_text("def load_tasks(path):\n    return []\n", encoding="utf-8")
    return repo


def test_corpus_loads_with_unique_ids():
    cases = load_cases(CORPUS, QuestionCase)

    assert len(cases) >= 5
    assert all(case.criteria for case in cases)


def test_question_with_unknown_field_is_refused(tmp_path):
    (tmp_path / "bad.toml").write_text(
        'id = "x"\nquestion = "q"\nreference = "r"\nexpected = "?"\n', encoding="utf-8"
    )

    with pytest.raises(CaseError, match="bad.toml"):
        load_cases(tmp_path, QuestionCase)


async def test_judge_sees_question_reference_criteria_and_answer():
    judge = FakeModelGateway([grade_json()])

    grade = await grade_answer(judge, CASE, "Elle renvoie [].")

    assert grade == Grade(verdict="correct", score=5, reason="Conforme à la référence.")
    prompt = judge.calls[0].messages[-1].content
    assert "Une liste vide." in prompt and "répond liste vide" in prompt
    assert "<answer>\nElle renvoie [].\n</answer>" in prompt
    assert judge.calls[0].schema is not None


async def test_empty_answer_is_incorrect_without_calling_the_judge():
    judge = FakeModelGateway()

    assert await grade_answer(judge, CASE, "  ") == NO_ANSWER
    assert judge.calls == []


async def test_judge_that_never_returns_a_valid_grade_leaves_the_answer_ungraded():
    judge = FakeModelGateway([Chunk("pas du JSON")], [grade_json(score=9)])

    assert await grade_answer(judge, CASE, "[]", max_repairs=1) is None


async def test_agent_answers_with_the_prompt_version_and_reads_the_copy(source):
    agent = FakeModelGateway(
        [ToolCall("read_file", {"path": "storage.py"})], [Chunk("Une liste vide.")]
    )

    answers, version, notes = await answer_questions([CASE], CONFIG, agent, source, index=False)

    assert answers[0].status == "completed"
    assert answers[0].answer == "Une liste vide."
    assert answers[0].calls == ["read_file"]
    assert version.startswith("v2@")
    assert agent.calls[0].messages[0].role == "system"
    assert "SENSAI" in agent.calls[0].messages[0].content
    assert notes == []


async def test_disabled_rag_is_noted_in_the_report(source):
    config = CONFIG.model_copy(update={"rag": CONFIG.rag.model_copy(update={"enabled": False})})
    agent = FakeModelGateway([Chunk("ok")])

    _, _, notes = await answer_questions([CASE], config, agent, source)

    assert notes == ["search_code désactivé : [rag] enabled = false"]


async def test_run_grades_every_answer():
    agent = FakeModelGateway([Chunk("Une liste vide.")], [Chunk("Je ne sais pas.")])
    judge = FakeModelGateway([grade_json()], [grade_json("incorrect", 1, "Hors sujet.")])
    other = CASE.model_copy(update={"id": "ev1-autre"})

    messages = []

    results, info = await ev1.run(
        [CASE, other], CONFIG, "juge", agent, judge, index=False, progress=messages.append
    )

    summary = summarize(results)
    assert summary["verdicts"] == {"correct": 1, "partial": 0, "incorrect": 1}
    assert summary["mean_score"] == 3
    assert info.judge_model == "juge" and info.prompt_version.startswith("v2@")
    assert messages == [
        "[1/2] ev1-test : réponse de l'agent",
        "[2/2] ev1-autre : réponse de l'agent",
        "[1/2] ev1-test : note du juge",
        "[2/2] ev1-autre : note du juge",
    ]


def test_markdown_report_lists_version_scores_and_ungraded_answers():
    answer = Answer(CASE, "completed", "Une liste vide.", ["read_file"])
    results = [
        GradedAnswer(answer, Grade(verdict="partial", score=3, reason="Incomplet.")),
        GradedAnswer(answer, None),
    ]
    info = RunInfo("m", "j", "v2@abcd1234", datetime(2026, 10, 25, 12, 0), ["note"])

    markdown = to_markdown(results, info)

    assert "Version de prompt : `v2@abcd1234`" in markdown
    assert "Score moyen : 3.00 / 5" in markdown
    assert "| non notée | 1 |" in markdown
    assert "| ev1-test | partielle | 3 | read_file |" in markdown
    assert "- Note : note" in markdown


def test_main_exits_2_on_unknown_prompt_version(tmp_path, capsys):
    assert ev1.main(["--questions", str(CORPUS), "--prompt", "absente"]) == 2
    assert "absente" in capsys.readouterr().err
