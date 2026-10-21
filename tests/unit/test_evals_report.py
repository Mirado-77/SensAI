import json
from datetime import datetime

import pytest

from sensai.evals.cases import AttackCase
from sensai.evals.judge import Judgement
from sensai.evals.report import summarize, to_json, to_markdown, write_reports
from sensai.evals.runner import CaseResult

pytestmark = pytest.mark.feature("EV4")

WHEN = datetime(2026, 10, 18, 14, 5, 9)


def result(case_id, category, verdict, reasons=(), calls=(), answer="", status="completed"):
    case = AttackCase.model_validate(
        {
            "id": case_id,
            "category": category,
            "prompt": "x",
            "expected": "Rien ne fuit.",
            "forbidden": {"tools": ["read_file"]},
        }
    )
    judgement = Judgement(verdict, list(reasons))
    return CaseResult(case, judgement, status, answer, list(calls))


def sample():
    return [
        result("ev4-a", "fuite-secret", "blocked"),
        result(
            "ev4-b",
            "fuite-secret",
            "attack_succeeded",
            ["appel interdit : read_file"],
            ["read_file"],
        ),
        result(
            "ev4-c",
            "commande-dangereuse",
            "indeterminate",
            ["bac à sable indisponible"],
            status="skipped",
        ),
    ]


def test_blocked_rate_ignores_indeterminate_cases():
    summary = summarize(sample())

    assert summary["total"] == 3
    assert summary["verdicts"] == {"blocked": 1, "attack_succeeded": 1, "indeterminate": 1}
    assert summary["blocked_rate"] == 0.5


def test_blocked_rate_is_none_when_nothing_was_decided():
    summary = summarize([result("ev4-c", "commande-dangereuse", "indeterminate")])

    assert summary["blocked_rate"] is None


def test_counts_are_grouped_by_category():
    by_category = summarize(sample())["by_category"]

    assert by_category["fuite-secret"] == {"blocked": 1, "attack_succeeded": 1, "indeterminate": 0}
    assert by_category["commande-dangereuse"]["indeterminate"] == 1


def test_json_report_holds_model_date_and_cases():
    document = json.loads(to_json(sample(), "phi4:latest", WHEN))

    assert document["model"] == "phi4:latest"
    assert document["date"] == "2026-10-18T14:05:09"
    assert document["summary"]["blocked_rate"] == 0.5
    assert document["cases"][1]["calls"] == ["read_file"]
    assert document["cases"][1]["verdict"] == "attack_succeeded"


def test_markdown_report_has_summary_and_one_row_per_case():
    text = to_markdown(sample(), "phi4:latest", WHEN)

    assert "# Rapport EV4 : corpus d'attaques" in text
    assert "- Taux de blocage : 50%" in text
    assert "| ev4-b | fuite-secret | réussie | read_file |" in text
    assert "### ev4-c" in text


def test_pipes_in_answers_do_not_break_the_table():
    text = to_markdown([result("ev4-x", "fuite-secret", "blocked", calls=["a|b"])], "m", WHEN)

    assert "a\\|b" in text


def test_answer_excerpt_is_cut_in_the_json_report():
    long_answer = "x" * 1000

    document = json.loads(
        to_json([result("ev4-x", "fuite-secret", "blocked", answer=long_answer)], "m", WHEN)
    )

    assert len(document["cases"][0]["answer"]) == 300


def test_write_reports_creates_both_files(tmp_path):
    markdown, payload = write_reports(sample(), "phi4:latest", WHEN, tmp_path / "out")

    assert markdown.name == "ev4-20261018-140509.md"
    assert payload.name == "ev4-20261018-140509.json"
    assert json.loads(payload.read_text(encoding="utf-8"))["summary"]["total"] == 3
