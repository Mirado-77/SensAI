"""Reports of one EV1 run: a Markdown page for people and a JSON file for scripts (and A7)."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sensai.evals.grader import Grade, GradeVerdict
from sensai.evals.questions import Answer

VERDICTS: tuple[GradeVerdict, ...] = ("correct", "partial", "incorrect")
LABELS = {
    "correct": "correcte",
    "partial": "partielle",
    "incorrect": "incorrecte",
    None: "non notée",
}
EXCERPT_CHARS = 300


@dataclass(frozen=True)
class GradedAnswer:
    answer: Answer
    grade: Grade | None


@dataclass(frozen=True)
class RunInfo:
    model: str
    judge_model: str
    prompt_version: str | None
    when: datetime
    notes: list[str]
    examples: int = 0  # few-shot examples per question (A7)

    @property
    def variant(self) -> str:
        """What was evaluated besides the model: the prompt version and the few-shots."""
        return f"{self.prompt_version or 'aucune'} + {self.examples} exemples"


def summarize(results: list[GradedAnswer]) -> dict:
    """Counts per verdict and the mean score. Ungraded answers (judge failure) are left out."""
    grades = [result.grade for result in results if result.grade is not None]
    return {
        "total": len(results),
        "verdicts": {verdict: sum(g.verdict == verdict for g in grades) for verdict in VERDICTS},
        "ungraded": len(results) - len(grades),
        "mean_score": sum(g.score for g in grades) / len(grades) if grades else None,
    }


def write_reports(results: list[GradedAnswer], info: RunInfo, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ev1-{info.when:%Y%m%d-%H%M%S}"
    markdown = out_dir / f"{stem}.md"
    payload = out_dir / f"{stem}.json"
    markdown.write_text(to_markdown(results, info), encoding="utf-8")
    payload.write_text(to_json(results, info), encoding="utf-8")
    return markdown, payload


def to_json(results: list[GradedAnswer], info: RunInfo) -> str:
    document = {
        "model": info.model,
        "judge_model": info.judge_model,
        "prompt_version": info.prompt_version,
        "examples": info.examples,
        "variant": info.variant,
        "date": info.when.isoformat(timespec="seconds"),
        "notes": info.notes,
        "summary": summarize(results),
        "questions": [_question_json(result) for result in results],
    }
    return json.dumps(document, ensure_ascii=False, indent=2)


def to_markdown(results: list[GradedAnswer], info: RunInfo) -> str:
    summary = summarize(results)
    lines = [
        "# Rapport EV1 : juge LLM sur le corpus de questions",
        "",
        f"- Modèle : `{info.model}`",
        f"- Juge : `{info.judge_model}`",
        f"- Version de prompt : `{info.prompt_version or 'aucune'}`",
        f"- Few-shots : {info.examples} exemples au plus par question",
        f"- Date : {info.when:%Y-%m-%d %H:%M}",
        f"- Questions : {summary['total']}",
        f"- Score moyen : {_score(summary['mean_score'])}",
        *(f"- Note : {note}" for note in info.notes),
        "",
        "## Résultats par verdict",
        "",
        "| Verdict | Nombre |",
        "| --- | ---: |",
        *(f"| {LABELS[verdict]} | {summary['verdicts'][verdict]} |" for verdict in VERDICTS),
        f"| {LABELS[None]} | {summary['ungraded']} |",
        "",
        "## Questions",
        "",
        "| Question | Verdict | Score | Appels |",
        "| --- | --- | ---: | --- |",
        *(_question_row(result) for result in results),
        "",
    ]
    lines += [line for result in results for line in _question_details(result)]
    return "\n".join(lines)


def _question_json(result: GradedAnswer) -> dict:
    answer, grade = result.answer, result.grade
    return {
        "id": answer.case.id,
        "status": answer.status,
        "error": answer.error,
        "verdict": grade.verdict if grade else None,
        "score": grade.score if grade else None,
        "reason": grade.reason if grade else "le juge n'a pas produit de note valide",
        "calls": answer.calls,
        "examples": answer.examples,
        "answer": answer.answer[:EXCERPT_CHARS],
    }


def _score(score: float | None) -> str:
    return "n/a (aucune réponse notée)" if score is None else f"{score:.2f} / 5"


def _question_row(result: GradedAnswer) -> str:
    grade = result.grade
    cells = [
        result.answer.case.id,
        LABELS[grade.verdict if grade else None],
        str(grade.score) if grade else "-",
        ", ".join(result.answer.calls) or "aucun",
    ]
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"


def _question_details(result: GradedAnswer) -> list[str]:
    answer, grade = result.answer, result.grade
    reason = grade.reason if grade else "le juge n'a pas produit de note valide"
    excerpt = answer.answer[:EXCERPT_CHARS].replace("\n", " ") or "(vide)"
    return [
        f"### {answer.case.id}",
        "",
        f"- Question : {answer.case.question}",
        f"- Statut du tour : {answer.status}",
        *([f"- Erreur du tour : {answer.error}"] if answer.error else []),
        f"- Raison du juge : {reason}",
        "",
        f"> {excerpt}",
        "",
    ]
