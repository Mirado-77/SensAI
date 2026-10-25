"""Reports of one EV4 run: a Markdown page for people and a JSON file for scripts."""

import json
from datetime import datetime
from pathlib import Path

from sensai.evals.judge import Verdict
from sensai.evals.runner import CaseResult

VERDICTS: tuple[Verdict, ...] = ("blocked", "attack_succeeded", "indeterminate")
LABELS = {
    "blocked": "bloquée",
    "attack_succeeded": "réussie",
    "indeterminate": "indéterminée",
}
EXCERPT_CHARS = 300


def summarize(results: list[CaseResult]) -> dict:
    """Counts per verdict and per category. The blocked rate ignores indeterminate cases."""
    counts = {verdict: _count(results, verdict) for verdict in VERDICTS}
    decided = counts["blocked"] + counts["attack_succeeded"]
    return {
        "total": len(results),
        "verdicts": counts,
        "blocked_rate": counts["blocked"] / decided if decided else None,
        "by_category": _by_category(results),
    }


def write_reports(
    results: list[CaseResult], model: str, when: datetime, out_dir: Path
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ev4-{when:%Y%m%d-%H%M%S}"
    markdown = out_dir / f"{stem}.md"
    payload = out_dir / f"{stem}.json"
    markdown.write_text(to_markdown(results, model, when), encoding="utf-8")
    payload.write_text(to_json(results, model, when), encoding="utf-8")
    return markdown, payload


def to_json(results: list[CaseResult], model: str, when: datetime) -> str:
    document = {
        "model": model,
        "date": when.isoformat(timespec="seconds"),
        "summary": summarize(results),
        "cases": [_case_json(result) for result in results],
    }
    return json.dumps(document, ensure_ascii=False, indent=2)


def to_markdown(results: list[CaseResult], model: str, when: datetime) -> str:
    summary = summarize(results)
    lines = [
        "# Rapport EV4 : corpus d'attaques",
        "",
        f"- Modèle : `{model}`",
        f"- Date : {when:%Y-%m-%d %H:%M}",
        f"- Cas : {summary['total']}",
        f"- Taux de blocage : {_rate(summary['blocked_rate'])}",
        "",
        "## Résultats par verdict",
        "",
        "| Verdict | Nombre |",
        "| --- | ---: |",
        *(f"| {LABELS[verdict]} | {summary['verdicts'][verdict]} |" for verdict in VERDICTS),
        "",
        "## Résultats par catégorie",
        "",
        "| Catégorie | Bloquées | Réussies | Indéterminées |",
        "| --- | ---: | ---: | ---: |",
        *_category_rows(summary["by_category"]),
        "",
        "## Cas",
        "",
        "| Cas | Catégorie | Verdict | Appels demandés |",
        "| --- | --- | --- | --- |",
        *(_case_row(result) for result in results),
        "",
    ]
    lines += [line for result in results for line in _case_details(result)]
    return "\n".join(lines)


def _count(results: list[CaseResult], verdict: Verdict) -> int:
    return sum(1 for result in results if result.judgement.verdict == verdict)


def _by_category(results: list[CaseResult]) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for result in results:
        row = table.setdefault(result.case.category, dict.fromkeys(VERDICTS, 0))
        row[result.judgement.verdict] += 1
    return table


def _case_json(result: CaseResult) -> dict:
    return {
        "id": result.case.id,
        "category": result.case.category,
        "verdict": result.judgement.verdict,
        "reasons": result.judgement.reasons,
        "status": result.status,
        "expected": result.case.expected,
        "calls": result.calls,
        "answer": result.answer[:EXCERPT_CHARS],
    }


def _rate(rate: float | None) -> str:
    return "n/a (aucun cas tranché)" if rate is None else f"{rate:.0%}"


def _category_rows(table: dict[str, dict[str, int]]) -> list[str]:
    return [
        f"| {category} | {row['blocked']} | {row['attack_succeeded']} | {row['indeterminate']} |"
        for category, row in sorted(table.items())
    ]


def _case_row(result: CaseResult) -> str:
    calls = ", ".join(result.calls) or "aucun"
    cells = [result.case.id, result.case.category, LABELS[result.judgement.verdict], calls]
    return "| " + " | ".join(_cell(cell) for cell in cells) + " |"


def _case_details(result: CaseResult) -> list[str]:
    lines = [f"### {result.case.id}", "", f"- Attendu : {result.case.expected}"]
    lines.append(f"- Statut du tour : {result.status}")
    lines += [f"- Raison : {reason}" for reason in result.judgement.reasons]
    excerpt = result.answer[:EXCERPT_CHARS].replace("\n", " ") or "(vide)"
    lines += ["", f"> {excerpt}", ""]
    return lines


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")
