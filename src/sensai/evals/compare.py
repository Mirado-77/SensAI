"""`python -m sensai.evals.compare A.json B.json` (A7): compare two EV1 reports, variant B against
variant A, and print a Markdown table: mean score and score per question. Exit codes: 0 table
printed, 2 unreadable report."""

import argparse
import json
import sys
from pathlib import Path


class ReportError(Exception):
    """A report file is missing, is not JSON, or is not an EV1 report."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sensai.evals.compare", description=__doc__)
    parser.add_argument("before", type=Path, help="rapport EV1 de la variante A (JSON)")
    parser.add_argument("after", type=Path, help="rapport EV1 de la variante B (JSON)")
    args = parser.parse_args(argv)
    try:
        before, after = load_report(args.before), load_report(args.after)
    except ReportError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2
    print(compare(before, after))
    return 0


def load_report(path: Path) -> dict:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"Rapport EV1 illisible ({path}) : {exc}") from exc
    if not isinstance(report, dict) or not {"summary", "questions"} <= report.keys():
        raise ReportError(f"Ce n'est pas un rapport EV1 : {path}")
    return report


def compare(before: dict, after: dict) -> str:
    """Markdown comparison; a question missing from one report shows `-` on that side."""
    scores_a, scores_b = _scores(before), _scores(after)
    mean_a, mean_b = before["summary"]["mean_score"], after["summary"]["mean_score"]
    lines = [
        "# Comparaison EV1 : variante B contre variante A",
        "",
        "| | Variante A | Variante B | Écart |",
        "| --- | --- | --- | ---: |",
        f"| Variante | {_variant(before)} | {_variant(after)} | |",
        f"| Modèle | `{before.get('model')}` | `{after.get('model')}` | |",
        f"| Score moyen | {_number(mean_a)} | {_number(mean_b)} | {_delta(mean_a, mean_b)} |",
        "",
        "| Question | A | B | Écart |",
        "| --- | ---: | ---: | ---: |",
    ]
    for question in [*scores_a, *(q for q in scores_b if q not in scores_a)]:
        a, b = scores_a.get(question), scores_b.get(question)
        lines.append(f"| {question} | {_number(a)} | {_number(b)} | {_delta(a, b)} |")
    return "\n".join(lines)


def _scores(report: dict) -> dict[str, float | None]:
    return {question["id"]: question.get("score") for question in report["questions"]}


def _variant(report: dict) -> str:
    # Reports written before A7 have no variant: rebuild it from the prompt version alone.
    return report.get("variant") or f"{report.get('prompt_version') or 'aucune'} + 0 exemples"


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}".rstrip("0").rstrip(".")


def _delta(a: float | None, b: float | None) -> str:
    return "-" if a is None or b is None else f"{b - a:+.2f}".rstrip("0").rstrip(".")


if __name__ == "__main__":
    raise SystemExit(main())
