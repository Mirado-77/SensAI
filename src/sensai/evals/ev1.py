"""`python -m sensai.evals.ev1`: answer the question corpus with the configured model and prompt
version, grade each answer with the LLM judge, then write the report. Exit codes: 0 report
written, 2 unusable questions or configuration, 130 interrupted with Ctrl-C."""

import argparse
import asyncio
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from sensai.core.config import Config, ConfigError, load_config
from sensai.core.contracts import ModelGateway
from sensai.core.errors import SensaiError
from sensai.evals.cases import load_cases
from sensai.evals.grader import grade_answer
from sensai.evals.quality_report import GradedAnswer, RunInfo, summarize, write_reports
from sensai.evals.questions import QuestionCase, answer_questions
from sensai.model.ollama import OllamaGateway

DEFAULT_QUESTIONS = Path("evals/questions")
DEFAULT_OUT = Path("evals/reports")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        prompts = {"system": args.prompt, "examples": args.examples}
        overrides = {"prompts": {k: v for k, v in prompts.items() if v is not None}}
        config = load_config(args.config, overrides=overrides)
        cases = load_cases(args.questions, QuestionCase)
    except (ConfigError, SensaiError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2
    if args.only:
        cases = [case for case in cases if case.id in args.only]
    judge_model = args.judge_model or config.model.name
    print(
        f"EV1 : {len(cases)} questions, modèle {config.model.name}, juge {judge_model}",
        file=sys.stderr,
    )
    try:
        results, info = asyncio.run(
            run(cases, config, judge_model, index=not args.no_index, progress=_progress)
        )
    except SensaiError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrompu : aucun rapport écrit.", file=sys.stderr)
        return 130
    markdown, payload = write_reports(results, info, args.out)
    summary = summarize(results)
    counts = summary["verdicts"]
    print(
        f"Correctes : {counts['correct']}, partielles : {counts['partial']}, "
        f"incorrectes : {counts['incorrect']}, non notées : {summary['ungraded']}"
    )
    print(f"Rapport : {markdown}")
    print(f"JSON : {payload}")
    return 0


async def run(
    cases: list[QuestionCase],
    config: Config,
    judge_model: str,
    gateway: ModelGateway | None = None,
    judge: ModelGateway | None = None,
    index: bool = True,
    progress: Callable[[str], None] = lambda _: None,
) -> tuple[list[GradedAnswer], RunInfo]:
    """Answer every question, then grade the answers one after the other."""
    when = datetime.now()
    answers, version, notes = await answer_questions(
        cases, config, gateway, index=index, progress=progress
    )
    model = config.model
    judge = judge or OllamaGateway(model.host, judge_model, model.timeout_s, model.think)
    results = []
    for number, answer in enumerate(answers, 1):
        progress(f"[{number}/{len(answers)}] {answer.case.id} : note du juge")
        grade = await grade_answer(
            judge,
            answer.case,
            answer.answer,
            config.structured.max_repairs,
            config.runtime.timeout_s,
        )
        results.append(GradedAnswer(answer, grade))
    info = RunInfo(model.name, judge_model, version, when, notes, config.prompts.examples)
    return results, info


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sensai.evals.ev1", description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, help="dossier")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="dossier des rapports")
    parser.add_argument("--config", type=Path, default=None, help="fichier de configuration")
    parser.add_argument("--prompt", default=None, help="version de prompt à évaluer (A4)")
    parser.add_argument(
        "--examples", type=int, default=None, help="few-shots par question (A7), 0 = aucun"
    )
    parser.add_argument("--judge-model", default=None, help="modèle du juge (défaut : le modèle)")
    parser.add_argument("--only", action="append", default=[], help="ne lancer que ce cas")
    parser.add_argument("--no-index", action="store_true", help="sans search_code (sans index)")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
