"""`python -m sensai.evals`: run the attack corpus against the configured model, then write the
report. Exit codes: 0 report written, 2 unusable cases or configuration."""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from sensai.core.config import ConfigError, load_config
from sensai.core.errors import SensaiError
from sensai.evals.cases import load_cases
from sensai.evals.report import summarize, write_reports
from sensai.evals.runner import run_cases

DEFAULT_CASES = Path("evals/attacks")
DEFAULT_OUT = Path("evals/reports")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        cases = load_cases(args.cases)
    except (ConfigError, SensaiError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2
    if args.only:
        cases = [case for case in cases if case.id in args.only]
    when = datetime.now()
    print(f"EV4 : {len(cases)} cas, modèle {config.model.name}", file=sys.stderr)
    results = asyncio.run(run_cases(cases, config, sandbox=not args.no_sandbox))
    markdown, payload = write_reports(results, config.model.name, when, args.out)
    counts = summarize(results)["verdicts"]
    print(
        f"Bloquées : {counts['blocked']}, réussies : {counts['attack_succeeded']}, "
        f"indéterminées : {counts['indeterminate']}"
    )
    print(f"Rapport : {markdown}")
    print(f"JSON : {payload}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sensai.evals", description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="dossier des cas")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="dossier des rapports")
    parser.add_argument("--config", type=Path, default=None, help="fichier de configuration")
    parser.add_argument("--only", action="append", default=[], help="ne lancer que ce cas")
    parser.add_argument("--no-sandbox", action="store_true", help="sans run_command (sans Docker)")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
