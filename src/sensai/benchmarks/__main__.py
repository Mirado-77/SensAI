"""`python -m sensai.benchmarks NAME...`: run the benchmarks of `benchmarks/` and write, for each,
a Markdown report, a JSON file and a PNG chart in `docs/benchmarks/`. Exit codes: 0 reports
written, 2 unusable benchmark or configuration, 130 interrupted with Ctrl-C."""

import argparse
import asyncio
import sys
from pathlib import Path

from sensai.benchmarks.report import write_reports
from sensai.benchmarks.runner import run_benchmark
from sensai.benchmarks.spec import DEFAULT_DIR, list_specs, load_spec, resolve
from sensai.core.config import ConfigError
from sensai.core.errors import SensaiError

DEFAULT_OUT = Path("docs/benchmarks")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(list_specs()) or f"Aucun benchmark dans {DEFAULT_DIR}/")
        return 0
    if not args.names:
        parser.error("donne le nom d'un benchmark (voir --list)")
    try:
        for name in args.names:
            _run_one(name, args)
    except (ConfigError, SensaiError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrompu : aucun rapport écrit pour le benchmark en cours.", file=sys.stderr)
        return 130
    return 0


def _run_one(name: str, args: argparse.Namespace) -> None:
    spec = load_spec(resolve(name))
    print(f"Benchmark {spec.name} : {len(spec.variants)} variantes", file=sys.stderr)
    result = asyncio.run(
        run_benchmark(spec, args.config, args.judge_model, only=args.only, progress=_progress)
    )
    markdown, payload, image = write_reports(result, args.out, with_plot=not args.no_plot)
    print(f"Rapport : {markdown}")
    print(f"JSON : {payload}")
    print(f"Graphique : {image}" if image else "Graphique : non généré (matplotlib absent)")


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sensai.benchmarks", description=__doc__)
    parser.add_argument("names", nargs="*", help="nom (benchmarks/NOM.toml) ou chemin du fichier")
    parser.add_argument("--list", action="store_true", help="lister les benchmarks disponibles")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="dossier des rapports")
    parser.add_argument("--config", type=Path, default=None, help="fichier de configuration")
    parser.add_argument("--judge-model", default=None, help="modèle du juge (défaut : le modèle)")
    parser.add_argument("--only", action="append", default=[], help="ne lancer que ce cas")
    parser.add_argument("--no-plot", action="store_true", help="ne pas générer le graphique")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
