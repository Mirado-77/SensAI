"""Reports of one benchmark run: Markdown for people, JSON for scripts, PNG chart."""

import json
from pathlib import Path

from sensai.benchmarks.metrics import METRICS, Sample, format_value
from sensai.benchmarks.plot import plot
from sensai.benchmarks.runner import BenchmarkResult, VariantResult


def write_reports(
    result: BenchmarkResult, out_dir: Path, with_plot: bool = True
) -> tuple[Path, Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{result.spec.name}-{result.when:%Y%m%d-%H%M%S}"
    image = out_dir / f"{stem}.png"
    drawn = with_plot and plot(result, image)
    markdown = out_dir / f"{stem}.md"
    payload = out_dir / f"{stem}.json"
    markdown.write_text(to_markdown(result, image.name if drawn else None), encoding="utf-8")
    payload.write_text(to_json(result), encoding="utf-8")
    return markdown, payload, image if drawn else None


def to_json(result: BenchmarkResult) -> str:
    document = {
        "benchmark": result.spec.name,
        "title": result.spec.title,
        "kind": result.spec.kind,
        "date": result.when.isoformat(timespec="seconds"),
        "model": result.model,
        "judge_model": result.judge_model,
        "variants": [_variant_json(variant) for variant in result.variants],
    }
    return json.dumps(document, ensure_ascii=False, indent=2)


def to_markdown(result: BenchmarkResult, image: str | None) -> str:
    spec = result.spec
    lines = [
        f"# Benchmark {spec.name} : {spec.title}",
        "",
        *([spec.description, ""] if spec.description else []),
        f"- Modèle : `{result.model}`",
        *([f"- Juge : `{result.judge_model}`"] if result.judge_model else []),
        f"- Date : {result.when:%Y-%m-%d %H:%M}",
        f"- Corpus : `{spec.corpus}` ({_case_count(result)} cas)",
        *(f"- Note ({v.variant.name}) : {note}" for v in result.variants for note in v.notes),
        "",
        *([f"![{spec.title}]({image})", ""] if image else []),
        "## Métriques",
        "",
        *_metrics_table(result),
        "",
        "## Détail par cas",
        "",
        *_cases_table(result),
        "",
    ]
    return "\n".join(lines)


def _case_count(result: BenchmarkResult) -> int:
    return max((len(variant.samples) for variant in result.variants), default=0)


def _metrics_table(result: BenchmarkResult) -> list[str]:
    names = [variant.variant.name for variant in result.variants]
    lines = ["| Métrique | " + " | ".join(names) + " |", "| --- |" + " ---: |" * len(names)]
    for name in result.spec.metrics:
        cells = [format_value(name, variant.metrics[name]) for variant in result.variants]
        lines.append(f"| {METRICS[name].label} | " + " | ".join(cells) + " |")
    return lines


def _cases_table(result: BenchmarkResult) -> list[str]:
    names = [variant.variant.name for variant in result.variants]
    lines = ["| Cas | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
    by_variant = [{sample.case_id: sample for sample in v.samples} for v in result.variants]
    case_ids = list(dict.fromkeys(case for samples in by_variant for case in samples))
    for case_id in case_ids:
        cells = [_cell(samples.get(case_id)) for samples in by_variant]
        lines.append(f"| {case_id} | " + " | ".join(cells) + " |")
    return lines


def _cell(sample: Sample | None) -> str:
    if sample is None:
        return "-"
    parts = []
    if sample.score is not None:
        parts.append(f"{sample.score:.0f}/5")
    if sample.blocked is not None:
        parts.append("bloquée" if sample.blocked else "réussie")
    parts.append(f"{sample.latency_s:.1f} s")
    parts.append(f"{sample.tokens} tok")
    if not sample.ok:
        parts.append("erreur")
    return " · ".join(parts)


def _variant_json(variant: VariantResult) -> dict:
    return {
        "name": variant.variant.name,
        "overrides": variant.variant.overrides,
        "metrics": variant.metrics,
        "notes": variant.notes,
        "samples": [
            {
                "case": sample.case_id,
                "ok": sample.ok,
                "latency_s": round(sample.latency_s, 3),
                "tokens": sample.tokens,
                "score": sample.score,
                "blocked": sample.blocked,
            }
            for sample in variant.samples
        ],
    }
