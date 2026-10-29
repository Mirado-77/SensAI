"""Play the corpus of a benchmark once per variant and collect one sample per case."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sensai.benchmarks.metrics import Sample, compute_metrics
from sensai.benchmarks.spec import BenchmarkSpec, Variant
from sensai.core.config import Config, load_config
from sensai.core.contracts import ModelGateway
from sensai.evals.cases import AttackCase, load_cases
from sensai.evals.grader import grade_answer
from sensai.evals.questions import Answer, QuestionCase, answer_questions
from sensai.evals.runner import DEMO_REPO, CaseResult, run_cases
from sensai.model.ollama import OllamaGateway


@dataclass(frozen=True)
class VariantResult:
    variant: Variant
    samples: list[Sample]
    metrics: dict[str, float | None]
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkResult:
    spec: BenchmarkSpec
    when: datetime
    model: str
    judge_model: str | None
    variants: list[VariantResult]


async def run_benchmark(
    spec: BenchmarkSpec,
    config_path: Path | None = None,
    judge_model: str | None = None,
    gateway: ModelGateway | None = None,
    judge: ModelGateway | None = None,
    source: Path = DEMO_REPO,
    only: list[str] | None = None,
    progress: Callable[[str], None] = lambda _: None,
) -> BenchmarkResult:
    """Variants run one after the other: a local model cannot serve them in parallel."""
    when = datetime.now()
    base = load_config(config_path)
    judge_name = (judge_model or base.model.name) if spec.judged else None
    results = []
    for number, variant in enumerate(spec.variants, 1):
        progress(f"Variante {variant.name} ({number}/{len(spec.variants)})")
        config = load_config(config_path, overrides=variant.overrides or None)
        grader = (judge or _judge_gateway(config, judge_name)) if spec.judged else None
        samples, notes = await _play(spec, config, gateway, grader, source, only or [], progress)
        metrics = compute_metrics(spec.metrics, samples)
        results.append(VariantResult(variant, samples, metrics, notes))
    return BenchmarkResult(spec, when, base.model.name, judge_name, results)


def _judge_gateway(config: Config, judge_model: str | None) -> ModelGateway:
    model = config.model
    return OllamaGateway(model.host, judge_model or model.name, model.timeout_s, model.temperature)


async def _play(
    spec: BenchmarkSpec,
    config: Config,
    gateway: ModelGateway | None,
    judge: ModelGateway | None,
    source: Path,
    only: list[str],
    progress: Callable[[str], None],
) -> tuple[list[Sample], list[str]]:
    if spec.kind == "questions":
        return await _play_questions(spec, config, gateway, judge, source, only, progress)
    return await _play_attacks(spec, config, gateway, source, only)


async def _play_questions(
    spec: BenchmarkSpec,
    config: Config,
    gateway: ModelGateway | None,
    judge: ModelGateway | None,
    source: Path,
    only: list[str],
    progress: Callable[[str], None],
) -> tuple[list[Sample], list[str]]:
    cases = _select(load_cases(spec.corpus, QuestionCase), only)
    answers, _, notes = await answer_questions(
        cases, config, gateway, source, index=spec.index, progress=progress
    )
    samples = []
    for answer in answers:
        score = await _score(judge, config, answer) if judge else None
        samples.append(_question_sample(answer, score))
    return samples, notes


async def _score(judge: ModelGateway, config: Config, answer: Answer) -> float | None:
    grade = await grade_answer(
        judge,
        answer.case,
        answer.answer,
        config.structured.max_repairs,
        config.runtime.timeout_s,
    )
    return float(grade.score) if grade else None


def _question_sample(answer: Answer, score: float | None) -> Sample:
    tokens = answer.prompt_tokens + answer.completion_tokens
    ok = answer.status == "completed"
    return Sample(answer.case.id, ok, answer.latency_s, tokens, score=score)


async def _play_attacks(
    spec: BenchmarkSpec,
    config: Config,
    gateway: ModelGateway | None,
    source: Path,
    only: list[str],
) -> tuple[list[Sample], list[str]]:
    cases = _select(load_cases(spec.corpus, AttackCase), only)
    results = await run_cases(cases, config, gateway, source)
    played = [result for result in results if result.status != "skipped"]
    notes = [
        f"{result.case.id} ignoré : {result.judgement.reasons[0]}"
        for result in results
        if result.status == "skipped"
    ]
    return [_attack_sample(result) for result in played], notes


def _attack_sample(result: CaseResult) -> Sample:
    verdict = result.judgement.verdict
    blocked = None if verdict == "indeterminate" else verdict == "blocked"
    tokens = result.prompt_tokens + result.completion_tokens
    ok = result.status == "completed"
    return Sample(result.case.id, ok, result.latency_s, tokens, blocked=blocked)


def _select(cases: list, only: list[str]) -> list:
    return [case for case in cases if case.id in only] if only else cases
