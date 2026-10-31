"""Metrics of a benchmark: each one reduces the samples of a variant to a single number."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    """What one case measured. `score` is set by the judge (questions), `blocked` by the attack
    judgement (attacks); both stay None when the case could not be rated."""

    case_id: str
    ok: bool
    latency_s: float
    tokens: int
    score: float | None = None
    blocked: bool | None = None


@dataclass(frozen=True)
class Metric:
    label: str
    unit: str
    kinds: frozenset[str]
    lower_is_better: bool
    compute: Callable[[list[Sample]], float | None]


ALL_KINDS = frozenset({"questions", "attacks"})


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percent(flags: list[bool]) -> float | None:
    return 100 * sum(flags) / len(flags) if flags else None


METRICS: dict[str, Metric] = {
    "latency": Metric(
        "Latence moyenne", "s", ALL_KINDS, True, lambda s: _mean([x.latency_s for x in s])
    ),
    "tokens": Metric("Tokens par cas", "", ALL_KINDS, True, lambda s: _mean([x.tokens for x in s])),
    "error_rate": Metric(
        "Taux d'erreur", "%", ALL_KINDS, True, lambda s: _percent([not x.ok for x in s])
    ),
    "judge_score": Metric(
        "Score du juge",
        "/5",
        frozenset({"questions"}),
        False,
        lambda s: _mean([x.score for x in s if x.score is not None]),
    ),
    "block_rate": Metric(
        "Attaques bloquées",
        "%",
        frozenset({"attacks"}),
        False,
        lambda s: _percent([x.blocked for x in s if x.blocked is not None]),
    ),
}


def compute_metrics(names: list[str], samples: list[Sample]) -> dict[str, float | None]:
    return {name: METRICS[name].compute(samples) for name in names}


def format_value(name: str, value: float | None) -> str:
    if value is None:
        return "n/a"
    metric = METRICS[name]
    text = f"{value:.0f}" if name == "tokens" else f"{value:.2f}"
    return f"{text} {metric.unit}".strip()
