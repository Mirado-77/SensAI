"""Chart of a benchmark: one panel per metric, one bar per variant."""

import math
from pathlib import Path

from sensai.benchmarks.metrics import METRICS, format_value
from sensai.benchmarks.runner import BenchmarkResult

PANEL_COLUMNS = 3
PANEL_SIZE = (4.6, 3.6)


def plot(result: BenchmarkResult, path: Path) -> bool:
    """Write the PNG. False when matplotlib is not installed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    names = result.spec.metrics
    columns = min(len(names), PANEL_COLUMNS)
    rows = math.ceil(len(names) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(PANEL_SIZE[0] * columns, PANEL_SIZE[1] * rows), squeeze=False
    )
    labels = [variant.variant.name for variant in result.variants]
    colors = [plt.get_cmap("tab10")(index) for index in range(len(labels))]
    for axis, name in zip(axes.flat, names, strict=False):
        _panel(axis, name, labels, colors, result)
    for axis in list(axes.flat)[len(names) :]:
        axis.set_visible(False)
    figure.suptitle(result.spec.title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def _panel(axis, name: str, labels: list[str], colors: list, result: BenchmarkResult) -> None:
    metric = METRICS[name]
    values = [variant.metrics[name] for variant in result.variants]
    heights = [value or 0.0 for value in values]
    bars = axis.bar(labels, heights, color=colors)
    for bar, value in zip(bars, values, strict=True):
        axis.annotate(
            format_value(name, value),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points",
        )
    hint = "plus bas = mieux" if metric.lower_is_better else "plus haut = mieux"
    axis.set_title(f"{metric.label} ({hint})", fontsize=9)
    axis.set_ylim(0, max(heights) * 1.15 or 1)
    axis.tick_params(axis="x", labelrotation=20, labelsize=8)
    axis.spines[["top", "right"]].set_visible(False)
