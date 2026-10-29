"""`sensai stats` (EV3): runs, tokens, latency, tools and errors from the event journal."""

import argparse
import json
import re
import time
from datetime import datetime
from typing import Any

from rich.text import Text

from sensai.cli.terminal import Terminal

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_PERIOD = re.compile(r"(\d+)([smhd])")


def parse_period(value: str) -> float:
    """argparse type of `--since`: `30m`, `2h`, `1d`... as a number of seconds."""
    match = _PERIOD.fullmatch(value.strip())
    if match is None or int(match.group(1)) == 0:
        raise argparse.ArgumentTypeError(f"invalid period {value!r} (e.g. 30m, 2h, 1d)")
    return int(match.group(1)) * _UNITS[match.group(2)]


def since_timestamp(period: float | None, now: float | None = None) -> float | None:
    return None if period is None else (time.time() if now is None else now) - period


def print_stats(stats: dict[str, Any], terminal: Terminal, as_json: bool = False) -> int:
    if as_json:
        terminal.console.print(Text(json.dumps(stats, indent=2)), soft_wrap=True)
        return 0
    overview = stats["overview"]
    if not overview["runs"]:
        terminal.console.print(Text("No run recorded yet.", "dim"))
        return 0
    _print_overview(overview, terminal)
    _print_latency(stats["latency"], terminal)
    _print_tools(stats["tools"], terminal)
    guards = stats["guards"]
    terminal.console.print(
        Text(f"{'Guards':<12} {guards['blocked']} blocked, {guards['redacted']} redacted")
    )
    _print_errors(stats["errors"], terminal)
    return 0


def _print_overview(overview: dict[str, Any], terminal: Terminal) -> None:
    statuses = ", ".join(
        f"{status} {overview[status]}"
        for status in ("completed", "cancelled", "failed", "running")
        if overview[status]
    )
    total = overview["prompt_tokens"] + overview["completion_tokens"]
    terminal.console.print(Text(f"{'Runs':<12} {overview['runs']} ({statuses})"))
    terminal.console.print(
        Text(
            f"{'Tokens':<12} {total} total, {overview['avg_tokens']:.0f} per run"
            f" (prompt {overview['prompt_tokens']}, completion {overview['completion_tokens']})"
        )
    )


def _print_latency(latency: dict[str, Any], terminal: Terminal) -> None:
    for label, summary in (("Turn", latency["turn"]), ("First token", latency["first_token"])):
        if not summary["count"]:
            continue
        values = ", ".join(f"{key} {summary[key]:.1f} s" for key in ("avg", "p50", "p95", "max"))
        terminal.console.print(Text(f"{label:<12} {values}"))


def _print_tools(tools: list[dict[str, Any]], terminal: Terminal) -> None:
    if not tools:
        return
    terminal.console.print()
    header = f"{'tool':<20} {'calls':>5} {'ok':>4} {'errors':>6} {'refused':>7} {'avg ms':>8}"
    terminal.console.print(Text(header, "bold"))
    for tool in tools:
        avg_ms = "-" if tool["avg_ms"] is None else f"{tool['avg_ms']:.0f}"
        terminal.console.print(
            Text(
                f"{tool['tool']:<20} {tool['calls']:>5} {tool['ok']:>4} {tool['errors']:>6} "
                f"{tool['refused']:>7} {avg_ms:>8}"
            )
        )
    terminal.console.print()


def _print_errors(errors: list[dict[str, Any]], terminal: Terminal) -> None:
    if not errors:
        return
    terminal.console.print()
    terminal.console.print(Text("Recent errors", "bold"))
    for error in errors:
        when = datetime.fromtimestamp(error["ts"]).strftime("%Y-%m-%d %H:%M")
        line = f"{when}  {error['type']:<11} {error['kind']}: {error['message']}"
        terminal.console.print(Text(line, "yellow"), soft_wrap=True)
