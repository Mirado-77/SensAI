"""Command-line entry point."""

import argparse
import asyncio
import contextlib
import signal
import sys
from collections.abc import Coroutine, Sequence
from pathlib import Path
from types import FrameType
from typing import Any

from rich.console import Console
from rich.text import Text

from sensai.cli.ask import run_ask
from sensai.cli.chat import LAST_SESSION, _open_database, print_sessions, run_chat
from sensai.cli.index import run_index
from sensai.cli.prompts import print_prompt_diff, print_prompt_stats, print_prompt_versions
from sensai.cli.schedule import run_schedule
from sensai.cli.stats import parse_period, print_stats, since_timestamp
from sensai.cli.terminal import Terminal
from sensai.core.config import Config, load_config
from sensai.core.errors import ConfigError
from sensai.observability.prompt_stats import prompt_stats
from sensai.observability.stats import run_stats

with contextlib.suppress(ImportError):
    import readline  # noqa: F401  (line editing and history for input())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sensai", description="SENSAI: local AI assistant built on Ollama."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    chat = commands.add_parser("chat", help="start an interactive chat")
    _add_common_options(chat)
    chat.add_argument(
        "--resume",
        nargs="?",
        const=LAST_SESSION,
        metavar="ID",
        help="continue a saved session (the most recent one without ID; an id prefix is enough)",
    )
    chat.add_argument("--prompt", metavar="NAME", help="system prompt version (e.g. v1)")
    chat.add_argument("--persona", metavar="NAME", help="persona to start with (e.g. reviewer)")
    sessions = commands.add_parser("sessions", help="list the saved sessions")
    _add_common_options(sessions)
    ask = commands.add_parser(
        "ask", help="answer one question with JSON conforming to a schema (stdout)"
    )
    _add_common_options(ask)
    ask.add_argument("--schema", type=Path, required=True, help="JSON Schema file of the answer")
    ask.add_argument("question", nargs="+", help="the question, or - to read it from stdin")
    prompts = commands.add_parser("prompts", help="list, compare and measure the prompt versions")
    _add_common_options(prompts)
    actions = prompts.add_subparsers(dest="action")
    actions.add_parser("list", help="list the versions, * marks the active one")
    diff = actions.add_parser("diff", help="show the changes between two versions")
    diff.add_argument("old")
    diff.add_argument("new")
    actions.add_parser("stats", help="compare the versions on the recorded runs")
    index = commands.add_parser("index", help="embed the codebase for search_code (R1)")
    _add_common_options(index)
    index.add_argument("paths", nargs="*", help="folders to index (default: the allowed roots)")
    schedule = commands.add_parser("schedule", help="run the scheduled tasks until Ctrl-C (X5)")
    _add_common_options(schedule)
    stats = commands.add_parser("stats", help="runs, tokens, latency, tools and errors (EV3)")
    _add_common_options(stats)
    stats.add_argument(
        "--since", type=parse_period, metavar="PERIOD", help="only the last 30m, 2h, 1d..."
    )
    stats.add_argument(
        "--errors", type=int, default=5, metavar="N", help="recent errors to show (default 5)"
    )
    stats.add_argument("--json", action="store_true", help="print the metrics as JSON")
    return parser


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", help="Ollama model to use (overrides the configuration)")
    parser.add_argument("--config", type=Path, help="TOML file overriding configs/default.toml")
    parser.add_argument("--temperature", type=float, help="model temperature (remplace configuration)")


def config_from_args(args: argparse.Namespace) -> Config:
    overrides: dict[str, Any] = {"model": {"name": args.model}} if args.model else {}
    prompts = {
        key: value
        for key, value in (
            ("system", getattr(args, "prompt", None)),
            ("persona", getattr(args, "persona", None)),
        )
        if value is not None
    }
    if prompts:
        overrides["prompts"] = prompts
    if args.temperature is not None:
        overrides.setdefault("model", {})["temperature"] = args.temperature
    return load_config(args.config, overrides or None)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    terminal = Terminal(Console(highlight=False), Console(stderr=True, highlight=False))
    try:
        config = config_from_args(args)
    except ConfigError as exc:
        terminal.errors.print(Text(f"Erreur : {exc}", style="bold red"))
        return 2
    if args.command == "ask":
        return _run(run_ask(config, terminal, args.schema, _question(args.question)), terminal)
    if args.command == "index":
        return _run(run_index(config, terminal, args.paths), terminal)
    if args.command == "sessions":
        return print_sessions(config, terminal)
    if args.command == "schedule":
        return _run(run_schedule(config, terminal), terminal)
    if args.command == "prompts":
        return _prompts(args, config, terminal)
    if args.command == "stats":
        return _stats(args, config, terminal)
    return _run(run_chat(config, terminal, resume=args.resume), terminal)


def _stats(args: argparse.Namespace, config: Config, terminal: Terminal) -> int:
    connection = _open_database(config, terminal)
    if connection is None:
        return 2
    try:
        stats = run_stats(connection, since_timestamp(args.since), max(args.errors, 0))
        return print_stats(stats, terminal, args.json)
    finally:
        connection.close()


def _prompts(args: argparse.Namespace, config: Config, terminal: Terminal) -> int:
    if args.action == "diff":
        return print_prompt_diff(terminal, args.old, args.new)
    if args.action == "stats":
        connection = _open_database(config, terminal)
        if connection is None:
            return 2
        try:
            return print_prompt_stats(prompt_stats(connection), terminal)
        finally:
            connection.close()
    return print_prompt_versions(terminal, config.prompts.system)


def _question(words: list[str]) -> str:
    return sys.stdin.read() if words == ["-"] else " ".join(words)


def _run(command: Coroutine[Any, Any, int], terminal: Terminal) -> int:
    """Run a command and return its exit code.

    asyncio.run installs its own SIGINT handler unless a custom one is set. That handler would
    swallow Ctrl-C while input() is waiting at the prompt, so a handler that raises
    KeyboardInterrupt is set first.
    """
    previous_handler = signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    try:
        return asyncio.run(command)
    except KeyboardInterrupt:
        terminal.console.print()
        return 130
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def _raise_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt
