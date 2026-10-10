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
from sensai.cli.chat import run_chat
from sensai.cli.terminal import Terminal
from sensai.core.config import Config, load_config
from sensai.core.errors import ConfigError

with contextlib.suppress(ImportError):
    import readline  # noqa: F401  (line editing and history for input())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sensai", description="SENSAI: local AI assistant built on Ollama."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    chat = commands.add_parser("chat", help="start an interactive chat")
    _add_common_options(chat)
    ask = commands.add_parser(
        "ask", help="answer one question with JSON conforming to a schema (stdout)"
    )
    _add_common_options(ask)
    ask.add_argument("--schema", type=Path, required=True, help="JSON Schema file of the answer")
    ask.add_argument("question", nargs="+", help="the question, or - to read it from stdin")
    return parser


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", help="Ollama model to use (overrides the configuration)")
    parser.add_argument("--config", type=Path, help="TOML file overriding configs/default.toml")


def config_from_args(args: argparse.Namespace) -> Config:
    overrides = {"model": {"name": args.model}} if args.model else None
    return load_config(args.config, overrides)


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
    return _run(run_chat(config, terminal), terminal)


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
