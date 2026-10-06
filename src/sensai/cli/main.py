"""Command-line entry point."""

import argparse
import asyncio
import contextlib
import signal
from collections.abc import Sequence
from pathlib import Path
from types import FrameType

from rich.console import Console
from rich.text import Text

from sensai.cli.chat import run_chat
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
    chat.add_argument("--model", help="Ollama model to use (overrides the configuration)")
    chat.add_argument("--config", type=Path, help="TOML file overriding configs/default.toml")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    overrides = {"model": {"name": args.model}} if args.model else None
    return load_config(args.config, overrides)


def _raise_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console(highlight=False)
    error_console = Console(stderr=True, highlight=False)
    try:
        config = config_from_args(args)
    except ConfigError as exc:
        error_console.print(Text(f"Erreur : {exc}", style="bold red"))
        return 2
    # asyncio.run installs its own SIGINT handler unless a custom one is set. That handler would
    # swallow Ctrl-C while input() is waiting at the prompt.
    previous_handler = signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    try:
        return asyncio.run(run_chat(config, console, error_console))
    except KeyboardInterrupt:
        console.print()
        return 130
    finally:
        signal.signal(signal.SIGINT, previous_handler)
