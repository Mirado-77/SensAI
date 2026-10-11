"""Interactive chat: wiring of the runtime and the prompt loop."""

import asyncio
import signal
import sqlite3
from collections.abc import Awaitable
from typing import Any

from rich.text import Text

from sensai.cli.channel import CliUserChannel
from sensai.cli.render import ConsoleRenderer
from sensai.cli.terminal import Terminal
from sensai.core.bus import EventBus
from sensai.core.config import Config
from sensai.core.contracts import ModelGateway
from sensai.core.errors import ConfigError, SensaiError
from sensai.core.runtime import AgentRuntime, RunResult, RuntimeDeps
from sensai.db.connection import connect
from sensai.memory.sqlite_store import SqliteSessionStore
from sensai.model.ollama import OllamaGateway
from sensai.observability.journal import SqliteEventJournal
from sensai.safety.policy import ApprovalPolicy
from sensai.tools import build_registry

EXIT_COMMANDS = {"/exit", "/quit"}


class _SigintCanceller:
    """While active, Ctrl-C cancels `task` instead of raising KeyboardInterrupt."""

    def __init__(self, task: asyncio.Future[Any]) -> None:
        self.triggered = False
        self._task = task
        self._installed = False

    def __enter__(self) -> "_SigintCanceller":
        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, self._cancel)
            self._installed = True
        except (NotImplementedError, RuntimeError, ValueError):
            pass
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._installed:
            asyncio.get_running_loop().remove_signal_handler(signal.SIGINT)

    def _cancel(self) -> None:
        self.triggered = True
        self._task.cancel()


async def run_interruptible(turn: Awaitable[RunResult]) -> RunResult | None:
    """Run `turn`; Ctrl-C cancels it instead of exiting. Return None if it was interrupted."""
    task = asyncio.ensure_future(turn)
    with _SigintCanceller(task) as canceller:
        try:
            return await task
        except asyncio.CancelledError:
            if not canceller.triggered:
                raise
            return None


async def chat_loop(runtime: AgentRuntime, session_id: str, terminal: Terminal) -> None:
    """Read lines and run one turn per line, until /exit, Ctrl-D or Ctrl-C at the prompt."""
    while True:
        try:
            text = terminal.read_line("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            terminal.console.print()
            return
        if text in EXIT_COMMANDS:
            return
        if text:
            await run_interruptible(runtime.run_turn(session_id, text))


async def run_chat(config: Config, terminal: Terminal, gateway: ModelGateway | None = None) -> int:
    """Open the database, build the runtime and chat. Return the process exit code."""
    connection = _open_database(config, terminal)
    if connection is None:
        return 2
    try:
        deps = _build_deps_or_report(config, terminal, connection, gateway)
        if deps is None:
            return 2
        runtime = AgentRuntime(deps, config.runtime, config.model.name)
        banner = f"SENSAI | model: {config.model.name} | /exit or Ctrl-D to quit"
        terminal.console.print(Text(banner, "dim"))
        await chat_loop(runtime, deps.store.create_session(), terminal)
        return 0
    finally:
        connection.close()


def _open_database(config: Config, terminal: Terminal) -> sqlite3.Connection | None:
    try:
        return connect(config.storage.db_path)
    except (OSError, sqlite3.Error, SensaiError) as exc:
        message = f"Erreur : base de données inutilisable : {exc}"
        terminal.errors.print(Text(message, style="bold red"))
        return None


def _build_deps_or_report(
    config: Config,
    terminal: Terminal,
    connection: sqlite3.Connection,
    gateway: ModelGateway | None,
) -> RuntimeDeps | None:
    try:
        return _build_deps(config, terminal, connection, gateway)
    except ConfigError as exc:
        terminal.errors.print(Text(f"Erreur : {exc}", style="bold red"))
        return None


def _build_deps(
    config: Config,
    terminal: Terminal,
    connection: sqlite3.Connection,
    gateway: ModelGateway | None,
) -> RuntimeDeps:
    model = config.model
    return RuntimeDeps(
        gateway=gateway or OllamaGateway(model.host, model.name, model.timeout_s),
        store=SqliteSessionStore(connection),
        sink=EventBus([SqliteEventJournal(connection), ConsoleRenderer(terminal)]),
        channel=CliUserChannel(terminal),
        registry=build_registry(config),
        policy=ApprovalPolicy.from_config(config.policy),
    )
