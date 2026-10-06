"""Interactive chat: wiring of the runtime and the prompt loop."""

import asyncio
import signal
import sqlite3
from collections.abc import Awaitable, Callable

from rich.console import Console
from rich.text import Text

from sensai.cli.channel import CliUserChannel
from sensai.cli.render import ConsoleRenderer
from sensai.core.bus import EventBus
from sensai.core.config import Config
from sensai.core.contracts import ModelGateway
from sensai.core.errors import SensaiError
from sensai.core.runtime import AgentRuntime, RunResult
from sensai.db.connection import connect
from sensai.memory.sqlite_store import SqliteSessionStore
from sensai.model.ollama import OllamaGateway
from sensai.observability.journal import SqliteEventJournal
from sensai.safety.policy import ApprovalPolicy
from sensai.tools.registry import ToolRegistry

EXIT_COMMANDS = {"/exit", "/quit"}


async def run_interruptible(turn: Awaitable[RunResult]) -> RunResult | None:
    """Run `turn`; Ctrl-C cancels it instead of exiting. Return None if it was interrupted."""
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(turn)
    interrupted = False

    def on_sigint() -> None:
        nonlocal interrupted
        interrupted = True
        task.cancel()

    try:
        loop.add_signal_handler(signal.SIGINT, on_sigint)
        handler_installed = True
    except (NotImplementedError, RuntimeError, ValueError):
        handler_installed = False
    try:
        return await task
    except asyncio.CancelledError:
        if not interrupted:
            raise
        return None
    finally:
        if handler_installed:
            loop.remove_signal_handler(signal.SIGINT)


async def chat_loop(
    runtime: AgentRuntime,
    session_id: str,
    console: Console,
    read_line: Callable[[str], str] = input,
) -> None:
    """Read lines and run one turn per line, until /exit, Ctrl-D or Ctrl-C at the prompt."""
    while True:
        try:
            text = read_line("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not text:
            continue
        if text in EXIT_COMMANDS:
            return
        await run_interruptible(runtime.run_turn(session_id, text))


async def run_chat(
    config: Config,
    console: Console,
    error_console: Console,
    *,
    gateway: ModelGateway | None = None,
    read_line: Callable[[str], str] = input,
) -> int:
    """Open the database, build the runtime and chat. Return the process exit code."""
    try:
        connection = connect(config.storage.db_path)
    except (OSError, sqlite3.Error, SensaiError) as exc:
        message = f"Erreur : base de données inutilisable : {exc}"
        error_console.print(Text(message, style="bold red"))
        return 2
    try:
        store = SqliteSessionStore(connection)
        renderer = ConsoleRenderer(console, error_console)
        runtime = AgentRuntime(
            gateway=gateway
            or OllamaGateway(config.model.host, config.model.name, config.model.timeout_s),
            store=store,
            sink=EventBus([SqliteEventJournal(connection), renderer]),
            channel=CliUserChannel(console, read_line),
            registry=ToolRegistry(),
            policy=ApprovalPolicy.from_config(config.policy),
            config=config.runtime,
            model_name=config.model.name,
        )
        console.print(Text(f"SENSAI | model: {config.model.name} | /exit or Ctrl-D to quit", "dim"))
        await chat_loop(runtime, store.create_session(), console, read_line)
        return 0
    finally:
        connection.close()
