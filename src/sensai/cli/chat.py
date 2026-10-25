"""Interactive chat: wiring of the runtime and the prompt loop."""

import asyncio
import signal
import sqlite3
from collections.abc import Awaitable
from datetime import datetime
from typing import Any

from rich.text import Text

from sensai.cli.channel import CliUserChannel
from sensai.cli.fork import FORK_COMMAND, HISTORY_COMMAND, print_history, run_fork_command
from sensai.cli.profile import (
    PROFILE_COMMAND,
    PROFILE_ID,
    load_profile,
    profile_prompt,
    run_profile_command,
)
from sensai.cli.prompts import PROMPT_COMMAND, ChatPrompt, run_prompt_command
from sensai.cli.render import ConsoleRenderer
from sensai.cli.terminal import Terminal
from sensai.core.bus import EventBus
from sensai.core.config import Config
from sensai.core.contracts import ModelGateway, Tool
from sensai.core.errors import ConfigError, SensaiError
from sensai.core.prompts import load_version
from sensai.core.runtime import AgentRuntime, RunResult, RuntimeDeps
from sensai.db.connection import connect
from sensai.memory.sqlite_store import SqliteSessionStore
from sensai.model.ollama import OllamaGateway
from sensai.observability.journal import SqliteEventJournal
from sensai.safety.guards import install_guards
from sensai.safety.policy import ApprovalPolicy
from sensai.safety.secrets import read_env_file
from sensai.tools import native_tools, sandbox_tools, search_tools
from sensai.tools.mcp.servers import McpServers
from sensai.tools.registry import ToolRegistry

EXIT_COMMANDS = {"/exit", "/quit"}
TOOLS_COMMAND = "/tools"
LAST_SESSION = ""


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


async def chat_loop(
    runtime: AgentRuntime,
    session_id: str,
    terminal: Terminal,
    store: SqliteSessionStore,
    registry: ToolRegistry | None = None,
    prompt: ChatPrompt | None = None,
) -> None:
    """Read lines and run one turn per line, until /exit, Ctrl-D or Ctrl-C at the prompt."""
    while True:
        try:
            text = terminal.read_line("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            terminal.console.print()
            return
        if text in EXIT_COMMANDS:
            return
        if text == TOOLS_COMMAND:
            print_tools(terminal, registry)
        elif text.split(" ", 1)[0] == PROFILE_COMMAND:
            prompt = prompt or ChatPrompt(None)
            prompt.profile = run_profile_command(store, terminal, text)
            prompt.apply(runtime)
        elif text.split(" ", 1)[0] == PROMPT_COMMAND:
            prompt = prompt or ChatPrompt(None)
            run_prompt_command(prompt, terminal, text)
            prompt.apply(runtime)
        elif text == HISTORY_COMMAND:
            print_history(store, terminal, session_id)
        elif text.split(" ", 1)[0] == FORK_COMMAND:
            session_id = run_fork_command(store, terminal, session_id, text)
        elif text:
            await run_interruptible(runtime.run_turn(session_id, text))


def print_tools(terminal: Terminal, registry: ToolRegistry | None) -> None:
    """List the tools the model can call, with their risk level (T1)."""
    tools = registry.tools() if registry is not None else []
    if not tools:
        terminal.console.print(Text("No tool available.", "dim"))
        return
    for tool in tools:
        line = Text(f"{tool.name} ", "bold")
        line.append(f"[{tool.risk}] ", "yellow")
        line.append(tool.description.splitlines()[0] if tool.description else "")
        terminal.console.print(line)


async def run_chat(
    config: Config,
    terminal: Terminal,
    gateway: ModelGateway | None = None,
    resume: str | None = None,
) -> int:
    """Open the database, build the runtime and chat. Return the process exit code.

    `resume` is the id (or id prefix) of a session to continue, LAST_SESSION for the most
    recent one, None for a new session.
    """
    try:
        version = load_version(config.prompts.system) if config.prompts.system else None
    except ConfigError as exc:
        terminal.errors.print(Text(f"Erreur : {exc}", style="bold red"))
        return 2
    connection = _open_database(config, terminal)
    if connection is None:
        return 2
    try:
        store = SqliteSessionStore(connection)
        session_id = _open_session(store, terminal, resume)
        if session_id is None:
            return 2
        native = _native_tools_or_report(config, terminal)
        if native is None:
            return 2
        native += await _sandbox_tools_or_report(config, terminal)
        native += search_tools(config, connection)
        async with McpServers(config.mcp) as mcp:
            registry = build_registry(native, mcp, terminal)
            deps = _build_deps(config, terminal, connection, gateway, registry)
            prompt = ChatPrompt(version, profile_prompt(load_profile(store)))
            runtime = AgentRuntime(
                deps,
                config.runtime,
                config.model.name,
                None,
                react=config.react,
                max_repairs=config.structured.max_repairs,
                questions=config.questions,
            )
            prompt.apply(runtime)
            install_guards(runtime, config.guards, read_env_file(config.guards.env_file))
            terminal.console.print(Text(_banner(config, registry, prompt), "dim"))
            await chat_loop(runtime, session_id, terminal, store, registry, prompt)
        return 0
    finally:
        connection.close()


def _open_session(store: SqliteSessionStore, terminal: Terminal, resume: str | None) -> str | None:
    """Return the session to chat in, or None after printing why it cannot be resumed."""
    if resume is None:
        return store.create_session(PROFILE_ID)
    sessions = store.list_sessions(2, resume)
    if not sessions or (resume and len(sessions) > 1):
        problem = "ambiguë" if sessions else "introuvable"
        message = f"session {problem} : {resume}" if resume else "aucune session enregistrée"
        terminal.errors.print(Text(f"Erreur : {message}", style="bold red"))
        return None
    session_id = sessions[0]["id"]
    count = len(store.history(session_id))
    terminal.console.print(Text(f"Resumed session {session_id} ({count} messages)", "dim"))
    return session_id


def print_sessions(config: Config, terminal: Terminal, limit: int = 20) -> int:
    """List the saved sessions, newest first. Return the process exit code."""
    connection = _open_database(config, terminal)
    if connection is None:
        return 2
    try:
        sessions = SqliteSessionStore(connection).list_sessions(limit)
    finally:
        connection.close()
    if not sessions:
        terminal.console.print(Text("No saved session.", "dim"))
    for row in sessions:
        line = Text(f"{row['id'][:8]} ", "bold")
        line.append(datetime.fromtimestamp(row["created_at"]).strftime("%Y-%m-%d %H:%M "), "dim")
        line.append((row["first_message"] or "").split("\n")[0][:60])
        terminal.console.print(line)
    return 0


def build_registry(native: list[Tool], mcp: McpServers, terminal: Terminal) -> ToolRegistry:
    """Register the native tools, then the MCP tools. A tool that cannot be used is reported."""
    registry = ToolRegistry(native)
    warnings = list(mcp.warnings)
    for tool in mcp.tools:
        try:
            registry.register(tool)
        except ValueError as exc:
            warnings.append(str(exc))
    for warning in warnings:
        terminal.errors.print(Text(f"Attention : {warning}", style="yellow"))
    return registry


def _banner(config: Config, registry: ToolRegistry, prompt: ChatPrompt) -> str:
    tools = f"{len(registry)} tools ({TOOLS_COMMAND})" if len(registry) else "no tools"
    version = prompt.version.label if prompt.version else "none"
    return (
        f"SENSAI | model: {config.model.name} | prompt: {version} ({PROMPT_COMMAND}) | {tools}"
        " | /exit or Ctrl-D to quit"
    )


def _open_database(config: Config, terminal: Terminal) -> sqlite3.Connection | None:
    try:
        return connect(config.storage.db_path)
    except (OSError, sqlite3.Error, SensaiError) as exc:
        message = f"Erreur : base de données inutilisable : {exc}"
        terminal.errors.print(Text(message, style="bold red"))
        return None


def _native_tools_or_report(config: Config, terminal: Terminal) -> list[Tool] | None:
    try:
        return native_tools(config)
    except ConfigError as exc:
        terminal.errors.print(Text(f"Erreur : {exc}", style="bold red"))
        return None


async def _sandbox_tools_or_report(config: Config, terminal: Terminal) -> list[Tool]:
    tools, reason = await sandbox_tools(config)
    if reason is not None:
        message = f"Attention : exécution de commandes désactivée (T2) : {reason}"
        terminal.errors.print(Text(message, style="yellow"))
    return tools


def _build_deps(
    config: Config,
    terminal: Terminal,
    connection: sqlite3.Connection,
    gateway: ModelGateway | None,
    registry: ToolRegistry,
) -> RuntimeDeps:
    model = config.model
    return RuntimeDeps(
        gateway=gateway or OllamaGateway(model.host, model.name, model.timeout_s),
        store=SqliteSessionStore(connection),
        sink=EventBus([SqliteEventJournal(connection), ConsoleRenderer(terminal)]),
        channel=CliUserChannel(terminal),
        registry=registry,
        policy=ApprovalPolicy.from_config(config.policy),
    )
