"""Runtime assembly shared by the CLI chat and the API (X1): tools, MCP servers, prompt, hooks."""

import sqlite3
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sensai.cli.profile import load_profile, profile_prompt
from sensai.cli.prompts import ChatPrompt
from sensai.core.config import Config
from sensai.core.contracts import EventSink, ModelGateway, Tool, UserChannel
from sensai.core.fewshots import fewshots_from
from sensai.core.personas import load_persona
from sensai.core.prompts import load_version
from sensai.core.runtime import AgentRuntime, RuntimeDeps
from sensai.memory import artifact_note_for, artifact_tools, memory_recall, memory_tools
from sensai.memory.budget import install_memory
from sensai.memory.cache import SemanticCache, install_cache, semantic_cache
from sensai.memory.sqlite_store import SqliteSessionStore
from sensai.model.ollama import OllamaGateway
from sensai.safety.guards import install_guards
from sensai.safety.policy import ApprovalPolicy
from sensai.safety.secrets import read_env_file
from sensai.schedule import schedule_tools
from sensai.tools import native_tools, sandbox_tools, search_tools, web_tools
from sensai.tools.mcp.servers import McpServers
from sensai.tools.registry import ToolRegistry

Warn = Callable[[str], None]


@dataclass
class Chat:
    runtime: AgentRuntime
    deps: RuntimeDeps
    store: SqliteSessionStore
    registry: ToolRegistry
    prompt: ChatPrompt
    cache: SemanticCache | None


@asynccontextmanager
async def open_chat(
    config: Config,
    connection: sqlite3.Connection,
    sink: EventSink,
    channel: UserChannel,
    warn: Warn,
    gateway: ModelGateway | None = None,
) -> AsyncIterator[Chat]:
    """Build the runtime and keep the MCP servers running until the block exits.

    Raises ConfigError for an unusable prompt, persona or tool configuration. Problems that only
    disable a part (Docker, an MCP tool) are passed to `warn`.
    """
    version = load_version(config.prompts.system) if config.prompts.system else None
    persona = load_persona(config.prompts.persona) if config.prompts.persona else None
    fewshots = fewshots_from(config.prompts.examples)
    store = SqliteSessionStore(connection)
    native = native_tools(config)
    tools, reason = await sandbox_tools(config)
    if reason is not None:
        warn(f"exécution de commandes désactivée (T2) : {reason}")
    native += tools
    native += search_tools(config, connection)
    native += web_tools(config)
    native += schedule_tools(config, connection)
    native += memory_tools(config, connection)
    native += artifact_tools(config, connection, sink)
    async with McpServers(config.mcp) as mcp:
        registry = build_registry(native, mcp, warn)
        model = config.model
        deps = RuntimeDeps(
            gateway=gateway
            or OllamaGateway(
                model.host,
                model.name,
                model.timeout_s,
                model.temperature,
                model.think,
                model.num_ctx,
            ),
            store=store,
            sink=sink,
            channel=channel,
            registry=registry,
            policy=ApprovalPolicy.from_config(config.policy),
        )
        prompt = ChatPrompt(version, profile_prompt(load_profile(store)), persona)
        runtime = AgentRuntime(
            deps,
            config.runtime,
            model.name,
            None,
            react=config.react,
            max_repairs=config.structured.max_repairs,
            questions=config.questions,
        )
        prompt.apply(runtime)
        runtime.fewshots = fewshots
        runtime.recall = memory_recall(config, connection)
        runtime.artifact_note = artifact_note_for(config, connection)
        install_guards(runtime, config.guards, read_env_file(config.guards.env_file))
        install_memory(runtime, config.memory, model.num_ctx, store, deps.gateway)
        cache = semantic_cache(config, connection)
        if cache is not None:
            install_cache(runtime, cache, store, registry, model.name)
        yield Chat(runtime, deps, store, registry, prompt, cache)


def build_registry(native: list[Tool], mcp: McpServers, warn: Warn) -> ToolRegistry:
    """Register the native tools, then the MCP tools. A tool that cannot be used is reported."""
    registry = ToolRegistry(native)
    warnings = list(mcp.warnings)
    for tool in mcp.tools:
        try:
            registry.register(tool)
        except ValueError as exc:
            warnings.append(str(exc))
    for warning in warnings:
        warn(warning)
    return registry
