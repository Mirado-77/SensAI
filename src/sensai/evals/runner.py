"""Run attack cases against the real runtime, on a copy of the demo repository."""

import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from sensai.core.config import AllowedRoot, Config
from sensai.core.contracts import EventSink, ModelGateway, Tool
from sensai.core.events import Event
from sensai.core.runtime import AgentRuntime, RunResult, RuntimeDeps
from sensai.core.types import ApprovalRequest
from sensai.db.connection import connect
from sensai.evals.cases import AttackCase, CaseError
from sensai.evals.judge import Judgement, judge, requested_calls
from sensai.memory.sqlite_store import SqliteSessionStore
from sensai.model.ollama import OllamaGateway
from sensai.safety.policy import ApprovalPolicy
from sensai.safety.secrets import SecretRedactor, read_env_file
from sensai.tools import native_tools, sandbox_tools
from sensai.tools.registry import ToolRegistry

DEMO_REPO = Path(__file__).resolve().parents[3] / "demo-repo"
IGNORED = shutil.ignore_patterns("__pycache__", ".pytest_cache", ".sensai")


@dataclass(frozen=True)
class CaseResult:
    case: AttackCase
    judgement: Judgement
    status: str
    answer: str = ""
    calls: list[str] = field(default_factory=list)
    error: str | None = None


class _Recorder:
    """EventSink that keeps every event of the turn for the judge."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)


class AutoDenyChannel:
    """Refuse every approval: an attack must not change the copy of the repository."""

    async def confirm(self, request: ApprovalRequest) -> bool:
        return False

    async def ask(self, question: dict) -> dict:
        return {}


async def run_cases(
    cases: list[AttackCase],
    config: Config,
    gateway: ModelGateway | None = None,
    source: Path = DEMO_REPO,
    sandbox: bool = True,
) -> list[CaseResult]:
    """Run the cases one after the other: a local model cannot serve them in parallel."""
    results = []
    for case in cases:
        results.append(await run_case(case, config, gateway, source, sandbox))
    return results


async def run_case(
    case: AttackCase,
    config: Config,
    gateway: ModelGateway | None = None,
    source: Path = DEMO_REPO,
    sandbox: bool = True,
) -> CaseResult:
    with tempfile.TemporaryDirectory(prefix=f"ev4-{case.id}-") as tmp:
        root = prepare_workspace(case, source, Path(tmp))
        case_config = _case_config(config, root, case)
        tools, skip = await _tools(case_config, case, sandbox)
        if skip:
            return CaseResult(case, Judgement("indeterminate", [skip]), "skipped")
        model = gateway or OllamaGateway(
            config.model.host, config.model.name, config.model.timeout_s
        )
        turn = _Turn(case, case_config, model, tools, _Recorder())
        return await _execute(turn, root)


def prepare_workspace(case: AttackCase, source: Path, tmp: Path) -> Path:
    """Copy the repository and write the case files in it. A path outside the copy is refused."""
    root = tmp / "repo"
    shutil.copytree(source, root, ignore=IGNORED)
    for relative, content in case.setup.files.items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()):
            raise CaseError(f"Fichier de cas hors du dépôt ({case.id}) : {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


def _case_config(config: Config, root: Path, case: AttackCase) -> Config:
    files = config.files.model_copy(
        update={"allowed_roots": [AllowedRoot(path=root, mode="write")]}
    )
    runtime = config.runtime
    if case.timeout_s is not None:
        runtime = runtime.model_copy(update={"timeout_s": case.timeout_s})
    return config.model_copy(update={"files": files, "runtime": runtime})


async def _tools(config: Config, case: AttackCase, sandbox: bool) -> tuple[list[Tool], str]:
    """Tools of the chat, plus the reason to skip the case when it needs a sandbox that is off."""
    tools: list[Tool] = list(native_tools(config))
    if not sandbox:
        return tools, ""
    extra, reason = await sandbox_tools(config)
    if extra:
        return tools + extra, ""
    reason = reason or "bac à sable désactivé dans la configuration"
    if "run_command" in case.forbidden.tools:
        return tools, f"bac à sable indisponible : {reason}"
    return tools, ""


@dataclass(frozen=True)
class _Turn:
    """What one attack turn needs: the case, its config, the model, the tools and the sink."""

    case: AttackCase
    config: Config
    gateway: ModelGateway
    tools: list[Tool]
    recorder: _Recorder


async def _execute(turn: _Turn, root: Path) -> CaseResult:
    secrets = read_env_file(root / ".env")
    connection = connect(root.parent / "sensai.db")
    try:
        result = await _run_turn(turn, connection)
    finally:
        connection.close()
    events = turn.recorder.events
    judgement = judge(turn.case, events, result.answer, result.status == "completed", secrets)
    # The error goes into the report: it must not carry a secret of the copied .env either.
    error = SecretRedactor(secrets.values()).redact(result.error)[0] if result.error else None
    calls = requested_calls(events)
    return CaseResult(turn.case, judgement, result.status, result.answer, calls, error)


async def _run_turn(turn: _Turn, connection: sqlite3.Connection) -> RunResult:
    store = SqliteSessionStore(connection)
    runtime = build_runtime(turn.config, turn.gateway, turn.tools, turn.recorder, store)
    return await runtime.run_turn(store.create_session(), turn.case.prompt)


def build_runtime(
    config: Config,
    gateway: ModelGateway,
    tools: list[Tool],
    sink: EventSink,
    store: SqliteSessionStore,
) -> AgentRuntime:
    """The chat runtime, with every approval refused so that no run changes the repository."""
    deps = RuntimeDeps(
        gateway=gateway,
        store=store,
        sink=sink,
        channel=AutoDenyChannel(),
        registry=ToolRegistry(tools),
        policy=ApprovalPolicy.from_config(config.policy),
    )
    return AgentRuntime(
        deps,
        config.runtime,
        config.model.name,
        react=config.react,
        max_repairs=config.structured.max_repairs,
    )
