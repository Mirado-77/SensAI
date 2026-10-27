"""Question corpus of EV1: one TOML file per question about the demo repository, answered by the
real runtime (active prompt version, search_code when the copy can be indexed), then graded."""

import shutil
import sqlite3
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from sensai.core.config import AllowedRoot, Config
from sensai.core.contracts import ModelGateway, Tool
from sensai.core.errors import ModelError
from sensai.core.events import PROMPT_EXAMPLES
from sensai.core.fewshots import FewShots, fewshots_from
from sensai.core.prompts import PromptVersion, load_version
from sensai.db.connection import connect
from sensai.evals.judge import requested_calls, token_usage
from sensai.evals.runner import DEMO_REPO, IGNORED, _Recorder, build_runtime
from sensai.memory.sqlite_store import SqliteSessionStore
from sensai.model.ollama import OllamaGateway
from sensai.rag.index import IndexReport
from sensai.tools import code_index, native_tools, search_tools


class QuestionCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    question: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    criteria: list[str] = Field(default_factory=list)
    timeout_s: float | None = Field(default=None, gt=0)


@dataclass(frozen=True)
class Answer:
    case: QuestionCase
    status: str
    answer: str = ""
    calls: list[str] = field(default_factory=list)
    error: str | None = None
    examples: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


async def answer_questions(
    cases: list[QuestionCase],
    config: Config,
    gateway: ModelGateway | None = None,
    source: Path = DEMO_REPO,
    index: bool = True,
    progress: Callable[[str], None] = lambda _: None,
) -> tuple[list[Answer], str | None, list[str]]:
    """Answer each question in a fresh session, on one read-only copy of `source`. Returns the
    answers, the prompt version label and notes on the setup (index skipped, for instance)."""
    model = gateway or OllamaGateway(
        config.model.host, config.model.name, config.model.timeout_s, config.model.think
    )
    version = load_version(config.prompts.system) if config.prompts.system else None
    fewshots = fewshots_from(config.prompts.examples)
    with tempfile.TemporaryDirectory(prefix="ev1-") as tmp:
        root = Path(tmp) / "repo"
        shutil.copytree(source, root, ignore=IGNORED)
        connection = connect(Path(tmp) / "sensai.db")
        try:
            notes: list[str] = []
            run_config = _read_only_copy_config(config, root)
            # Questions only need reading: a write tool would just burn steps on refused edits.
            tools = [tool for tool in native_tools(run_config) if tool.risk == "read"]
            if index:
                progress("Indexation de la copie du dépôt (search_code)")
                tools += await _search_tools(run_config, root, connection, notes)
            store = SqliteSessionStore(connection)
            answers = []
            for number, case in enumerate(cases, 1):
                progress(f"[{number}/{len(cases)}] {case.id} : réponse de l'agent")
                answers.append(
                    await _answer(case, run_config, model, tools, store, version, fewshots)
                )
        finally:
            connection.close()
    return answers, version.label if version else None, notes


def _read_only_copy_config(config: Config, root: Path) -> Config:
    files = config.files.model_copy(update={"allowed_roots": [AllowedRoot(path=root, mode="read")]})
    return config.model_copy(update={"files": files})


async def _search_tools(
    config: Config, root: Path, connection: sqlite3.Connection, notes: list[str]
) -> list[Tool]:
    """Index the copy for search_code (R1). Without the embedding model, run without it."""
    if not config.rag.enabled:
        notes.append("search_code désactivé : [rag] enabled = false")
        return []
    try:
        await code_index(config, connection).index_root(
            root, config.files.max_read_bytes, IndexReport()
        )
    except ModelError as exc:
        notes.append(f"search_code indisponible, index impossible : {exc}")
        return []
    return search_tools(config, connection)


async def _answer(
    case: QuestionCase,
    config: Config,
    gateway: ModelGateway,
    tools: list[Tool],
    store: SqliteSessionStore,
    version: PromptVersion | None,
    fewshots: FewShots | None,
) -> Answer:
    if case.timeout_s is not None:
        config = config.model_copy(
            update={"runtime": config.runtime.model_copy(update={"timeout_s": case.timeout_s})}
        )
    recorder = _Recorder()
    runtime = build_runtime(config, gateway, tools, recorder, store)
    runtime.system_prompt = version.text if version else None
    runtime.prompt_version = version.label if version else None
    runtime.fewshots = fewshots
    started = time.perf_counter()
    result = await runtime.run_turn(store.create_session(), case.question)
    latency = time.perf_counter() - started
    events = recorder.events
    examples = [id_ for e in events if e.type == PROMPT_EXAMPLES for id_ in e.payload["examples"]]
    prompt_tokens, completion_tokens = token_usage(events)
    return Answer(
        case,
        result.status,
        result.answer,
        requested_calls(events),
        result.error,
        examples,
        latency,
        prompt_tokens,
        completion_tokens,
    )
