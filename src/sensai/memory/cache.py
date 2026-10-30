"""Semantic cache (A6): the answer to a question close enough to one already answered is served
again without calling the model.

Only the first message of a session is looked up and stored: a later answer depends on the
conversation. A turn that ran a write or exec tool is not stored, its answer reports an action,
nor a turn where a tool failed, its answer reports the failure.
Entries are scoped by model, embedding model, prompt version and persona, and expire after
`ttl_hours`, so that answers about a codebase that changed do not live long.

Embeddings barely see a different identifier or a negation (nomic-embed-text gives 0.85 between
"what does count_done return" and the same about completion_rate, 0.955 for "what does it not
return"), so a hit also needs the same code identifiers and the default `min_score` is high.

Event payloads (all events also carry `run_id`):

- cache.hit: score (cosine similarity), question (the cached question that matched)
- cache.stored: question
"""

import math
import re
import sqlite3
import time
from array import array

from sensai.core.config import CacheConfig, Config
from sensai.core.contracts import SessionStore
from sensai.core.errors import SensaiError
from sensai.core.events import CACHE_HIT, CACHE_STORED
from sensai.core.runtime import AgentRuntime, RunContext
from sensai.core.types import new_id
from sensai.model.ollama import OllamaEmbedder
from sensai.rag.index import Embedder
from sensai.tools.registry import ToolRegistry

STORED_RISKS = frozenset({"read"})
# Code identifiers: snake_case, dotted or path names, calls.
_IDENTIFIER = re.compile(r"\w+\(\)|\w*[_./]\w[\w./]*")


class SemanticCache:
    def __init__(
        self,
        connection: sqlite3.Connection,
        embedder: Embedder,
        config: CacheConfig,
        prefix: str = "",
    ) -> None:
        self._db = connection
        self._embedder = embedder
        self._config = config
        self._prefix = prefix  # the same on both sides: question against question

    async def embed(self, question: str) -> array:
        [vector] = await self._embedder.embed([self._prefix + question])
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return array("f", (x / norm for x in vector))

    def lookup(self, scope: str, question: str, vector: array) -> tuple[str, str, float] | None:
        """The closest live entry of `scope` as (question, answer, score), if over `min_score` and
        naming the same code identifiers as `question`."""
        rows = self._db.execute(
            "SELECT question, answer, embedding FROM semantic_cache"
            " WHERE scope = ? AND created_at >= ?",
            (self._scope(scope), self._cutoff()),
        ).fetchall()
        names = identifiers(question)
        best = None
        for cached_question, answer, blob in rows:
            cached = array("f")
            cached.frombytes(blob)
            score = sum(a * b for a, b in zip(vector, cached, strict=False))
            if score < self._config.min_score or (best is not None and score <= best[2]):
                continue
            if identifiers(cached_question) == names:
                best = (cached_question, answer, score)
        return best

    def store(self, scope: str, question: str, vector: array, answer: str) -> None:
        """Save one entry and drop the expired ones."""
        with self._db:
            self._db.execute("DELETE FROM semantic_cache WHERE created_at < ?", (self._cutoff(),))
            self._db.execute(
                "INSERT INTO semantic_cache (id, scope, question, answer, embedding, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (new_id(), self._scope(scope), question, answer, vector.tobytes(), time.time()),
            )

    def clear(self) -> int:
        with self._db:
            return self._db.execute("DELETE FROM semantic_cache").rowcount

    def _scope(self, scope: str) -> str:
        return f"{scope}|{self._embedder.model}"

    def _cutoff(self) -> float:
        return time.time() - self._config.ttl_hours * 3600


def semantic_cache(config: Config, connection: sqlite3.Connection) -> SemanticCache | None:
    """The cache of the chat, embedded with `[rag] embed_model`, or None when `[cache]` is off."""
    if not config.cache.enabled:
        return None
    model, rag = config.model, config.rag
    embedder = OllamaEmbedder(model.host, rag.embed_model, model.timeout_s)
    return SemanticCache(connection, embedder, config.cache, rag.query_prefix)


def install_cache(
    runtime: AgentRuntime,
    cache: SemanticCache,
    store: SessionStore,
    registry: ToolRegistry,
    model: str,
) -> None:
    """Look the question up before the model (entry hook), store the answer after (exit hook).

    Install it after the guards, so that a blocked question is never looked up and the stored
    answer is the redacted one.
    """
    pending: dict[str, array] = {}

    def scope() -> str:
        return f"{model}|{runtime.prompt_version}|{runtime.persona}"

    async def look_up(ctx: RunContext) -> None:
        if store.history(ctx.session_id):
            return
        try:
            vector = await cache.embed(ctx.user_text)
        except SensaiError:
            return  # no embedding model: the turn runs as without the cache
        hit = cache.lookup(scope(), ctx.user_text, vector)
        if hit is None:
            pending[ctx.run_id] = vector
            return
        question, answer, score = hit
        ctx.answer = answer
        ctx.notices.append((CACHE_HIT, {"score": round(score, 4), "question": question}))

    async def save(ctx: RunContext) -> None:
        vector = pending.pop(ctx.run_id, None)
        if vector is None or not ctx.answer.strip() or ctx.tool_errors:
            return
        if not _read_only(ctx, registry):
            return
        cache.store(scope(), ctx.user_text, vector, ctx.answer)
        ctx.notices.append((CACHE_STORED, {"question": ctx.user_text}))

    runtime.entry_hooks.append(look_up)
    runtime.exit_hooks.append(save)


def identifiers(text: str) -> set[str]:
    return {name.rstrip(".") for name in _IDENTIFIER.findall(text)}  # no full stop


def _read_only(ctx: RunContext, registry: ToolRegistry) -> bool:
    """No tool of this turn (the messages after the user's one) could change anything."""
    for message in reversed(ctx.messages):
        if message.role == "user":
            break
        for call in message.tool_calls:
            tool = registry.get(call.name)
            if tool is None or tool.risk not in STORED_RISKS:
                return False
    return True
