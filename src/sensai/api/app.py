"""FastAPI application (X1): REST routes under /api, the session WebSocket, the built web UI."""

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sensai.api.socket import Outbox, QueueSink, SessionSocket, WebUserChannel
from sensai.assembly import open_chat
from sensai.cli.profile import PROFILE_ID
from sensai.core.bus import EventBus
from sensai.core.config import Config
from sensai.core.contracts import ModelGateway
from sensai.core.errors import ConfigError, SensaiError
from sensai.core.personas import list_personas
from sensai.core.prompts import list_versions
from sensai.db.connection import connect
from sensai.memory.artifact import ArtifactStore
from sensai.memory.sqlite_store import SqliteSessionStore
from sensai.observability.journal import SqliteEventJournal

API_NAME = "sensai"
SESSION_NOT_FOUND = 4404


class ForkRequest(BaseModel):
    message_id: str


def create_app(config: Config, gateway: ModelGateway | None = None) -> FastAPI:
    """`gateway` replaces the Ollama gateway (tests)."""
    app = FastAPI(title="SENSAI API")

    async def database() -> AsyncIterator[sqlite3.Connection]:
        """Async so the connection is opened on the event loop thread, where the routes use it."""
        try:
            connection = connect(config.storage.db_path)
        except (OSError, sqlite3.Error, SensaiError) as exc:
            raise HTTPException(503, f"Base de données inutilisable : {exc}") from exc
        try:
            yield connection
        finally:
            connection.close()

    Database = Annotated[sqlite3.Connection, Depends(database)]

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"name": API_NAME}

    @app.get("/api/config")
    async def settings() -> dict[str, Any]:
        try:
            personas = [
                {"name": persona.name, "description": persona.description}
                for persona in list_personas()
            ]
        except ConfigError as exc:
            raise HTTPException(500, str(exc)) from exc
        return {
            "model": config.model.name,
            "temperature": config.model.temperature,
            "prompt": config.prompts.system,
            "prompts": [version.name for version in list_versions()],
            "persona": config.prompts.persona,
            "personas": personas,
            "max_steps": config.runtime.max_steps,
            "max_tokens": config.runtime.max_tokens,
        }

    @app.get("/api/sessions")
    async def sessions(
        connection: Database, limit: Annotated[int, Query(ge=1, le=200)] = 20
    ) -> list[dict[str, Any]]:
        rows = SqliteSessionStore(connection).list_sessions(limit)
        return [dict(row) for row in rows]

    @app.post("/api/sessions", status_code=201)
    async def create_session(connection: Database) -> dict[str, str]:
        return {"id": SqliteSessionStore(connection).create_session(PROFILE_ID)}

    @app.get("/api/sessions/{session_id}/messages")
    async def messages(session_id: str, connection: Database) -> list[dict[str, Any]]:
        try:
            history = SqliteSessionStore(connection).history(session_id)
        except KeyError as exc:
            raise HTTPException(404, exc.args[0]) from exc
        return [asdict(message) for message in history]

    @app.post("/api/sessions/{session_id}/fork", status_code=201)
    async def fork(session_id: str, body: ForkRequest, connection: Database) -> dict[str, str]:
        try:
            forked = SqliteSessionStore(connection).fork(session_id, body.message_id)
        except KeyError as exc:
            raise HTTPException(404, exc.args[0]) from exc
        if config.artifact.enabled:
            ArtifactStore(connection).copy(session_id, forked)
        return {"id": forked}

    @app.websocket("/ws/sessions/{session_id}")
    async def session_socket(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        try:
            connection = connect(config.storage.db_path)
        except (OSError, sqlite3.Error, SensaiError) as exc:
            await _refuse(websocket, 1011, f"base de données inutilisable : {exc}")
            return
        try:
            await _serve_session(websocket, connection, session_id)
        finally:
            connection.close()

    async def _serve_session(
        websocket: WebSocket, connection: sqlite3.Connection, session_id: str
    ) -> None:
        try:
            SqliteSessionStore(connection).history(session_id)
        except KeyError:
            await _refuse(websocket, SESSION_NOT_FOUND, f"session introuvable : {session_id}")
            return
        outbox: Outbox = asyncio.Queue()
        sink = EventBus([SqliteEventJournal(connection), QueueSink(outbox)])
        channel = WebUserChannel(outbox)
        warnings: list[str] = []
        try:
            async with open_chat(
                config, connection, sink, channel, warnings.append, gateway
            ) as chat:
                for warning in warnings:
                    outbox.put_nowait({"kind": "warning", "message": f"Attention : {warning}"})
                persona = chat.prompt.persona
                outbox.put_nowait(
                    {
                        "kind": "ready",
                        "session_id": session_id,
                        "model": config.model.name,
                        "persona": persona.name if persona else None,
                        "tools": [
                            {"name": tool.name, "risk": tool.risk, "description": tool.description}
                            for tool in chat.registry.tools()
                        ],
                    }
                )
                await SessionSocket(websocket, chat, session_id, outbox, channel).serve()
        except ConfigError as exc:
            await _refuse(websocket, 1011, str(exc))

    web_dir = config.api.web_dir.expanduser()
    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app


async def _refuse(websocket: WebSocket, code: int, message: str) -> None:
    """Send the reason as an error frame (a close reason is limited to 123 bytes), then close."""
    await websocket.send_json({"kind": "error", "message": f"Erreur : {message}"})
    await websocket.close(code)
