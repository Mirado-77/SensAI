"""WebSocket of a session (X1): runs the turns, streams the events and asks the client.

Server to client, one JSON object per frame, told apart by `kind`:
`ready`, `event` (an Event of the shared stream), `confirm` and `ask` (wait for a reply with the
same `id`), `persona`, `idle` (the turn is over), `warning`, `error`.

Client to server, told apart by `type`:
`message` {text}, `cancel`, `approval` {id, allowed}, `answer` {id, answer}, `persona` {name}.
"""

import asyncio
import contextlib
import json
import logging
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

from sensai.assembly import Chat
from sensai.core.errors import ConfigError
from sensai.core.events import Event
from sensai.core.personas import load_persona
from sensai.core.types import ApprovalRequest

logger = logging.getLogger(__name__)

Outbox = asyncio.Queue[dict[str, Any]]


class QueueSink:
    """EventSink that queues each event for the WebSocket."""

    def __init__(self, outbox: Outbox) -> None:
        self._outbox = outbox

    def emit(self, event: Event) -> None:
        self._outbox.put_nowait({"kind": "event", "event": asdict(event)})


class WebUserChannel:
    """UserChannel that sends the request to the client and waits for its reply.

    No reply before the client leaves means a refusal for an approval and an empty answer for a
    question. The runtime timeout bounds the wait.
    """

    def __init__(self, outbox: Outbox) -> None:
        self._outbox = outbox
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}

    async def confirm(self, request: ApprovalRequest) -> bool:
        reply = await self._request("confirm", {"request": asdict(request)})
        return reply.get("allowed") is True

    async def ask(self, question: dict[str, Any]) -> dict[str, Any]:
        reply = await self._request("ask", {"question": question})
        return {"answer": str(reply.get("answer") or "")}

    def resolve(self, request_id: str, reply: dict[str, Any]) -> bool:
        """Deliver the reply to `request_id`. Return False if no request waits for it."""
        future = self._pending.get(request_id)
        if future is None or future.done():
            return False
        future.set_result(reply)
        return True

    def close(self) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_result({})

    async def _request(self, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        request_id = uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        self._outbox.put_nowait({"kind": kind, "id": request_id, **body})
        try:
            return await future
        finally:
            self._pending.pop(request_id, None)


class SessionSocket:
    """Reads the client messages while a turn runs, so it can be cancelled or answered."""

    def __init__(
        self,
        websocket: WebSocket,
        chat: Chat,
        session_id: str,
        outbox: Outbox,
        channel: WebUserChannel,
    ) -> None:
        self._websocket = websocket
        self._chat = chat
        self._session_id = session_id
        self._outbox = outbox
        self._channel = channel
        self._turn: asyncio.Task[None] | None = None

    async def serve(self) -> None:
        sender = asyncio.create_task(self._send_loop())
        try:
            await self._receive_loop()
        finally:
            self._channel.close()
            await self._stop_turn()
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender

    @property
    def busy(self) -> bool:
        return self._turn is not None and not self._turn.done()

    async def _receive_loop(self) -> None:
        while True:
            try:
                data = await self._websocket.receive_json()
            except WebSocketDisconnect:
                return
            except (ValueError, KeyError):
                self._error("message JSON invalide")
                continue
            if isinstance(data, dict):
                self._handle(data)
            else:
                self._error("message JSON invalide")

    def _handle(self, data: dict[str, Any]) -> None:
        kind = data.get("type")
        if kind == "message":
            self._start_turn(str(data.get("text") or "").strip())
        elif kind == "cancel":
            if self._turn is not None:
                self._turn.cancel()
        elif kind == "approval":
            self._reply(data, {"allowed": data.get("allowed") is True})
        elif kind == "answer":
            self._reply(data, {"answer": data.get("answer")})
        elif kind == "persona":
            self._set_persona(data.get("name") or None)
        else:
            self._error(f"type de message inconnu : {kind!r}")

    def _start_turn(self, text: str) -> None:
        if not text:
            self._error("message vide")
        elif self.busy:
            self._error("un tour est déjà en cours")
        else:
            self._turn = asyncio.create_task(self._run_turn(text))

    async def _run_turn(self, text: str) -> None:
        try:
            await self._chat.runtime.run_turn(self._session_id, text)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Turn failed in session %s", self._session_id)
            self._error("échec inattendu du tour")
        finally:
            self._outbox.put_nowait({"kind": "idle"})

    def _reply(self, data: dict[str, Any], reply: dict[str, Any]) -> None:
        if not self._channel.resolve(str(data.get("id")), reply):
            self._error(f"aucune demande en attente avec l'id {data.get('id')!r}")

    def _set_persona(self, name: str | None) -> None:
        if self.busy:
            self._error("impossible de changer de persona pendant un tour")
            return
        try:
            persona = load_persona(str(name)) if name else None
        except ConfigError as exc:
            self._error(str(exc))
            return
        prompt = self._chat.prompt
        prompt.persona = persona
        prompt.apply(self._chat.runtime)
        self._outbox.put_nowait({"kind": "persona", "name": persona.name if persona else None})

    async def _stop_turn(self) -> None:
        if self._turn is None:
            return
        self._turn.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._turn

    async def _send_loop(self) -> None:
        while True:
            item = await self._outbox.get()
            text = json.dumps(item, ensure_ascii=False, default=str)
            try:
                await self._websocket.send_text(text)
            except (WebSocketDisconnect, RuntimeError):
                return

    def _error(self, message: str) -> None:
        self._outbox.put_nowait({"kind": "error", "message": f"Erreur : {message}"})
