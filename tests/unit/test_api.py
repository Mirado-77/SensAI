import asyncio

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from sensai.api.app import SESSION_NOT_FOUND, create_app
from sensai.cli.main import build_parser
from sensai.core.config import load_config
from sensai.core.events import (
    APPROVAL_REQUESTED,
    APPROVAL_RESOLVED,
    MODEL_CHUNK,
    RUN_CANCELLED,
    RUN_COMPLETED,
)
from sensai.core.fakes import FakeModelGateway
from sensai.core.types import Chunk, Done, ToolCall

pytestmark = pytest.mark.feature("X1")


def api_config(tmp_path, **overrides):
    defaults = {
        "storage": {"db_path": str(tmp_path / "sensai.db")},
        "react": {"enabled": False, "max_plan_steps": 5},
        "questions": {"enabled": False, "max_per_turn": 2},
        "prompts": {"system": "", "persona": ""},
        "sandbox": {"enabled": False},
        "web": {"enabled": False},
        "scheduler": {"enabled": False},
        "memory": {"enabled": False},
        "artifact": {"enabled": False},
        "cache": {"enabled": False, "min_score": 0.92, "ttl_hours": 24},
        "api": {"port": 8000, "web_dir": str(tmp_path / "dist")},
    }
    return load_config(overrides={**defaults, **overrides})


def client_for(config, gateway=None):
    return TestClient(create_app(config, gateway or FakeModelGateway()))


def new_session(client):
    response = client.post("/api/sessions")
    assert response.status_code == 201
    return response.json()["id"]


def receive_until(socket, kind, event_type=None):
    """Frames up to and including the first one of `kind` (and `event_type` for events)."""
    frames = []
    while True:
        frame = socket.receive_json()
        frames.append(frame)
        if frame["kind"] == kind and (event_type is None or frame["event"]["type"] == event_type):
            return frames


def event_types(frames):
    return [frame["event"]["type"] for frame in frames if frame["kind"] == "event"]


class BlockingGateway:
    """Streams one chunk, then waits until cancelled."""

    async def stream_chat(self, messages, tools=None, schema=None):
        yield Chunk("partial")
        await asyncio.Event().wait()
        yield Done()


def test_health_names_the_api(tmp_path):
    with client_for(api_config(tmp_path)) as client:
        assert client.get("/api/health").json() == {"name": "sensai"}


def test_config_lists_the_model_and_the_personas(tmp_path):
    config = api_config(tmp_path, model={"name": "llama3.2:3b"})
    with client_for(config) as client:
        body = client.get("/api/config").json()
    assert body["model"] == "llama3.2:3b"
    assert {"dev", "reviewer"} <= {persona["name"] for persona in body["personas"]}


def test_an_unknown_session_is_a_404(tmp_path):
    with client_for(api_config(tmp_path)) as client:
        response = client.get("/api/sessions/nope/messages")
    assert response.status_code == 404
    assert "nope" in response.json()["detail"]


def test_a_turn_streams_its_events_and_is_saved(tmp_path):
    gateway = FakeModelGateway([Chunk("Hel"), Chunk("lo")])
    with client_for(api_config(tmp_path), gateway) as client:
        session_id = new_session(client)
        with client.websocket_connect(f"/ws/sessions/{session_id}") as socket:
            ready = socket.receive_json()
            socket.send_json({"type": "message", "text": "hi"})
            frames = receive_until(socket, "idle")
        messages = client.get(f"/api/sessions/{session_id}/messages").json()
        sessions = client.get("/api/sessions").json()
    assert ready["kind"] == "ready" and ready["session_id"] == session_id
    chunks = [
        f["event"]["payload"]["text"]
        for f in frames
        if f.get("event", {}).get("type") == MODEL_CHUNK
    ]
    assert "".join(chunks) == "Hello"
    assert RUN_COMPLETED in event_types(frames)
    assert [(m["role"], m["content"]) for m in messages] == [("user", "hi"), ("assistant", "Hello")]
    assert [row["id"] for row in sessions] == [session_id]


def test_an_unknown_session_closes_the_socket(tmp_path):
    with client_for(api_config(tmp_path)) as client:
        with client.websocket_connect("/ws/sessions/nope") as socket:
            error = socket.receive_json()
            with pytest.raises(WebSocketDisconnect) as closed:
                socket.receive_json()
    assert error["kind"] == "error" and "nope" in error["message"]
    assert closed.value.code == SESSION_NOT_FOUND


def test_empty_and_unknown_messages_are_errors(tmp_path):
    with client_for(api_config(tmp_path)) as client:
        session_id = new_session(client)
        with client.websocket_connect(f"/ws/sessions/{session_id}") as socket:
            socket.receive_json()
            socket.send_json({"type": "message", "text": "   "})
            empty = socket.receive_json()
            socket.send_json({"type": "dance"})
            unknown = socket.receive_json()
            socket.send_text("not json")
            invalid = socket.receive_json()
    assert empty == {"kind": "error", "message": "Erreur : message vide"}
    assert "dance" in unknown["message"]
    assert invalid == {"kind": "error", "message": "Erreur : message JSON invalide"}


@pytest.mark.feature("A3")
@pytest.mark.parametrize("allowed", [True, False])
def test_a_write_waits_for_the_client_approval(tmp_path, allowed):
    root = tmp_path / "repo"
    root.mkdir()
    files = {"allowed_roots": [{"path": str(root), "mode": "write"}]}
    write = ToolCall("write_file", {"path": "a.py", "content": "x = 1\n"})
    gateway = FakeModelGateway([write], [Chunk("done")])
    with client_for(api_config(tmp_path, files=files), gateway) as client:
        session_id = new_session(client)
        with client.websocket_connect(f"/ws/sessions/{session_id}") as socket:
            socket.receive_json()
            socket.send_json({"type": "message", "text": "write it"})
            frames = receive_until(socket, "confirm")
            confirm = frames[-1]
            socket.send_json({"type": "approval", "id": confirm["id"], "allowed": allowed})
            frames += receive_until(socket, "idle")
    assert confirm["request"]["tool"] == "write_file"
    assert "+x = 1" in confirm["request"]["preview"]
    assert APPROVAL_REQUESTED in event_types(frames)
    resolved = [
        f["event"]["payload"] for f in frames if f.get("event", {}).get("type") == APPROVAL_RESOLVED
    ]
    assert resolved[0]["approved"] is allowed
    assert (root / "a.py").exists() is allowed


def test_a_reply_without_a_pending_request_is_an_error(tmp_path):
    with client_for(api_config(tmp_path)) as client:
        session_id = new_session(client)
        with client.websocket_connect(f"/ws/sessions/{session_id}") as socket:
            socket.receive_json()
            socket.send_json({"type": "approval", "id": "x", "allowed": True})
            error = socket.receive_json()
    assert error["kind"] == "error" and "'x'" in error["message"]


@pytest.mark.feature("X2")
def test_cancel_interrupts_the_turn_and_keeps_the_partial_text(tmp_path):
    with client_for(api_config(tmp_path), BlockingGateway()) as client:
        session_id = new_session(client)
        with client.websocket_connect(f"/ws/sessions/{session_id}") as socket:
            socket.receive_json()
            socket.send_json({"type": "message", "text": "go"})
            receive_until(socket, "event", MODEL_CHUNK)
            socket.send_json({"type": "message", "text": "again"})
            busy = socket.receive_json()
            socket.send_json({"type": "cancel"})
            frames = receive_until(socket, "idle")
        messages = client.get(f"/api/sessions/{session_id}/messages").json()
    assert busy == {"kind": "error", "message": "Erreur : un tour est déjà en cours"}
    assert RUN_CANCELLED in event_types(frames)
    assert (messages[-1]["content"], messages[-1]["status"]) == ("partial", "interrupted")


@pytest.mark.feature("A5")
def test_the_persona_can_be_switched_between_turns(tmp_path):
    gateway = FakeModelGateway([Chunk("ok")])
    with client_for(api_config(tmp_path), gateway) as client:
        session_id = new_session(client)
        with client.websocket_connect(f"/ws/sessions/{session_id}") as socket:
            assert socket.receive_json()["persona"] is None
            socket.send_json({"type": "persona", "name": "reviewer"})
            switched = socket.receive_json()
            socket.send_json({"type": "persona", "name": "ghost"})
            unknown = socket.receive_json()
            socket.send_json({"type": "message", "text": "review"})
            frames = receive_until(socket, "idle")
    assert switched == {"kind": "persona", "name": "reviewer"}
    assert unknown["kind"] == "error"
    started = [
        f["event"]["payload"] for f in frames if f.get("event", {}).get("type") == "run.started"
    ]
    assert started[0]["persona"] == "reviewer"


@pytest.mark.feature("X2")
def test_fork_creates_a_session_from_a_message(tmp_path):
    gateway = FakeModelGateway([Chunk("one")], [Chunk("two")])
    with client_for(api_config(tmp_path), gateway) as client:
        session_id = new_session(client)
        with client.websocket_connect(f"/ws/sessions/{session_id}") as socket:
            socket.receive_json()
            for text in ("first", "second"):
                socket.send_json({"type": "message", "text": text})
                receive_until(socket, "idle")
        first_answer = client.get(f"/api/sessions/{session_id}/messages").json()[1]
        response = client.post(
            f"/api/sessions/{session_id}/fork", json={"message_id": first_answer["id"]}
        )
        forked = client.get(f"/api/sessions/{response.json()['id']}/messages").json()
        missing = client.post(f"/api/sessions/{session_id}/fork", json={"message_id": "nope"})
    assert response.status_code == 201
    assert [m["content"] for m in forked] == ["first", "one"]
    assert missing.status_code == 404


def test_the_built_web_ui_is_served_at_the_root(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<h1>SENSAI</h1>", encoding="utf-8")
    with client_for(api_config(tmp_path)) as client:
        page = client.get("/")
        health = client.get("/api/health")
    assert page.text == "<h1>SENSAI</h1>"
    assert health.json() == {"name": "sensai"}


def test_start_api_is_a_command():
    args = build_parser().parse_args(["start:api", "--model", "m"])
    assert (args.command, args.model) == ("start:api", "m")
