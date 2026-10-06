import asyncio
import io
import os
import signal
import subprocess
import sys

import pytest
from rich.console import Console

from sensai.cli.channel import CliUserChannel
from sensai.cli.chat import run_chat, run_interruptible
from sensai.cli.main import build_parser, config_from_args, main
from sensai.cli.render import ConsoleRenderer
from sensai.core.config import load_config
from sensai.core.errors import ModelConnectionError
from sensai.core.events import (
    APPROVAL_REQUESTED,
    APPROVAL_RESOLVED,
    MODEL_CHUNK,
    MODEL_USAGE,
    RUN_CANCELLED,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_STARTED,
    TOOL_FAILED,
    TOOL_FINISHED,
    TOOL_STARTED,
    make_event,
)
from sensai.core.fakes import FakeModelGateway
from sensai.core.runtime import RunResult
from sensai.core.types import ApprovalRequest, Chunk, Usage
from sensai.db.connection import connect
from sensai.memory.sqlite_store import SqliteSessionStore

pytestmark = pytest.mark.feature("B1")


def make_console():
    buffer = io.StringIO()
    console = Console(
        file=buffer, width=100, color_system=None, force_terminal=False, highlight=False
    )
    return console, buffer


def render(events):
    console, out = make_console()
    errors, err = make_console()
    renderer = ConsoleRenderer(console, errors)
    for type_, payload in events:
        renderer.emit(make_event(type_, "r1", payload))
    return out.getvalue(), err.getvalue()


def scripted(*lines):
    remaining = iter(lines)

    def read_line(prompt):
        try:
            return next(remaining)
        except StopIteration:
            raise EOFError from None

    return read_line


STARTED = (RUN_STARTED, {"session_id": "s", "model": "m"})
COMPLETED = (RUN_COMPLETED, {"session_id": "s", "steps": 1, "tool_calls": 0})


def test_renderer_prints_the_streamed_answer_then_the_token_count():
    out, err = render(
        [
            STARTED,
            (MODEL_CHUNK, {"text": "Hel"}),
            (MODEL_CHUNK, {"text": "lo"}),
            (MODEL_USAGE, {"prompt_tokens": 5, "completion_tokens": 2}),
            COMPLETED,
        ]
    )
    assert out == "sensai> Hello\n7 tokens (prompt 5, completion 2)\n"
    assert err == ""


def test_renderer_does_not_interpret_markup_in_the_answer():
    out, _ = render([STARTED, (MODEL_CHUNK, {"text": "[bold]x[/bold] 42"}), COMPLETED])
    assert "sensai> [bold]x[/bold] 42\n" in out


def test_renderer_prints_nothing_for_a_turn_without_output():
    out, err = render([STARTED, COMPLETED])
    assert (out, err) == ("", "")


def test_renderer_counts_tokens_per_turn():
    usage = (MODEL_USAGE, {"prompt_tokens": 5, "completion_tokens": 2})
    out, _ = render([STARTED, usage, COMPLETED, STARTED, COMPLETED])
    assert out.count("tokens") == 1


def test_renderer_shows_an_interrupted_turn_on_its_own_line():
    out, _ = render(
        [
            STARTED,
            (MODEL_CHUNK, {"text": "par"}),
            (RUN_CANCELLED, {"session_id": "s", "partial": True}),
        ]
    )
    assert out == "sensai> par\n(interrupted)\n"


def test_renderer_prints_errors_with_a_hint_on_stderr():
    failed = {"session_id": "s", "error_type": "ModelConnectionError", "error": "Ollama is down"}
    out, err = render([STARTED, (RUN_FAILED, failed)])
    assert out == ""
    assert "Erreur : Ollama is down" in err
    assert "ollama serve" in err


def test_renderer_prints_an_error_without_hint_for_other_types():
    failed = {"session_id": "s", "error_type": "RunLimitError", "error": "Limite atteinte"}
    _, err = render([STARTED, (RUN_FAILED, failed)])
    assert err == "Erreur : Limite atteinte\n"


def test_renderer_shows_tool_activity():
    out, _ = render(
        [
            STARTED,
            (
                TOOL_STARTED,
                {"call_id": "c", "tool": "read_file", "risk": "read", "args": {"path": "a.py"}},
            ),
            (
                TOOL_FINISHED,
                {
                    "call_id": "c",
                    "tool": "read_file",
                    "is_error": False,
                    "duration_ms": 12.4,
                    "content": "x",
                },
            ),
            (TOOL_FAILED, {"call_id": "d", "tool": "write_file", "error": "denied"}),
            COMPLETED,
        ]
    )
    assert out == (
        'tool> read_file({"path": "a.py"})\n'
        "tool> read_file finished (12 ms)\n"
        "tool> write_file failed: denied\n"
    )


def test_renderer_ends_the_streamed_line_before_a_tool_line():
    out, _ = render(
        [
            STARTED,
            (MODEL_CHUNK, {"text": "Let me look."}),
            (TOOL_STARTED, {"call_id": "c", "tool": "read_file", "risk": "read", "args": {}}),
        ]
    )
    assert out == "sensai> Let me look.\ntool> read_file({})\n"


def test_renderer_prints_nothing_for_approval_events():
    out, err = render(
        [
            STARTED,
            (APPROVAL_REQUESTED, {"call_id": "c", "tool": "w", "risk": "write", "preview": "w()"}),
            (APPROVAL_RESOLVED, {"call_id": "c", "approved": True}),
            COMPLETED,
        ]
    )
    assert (out, err) == ("", "")


REQUEST = ApprovalRequest("r1", "write_file", "write", 'write_file({"path": "a.py"})', {})


@pytest.mark.parametrize("answer", ["y", "Y", "yes", " YES "])
async def test_confirm_accepts_a_yes(answer):
    console, _ = make_console()
    assert await CliUserChannel(console, lambda prompt: answer).confirm(REQUEST) is True


@pytest.mark.parametrize("answer", ["", "n", "no", "maybe"])
async def test_confirm_refuses_anything_else(answer):
    console, _ = make_console()
    assert await CliUserChannel(console, lambda prompt: answer).confirm(REQUEST) is False


async def test_confirm_refuses_at_end_of_input_and_shows_the_preview():
    console, out = make_console()
    assert await CliUserChannel(console, scripted()).confirm(REQUEST) is False
    assert out.getvalue() == 'Approval required (write): write_file({"path": "a.py"})\n'


async def test_ask_is_not_supported_yet():
    console, _ = make_console()
    with pytest.raises(NotImplementedError):
        await CliUserChannel(console).ask({"question": "?"})


def chat_setup(tmp_path, **overrides):
    storage = {"db_path": str(tmp_path / "sensai.db")}
    config = load_config(overrides={"storage": storage, **overrides})
    console, out = make_console()
    errors, err = make_console()
    return config, console, out, errors, err


async def test_each_line_is_a_turn_and_empty_lines_are_ignored(tmp_path):
    config, console, out, errors, err = chat_setup(tmp_path)
    gateway = FakeModelGateway([Chunk("one")], [Chunk("two")])
    lines = scripted("hello", "", "   ", "again", "/exit")
    assert await run_chat(config, console, errors, gateway=gateway, read_line=lines) == 0

    assert len(gateway.calls) == 2
    assert "sensai> one\n" in out.getvalue()
    assert "sensai> two\n" in out.getvalue()
    second = [(m.role, m.content) for m in gateway.calls[1].messages]
    assert second == [("user", "hello"), ("assistant", "one"), ("user", "again")]
    assert err.getvalue() == ""


async def test_the_banner_shows_the_model(tmp_path):
    config, console, out, errors, _ = chat_setup(tmp_path, model={"name": "llama3.2:3b"})
    await run_chat(config, console, errors, gateway=FakeModelGateway(), read_line=scripted())
    assert "model: llama3.2:3b" in out.getvalue()


@pytest.mark.parametrize("ending", [["/exit"], ["/quit"], []])
async def test_exit_command_and_end_of_input_stop_the_chat_cleanly(tmp_path, ending):
    config, console, _, errors, _ = chat_setup(tmp_path)
    gateway = FakeModelGateway([Chunk("one")])
    lines = scripted("hi", *ending)
    code = await run_chat(config, console, errors, gateway=gateway, read_line=lines)
    assert code == 0
    assert len(gateway.calls) == 1


async def test_ctrl_c_at_the_prompt_stops_the_chat_cleanly(tmp_path):
    config, console, _, errors, _ = chat_setup(tmp_path)

    def interrupt(prompt):
        raise KeyboardInterrupt

    gateway = FakeModelGateway()
    code = await run_chat(config, console, errors, gateway=gateway, read_line=interrupt)
    assert code == 0


async def test_a_model_error_is_shown_and_the_chat_continues(tmp_path):
    config, console, _, errors, err = chat_setup(tmp_path)
    gateway = FakeModelGateway(error=ModelConnectionError("Ollama is down"))
    code = await run_chat(config, console, errors, gateway=gateway, read_line=scripted("a", "b"))
    assert code == 0
    assert len(gateway.calls) == 2
    assert err.getvalue().count("Erreur : Ollama is down") == 2


async def test_an_unusable_database_ends_with_exit_code_2(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    config, console, _, errors, err = chat_setup(tmp_path, storage={"db_path": str(blocker / "db")})
    code = await run_chat(config, console, errors, gateway=FakeModelGateway(), read_line=scripted())
    assert code == 2
    assert "base de données" in err.getvalue()


async def test_turns_and_events_are_saved_in_the_database(tmp_path):
    config, console, _, errors, _ = chat_setup(tmp_path)
    gateway = FakeModelGateway([Chunk("one"), Usage(5, 2)], [Chunk("two")])
    await run_chat(config, console, errors, gateway=gateway, read_line=scripted("a", "b"))

    db = connect(config.storage.db_path)
    [session_id] = [row[0] for row in db.execute("SELECT id FROM sessions")]
    history = SqliteSessionStore(db).history(session_id)
    assert [(m.role, m.content) for m in history] == [
        ("user", "a"),
        ("assistant", "one"),
        ("user", "b"),
        ("assistant", "two"),
    ]
    assert [row[0] for row in db.execute("SELECT status FROM runs")] == ["completed", "completed"]
    assert db.execute("SELECT COUNT(*) FROM events").fetchone()[0] > 0


async def test_run_interruptible_returns_the_result():
    async def turn():
        return RunResult("r1", "completed", "ok")

    assert (await run_interruptible(turn())).answer == "ok"


async def test_run_interruptible_works_without_signal_support(monkeypatch):
    def unsupported(*args):
        raise NotImplementedError

    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", unsupported)

    async def turn():
        return RunResult("r1", "completed", "ok")

    assert (await run_interruptible(turn())).answer == "ok"


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT handling is Unix-only")
async def test_ctrl_c_cancels_the_running_turn_and_restores_the_default_handler():
    cancelled = False

    async def slow_turn():
        nonlocal cancelled
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled = True
            raise

    asyncio.get_running_loop().call_later(0.05, os.kill, os.getpid(), signal.SIGINT)
    assert await run_interruptible(slow_turn()) is None
    assert cancelled
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


async def test_a_cancellation_that_is_not_ctrl_c_is_not_swallowed():
    async def slow_turn():
        await asyncio.sleep(30)

    outer = asyncio.create_task(run_interruptible(slow_turn()))
    await asyncio.sleep(0.01)
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer


def test_parser_reads_the_chat_options(tmp_path):
    args = build_parser().parse_args(["chat", "--model", "llama3.2:3b", "--config", "c.toml"])
    assert (args.command, args.model, str(args.config)) == ("chat", "llama3.2:3b", "c.toml")


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_model_option_overrides_the_configuration():
    default = config_from_args(build_parser().parse_args(["chat"]))
    chosen = config_from_args(build_parser().parse_args(["chat", "--model", "llama3.2:3b"]))
    assert chosen.model.name == "llama3.2:3b"
    assert chosen.model.host == default.model.host


def test_config_file_option_is_applied(tmp_path):
    user_file = tmp_path / "user.toml"
    user_file.write_text('[model]\nname = "from-file"\n', encoding="utf-8")
    args = build_parser().parse_args(["chat", "--config", str(user_file)])
    assert config_from_args(args).model.name == "from-file"


@pytest.mark.parametrize("content", [None, "[model\nname ="])
def test_main_exits_with_code_2_on_a_bad_config_file(tmp_path, capsys, content):
    path = tmp_path / "user.toml"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    assert main(["chat", "--config", str(path)]) == 2
    assert "Erreur" in capsys.readouterr().err


def test_the_module_can_be_run_with_python_dash_m():
    result = subprocess.run(
        [sys.executable, "-m", "sensai", "chat", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--model" in result.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="needs a pseudo-terminal")
@pytest.mark.parametrize("key", [b"\x03", b"\x04"], ids=["ctrl-c", "ctrl-d"])
def test_ctrl_c_and_ctrl_d_at_the_first_prompt_quit_in_a_real_terminal(tmp_path, key):
    import pty
    import select
    import time

    config = tmp_path / "config.toml"
    config.write_text(f'[storage]\ndb_path = "{tmp_path / "sensai.db"}"\n', encoding="utf-8")
    pid, fd = pty.fork()
    if pid == 0:
        os.execv(sys.executable, [sys.executable, "-m", "sensai", "chat", "--config", str(config)])

    output = b""
    deadline = time.time() + 20
    while b"you> " not in output and time.time() < deadline:
        if select.select([fd], [], [], 0.2)[0]:
            output += os.read(fd, 4096)
    assert b"you> " in output
    os.write(fd, key)

    status = None
    deadline = time.time() + 10
    while status is None and time.time() < deadline:
        finished, raw_status = os.waitpid(pid, os.WNOHANG)
        if finished:
            status = os.waitstatus_to_exitcode(raw_status)
        else:
            time.sleep(0.05)
    if status is None:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    os.close(fd)
    assert status == 0
