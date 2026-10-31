"""EV3: run metrics (latency, tokens, tools, errors) read from the event journal."""

import argparse
import json

import pytest
from tests.unit.test_cli import chat_setup

from sensai.cli.chat import run_chat
from sensai.cli.main import main
from sensai.cli.stats import parse_period, since_timestamp
from sensai.core.config import load_config
from sensai.core.events import make_event
from sensai.core.fakes import FakeModelGateway
from sensai.core.types import Chunk
from sensai.db.connection import connect
from sensai.observability.journal import SqliteEventJournal
from sensai.observability.stats import percentile, run_stats

pytestmark = pytest.mark.feature("EV3")


@pytest.fixture
def journal():
    connection = connect(":memory:")
    yield connection, SqliteEventJournal(connection)
    connection.close()


def play(
    journal,
    run_id,
    start,
    *,
    end="run.completed",
    duration=2.0,
    first_token=0.5,
    tokens=(100, 20),
    tools=(),
):
    """Journal one run started at `start`: tools are (name, status, duration_ms) triples."""
    emit = journal.emit
    emit(make_event("run.started", run_id, {"session_id": "s", "model": "m"}, ts=start))
    if first_token is not None:
        emit(make_event("model.chunk", run_id, {"text": "a"}, ts=start + first_token))
        emit(make_event("model.chunk", run_id, {"text": "b"}, ts=start + first_token + 0.1))
    for index, (name, status, duration_ms) in enumerate(tools):
        call = {"call_id": f"{run_id}-{index}", "tool": name}
        emit(make_event("model.tool_call", run_id, {**call, "name": name, "args": {}}, ts=start))
        if status == "denied":
            emit(make_event("tool.failed", run_id, {**call, "error": "denied"}, ts=start))
        elif status == "failed":
            emit(make_event("tool.failed", run_id, {**call, "error": "boom"}, ts=start + 1))
        else:
            payload = {
                **call,
                "is_error": status == "error",
                "duration_ms": duration_ms,
                "content": "",
            }
            emit(make_event("tool.finished", run_id, payload, ts=start))
    prompt, completion = tokens
    usage = {"prompt_tokens": prompt, "completion_tokens": completion}
    emit(make_event("model.usage", run_id, usage, ts=start + duration))
    payload = {"session_id": "s"}
    if end == "run.failed":
        payload.update(error_type="RunLimitError", error="Délai dépassé")
    emit(make_event(end, run_id, payload, ts=start + duration))


def test_ev3_overview_counts_runs_by_status_and_tokens(journal):
    connection, sink = journal
    play(sink, "r1", 1000, tokens=(100, 20))
    play(sink, "r2", 1010, end="run.cancelled", tokens=(50, 5))
    play(sink, "r3", 1020, end="run.failed", tokens=(30, 0))
    overview = run_stats(connection)["overview"]
    assert overview["runs"] == 3
    assert (overview["completed"], overview["cancelled"], overview["failed"]) == (1, 1, 1)
    assert overview["prompt_tokens"] == 180 and overview["completion_tokens"] == 25
    assert overview["avg_tokens"] == pytest.approx(205 / 3)


def test_ev3_latency_measures_the_turn_and_the_first_token(journal):
    connection, sink = journal
    for index, duration in enumerate([1.0, 2.0, 3.0, 4.0, 10.0]):
        play(sink, f"r{index}", 1000 + 100 * index, duration=duration, first_token=duration / 4)
    play(sink, "silent", 2000, first_token=None)
    play(sink, "failed", 2100, end="run.failed", duration=0.01, first_token=None)
    play(sink, "stopped", 2200, end="run.cancelled", duration=60.0, first_token=0.1)
    latency = run_stats(connection)["latency"]
    turn, first = latency["turn"], latency["first_token"]
    assert turn["count"] == 6 and turn["max"] == pytest.approx(10.0)
    assert turn["p50"] == pytest.approx(2.0) and turn["p95"] == pytest.approx(10.0)
    assert first["count"] == 5
    assert first["avg"] == pytest.approx(sum([0.25, 0.5, 0.75, 1.0, 2.5]) / 5)


def test_ev3_percentile_uses_the_nearest_rank():
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.0
    assert percentile(list(map(float, range(1, 21))), 0.95) == 19.0


def test_ev3_tools_split_ok_errors_and_refusals(journal):
    connection, sink = journal
    tools = [
        ("read_file", "ok", 10.0),
        ("read_file", "ok", 30.0),
        ("run_command", "error", 2000.0),
        ("run_command", "failed", None),
        ("write_file", "denied", None),
    ]
    play(sink, "r1", 1000, tools=tools)
    rows = {row["tool"]: row for row in run_stats(connection)["tools"]}
    assert rows["read_file"]["calls"] == 2 and rows["read_file"]["ok"] == 2
    assert rows["read_file"]["avg_ms"] == pytest.approx(20.0)
    assert rows["run_command"]["errors"] == 2 and rows["run_command"]["ok"] == 0
    assert rows["write_file"]["refused"] == 1 and rows["write_file"]["errors"] == 0


def test_ev3_errors_list_failures_newest_first_without_refusals(journal):
    connection, sink = journal
    play(sink, "r1", 1000, end="run.failed")
    play(sink, "r2", 1100, tools=[("run_command", "failed", None), ("edit", "denied", None)])
    errors = run_stats(connection)["errors"]
    assert [(error["type"], error["kind"]) for error in errors] == [
        ("tool.failed", "run_command"),
        ("run.failed", "RunLimitError"),
    ]
    assert errors[1]["message"] == "Délai dépassé"
    assert len(run_stats(connection, errors=1)["errors"]) == 1


def test_ev3_guards_are_counted(journal):
    connection, sink = journal
    play(sink, "r1", 1000)
    sink.emit(make_event("guard.blocked", "r1", {"stage": "entry", "reason": "x"}, ts=1001))
    sink.emit(make_event("guard.redacted", "r1", {"stage": "answer", "count": 2}, ts=1001))
    sink.emit(make_event("guard.redacted", "r1", {"stage": "tool", "count": 1}, ts=1001))
    assert run_stats(connection)["guards"] == {"blocked": 1, "redacted": 2}


def test_ev3_since_keeps_only_the_recent_runs(journal):
    connection, sink = journal
    play(sink, "old", 1000, end="run.failed", tools=[("read_file", "ok", 5.0)])
    play(sink, "new", 5000, tokens=(7, 3))
    stats = run_stats(connection, since=4000)
    assert stats["overview"]["runs"] == 1 and stats["overview"]["prompt_tokens"] == 7
    assert stats["tools"] == [] and stats["errors"] == []
    assert stats["latency"]["turn"]["count"] == 1


def test_ev3_empty_journal_gives_empty_metrics(journal):
    stats = run_stats(journal[0])
    assert stats["overview"]["runs"] == 0 and stats["overview"]["avg_tokens"] is None
    assert stats["latency"]["turn"] == {
        "count": 0,
        "avg": None,
        "p50": None,
        "p95": None,
        "max": None,
    }


@pytest.mark.parametrize("value, seconds", [("30s", 30), ("15m", 900), ("2h", 7200), ("1d", 86400)])
def test_ev3_since_periods_are_parsed(value, seconds):
    assert parse_period(value) == seconds
    assert since_timestamp(seconds, now=100000) == 100000 - seconds
    assert since_timestamp(None) is None


@pytest.mark.parametrize("value", ["", "2", "h", "0m", "-1h", "1w", "1.5h"])
def test_ev3_invalid_periods_are_refused(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_period(value)


def write_config(tmp_path):
    path = tmp_path / "c.toml"
    db = tmp_path / "sensai.db"
    path.write_text(f'[storage]\ndb_path = "{db}"\n[react]\nenabled = false\n', encoding="utf-8")
    return path


async def test_ev3_stats_command_reports_the_chat_runs(tmp_path, capsys):
    path = write_config(tmp_path)
    assert main(["stats", "--config", str(path)]) == 0
    assert "No run recorded yet." in capsys.readouterr().out
    _, terminal, _, _ = chat_setup(tmp_path, "hi", "again")
    gateway = FakeModelGateway([Chunk("one")], [Chunk("two")])
    await run_chat(load_config(path), terminal, gateway)
    assert main(["stats", "--config", str(path)]) == 0
    out = capsys.readouterr().out
    assert "Runs         2 (completed 2)" in out
    assert "Turn" in out and "First token" in out and "p95" in out
    assert main(["stats", "--config", str(path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["overview"]["completed"] == 2 and data["latency"]["first_token"]["count"] == 2


def test_ev3_stats_command_shows_tools_and_errors(tmp_path, capsys):
    path = write_config(tmp_path)
    connection = connect(tmp_path / "sensai.db")
    sink = SqliteEventJournal(connection)
    play(sink, "r1", 1000, tools=[("read_file", "ok", 12.0)])
    play(sink, "r2", 1100, end="run.failed")
    connection.close()
    assert main(["stats", "--config", str(path)]) == 0
    out = capsys.readouterr().out
    assert "read_file" in out and "Recent errors" in out
    assert "RunLimitError: Délai dépassé" in out
    assert main(["stats", "--config", str(path), "--since", "1h"]) == 0
    assert "No run recorded yet." in capsys.readouterr().out


def test_ev3_stats_command_reports_an_unusable_database(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("", encoding="utf-8")
    path = tmp_path / "c.toml"
    path.write_text(f'[storage]\ndb_path = "{blocker / "sensai.db"}"\n', encoding="utf-8")
    assert main(["stats", "--config", str(path)]) == 2
    assert "base de données inutilisable" in capsys.readouterr().err


def test_ev3_invalid_since_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["stats", "--since", "soon"])
    assert exit_info.value.code == 2
    assert "invalid period" in capsys.readouterr().err
