import asyncio
from pathlib import Path

import pytest

from sensai.core.config import load_config
from sensai.tools.sandbox import docker
from sensai.tools.sandbox.docker import (
    docker_run_args,
    docker_unavailable_reason,
    run_container,
)

pytestmark = pytest.mark.feature("T2")


def _args(command="python -m pytest"):
    sandbox = load_config().sandbox
    return docker_run_args(sandbox, "sensai-test", Path("/repo"), command)


def _value_after(args, flag):
    return args[args.index(flag) + 1]


def test_run_args_isolate_the_container():
    args = _args()
    assert args[:3] == ["docker", "run", "--rm"]
    assert _value_after(args, "--name") == "sensai-test"
    assert _value_after(args, "--network") == "none"
    assert "--read-only" in args
    assert _value_after(args, "--cap-drop") == "ALL"
    assert _value_after(args, "--security-opt") == "no-new-privileges"


def test_run_args_apply_the_configured_limits():
    sandbox = load_config().sandbox
    args = _args()
    assert _value_after(args, "--cpus") == str(sandbox.cpus)
    assert _value_after(args, "--memory") == f"{sandbox.memory_mb}m"
    assert _value_after(args, "--memory-swap") == f"{sandbox.memory_mb}m"
    assert _value_after(args, "--pids-limit") == str(sandbox.pids_limit)


def test_run_args_mount_the_workspace_read_only():
    args = _args()
    assert _value_after(args, "-v") == "/repo:/workspace:ro"
    assert _value_after(args, "-w") == "/workspace"
    assert _value_after(args, "--tmpfs").startswith("/tmp:rw")


def test_the_command_is_passed_as_one_shell_argument():
    args = _args("pytest -x && echo done")
    assert args[-4:] == [load_config().sandbox.image, "sh", "-c", "pytest -x && echo done"]


async def test_reason_when_docker_is_not_installed(monkeypatch):
    monkeypatch.setattr(docker.shutil, "which", lambda _: None)
    assert "introuvable" in await docker_unavailable_reason("img")


def _fake_quiet(failing):
    async def quiet(*args):
        return 1 if args[1] == failing else 0

    return quiet


async def test_reason_when_the_daemon_does_not_answer(monkeypatch):
    monkeypatch.setattr(docker.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(docker, "_quiet", _fake_quiet("info"))
    assert "démon" in await docker_unavailable_reason("img")


async def test_reason_when_the_image_is_missing(monkeypatch):
    monkeypatch.setattr(docker.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(docker, "_quiet", _fake_quiet("image"))
    reason = await docker_unavailable_reason("sensai-sandbox:latest")
    assert "sensai-sandbox:latest" in reason
    assert "make sandbox-image" in reason


async def test_no_reason_when_everything_is_ready(monkeypatch):
    monkeypatch.setattr(docker.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(docker, "_quiet", _fake_quiet(None))
    assert await docker_unavailable_reason("img") is None


async def test_run_container_returns_the_exit_code_and_merged_output():
    result = await run_container(["sh", "-c", "echo out; echo err >&2; exit 3"], "n", 5)
    assert result.exit_code == 3
    assert result.output == b"out\nerr\n"
    assert not result.timed_out


def _record_stops(monkeypatch):
    stopped = []

    async def stop(proc, name):
        stopped.append(name)
        proc.kill()
        await proc.wait()

    monkeypatch.setattr(docker, "_stop", stop)
    return stopped


async def test_timeout_stops_the_container_and_keeps_the_partial_output(monkeypatch):
    stopped = _record_stops(monkeypatch)
    result = await run_container(["sh", "-c", "echo started; exec sleep 5"], "box", 0.5)
    assert result.timed_out
    assert result.exit_code is None
    assert result.output == b"started\n"
    assert stopped == ["box"]


async def test_cancel_stops_the_container_and_propagates(monkeypatch):
    stopped = _record_stops(monkeypatch)
    task = asyncio.create_task(run_container(["sleep", "5"], "box", 10))
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped == ["box"]
