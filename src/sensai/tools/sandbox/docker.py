import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path

from sensai.core.config import SandboxConfig

READ_CHUNK_BYTES = 4096


@dataclass(frozen=True)
class ContainerResult:
    exit_code: int | None
    output: bytes
    timed_out: bool


def docker_run_args(sandbox: SandboxConfig, name: str, workspace: Path, command: str) -> list[str]:
    """Build the `docker run` command line. Nothing is started."""
    return [
        "docker", "run", "--rm", "--name", name,
        "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=64m",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--cpus", str(sandbox.cpus),
        "--memory", f"{sandbox.memory_mb}m", "--memory-swap", f"{sandbox.memory_mb}m",
        "--pids-limit", str(sandbox.pids_limit),
        "-v", f"{workspace}:/workspace:ro", "-w", "/workspace",
        sandbox.image, "sh", "-c", command,
    ]


async def _quiet(*args: str) -> int:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    return await proc.wait()


async def docker_unavailable_reason(image: str) -> str | None:
    """Return why the sandbox cannot run, in French, or None when it can."""
    if shutil.which("docker") is None:
        return "Docker est introuvable : la commande `docker` n'est pas installée."
    if await _quiet("docker", "info") != 0:
        return "Le démon Docker ne répond pas : démarre Docker puis relance SENSAI."
    if await _quiet("docker", "image", "inspect", image) != 0:
        return f"L'image {image} est absente : lance `make sandbox-image`."
    return None


async def kill_container(name: str) -> None:
    await _quiet("docker", "kill", name)


async def _read_until_exit(proc: asyncio.subprocess.Process, chunks: list[bytes]) -> None:
    assert proc.stdout is not None
    while chunk := await proc.stdout.read(READ_CHUNK_BYTES):
        chunks.append(chunk)
    await proc.wait()


async def _stop(proc: asyncio.subprocess.Process, name: str) -> None:
    await kill_container(name)
    await proc.wait()


async def run_container(args: list[str], name: str, timeout_s: float) -> ContainerResult:
    """Run the container and collect its merged output. A timeout or a cancel kills it."""
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    chunks: list[bytes] = []
    try:
        await asyncio.wait_for(_read_until_exit(proc, chunks), timeout_s)
    except asyncio.TimeoutError:
        await _stop(proc, name)
        return ContainerResult(None, b"".join(chunks), timed_out=True)
    except asyncio.CancelledError:
        await _stop(proc, name)
        raise
    return ContainerResult(proc.returncode, b"".join(chunks), timed_out=False)
