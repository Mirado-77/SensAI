import secrets
from pathlib import Path
from typing import Any

from sensai.core.config import SandboxConfig
from sensai.core.types import Risk, ToolContext, ToolResult
from sensai.tools.filesystem.guard import PathDeniedError, PathGuard
from sensai.tools.sandbox.docker import ContainerResult, docker_run_args, run_container

DOCKER_RUN_FAILED = 125
KILLED = 137
NOT_AN_ERROR_ON_ONE = {
    "grep": "Aucune correspondance trouvée.",
    "rg": "Aucune correspondance trouvée.",
    "diff": "Les fichiers diffèrent.",
    "test": "La condition est fausse.",
    "[": "La condition est fausse.",
}
NO_NETWORK_HINT = (
    "Le conteneur n'a pas de réseau : impossible de télécharger ou d'installer des dépendances."
)
SANDBOX_HINTS = (
    (
        "Read-only file system",
        "Le dépôt est monté en lecture seule : modifie les fichiers avec edit_file ou write_file,"
        " et écris les fichiers temporaires dans /tmp.",
    ),
    ("Temporary failure in name resolution", NO_NETWORK_HINT),
    ("Network is unreachable", NO_NETWORK_HINT),
    ("EAI_AGAIN", NO_NETWORK_HINT),
)


def truncate_output(data: bytes, max_bytes: int) -> str:
    """Keep the start and the end of the output, where tools print their summary."""
    if len(data) <= max_bytes:
        return data.decode("utf-8", errors="replace")
    half = max_bytes // 2
    head = data[:half].decode("utf-8", errors="replace")
    tail = data[-half:].decode("utf-8", errors="replace")
    removed = data[half:-half].count(b"\n")
    return f"{head}\n[... {removed} lines truncated ...]\n{tail}"


def interpret_exit_code(command: str, exit_code: int) -> tuple[bool, str | None]:
    """Return (is_error, note). Some programs use the exit code 1 for a normal answer."""
    if exit_code == 0:
        return False, None
    words = command.split()
    program = Path(words[0]).name if words else ""
    if exit_code == 1 and program in NOT_AN_ERROR_ON_ONE:
        return False, NOT_AN_ERROR_ON_ONE[program]
    return True, f"La commande a échoué avec le code {exit_code}."


def sandbox_hints(output: str, exit_code: int | None) -> list[str]:
    """Explain the failures caused by the sandbox, so the model does not retry them."""
    hints = []
    for marker, hint in SANDBOX_HINTS:
        if marker in output and hint not in hints:
            hints.append(hint)
    if exit_code == KILLED:
        hints.append("Le processus a été tué (code 137) : limite de mémoire atteinte.")
    return hints


def to_tool_result(command: str, result: ContainerResult, sandbox: SandboxConfig) -> ToolResult:
    full_output = result.output.decode("utf-8", errors="replace")
    if result.exit_code == DOCKER_RUN_FAILED:
        message = f"Docker n'a pas pu lancer le conteneur (code 125) : {full_output.strip()}"
        return ToolResult(message, is_error=True)
    notes = sandbox_hints(full_output, result.exit_code)
    if result.timed_out:
        is_error = True
        notes.append("La commande a dépassé le délai et a été arrêtée.")
    else:
        is_error, note = interpret_exit_code(command, result.exit_code or 0)
        notes += [note] if note else []
    text = truncate_output(result.output, sandbox.max_output_bytes).rstrip() or "(no output)"
    return ToolResult("\n".join([text, *notes]), is_error=is_error, untrusted=True)


def _check_arguments(args: dict[str, Any]) -> ToolResult | None:
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        return ToolResult('Argument "command" manquant ou invalide.', is_error=True)
    for key in ("description", "path"):
        if args.get(key) is not None and not isinstance(args[key], str):
            return ToolResult(f'Argument "{key}" invalide.', is_error=True)
    timeout = args.get("timeout_s")
    valid_number = isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
    if timeout is not None and not (valid_number and timeout > 0):
        return ToolResult('Argument "timeout_s" invalide.', is_error=True)
    return None


class RunCommandTool:
    """A tool that runs a shell command in an isolated Docker container"""

    name = "run_command"
    description = (
        "Run a shell command (for example the tests of the project) in an isolated container."
        " The directory is mounted read-only at /workspace: change files with edit_file or"
        " write_file, write temporary files in /tmp. There is no network, so dependencies"
        " cannot be installed. Returns the merged stdout and stderr, truncated if too long."
    )
    risk: Risk = "exec"
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run, run by sh -c"},
            "description": {
                "type": "string",
                "description": "Short sentence that says what the command does, shown to the user",
            },
            "path": {
                "type": "string",
                "description": "Directory to run in. Defaults to the first allowed root.",
            },
            "timeout_s": {"type": "number", "description": "Optional timeout in seconds"},
        },
        "required": ["command"],
    }

    def __init__(self, guard: PathGuard, sandbox: SandboxConfig) -> None:
        self._guard = guard
        self._sandbox = sandbox

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        error = _check_arguments(args)
        if error is not None:
            return error
        try:
            workspace = self._workspace(args)
        except PathDeniedError as denied:
            return ToolResult(str(denied), is_error=True)
        name = f"sensai-{secrets.token_hex(6)}"
        run_args = docker_run_args(self._sandbox, name, workspace, args["command"])
        result = await run_container(run_args, name, self._timeout(args))
        return to_tool_result(args["command"], result, self._sandbox)

    async def preview(self, args: dict[str, Any]) -> str | None:
        if _check_arguments(args) is not None:
            return None
        workspace = self._workspace(args)
        lines = [f"# {args['description']}"] if args.get("description") else []
        lines.append(f"$ {args['command']}")
        lines.append(f"[{self._sandbox.image}, {workspace} read-only, no network]")
        return "\n".join(lines)

    def _workspace(self, args: dict[str, Any]) -> Path:
        raw_path = args.get("path") or "."
        workspace = self._guard.resolve(raw_path, write=False)
        if not workspace.is_dir():
            raise PathDeniedError(f"Dossier introuvable : {raw_path}")
        return workspace

    def _timeout(self, args: dict[str, Any]) -> float:
        requested = args.get("timeout_s")
        if requested is None:
            return self._sandbox.timeout_s
        return min(float(requested), self._sandbox.timeout_s)
