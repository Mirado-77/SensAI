"""Console rendering of the event stream."""

from typing import Any

from rich.status import Status
from rich.text import Text

from sensai.cli.terminal import Terminal
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
    Event,
)
from sensai.safety.policy import preview_call

_HINTS = {
    "ModelConnectionError": (
        "Vérifiez qu'Ollama est lancé (`make ollama` ou `ollama serve`) "
        "et que `model.host` est correct."
    ),
    "ModelNotFoundError": (
        "Téléchargez le modèle avec `ollama pull <modèle>` ou choisissez-en un autre avec --model."
    ),
}


class ConsoleRenderer:
    """EventSink that prints the streamed answer, tool activity and errors."""

    def __init__(self, terminal: Terminal) -> None:
        self._console = terminal.console
        self._errors = terminal.errors
        self._status: Status | None = None
        self._streaming = False
        self._tokens = [0, 0]
        self._handlers = {
            RUN_STARTED: self._run_started,
            MODEL_CHUNK: self._chunk,
            MODEL_USAGE: self._usage,
            APPROVAL_REQUESTED: self._approval_requested,
            APPROVAL_RESOLVED: self._approval_resolved,
            TOOL_STARTED: self._tool_started,
            TOOL_FINISHED: self._tool_finished,
            TOOL_FAILED: self._tool_failed,
            RUN_COMPLETED: self._run_completed,
            RUN_CANCELLED: self._run_cancelled,
            RUN_FAILED: self._run_failed,
        }

    def emit(self, event: Event) -> None:
        handler = self._handlers.get(event.type)
        if handler is not None:
            handler(event.payload)

    def _run_started(self, payload: dict[str, Any]) -> None:
        self._tokens = [0, 0]
        self._streaming = False
        self._start_status()

    def _chunk(self, payload: dict[str, Any]) -> None:
        self._stop_status()
        if not self._streaming:
            self._console.out("sensai> ", style="bold cyan", end="", highlight=False)
            self._streaming = True
        self._console.out(payload["text"], end="", highlight=False)

    def _usage(self, payload: dict[str, Any]) -> None:
        self._tokens[0] += payload["prompt_tokens"]
        self._tokens[1] += payload["completion_tokens"]

    def _approval_requested(self, payload: dict[str, Any]) -> None:
        self._pause()

    def _approval_resolved(self, payload: dict[str, Any]) -> None:
        self._start_status()

    def _tool_started(self, payload: dict[str, Any]) -> None:
        self._pause()
        self._dim(f"tool> {preview_call(payload['tool'], payload['args'])}")

    def _tool_finished(self, payload: dict[str, Any]) -> None:
        outcome = "returned an error" if payload["is_error"] else "finished"
        self._dim(f"tool> {payload['tool']} {outcome} ({payload['duration_ms']:.0f} ms)")
        self._start_status()

    def _tool_failed(self, payload: dict[str, Any]) -> None:
        self._pause()
        message = f"tool> {payload['tool']} failed: {payload['error']}"
        self._console.print(Text(message, style="yellow"))
        self._start_status()

    def _run_completed(self, payload: dict[str, Any]) -> None:
        self._pause()
        if any(self._tokens):
            prompt, completion = self._tokens
            self._dim(f"{prompt + completion} tokens (prompt {prompt}, completion {completion})")

    def _run_cancelled(self, payload: dict[str, Any]) -> None:
        self._pause()
        self._dim("(interrupted)")

    def _run_failed(self, payload: dict[str, Any]) -> None:
        self._pause()
        message = Text("Erreur : ", style="bold red")
        message.append(payload["error"])
        self._errors.print(message)
        hint = _HINTS.get(payload["error_type"])
        if hint is not None:
            self._errors.print(Text(hint, style="dim"))

    def _dim(self, message: str) -> None:
        self._console.print(Text(message, style="dim"))

    def _start_status(self) -> None:
        self._stop_status()
        self._status = self._console.status("thinking...", spinner="dots")
        self._status.start()

    def _stop_status(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def _pause(self) -> None:
        """Stop the spinner and end the line of streamed text, before printing anything else."""
        self._stop_status()
        if self._streaming:
            self._console.out("")
            self._streaming = False
