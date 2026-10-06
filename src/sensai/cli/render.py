"""Console rendering of the event stream."""

from rich.console import Console
from rich.status import Status
from rich.text import Text

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

    def __init__(self, console: Console, error_console: Console) -> None:
        self._console = console
        self._errors = error_console
        self._status: Status | None = None
        self._streaming = False
        self._tokens = [0, 0]

    def emit(self, event: Event) -> None:
        payload = event.payload
        if event.type == RUN_STARTED:
            self._tokens = [0, 0]
            self._streaming = False
            self._start_status()
        elif event.type == MODEL_CHUNK:
            self._stop_status()
            if not self._streaming:
                self._console.out("sensai> ", style="bold cyan", end="", highlight=False)
                self._streaming = True
            self._console.out(payload["text"], end="", highlight=False)
        elif event.type == MODEL_USAGE:
            self._tokens[0] += payload["prompt_tokens"]
            self._tokens[1] += payload["completion_tokens"]
        elif event.type == APPROVAL_REQUESTED:
            self._pause()
        elif event.type == APPROVAL_RESOLVED:
            self._start_status()
        elif event.type == TOOL_STARTED:
            self._pause()
            self._dim(f"tool> {preview_call(payload['tool'], payload['args'])}")
        elif event.type == TOOL_FINISHED:
            outcome = "returned an error" if payload["is_error"] else "finished"
            self._dim(f"tool> {payload['tool']} {outcome} ({payload['duration_ms']:.0f} ms)")
            self._start_status()
        elif event.type == TOOL_FAILED:
            self._pause()
            message = f"tool> {payload['tool']} failed: {payload['error']}"
            self._console.print(Text(message, style="yellow"))
            self._start_status()
        elif event.type == RUN_COMPLETED:
            self._pause()
            if any(self._tokens):
                prompt, completion = self._tokens
                total = prompt + completion
                self._dim(f"{total} tokens (prompt {prompt}, completion {completion})")
        elif event.type == RUN_CANCELLED:
            self._pause()
            self._dim("(interrupted)")
        elif event.type == RUN_FAILED:
            self._pause()
            message = Text("Erreur : ", style="bold red")
            message.append(payload["error"])
            self._errors.print(message)
            if payload["error_type"] in _HINTS:
                self._errors.print(Text(_HINTS[payload["error_type"]], style="dim"))

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
