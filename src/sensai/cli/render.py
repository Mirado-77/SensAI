"""Console rendering of the event stream."""

from typing import Any

from rich.status import Status
from rich.text import Text

from sensai.cli.terminal import Terminal
from sensai.core.events import (
    APPROVAL_REQUESTED,
    APPROVAL_RESOLVED,
    CACHE_HIT,
    MEMORY_BUDGET,
    MEMORY_COMPRESSED,
    MODEL_CHUNK,
    MODEL_USAGE,
    QUESTION_ANSWERED,
    QUESTION_REQUESTED,
    REACT_ACTION,
    REACT_PLAN,
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
        self._budget: dict[str, Any] | None = None
        self._handlers = {
            RUN_STARTED: self._run_started,
            CACHE_HIT: self._cache_hit,
            MODEL_CHUNK: self._chunk,
            MODEL_USAGE: self._usage,
            MEMORY_COMPRESSED: self._memory_compressed,
            MEMORY_BUDGET: self._memory_budget,
            REACT_PLAN: self._react_plan,
            REACT_ACTION: self._react_action,
            QUESTION_REQUESTED: self._approval_requested,
            QUESTION_ANSWERED: self._approval_resolved,
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
        self._budget = None
        self._streaming = False
        self._start_status()

    def _cache_hit(self, payload: dict[str, Any]) -> None:
        self._pause()
        self._dim(
            f"cache> déjà répondu à « {payload['question']} » (similarité {payload['score']:.2f}),"
            " /cache clear pour l'oublier"
        )

    def _chunk(self, payload: dict[str, Any]) -> None:
        self._stop_status()
        if not self._streaming:
            self._console.out("sensai> ", style="bold cyan", end="", highlight=False)
            self._streaming = True
        self._console.out(payload["text"], end="", highlight=False)

    def _usage(self, payload: dict[str, Any]) -> None:
        self._tokens[0] += payload["prompt_tokens"]
        self._tokens[1] += payload["completion_tokens"]

    def _memory_compressed(self, payload: dict[str, Any]) -> None:
        self._pause()
        self._dim(
            f"memory> summarized {payload['summarized']} old messages"
            f" (~{payload['before']} -> ~{payload['after']} tokens)"
        )
        self._start_status()

    def _memory_budget(self, payload: dict[str, Any]) -> None:
        self._budget = payload

    def _react_plan(self, payload: dict[str, Any]) -> None:
        if not payload["steps"]:
            return
        self._pause()
        self._console.print(Text(f"plan> {payload['goal']}", style="cyan"))
        for index, step in enumerate(payload["steps"], 1):
            self._dim(f"  {index}. {step}")
        self._start_status()

    def _react_action(self, payload: dict[str, Any]) -> None:
        self._pause()
        self._console.print(Text(f"step {payload['step']}> {', '.join(payload['tools'])}", "cyan"))

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
        line = []
        if any(self._tokens):
            prompt, completion = self._tokens
            line.append(f"{prompt + completion} tokens (prompt {prompt}, completion {completion})")
        if self._budget is not None:
            used, limit = self._budget["used"], self._budget["limit"]
            line.append(f"context ~{used}/{limit} ({used * 100 // limit}%)")
        if line:
            self._dim(" | ".join(line))

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
