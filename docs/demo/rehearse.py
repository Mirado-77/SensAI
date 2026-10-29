"""Unattended rehearsal of the keynote scenario (docs/demo/SCENARIO.md), recorded for replay.

Plays every step in a real terminal (pty) against the real model, approves each request with
"y", picks option 1 of a question, and presses Ctrl-C in the interrupted step. The whole session
is recorded in docs/demo/recordings/ in the format of `script`, with the waits on the model
shortened (see Recorder), so it can be replayed at normal speed:

    scriptreplay -t docs/demo/recordings/<date>.timing docs/demo/recordings/<date>.typescript

Run it through `make demo-rehearse` (which resets the demo first). Step markers and timings go to
stderr only, so the recording shows exactly what the audience would see.
"""

import os
import pty
import re
import select
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORDINGS = ROOT / "docs" / "demo" / "recordings"
BIN = Path(sys.executable).parent
CONFIG = ["--config", "configs/demo.toml"]

PROMPT = "you> "
APPROVAL = "Allow? [y/N] "
QUESTION = re.compile(r"Answer (\[[^\]]*\] )?> $")
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# One redraw of the "thinking..." status line (rich spinner): what the time-lapse shortens.
SPINNER = re.compile(r"(\r|\x1b\[[0-9;?]*[A-Za-z]|[⠀-⣿]| thinking\.\.\.)+")
SPINNER_KEEP_S = 3
MAX_PAUSE_S = 2  # a silent wait (model loading, MCP server starting) is shortened too
TURN_TIMEOUT_S = 3600
INTERRUPT_AFTER_S = 25  # on a CPU, enough for a paragraph to appear before Ctrl-C
LONG_REQUEST = "Écris une documentation détaillée de chaque fonction du projet, avec un exemple"

# Each act is one `sensai chat` process; "/fork" without a number forks just before LONG_REQUEST.
ACTS = [
    (
        [],
        [
            "Je reprends ce projet que je ne connais pas. Explique-moi en quelques lignes ce qu'il"
            " fait et comment il est organisé.",
            "Lis CONTRIBUTING.md et résume les règles à respecter pour contribuer.",
            "Lance les tests, trouve pourquoi un test échoue, corrige-le et relance les tests.",
            "/tools",
            "Avec l'outil code__python_symbols, liste les fonctions de taskman/stats.py.",
            "Avec l'outil fs__read_text_file du serveur MCP fs, lis tests/test_stats.py et"
            " compte les fonctions dont le nom commence par test_.",
            LONG_REQUEST + " pour chacune.",
            "/history",
            "/fork",
            "Résume plutôt en une phrase ce que fait chaque module.",
            "/persona reviewer",
            "Relis la correction que tu as faite dans taskman/stats.py : est-ce qu'il manque un"
            " test ou un cas limite ?",
            "/persona dev",
            "Retiens comme convention du projet : les docstrings sont écrites en français.",
            "/exit",
        ],
    ),
    (
        ["--resume"],
        [
            "Où en est-on sur ce projet ? Qu'est-ce qui a été corrigé ?",
            "/memory",
            "/exit",
        ],
    ),
]
COMMANDS = [
    ["git", "-C", "demo-work", "diff", "--stat"],
    [str(BIN / "sensai"), "stats", *CONFIG],
]


class Recorder:
    """Writes the output as `script` does: a typescript and a timing file of `delay length`.

    Time-lapse: on a CPU-only machine a turn can wait minutes on the "thinking..." spinner, so
    only the first SPINNER_KEEP_S of each wait are recorded, and a silent pause lasts at most
    MAX_PAUSE_S. The streamed text keeps its real rhythm, and the recording replays at normal
    speed.
    """

    def __init__(self, stem: Path) -> None:
        self.typescript = stem.with_suffix(".typescript").open("wb")
        self.timing = stem.with_suffix(".timing").open("w")
        self.typescript.write(f"Script started on {datetime.now():%Y-%m-%d %H:%M:%S}\n".encode())
        self._last = time.monotonic()
        self._spinner_since: float | None = None
        self._skipped = False

    def write(self, data: bytes) -> None:
        now = time.monotonic()
        sys.stdout.buffer.write(data)
        sys.stdout.flush()
        if SPINNER.fullmatch(data.decode(errors="replace")):
            self._spinner_since = self._spinner_since or now
            if now - self._spinner_since > SPINNER_KEEP_S:
                self._skipped = True
                return
        else:
            self._spinner_since = None
        delay = min(now - self._last, 0.3 if self._skipped else MAX_PAUSE_S)
        self.timing.write(f"{delay:.6f} {len(data)}\n")
        self.typescript.write(data)
        self._last, self._skipped = now, False

    def close(self) -> None:
        self.typescript.close()
        self.timing.close()


def log(message: str) -> None:
    print(f"\n[rehearse {datetime.now():%H:%M:%S}] {message}", file=sys.stderr, flush=True)


class Chat:
    def __init__(self, args: list[str], recorder: Recorder) -> None:
        argv = [str(BIN / "sensai"), "chat", *CONFIG, *args]
        env = {**os.environ, "COLUMNS": "100", "LINES": "40", "TERM": "xterm-256color"}
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.chdir(ROOT)
            os.execvpe(argv[0], argv, env)
        self.recorder = recorder
        self.screen = ""

    def read(self, timeout: float) -> bool:
        """Read what is available within `timeout`; False once the process has closed the pty."""
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return True
        try:
            data = os.read(self.fd, 4096)
        except OSError:
            return False
        if not data:
            return False
        self.recorder.write(data)
        self.screen = (self.screen + ANSI.sub("", data.decode(errors="replace")))[-20000:]
        return True

    def wait_for_input(self, interrupt_after: float | None = None) -> str | None:
        """Answer approvals and questions until the prompt is back; return None at exit."""
        start, streaming_since = time.monotonic(), None
        while time.monotonic() - start < TURN_TIMEOUT_S:
            if not self.read(0.2):
                return None
            tail = self.screen[-200:]
            if tail.endswith(PROMPT):
                return self.screen
            if tail.endswith(APPROVAL):
                self.send("y")
            elif QUESTION.search(tail):
                self.send("1")
            if interrupt_after is not None and streaming_since is None and "sensai> " in tail:
                streaming_since = time.monotonic()
            if streaming_since and time.monotonic() - streaming_since > interrupt_after:
                log("Ctrl-C")
                os.write(self.fd, b"\x03")
                interrupt_after, streaming_since = None, None
        raise TimeoutError("no prompt after the turn timeout")

    def send(self, line: str) -> None:
        time.sleep(0.4)  # leave the audience time to see the prompt
        os.write(self.fd, line.encode() + b"\r")

    def close(self) -> None:
        os.waitpid(self.pid, 0)


def fork_number(screen: str) -> str:
    """The message just before LONG_REQUEST in the last /history listing."""
    for line in screen.splitlines():
        match = re.match(r"\s*(\d+)\s+user\s+(.*)", line)
        if match and match.group(2).startswith(LONG_REQUEST[:40]):
            return str(int(match.group(1)) - 1)
    raise RuntimeError("the long request is not in /history")


def play(recorder: Recorder) -> None:
    for number, (args, lines) in enumerate(ACTS, 1):
        log(f"act {number}: sensai chat {' '.join(args)}")
        chat = Chat(args, recorder)
        screen = chat.wait_for_input()
        for line in lines:
            if screen is None:
                raise RuntimeError(f"the chat exited before: {line}")
            if line == "/fork":
                line = f"/fork {fork_number(screen)}"
            log(f"step: {line[:70]}")
            started = time.monotonic()
            chat.screen = ""
            chat.send(line)
            interrupt = INTERRUPT_AFTER_S if line.startswith(LONG_REQUEST) else None
            screen = chat.wait_for_input(interrupt)
            log(f"done in {time.monotonic() - started:.0f} s")
        chat.close()
    for command in COMMANDS:
        log(f"$ {' '.join(command)}")
        result = subprocess.run(command, cwd=ROOT, capture_output=True)
        shown = [Path(command[0]).name, *command[1:]]
        recorder.write(f"$ {' '.join(shown)}\r\n".encode())
        recorder.write((result.stdout + result.stderr).replace(b"\n", b"\r\n"))


def main() -> int:
    RECORDINGS.mkdir(parents=True, exist_ok=True)
    stem = RECORDINGS / f"{datetime.now():%Y-%m-%d-%H%M}"
    recorder = Recorder(stem)
    started = time.monotonic()
    try:
        play(recorder)
    finally:
        recorder.close()
        log(f"total {time.monotonic() - started:.0f} s, recording: {stem}.typescript")
    return 0


if __name__ == "__main__":
    sys.exit(main())
