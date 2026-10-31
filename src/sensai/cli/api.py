"""`sensai start:api` serves the API (X1) until Ctrl-C, `sensai start:web` opens the web UI."""

import errno
import socket
import webbrowser
from collections.abc import Callable

import httpx
import uvicorn
from rich.text import Text

from sensai.api.app import API_NAME, create_app
from sensai.cli.terminal import Terminal
from sensai.core.config import Config

Opener = Callable[[str], object]


def run_api(config: Config, terminal: Terminal) -> int:
    return _serve(config, terminal)


def run_web(config: Config, terminal: Terminal, opener: Opener = webbrowser.open) -> int:
    """Open the web UI; start the API first, until Ctrl-C, unless it is already running."""
    url = f"http://{config.api.host}:{config.api.port}/"
    web_dir = config.api.web_dir.expanduser()
    if not web_dir.is_dir():
        message = f"Erreur : interface web non construite ({web_dir} absent), lance `make web`"
        terminal.errors.print(Text(message, style="bold red"))
        return 2
    if api_running(config):
        terminal.console.print(Text(f"SENSAI API already running, opening {url}", "dim"))
        opener(url)
        return 0
    return _serve(config, terminal, lambda: opener(url))


def api_running(config: Config) -> bool:
    url = f"http://{config.api.host}:{config.api.port}/api/health"
    try:
        response = httpx.get(url, timeout=1)
        return response.json().get("name") == API_NAME
    except (httpx.HTTPError, ValueError, AttributeError):
        return False


def _serve(
    config: Config, terminal: Terminal, on_listen: Callable[[], object] | None = None
) -> int:
    """Bind first so a busy port is reported cleanly; requests wait in the backlog until uvicorn
    accepts them, so `on_listen` may open the browser before the server runs."""
    host, port = config.api.host, config.api.port
    try:
        listener = socket.create_server((host, port))
    except OSError as exc:
        reason = "port déjà utilisé" if exc.errno == errno.EADDRINUSE else str(exc)
        message = f"Erreur : impossible d'écouter sur {host}:{port} : {reason}"
        terminal.errors.print(Text(message, style="bold red"))
        return 2
    terminal.console.print(Text(_banner(config), "dim"))
    server = uvicorn.Server(uvicorn.Config(create_app(config), log_level="warning"))
    if on_listen is not None:
        on_listen()
    try:
        server.run(sockets=[listener])
    except KeyboardInterrupt:
        terminal.console.print()
    finally:
        listener.close()
    return 0


def _banner(config: Config) -> str:
    address = f"{config.api.host}:{config.api.port}"
    web_dir = config.api.web_dir.expanduser()
    web = f"http://{address}/" if web_dir.is_dir() else f"not built yet ({web_dir} is missing)"
    return (
        f"SENSAI API on http://{address}/api | model: {config.model.name}\n"
        f"  Web UI:         {web}\n"
        f"  Session stream: ws://{address}/ws/sessions/<id>\n"
        f"  Routes:         http://{address}/docs\n"
        "  Ctrl-C to stop."
    )
