"""`sensai start:api`: serve the API (X1) until Ctrl-C."""

import errno
import socket

import uvicorn
from rich.text import Text

from sensai.api.app import create_app
from sensai.cli.terminal import Terminal
from sensai.core.config import Config


def run_api(config: Config, terminal: Terminal) -> int:
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
