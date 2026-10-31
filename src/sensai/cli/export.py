"""`sensai export` and `sensai share` (X3): a session as JSON, Markdown or PDF, or as a link.

Both pass the session through the EV2 redactor first, so a secret read during the session does
not end up in a file or on a shared page.
"""

import json
import sqlite3
from pathlib import Path

from rich.text import Text

from sensai.cli.chat import _open_database
from sensai.cli.terminal import Terminal
from sensai.core.config import Config
from sensai.core.errors import ExportError
from sensai.export.pdf import to_pdf
from sensai.export.render import to_html, to_markdown
from sensai.export.session import session_export
from sensai.export.share import create_share, make_share_server, revoke_share, share_url
from sensai.memory.artifact import ArtifactStore
from sensai.memory.sqlite_store import SqliteSessionStore
from sensai.safety.secrets import SecretRedactor, read_env_file

FORMATS = ("json", "md", "pdf")


def run_export(
    config: Config, terminal: Terminal, session: str, format_: str, output: Path | None
) -> int:
    if format_ == "pdf" and output is None:
        return _error(terminal, "le PDF s'écrit dans un fichier : ajoute -o <fichier>.")
    try:
        export = _load(config, terminal, session)
        if export is None:
            return 2
        if format_ == "json":
            content: str | bytes = json.dumps(export, ensure_ascii=False, indent=2) + "\n"
        elif format_ == "md":
            content = to_markdown(export)
        else:
            content = to_pdf(to_html(export))
    except ExportError as exc:
        return _error(terminal, str(exc))
    if output is None:
        terminal.console.print(Text(str(content)), soft_wrap=True, end="")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        output.write_bytes(content)
    else:
        output.write_text(content, encoding="utf-8")
    terminal.errors.print(Text(f"Exporté : {output}", "dim"))
    return 0


def run_share(config: Config, terminal: Terminal, session: str) -> int:
    try:
        export = _load(config, terminal, session)
    except ExportError as exc:
        return _error(terminal, str(exc))
    if export is None:
        return 2
    token = create_share(to_html(export), _shares_dir(config))
    terminal.errors.print(Text("Lien sans authentification (sensai share --serve) :", "dim"))
    terminal.console.print(Text(share_url(token, config.export.host, config.export.port)))
    return 0


def run_revoke(config: Config, terminal: Terminal, token: str) -> int:
    if not revoke_share(token, _shares_dir(config)):
        return _error(terminal, f"aucun partage pour le token {token}.")
    terminal.console.print(Text("Lien révoqué.", "dim"))
    return 0


def run_serve(config: Config, terminal: Terminal) -> int:
    shares_dir = _shares_dir(config)
    shares_dir.mkdir(parents=True, exist_ok=True)
    try:
        server = make_share_server(shares_dir, config.export.host, config.export.port)
    except OSError as exc:
        return _error(terminal, f"serveur de partage impossible à lancer : {exc}")
    address = f"http://{config.export.host}:{config.export.port}/s/<token>"
    terminal.console.print(Text(f"Partages servis sur {address}, Ctrl-C pour arrêter.", "dim"))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        terminal.console.print()
    finally:
        server.server_close()
    return 0


def _load(config: Config, terminal: Terminal, session: str) -> dict | None:
    connection = _open_database(config, terminal)
    if connection is None:
        return None
    try:
        return _export(config, connection, session)
    finally:
        connection.close()


def _export(config: Config, connection: sqlite3.Connection, session: str) -> dict:
    store = SqliteSessionStore(connection)
    matches = store.list_sessions(2, session)
    if len(matches) != 1:
        problem = "ambiguë" if matches else "inconnue ou vide"
        raise ExportError(f"session {problem} : {session}")
    secrets = read_env_file(config.guards.env_file)
    redactor = SecretRedactor(secrets.values(), pii=config.guards.pii)
    return session_export(
        store, ArtifactStore(connection), matches[0]["id"], config.artifact.sections, redactor
    )


def _shares_dir(config: Config) -> Path:
    return config.export.shares_dir.expanduser()


def _error(terminal: Terminal, message: str) -> int:
    text = Text("Erreur : ", style="bold red")
    text.append(message)
    terminal.errors.print(text)
    return 2
