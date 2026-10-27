"""`sensai index [PATH ...]`: embed the codebase so the chat can search it (R1).

Exit codes: 0 success, 1 Ollama failure, 2 unusable config, path or database.
"""

import sqlite3

from rich.text import Text

from sensai.cli.terminal import Terminal
from sensai.core.config import Config
from sensai.core.errors import ModelError, SensaiError
from sensai.db.connection import connect
from sensai.rag.index import CodeIndex, Embedder, IndexReport
from sensai.tools import code_index
from sensai.tools.filesystem.guard import PathGuard


async def run_index(
    config: Config, terminal: Terminal, paths: list[str], embedder: Embedder | None = None
) -> int:
    if not config.files.allowed_roots:
        _error(terminal, "aucune racine autorisée : renseigner [files] allowed_roots.")
        return 2
    try:
        guard = PathGuard(config.files.allowed_roots)
        roots = [guard.resolve(path, write=False) for path in paths] or [
            root.path.expanduser().resolve() for root in config.files.allowed_roots
        ]
        connection = connect(config.storage.db_path)
    except (OSError, sqlite3.Error, SensaiError) as exc:
        _error(terminal, str(exc))
        return 2
    try:
        index = code_index(config, connection)
        if embedder is not None:
            index = CodeIndex(
                connection,
                embedder,
                config.rag.chunk_lines,
                config.rag.overlap,
                document_prefix=config.rag.document_prefix,
                query_prefix=config.rag.query_prefix,
            )
        report = IndexReport()
        for root in roots:
            if not root.is_dir():
                _error(terminal, f"dossier introuvable : {root}")
                return 2
            terminal.console.print(Text(f"Indexing {root}", "dim"))
            await index.index_root(root, config.files.max_read_bytes, report)
    except ModelError as exc:
        _error(terminal, str(exc))
        return 1
    finally:
        connection.close()
    terminal.console.print(
        f"{report.indexed} files indexed ({report.chunks} chunks), "
        f"{report.unchanged} unchanged, {report.removed} removed"
    )
    return 0


def _error(terminal: Terminal, message: str) -> None:
    terminal.errors.print(Text(f"Erreur : {message}", style="bold red"))
