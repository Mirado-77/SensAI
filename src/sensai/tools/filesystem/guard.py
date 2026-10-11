from pathlib import Path

from sensai.core.config import AllowedRoot
from sensai.core.errors import ConfigError, SensaiError


class PathDeniedError(SensaiError):
    """A requested path is outside the allowed roots or not writable."""


class PathGuard:
    def __init__(self, allowed_roots: list[AllowedRoot]) -> None:
        self._roots: list[tuple[Path, str]] = []
        for root in allowed_roots:
            resolved = root.path.expanduser().resolve()
            if not resolved.is_dir():
                raise ConfigError(f"Racine autorisée introuvable : {resolved}")
            self._roots.append((resolved, root.mode))

    def _real_path(self, raw_path: str) -> Path:
        if not raw_path.strip():
            raise PathDeniedError("Chemin vide.")
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = self._roots[0][0] / candidate
        return candidate.resolve()

    def _modes_for(self, path: Path) -> list[str]:
        modes = []
        for root_path, mode in self._roots:
            if path.is_relative_to(root_path):
                modes.append(mode)
        return modes

    def resolve(self, raw_path: str, write: bool) -> Path:
        path = self._real_path(raw_path)
        modes = self._modes_for(path)
        if not modes:
            raise PathDeniedError(f"Chemin hors des racines autorisées : {raw_path}")
        if write and "write" not in modes:
            raise PathDeniedError(f"Racine en lecture seule : {raw_path}")
        return path
