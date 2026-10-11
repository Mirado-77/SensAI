import pytest

from sensai.core.config import AllowedRoot
from sensai.core.errors import ConfigError
from sensai.tools.filesystem.guard import PathDeniedError, PathGuard

pytestmark = pytest.mark.feature("T4")


@pytest.fixture
def root(tmp_path):
    folder = tmp_path / "root"
    folder.mkdir()
    return folder


@pytest.fixture
def outside(tmp_path):
    folder = tmp_path / "outside"
    folder.mkdir()
    (folder / "secret.txt").write_text("secret", encoding="utf-8")
    return folder


@pytest.fixture
def guard(root):
    return PathGuard([AllowedRoot(path=root, mode="read")])


def test_guard_accepts_a_relative_path_inside_a_root(root, guard):
    (root / "a.txt").write_text("hello", encoding="utf-8")
    result = guard.resolve("a.txt", write=False)
    assert result == (root / "a.txt").resolve()


def test_guard_refuses_a_path_outside_the_roots(outside, guard):
    with pytest.raises(PathDeniedError, match="Chemin hors des racines"):
        guard.resolve("../outside/secret.txt", write=False)


def test_guard_refuses_symlink_outside_a_root(root, outside, guard):
    (root / "link.txt").symlink_to(outside / "secret.txt")
    with pytest.raises(PathDeniedError, match="Chemin hors des racines"):
        guard.resolve("link.txt", write=False)


def test_guard_accepts_symlink_inside_a_root(root, guard):
    (root / "real.txt").write_text("real", encoding="utf-8")
    (root / "alias.txt").symlink_to(root / "real.txt")
    assert guard.resolve("alias.txt", write=False) == (root / "real.txt").resolve()


def test_guard_accepts_an_absolute_path_inside_a_root(root, guard):
    (root / "a.txt").write_text("hello", encoding="utf-8")
    assert guard.resolve(str(root / "a.txt"), write=False) == (root / "a.txt").resolve()


def test_guard_refuses_a_write_on_a_read_only_root(root, guard):
    with pytest.raises(PathDeniedError, match="lecture seule"):
        guard.resolve("a.txt", write=True)


def test_guard_accepts_a_new_file_on_a_write_root(root):
    guard = PathGuard([AllowedRoot(path=root, mode="write")])
    assert guard.resolve("new.txt", write=True) == (root / "new.txt").resolve()


def test_guard_accepts_a_write_root_nested_in_a_read_root(root):
    (root / "sub").mkdir()
    roots = [AllowedRoot(path=root, mode="read"), AllowedRoot(path=root / "sub", mode="write")]
    guard = PathGuard(roots)
    assert guard.resolve("sub/new.txt", write=True) == (root / "sub" / "new.txt").resolve()
    with pytest.raises(PathDeniedError, match="lecture seule"):
        guard.resolve("other.txt", write=True)


def test_guard_refuses_an_empty_path(guard):
    with pytest.raises(PathDeniedError, match="vide"):
        guard.resolve("", write=False)
    with pytest.raises(PathDeniedError, match="vide"):
        guard.resolve("   ", write=False)


def test_guard_refuses_a_sibling_folder_sharing_the_root_name_prefix(tmp_path, guard):
    sibling = tmp_path / "root-evil"
    sibling.mkdir()
    (sibling / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(PathDeniedError, match="hors des racines"):
        guard.resolve("../root-evil/x.txt", write=False)


def test_guard_rejects_a_missing_root(tmp_path):
    with pytest.raises(ConfigError, match="introuvable"):
        PathGuard([AllowedRoot(path=tmp_path / "missing", mode="read")])
