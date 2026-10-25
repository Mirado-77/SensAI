import os
import stat

import pytest

from sensai.tools.filesystem import atomic
from sensai.tools.filesystem.atomic import atomic_write

pytestmark = pytest.mark.feature("T4")


def test_atomic_write_creates_a_file(tmp_path):
    target = tmp_path / "a.txt"
    atomic_write(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello"


def test_atomic_write_replaces_the_whole_content(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("old content that is longer", encoding="utf-8")
    atomic_write(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


def test_atomic_write_creates_the_missing_parent_folders(tmp_path):
    target = tmp_path / "one" / "two" / "a.txt"
    atomic_write(target, "deep")
    assert target.read_text(encoding="utf-8") == "deep"


def test_atomic_write_writes_utf8(tmp_path):
    target = tmp_path / "a.txt"
    atomic_write(target, "café ✓")
    assert target.read_bytes() == "café ✓".encode()


def test_atomic_write_keeps_the_permissions_of_an_existing_file(tmp_path):
    target = tmp_path / "run.sh"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o755)
    atomic_write(target, "#!/bin/sh\necho hi\n")
    assert stat.S_IMODE(target.stat().st_mode) == 0o755


def test_atomic_write_leaves_no_temporary_file(tmp_path):
    atomic_write(tmp_path / "a.txt", "hello")
    assert [entry.name for entry in tmp_path.iterdir()] == ["a.txt"]


def test_a_failed_write_keeps_the_old_content_and_leaves_no_temporary_file(tmp_path, monkeypatch):
    target = tmp_path / "a.txt"
    target.write_text("old", encoding="utf-8")

    def failing_replace(source, destination):
        raise OSError("disk error")

    monkeypatch.setattr(atomic.os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk error"):
        atomic_write(target, "new")
    assert target.read_text(encoding="utf-8") == "old"
    assert [entry.name for entry in tmp_path.iterdir()] == ["a.txt"]


def test_a_write_in_a_folder_without_permission_raises_and_leaves_nothing(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root can write anywhere")
    folder = tmp_path / "locked"
    folder.mkdir()
    folder.chmod(0o500)
    try:
        with pytest.raises(OSError, match="Permission"):
            atomic_write(folder / "a.txt", "hello")
    finally:
        folder.chmod(0o700)
    assert list(folder.iterdir()) == []
