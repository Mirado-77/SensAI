from taskman.models import Task
from taskman.storage import load_tasks, save_tasks


def test_load_missing_file_returns_empty_list(tmp_path):
    assert load_tasks(tmp_path / "absent.json") == []


def test_save_then_load_roundtrip(tmp_path):
    path = tmp_path / "tasks.json"
    tasks = [Task("acheter du café", priority=2), Task("déployer", done=True)]

    save_tasks(path, tasks)

    assert load_tasks(path) == tasks


def test_load_applies_defaults(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text('[{"title": "minimal"}]', encoding="utf-8")

    assert load_tasks(path) == [Task("minimal", priority=1, done=False)]
