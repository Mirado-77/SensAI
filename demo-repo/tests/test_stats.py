from taskman.models import Task
from taskman.stats import completion_rate, count_done, highest_priority


def make_tasks():
    return [
        Task("écrire le rapport", priority=2, done=True),
        Task("relire la PR", priority=3),
        Task("mettre à jour le README", priority=1),
        Task("corriger le bug", priority=5, done=True),
    ]


def test_count_done():
    assert count_done(make_tasks()) == 2


def test_completion_rate():
    assert completion_rate(make_tasks()) == 0.5


def test_completion_rate_empty_list():
    assert completion_rate([]) == 0.0


def test_highest_priority():
    assert highest_priority(make_tasks()).title == "relire la PR"


def test_highest_priority_all_done():
    assert highest_priority([Task("fini", done=True)]) is None
