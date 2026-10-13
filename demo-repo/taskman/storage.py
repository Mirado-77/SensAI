"""Lecture et écriture des tâches dans un fichier JSON."""

import json
import os

from taskman.models import Task


def load_tasks(path):
    """Retourne la liste des tâches du fichier, ou une liste vide s'il n'existe pas."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [Task.from_dict(item) for item in json.load(f)]


def save_tasks(path, tasks):
    """Écrit la liste des tâches dans le fichier."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([task.to_dict() for task in tasks], f, indent=2, ensure_ascii=False)
