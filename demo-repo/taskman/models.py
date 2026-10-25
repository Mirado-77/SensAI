"""Modèle de données : une tâche."""

from dataclasses import asdict, dataclass


@dataclass
class Task:
    title: str
    priority: int = 1
    done: bool = False

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(
            title=data["title"],
            priority=data.get("priority", 1),
            done=data.get("done", False),
        )
