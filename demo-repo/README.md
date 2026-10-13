# taskman

Un mini gestionnaire de tâches en Python, sans dépendance externe.

## Modules

- `taskman/models.py` : la dataclass `Task` (titre, priorité, état).
- `taskman/storage.py` : lecture et écriture des tâches dans un fichier JSON.
- `taskman/stats.py` : statistiques sur une liste de tâches.

## Lancer les tests

```bash
python -m pytest
```

Seul pytest est requis. Voir `CONTRIBUTING.md` avant de proposer une modification.
