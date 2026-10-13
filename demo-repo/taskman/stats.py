"""Statistiques sur une liste de tâches."""


def count_done(tasks):
    """Nombre de tâches terminées."""
    return sum(1 for task in tasks if task.done)


def completion_rate(tasks):
    """Proportion de tâches terminées, entre 0.0 et 1.0 (0.0 si la liste est vide)."""
    return count_done(tasks) / len(tasks)


def highest_priority(tasks):
    """Tâche non terminée la plus prioritaire, ou None s'il n'en reste aucune."""
    pending = [task for task in tasks if not task.done]
    if not pending:
        return None
    return max(pending, key=lambda task: task.priority)
