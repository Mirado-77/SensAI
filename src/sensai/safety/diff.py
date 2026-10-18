import difflib


def unified_diff(name: str, old: str | None, new: str, max_lines: int) -> str:
    if old == new:
        return "(aucun changement)"
    if old is None:
        fromfile = "/dev/null"
    else:
        fromfile = f"a/{name}"
    tofile = f"b/{name}"
    lines: list = list(
        difflib.unified_diff(
            (old or "").splitlines(), new.splitlines(), fromfile, tofile, lineterm=""
        )
    )

    if old is None and new == "":
        return "(création d'un fichier vide)"
    if not lines:
        return "(seules les fins de ligne changent)"

    if len(lines) > max_lines:
        hidden = len(lines) - max_lines
        lines = lines[:max_lines]
        lines.append(f"... {hidden} lignes de plus")

    return "\n".join(lines)
