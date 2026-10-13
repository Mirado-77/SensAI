"""The user profile (M1): custom instructions and preferences sent to the model every turn.

One profile per user, stored in the `profiles` table and edited with /profile in the chat.
"""

from typing import Any

from rich.text import Text

from sensai.cli.terminal import Terminal
from sensai.memory.sqlite_store import SqliteSessionStore

PROFILE_ID = "default"
PROFILE_COMMAND = "/profile"
PROFILE_HELP = """\
/profile                       show the profile
/profile instructions <text>   set the custom instructions
/profile set <key> <value>     set a preference
/profile unset <key>           remove a preference
/profile clear                 erase the profile"""


def load_profile(store: SqliteSessionStore) -> dict[str, Any]:
    try:
        return store.profile(PROFILE_ID)
    except KeyError:
        return {}


def profile_prompt(profile: dict[str, Any]) -> str | None:
    """System prompt built from the profile, or None for an empty profile."""
    instructions = profile.get("instructions", "")
    preferences = profile.get("preferences", {})
    if not instructions and not preferences:
        return None
    lines = ["User profile. Apply it to every answer; the user will not repeat it."]
    if instructions:
        lines.append(f"Instructions: {instructions}")
    lines.extend(f"- {key}: {value}" for key, value in preferences.items())
    return "\n".join(lines)


def run_profile_command(store: SqliteSessionStore, terminal: Terminal, text: str) -> str | None:
    """Apply a /profile command, print the outcome and return the new system prompt."""
    action, _, rest = text.removeprefix(PROFILE_COMMAND).strip().partition(" ")
    rest = rest.strip()
    profile = load_profile(store)
    preferences = profile.setdefault("preferences", {})
    key, _, value = rest.partition(" ")
    if action == "instructions" and rest:
        profile["instructions"] = rest
    elif action == "set" and key and value.strip():
        preferences[key] = value.strip()
    elif action == "unset" and key in preferences:
        del preferences[key]
    elif action == "clear":
        profile = {}
    elif action:
        terminal.errors.print(Text(f"Commande /profile invalide.\n{PROFILE_HELP}", style="yellow"))
        return profile_prompt(profile)
    if action:
        store.add_profile(PROFILE_ID, profile)
    prompt = profile_prompt(profile)
    terminal.console.print(Text(prompt or f"Empty profile.\n{PROFILE_HELP}", "dim"))
    return prompt
