"""Detection of prompt-injection phrases in text coming from the user or from a tool."""

import re

_INVISIBLE = re.compile("[​-‏⁠﻿]")
_SPACES = re.compile(r"\s+")

_PREVIOUS = r"(previous|prior|above|earlier)"
_PHRASES = [
    rf"(ignore|disregard|forget) (all |the |any |your )?{_PREVIOUS}"
    r" (instructions?|prompts?|messages|rules)",
    r"ignore (toutes? )?les (instructions|consignes|r[èe]gles)"
    r" (pr[ée]c[ée]dentes?|ci-dessus|du dessus)",
    r"oublie (toutes? )?les (instructions|consignes|r[èe]gles)",
    r"new instructions? (from|by) (the )?(system|developer|administrator)",
    r"nouvelles? instructions? (du|de) (syst[èe]me|d[ée]veloppeur|administrateur)",
    r"</?(system|assistant)>|<\|im_(start|end)\|>|\[/?inst\]",
    r"(do not|don[’']t) (tell|inform|mention to) the user",
    r"ne (le |la )?(dis|dites|mentionne[sz]?) pas [àa] l[’']utilisateur",
    r"(send|envoie|transmets|upload)\b.{0,60}(\.env\b|api[ _-]?key|mot de passe|password)",
]

_PATTERNS = [re.compile(phrase, re.IGNORECASE) for phrase in _PHRASES]


def is_injection(text: str) -> bool:
    """Tell whether `text` contains a phrase that tries to override the instructions.

    Invisible characters are removed and whitespace collapsed first, so that a phrase split
    by a zero-width character or a line break is still found.
    """
    normalized = _SPACES.sub(" ", _INVISIBLE.sub("", text))
    return any(pattern.search(normalized) for pattern in _PATTERNS)
