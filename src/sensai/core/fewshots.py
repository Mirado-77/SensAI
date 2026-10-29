"""Few-shot examples (A7): question/answer pairs sent before the conversation to show the model
the expected style.

Each example is a TOML file `configs/prompts/examples/<id>.toml` with `question` and `answer`.
At every turn the `k` examples sharing the most words with the user's message are selected; an
example that shares none is never sent. They are sent as user/assistant messages after the
system prompt and are never saved in the session.
"""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sensai.core.config import DEFAULT_CONFIG_PATH
from sensai.core.errors import ConfigError
from sensai.core.types import Message

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

EXAMPLES_DIR = DEFAULT_CONFIG_PATH.parent / "prompts" / "examples"
MIN_WORD = 4  # shorter words are mostly articles and pronouns, they would match everything


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)

    def messages(self) -> list[Message]:
        return [Message("user", self.question), Message("assistant", self.answer)]


@dataclass(frozen=True)
class FewShots:
    """The examples and how many to send per turn (`k` = 0 sends none)."""

    examples: tuple[Example, ...]
    k: int

    def select(self, text: str) -> list[Example]:
        """The `k` examples sharing the most words with `text`, best first; ties keep file order.
        shortcut: word overlap, not meaning; switch to the R1 embeddings if it picks badly."""
        words = keywords(text)
        scored = [(len(words & keywords(example.question)), example) for example in self.examples]
        ranked = sorted((pair for pair in scored if pair[0]), key=lambda pair: -pair[0])
        return [example for _, example in ranked[: self.k]]


def fewshots_from(k: int, directory: Path = EXAMPLES_DIR) -> FewShots | None:
    """The few-shots for `[prompts] examples` = `k`, or None when it is 0."""
    return FewShots(load_examples(directory), k) if k else None


def load_examples(directory: Path = EXAMPLES_DIR) -> tuple[Example, ...]:
    """Every `*.toml` example of `directory`, sorted by file name. Ids must be unique."""
    examples = tuple(_load(path) for path in sorted(directory.glob("*.toml")))
    ids = [example.id for example in examples]
    duplicates = sorted({example_id for example_id in ids if ids.count(example_id) > 1})
    if duplicates:
        raise ConfigError(f"Identifiants d'exemples en double : {', '.join(duplicates)}")
    return examples


def _load(path: Path) -> Example:
    try:
        return Example.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Exemple illisible ({path.name}) : {exc}") from exc
    except ValidationError as exc:
        details = "; ".join(error["msg"] for error in exc.errors())
        raise ConfigError(f"Exemple invalide ({path.name}) : {details}") from exc


def keywords(text: str) -> set[str]:
    """Lowercase words without accents, long enough to carry meaning (A7 few-shots, M3 recall)."""
    plain = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return {word for word in re.findall(r"[a-z0-9_]+", plain) if len(word) >= MIN_WORD}
