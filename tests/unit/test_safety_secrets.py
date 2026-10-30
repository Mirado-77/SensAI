import pytest

from sensai.safety.secrets import REDACTED, SecretRedactor, read_env_file


def test_read_env_file_keeps_values_and_skips_comments_and_empty_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nA=1\nB='quoted'\nC=\n\n=orphan\nD=deux\n",
        encoding="utf-8",
    )
    assert read_env_file(env) == {"A": "1", "B": "quoted", "D": "deux"}


def test_read_env_file_returns_empty_when_missing(tmp_path):
    assert read_env_file(tmp_path / "missing.env") == {}


def test_known_values_are_redacted():
    redactor = SecretRedactor(["hunter2-secret"])
    text, count = redactor.redact("the password is hunter2-secret, again hunter2-secret")
    assert text == f"the password is {REDACTED}, again {REDACTED}"
    assert count == 2


def test_short_values_are_ignored():
    text, count = SecretRedactor(["1234"]).redact("port 1234")
    assert (text, count) == ("port 1234", 0)


def test_longest_value_wins_when_one_contains_another():
    redactor = SecretRedactor(["abcdefgh", "abcdefghijkl"])
    text, _ = redactor.redact("value abcdefghijkl")
    assert text == f"value {REDACTED}"


def test_common_key_formats_are_redacted_without_any_value():
    redactor = SecretRedactor([])
    key = "sk-" + "a" * 24
    aws = "AKIA" + "B" * 16
    text, count = redactor.redact(f"{key} and {aws}")
    assert text == f"{REDACTED} and {REDACTED}"
    assert count == 2


@pytest.mark.parametrize(
    "text, expected",
    [
        ("mail jane.doe+dev@mail.example.fr now", "mail [EMAIL] now"),
        ("call +33 6 12 34 56 78", "call [PHONE]"),
        ("call +1-415-555-0132", "call [PHONE]"),
        ("call 01.23.45.67.89", "call [PHONE]"),
    ],
)
def test_pii_is_masked_when_enabled(text, expected):
    assert SecretRedactor([], pii=True).redact(text) == (expected, 1)


@pytest.mark.parametrize(
    "text",
    [
        "version 3.12.4, port 8080, id 1234567890",
        "2026-10-30 12:34:56",
        "result = a @ b.T and @property",
        "x = 0612345678901",
    ],
)
def test_code_and_numbers_are_not_taken_for_pii(text):
    assert SecretRedactor([], pii=True).redact(text) == (text, 0)


def test_pii_is_kept_when_disabled():
    text = "jane@example.com 06 12 34 56 78"
    assert SecretRedactor([]).redact(text) == (text, 0)
