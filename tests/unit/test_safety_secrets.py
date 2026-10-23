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
