import io

import pytest
from rich.console import Console

from sensai.cli.channel import CliUserChannel, _styled_preview
from sensai.cli.terminal import Terminal
from sensai.core.types import ApprovalRequest

pytestmark = pytest.mark.feature("A3")

DIFF = "\n".join(
    [
        "--- a/a.py",
        "+++ b/a.py",
        "@@ -1,2 +1,2 @@",
        " keep",
        "-old = [1]",
        "+new = [2]",
    ]
)
GENERIC = 'write_file({"path": "a.py"})'
RED = "\x1b[31m"
GREEN = "\x1b[32m"
CYAN = "\x1b[36m"
BOLD = "\x1b[1m"


def styles_by_line(text):
    return {text.plain[span.start : span.end]: str(span.style) for span in text.spans}


def make_terminal(color, answer="n"):
    out, err = io.StringIO(), io.StringIO()
    system = "standard" if color else None
    console = Console(
        file=out, width=100, color_system=system, force_terminal=color, highlight=False
    )
    errors = Console(file=err, width=100, color_system=None, force_terminal=False)
    return Terminal(console, errors, lambda prompt: answer), out


def request(preview):
    return ApprovalRequest("r1", "write_file", "write", preview, {})


def test_styled_preview_keeps_the_text_of_a_diff_unchanged():
    assert _styled_preview(DIFF).plain == DIFF


def test_styled_preview_colors_each_kind_of_diff_line():
    styles = styles_by_line(_styled_preview(DIFF))
    assert styles["--- a/a.py"] == "bold"
    assert styles["+++ b/a.py"] == "bold"
    assert styles["@@ -1,2 +1,2 @@"] == "cyan"
    assert styles["-old = [1]"] == "red"
    assert styles["+new = [2]"] == "green"


def test_styled_preview_leaves_the_context_lines_unstyled():
    styles = styles_by_line(_styled_preview(DIFF))
    assert " keep" not in styles
    assert len(styles) == 5


def test_styled_preview_does_not_color_the_headers_as_changes():
    text = _styled_preview("--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x = 1")
    styles = styles_by_line(text)
    assert styles["--- /dev/null"] == "bold"
    assert styles["+++ b/new.py"] == "bold"
    assert styles["+x = 1"] == "green"


def test_styled_preview_does_not_interpret_square_brackets_as_markup():
    preview = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = [1]\n+print('[bold]hi[/bold]')"
    assert _styled_preview(preview).plain == preview


def test_styled_preview_adds_no_trailing_newline():
    assert not _styled_preview(DIFF).plain.endswith("\n")
    assert not _styled_preview(DIFF + "\n").plain.endswith("\n\n")


def test_styled_preview_leaves_the_truncation_line_unstyled():
    preview = "--- /dev/null\n+++ b/a.py\n@@ -0,0 +1,9 @@\n+1\n... 6 lignes de plus"
    text = _styled_preview(preview)
    assert text.plain == preview
    assert "... 6 lignes de plus" not in styles_by_line(text)


@pytest.mark.parametrize(
    "preview",
    [GENERIC, "pytest -q", "-rf /tmp/x", "+not a diff", "(aucun changement)", ""],
)
def test_styled_preview_leaves_a_non_diff_preview_unstyled(preview):
    text = _styled_preview(preview)
    assert text.plain == preview
    assert text.spans == []


async def test_confirm_prints_a_diff_below_the_header_with_colors():
    terminal, out = make_terminal(color=True)
    await CliUserChannel(terminal).confirm(request(DIFF))

    printed = out.getvalue()
    assert f"{RED}-old = [1]" in printed
    assert f"{GREEN}+new = [2]" in printed
    assert f"{CYAN}@@ -1,2 +1,2 @@" in printed
    assert f"{BOLD}--- a/a.py" in printed


async def test_confirm_puts_a_diff_on_its_own_lines():
    terminal, out = make_terminal(color=False)
    await CliUserChannel(terminal).confirm(request(DIFF))

    lines = out.getvalue().split("\n")
    assert lines[0].strip() == "Approval required (write):"
    assert lines[1:7] == DIFF.split("\n")


async def test_confirm_prints_no_color_codes_when_the_terminal_has_no_colors():
    terminal, out = make_terminal(color=False)
    await CliUserChannel(terminal).confirm(request(DIFF))

    assert "\x1b" not in out.getvalue()


async def test_confirm_keeps_square_brackets_of_the_code_in_the_output():
    terminal, out = make_terminal(color=False)
    preview = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = [1]\n+x = [2, 3]"
    await CliUserChannel(terminal).confirm(request(preview))

    assert "-x = [1]" in out.getvalue()
    assert "+x = [2, 3]" in out.getvalue()


async def test_confirm_keeps_a_generic_preview_on_the_header_line_without_colors():
    terminal, out = make_terminal(color=True)
    await CliUserChannel(terminal).confirm(request(GENERIC))

    printed = out.getvalue()
    assert f"Approval required (write): \x1b[0m{GENERIC}" in printed
    assert RED not in printed
    assert GREEN not in printed
    assert CYAN not in printed


@pytest.mark.parametrize(("answer", "expected"), [("y", True), ("n", False), ("", False)])
async def test_confirm_returns_the_answer_for_a_diff(answer, expected):
    terminal, _ = make_terminal(color=False, answer=answer)
    assert await CliUserChannel(terminal).confirm(request(DIFF)) is expected
