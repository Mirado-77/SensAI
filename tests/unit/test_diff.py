import pytest

from sensai.safety.diff import unified_diff

pytestmark = pytest.mark.feature("A3")

NO_CHANGE = "(aucun changement)"
EOL_ONLY = "(seules les fins de ligne changent)"
MAX_LINES = 50


def lines_of(text):
    return text.split("\n")


def test_new_file_is_shown_as_a_creation():
    result = unified_diff("a.py", None, "x\ny\n", MAX_LINES)
    assert lines_of(result) == [
        "--- /dev/null",
        "+++ b/a.py",
        "@@ -0,0 +1,2 @@",
        "+x",
        "+y",
    ]


def test_modified_file_shows_removed_added_and_context_lines():
    result = unified_diff("a.py", "one\ntwo\nthree\n", "one\nTWO\nthree\n", MAX_LINES)
    assert lines_of(result)[:2] == ["--- a/a.py", "+++ b/a.py"]
    assert " one" in lines_of(result)
    assert "-two" in lines_of(result)
    assert "+TWO" in lines_of(result)
    assert " three" in lines_of(result)


def test_identical_content_reports_no_change():
    assert unified_diff("a.py", "same\n", "same\n", MAX_LINES) == NO_CHANGE


def test_identical_empty_content_reports_no_change():
    assert unified_diff("a.py", "", "", MAX_LINES) == NO_CHANGE


def test_final_newline_change_is_reported_as_an_end_of_line_change():
    assert unified_diff("a.py", "a\nb", "a\nb\n", MAX_LINES) == EOL_ONLY


def test_crlf_to_lf_change_is_reported_as_an_end_of_line_change():
    assert unified_diff("a.py", "a\r\nb\r\n", "a\nb\n", MAX_LINES) == EOL_ONLY


def test_existing_empty_file_is_not_shown_as_a_creation():
    result = unified_diff("a.py", "", "x\n", MAX_LINES)
    assert lines_of(result)[0] == "--- a/a.py"


def test_creating_an_empty_file_has_its_own_message():
    result = unified_diff("a.py", None, "", MAX_LINES)
    assert result not in (NO_CHANGE, EOL_ONLY)
    assert "vide" in result


def test_emptying_a_file_shows_the_removed_lines():
    result = unified_diff("a.py", "x\ny\n", "", MAX_LINES)
    assert "-x" in lines_of(result)
    assert "-y" in lines_of(result)


def test_long_diff_is_truncated_with_a_count_of_hidden_lines():
    content = "1\n2\n3\n4\n5\n"
    full = unified_diff("a.py", None, content, MAX_LINES)
    assert len(lines_of(full)) == 8

    result = lines_of(unified_diff("a.py", None, content, 4))
    assert result[:4] == lines_of(full)[:4]
    assert len(result) == 5
    assert result[-1].endswith("4 lignes de plus")


def test_diff_with_exactly_max_lines_is_not_truncated():
    result = unified_diff("a.py", None, "1\n2\n3\n4\n5\n", 8)
    assert len(lines_of(result)) == 8
    assert "lignes de plus" not in result


def test_name_appears_in_both_headers():
    result = lines_of(unified_diff("src/mod.py", "a\n", "b\n", MAX_LINES))
    assert result[0] == "--- a/src/mod.py"
    assert result[1] == "+++ b/src/mod.py"
