import json
import sys
import threading
import urllib.error
import urllib.request

import pytest

from sensai.cli.main import main
from sensai.core.config import load_config
from sensai.core.errors import ExportError
from sensai.core.types import Message
from sensai.db.connection import connect
from sensai.export.pdf import to_pdf
from sensai.export.render import to_html, to_markdown
from sensai.export.session import session_export
from sensai.export.share import create_share, make_share_server, revoke_share, share_url
from sensai.memory.artifact import ArtifactStore
from sensai.memory.sqlite_store import SqliteSessionStore
from sensai.safety.secrets import SecretRedactor

pytestmark = pytest.mark.feature("X3")

SECTIONS = ["Objectif", "Décisions", "Fait"]
SECRET = "sk-demo-1234567890abcdef"


@pytest.fixture
def db(tmp_path):
    return connect(tmp_path / "sensai.db")


@pytest.fixture
def store(db):
    return SqliteSessionStore(db)


@pytest.fixture
def artifacts(db):
    return ArtifactStore(db)


def add_messages(store, session_id, *messages):
    parent = None
    for message in messages:
        parent = store.append(session_id, message, parent)


@pytest.fixture
def session(store, artifacts):
    session_id = store.create_session()
    add_messages(
        store,
        session_id,
        Message("user", "Pourquoi test_stats échoue ?"),
        Message("assistant", "Division par zéro dans stats.py."),
        Message("tool", "sortie brute de pytest"),
        Message("assistant", "Je corr", status="interrupted"),
    )
    artifacts.apply(session_id, None, "Fait", "append", "- bug corrigé")
    artifacts.apply(session_id, None, "Objectif", "replace", "Faire passer les tests.")
    return session_id


@pytest.fixture
def export(store, artifacts, session):
    return session_export(store, artifacts, session, SECTIONS)


def test_the_export_holds_the_conversation_and_the_document(export, session):
    assert export["session_id"] == session
    assert isinstance(export["created_at"], float)
    assert export["messages"] == [
        {"role": "user", "content": "Pourquoi test_stats échoue ?", "status": "complete"},
        {"role": "assistant", "content": "Division par zéro dans stats.py.", "status": "complete"},
        {"role": "assistant", "content": "Je corr", "status": "interrupted"},
    ]
    assert export["document"] == [
        {"section": "Objectif", "content": "Faire passer les tests."},
        {"section": "Fait", "content": "- bug corrigé"},
    ]
    assert json.loads(json.dumps(export)) == export


def test_the_export_redacts_the_secrets(store, artifacts):
    session_id = store.create_session()
    add_messages(store, session_id, Message("user", f"Ma clé est {SECRET}"))
    artifacts.apply(session_id, None, "Fait", "append", f"- clé {SECRET} ajoutée")

    export = session_export(store, artifacts, session_id, SECTIONS, SecretRedactor([SECRET]))

    assert SECRET not in json.dumps(export)


def test_an_unknown_or_empty_session_cannot_be_exported(store, artifacts):
    with pytest.raises(ExportError):
        session_export(store, artifacts, "inconnue", SECTIONS)
    with pytest.raises(ExportError):
        session_export(store, artifacts, store.create_session(), SECTIONS)


def test_markdown_shows_the_conversation_then_the_document(export, session):
    markdown = to_markdown(export)

    assert markdown.startswith(f"# SENSAI session {session}")
    conversation, document = markdown.index("## Conversation"), markdown.index("## State document")
    assert conversation < document
    assert markdown.index("### You") < markdown.index("### SENSAI")
    assert "Division par zéro dans stats.py." in markdown
    assert "(interrupted)" in markdown
    assert "### Objectif\nFaire passer les tests." in markdown
    assert "### Fait\n- bug corrigé" in markdown
    assert "sortie brute de pytest" not in markdown


def test_markdown_has_no_document_part_when_it_is_empty(export):
    assert "## State document" not in to_markdown({**export, "document": []})


def test_html_is_a_full_page_with_escaped_content(export):
    message = {"role": "user", "content": "<script>alert(1)</script>", "status": "complete"}

    page = to_html({**export, "messages": [message]})

    assert page.lower().startswith("<!doctype html>")
    assert '<meta charset="utf-8">' in page
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "Faire passer les tests." in page


def test_pdf_starts_with_the_pdf_signature(export):
    pytest.importorskip("weasyprint")

    assert to_pdf(to_html(export)).startswith(b"%PDF")


def test_pdf_without_weasyprint_is_a_clear_error(monkeypatch, export):
    monkeypatch.setitem(sys.modules, "weasyprint", None)

    with pytest.raises(ExportError, match="WeasyPrint"):
        to_pdf(to_html(export))


def test_a_share_is_a_file_named_by_an_unguessable_token(tmp_path):
    shares = tmp_path / "shares"

    first, second = create_share("<p>a</p>", shares), create_share("<p>b</p>", shares)

    assert first != second and len(first) >= 20
    assert (shares / f"{first}.html").read_text(encoding="utf-8") == "<p>a</p>"
    assert share_url(first, "127.0.0.1", 8765) == f"http://127.0.0.1:8765/s/{first}"


def test_revoking_deletes_only_a_valid_existing_share(tmp_path):
    shares = tmp_path / "shares"
    token = create_share("<p>a</p>", shares)
    outside = tmp_path / "keep.html"
    outside.write_text("x", encoding="utf-8")

    assert revoke_share(token, shares) is True
    assert not (shares / f"{token}.html").exists()
    assert revoke_share(token, shares) is False
    assert revoke_share("../keep", shares) is False
    assert outside.exists()


@pytest.fixture
def server(tmp_path):
    shares = tmp_path / "shares"
    shares.mkdir()
    httpd = make_share_server(shares, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield shares, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def status_of(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def test_the_server_serves_a_share_without_auth(server):
    shares, base = server
    token = create_share("<p>Partagé</p>", shares)

    with urllib.request.urlopen(f"{base}/s/{token}", timeout=5) as response:
        assert response.status == 200
        assert response.headers["Content-Type"].startswith("text/html")
        assert response.read().decode("utf-8") == "<p>Partagé</p>"


def test_the_server_answers_404_for_anything_else(server, tmp_path):
    shares, base = server
    (tmp_path / "secret.html").write_text("secret", encoding="utf-8")

    assert status_of(f"{base}/s/inconnu") == 404
    assert status_of(f"{base}/s/..%2Fsecret") == 404
    assert status_of(f"{base}/") == 404
    assert status_of(f"{base}/sensai.db") == 404


def test_the_default_config_keeps_the_share_server_local():
    config = load_config()

    assert config.export.host == "127.0.0.1"
    assert config.export.port > 0


def write_config(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(
        f'[storage]\ndb_path = "{tmp_path / "sensai.db"}"\n'
        f'[export]\nshares_dir = "{tmp_path / "shares"}"\nhost = "127.0.0.1"\nport = 8765\n',
        encoding="utf-8",
    )
    return path


def test_export_command_prints_json_and_markdown(tmp_path, session, capsys):
    path = write_config(tmp_path)

    assert main(["export", session, "--format", "json", "--config", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["session_id"] == session
    assert main(["export", session, "--format", "md", "--config", str(path)]) == 0
    assert "## Conversation" in capsys.readouterr().out


def test_export_command_writes_a_file(tmp_path, session):
    path, target = write_config(tmp_path), tmp_path / "out" / "session.md"

    assert (
        main(["export", session, "--format", "md", "-o", str(target), "--config", str(path)]) == 0
    )
    assert "## State document" in target.read_text(encoding="utf-8")


def test_export_command_errors(tmp_path, session, capsys):
    path = write_config(tmp_path)

    assert main(["export", "inconnue", "--format", "md", "--config", str(path)]) == 2
    assert "Erreur" in capsys.readouterr().err
    assert main(["export", session, "--format", "pdf", "--config", str(path)]) == 2
    assert "-o" in capsys.readouterr().err


def test_share_command_prints_a_link_and_revokes_it(tmp_path, session, capsys):
    path = write_config(tmp_path)

    assert main(["share", session, "--config", str(path)]) == 0
    url = capsys.readouterr().out.strip().splitlines()[-1]
    token = url.rsplit("/", 1)[1]
    assert url == f"http://127.0.0.1:8765/s/{token}"
    assert (tmp_path / "shares" / f"{token}.html").exists()

    assert main(["share", "--revoke", token, "--config", str(path)]) == 0
    assert not (tmp_path / "shares" / f"{token}.html").exists()
    assert main(["share", "--revoke", token, "--config", str(path)]) == 2
