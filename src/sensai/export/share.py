"""No-auth sharing (X3): a page saved under a random token, served read-only over HTTP.

The link carries about 128 random bits, so it cannot be guessed; anyone who has it can read the
page. Revoking a share deletes its file. Only `/s/<token>` is served, nothing else in the folder.
"""

import re
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SHARE_PATH = "/s/"
_TOKEN = re.compile(r"[A-Za-z0-9_-]+")


def create_share(html: str, shares_dir: Path) -> str:
    shares_dir.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(16)
    _page(shares_dir, token).write_text(html, encoding="utf-8")
    return token


def revoke_share(token: str, shares_dir: Path) -> bool:
    if not _TOKEN.fullmatch(token):
        return False
    page = _page(shares_dir, token)
    if not page.is_file():
        return False
    page.unlink()
    return True


def share_url(token: str, host: str, port: int) -> str:
    return f"http://{host}:{port}{SHARE_PATH}{token}"


def make_share_server(shares_dir: Path, host: str, port: int) -> ThreadingHTTPServer:
    class ShareHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            token = self.path.removeprefix(SHARE_PATH) if self.path.startswith(SHARE_PATH) else ""
            page = _page(shares_dir, token) if _TOKEN.fullmatch(token) else None
            if page is None or not page.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content = page.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: object) -> None:
            pass

    return ThreadingHTTPServer((host, port), ShareHandler)


def _page(shares_dir: Path, token: str) -> Path:
    return shares_dir / f"{token}.html"
