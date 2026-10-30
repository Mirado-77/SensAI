"""Markdown and HTML views of a session export. The HTML is the shared page and the PDF source."""

import html
from typing import Any

ROLE_TITLES = {"user": "You", "assistant": "SENSAI"}
INTERRUPTED = " (interrupted)"

_STYLE = """
body { font-family: system-ui, sans-serif; max-width: 48rem; margin: 2rem auto; padding: 0 1rem;
       line-height: 1.5; color: #1f2328; }
h1 { font-size: 1.4rem; } h2 { margin-top: 2rem; border-bottom: 1px solid #d0d7de; }
.message { margin: 1rem 0; } .role { font-weight: 600; } .interrupted { color: #9a6700; }
pre { white-space: pre-wrap; background: #f6f8fa; padding: 0.75rem; border-radius: 6px; }
"""


def to_markdown(export: dict[str, Any]) -> str:
    lines = [f"# SENSAI session {export['session_id']}", "", "## Conversation"]
    for message in export["messages"]:
        lines += ["", f"### {_title(message)}", message["content"]]
    if export["document"]:
        lines += ["", "## State document"]
        for part in export["document"]:
            lines += ["", f"### {part['section']}\n{part['content']}"]
    return "\n".join(lines) + "\n"


def to_html(export: dict[str, Any]) -> str:
    title = html.escape(f"SENSAI session {export['session_id']}")
    body = [f"<h1>{title}</h1>", "<h2>Conversation</h2>"]
    for message in export["messages"]:
        status = ' class="interrupted"' if message["status"] == "interrupted" else ""
        body.append(
            f'<div class="message"><div class="role"{status}>{html.escape(_title(message))}</div>'
            f"<pre>{html.escape(message['content'])}</pre></div>"
        )
    if export["document"]:
        body.append("<h2>State document</h2>")
        for part in export["document"]:
            body.append(
                f"<h3>{html.escape(part['section'])}</h3><pre>{html.escape(part['content'])}</pre>"
            )
    return (
        '<!doctype html>\n<html lang="fr">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{title}</title>\n<style>{_STYLE}</style>\n</head>\n<body>\n"
        + "\n".join(body)
        + "\n</body>\n</html>\n"
    )


def _title(message: dict[str, Any]) -> str:
    title = ROLE_TITLES.get(message["role"], message["role"])
    return title + (INTERRUPTED if message["status"] == "interrupted" else "")
