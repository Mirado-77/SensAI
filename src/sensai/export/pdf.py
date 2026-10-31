"""PDF export from the HTML page, with WeasyPrint (optional dependency)."""

from sensai.core.errors import ExportError

MISSING = (
    "Export PDF impossible : WeasyPrint n'est pas installé. Installe-le avec"
    " `pip install weasyprint` (il a besoin de Pango sur la machine)."
)


def to_pdf(html: str) -> bytes:
    """WeasyPrint is imported here so that the rest of SENSAI works without it."""
    try:
        import weasyprint
    except (ImportError, OSError) as exc:
        raise ExportError(MISSING) from exc
    return weasyprint.HTML(string=html).write_pdf()
