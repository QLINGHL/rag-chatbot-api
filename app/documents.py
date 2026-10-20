"""Text extraction from uploaded documents.

Supported formats:

* ``.txt`` / ``.md`` - plain UTF-8 text.
* ``.pdf``           - text extracted page by page with ``pypdf``.

Extraction returns plain text; chunking/embedding/storage happen downstream
in :mod:`app.rag`, exactly the same for every format.
"""

import io

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf")


def is_supported(filename: str) -> bool:
    return filename.lower().endswith(SUPPORTED_EXTENSIONS)


def extract_text(filename: str, raw: bytes) -> str:
    """Extract plain text from an uploaded file's bytes.

    Raises ``ValueError`` with a client-friendly message for anything the
    API should reject with HTTP 400 (unsupported type, undecodable text,
    unreadable PDF, PDF with no extractable text).
    """
    lower = filename.lower()
    if lower.endswith((".txt", ".md")):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("File must be UTF-8 text.")
    if lower.endswith(".pdf"):
        return _extract_pdf_text(raw)
    raise ValueError("Only .txt, .md and .pdf files are supported.")


def _extract_pdf_text(raw: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception:
        raise ValueError("File could not be read as a valid PDF.")
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text:
        raise ValueError(
            "No extractable text found in the PDF "
            "(scanned/image-only PDFs are not supported)."
        )
    return text
