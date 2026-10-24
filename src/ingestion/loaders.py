
from pathlib import Path
from pypdf import PdfReader
from docx import Document

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}
MAX_FILE_BYTES = 10 * 1024 * 1024


def _extract_docx_blocks(doc):
    """Extract paragraphs and tables in document order."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    parts = []

    for block in doc.iter_inner_content():
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if text:
                parts.append(text)

        elif isinstance(block, Table):
            for row in block.rows:
                cells = [
                    cell.text.strip().replace("\n", " ")
                    for cell in row.cells
                ]
                if any(cells):
                    parts.append(" | ".join(cells))

    return "\n".join(parts).strip()


def extract_text(filename: str, content: bytes) -> list[dict]:
    """Extract text into page/section records. Raises ValueError for invalid input."""
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext or '(no extension)'}"
        )

    if len(content) > MAX_FILE_BYTES:
        raise ValueError("File exceeds the 10 MB upload limit")

    if not content:
        raise ValueError("File is empty")

    try:
        if ext == ".txt":
            text = content.decode(
                "utf-8-sig", errors="replace"
            ).strip()
            records = [{"page": 1, "text": text}]

        elif ext == ".pdf":
            import io

            reader = PdfReader(io.BytesIO(content))
            records = [
                {
                    "page": i + 1,
                    "text": (page.extract_text() or "").strip(),
                }
                for i, page in enumerate(reader.pages)
            ]

        else:
            import io

            doc = Document(io.BytesIO(content))
            text = _extract_docx_blocks(doc)
            records = [{"page": None, "text": text}]

    except Exception as exc:
        raise ValueError(
            f"Could not extract text from {filename}: "
            f"{exc.__class__.__name__}"
        ) from exc

    records = [record for record in records if record["text"]]

    if not records:
        raise ValueError("No extractable text was found in the file")

    return records
