import re
from app.core.config import settings


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Splits each page's text into overlapping chunks.
    Returns: [{"page_number": int, "chunk_text": str, "section": str|None}, ...]
    Chunking per-page (rather than across the whole doc) keeps page-number
    citations accurate, which matters a lot for clinical traceability.
    """
    chunks = []
    size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP

    for page in pages:
        text = clean_text(page["text"])
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append({
                    "page_number": page["page_number"],
                    "chunk_text": chunk_text,
                    "section": None,  # could be enriched later via heading detection
                })
            if end == len(text):
                break
            start = end - overlap  # overlap for context continuity

    return chunks
