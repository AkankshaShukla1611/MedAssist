from app.services.pdf_service import extract_text_by_page


def load_document_pages(file_path: str) -> list[dict]:
    """Thin wrapper kept separate so future loaders (docx, html, etc.) plug in here."""
    return extract_text_by_page(file_path)
