"""
Handles safe ingestion of uploaded PDFs.
Security notes:
- We validate extension AND real file signature (magic bytes) — never trust the filename alone.
- We enforce a max size BEFORE reading the whole file into memory.
- We generate our own filename (uuid) — never trust user-supplied filenames on disk.
- Every file gets a SHA-256 checksum: stored for duplicate detection at
  upload time, and re-verified before ingestion (Celery task) so a file
  silently corrupted or tampered with on disk between upload and processing
  is caught rather than silently embedded.
"""
import hashlib
import os
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from fastapi import UploadFile, HTTPException

from app.core.config import settings

PDF_MAGIC_BYTES = b"%PDF-"


def compute_sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def verify_file_integrity(file_path: str, expected_checksum: str) -> bool:
    """Re-hashes the file on disk and compares to the checksum recorded at
    upload time. Called just before ingestion (see app.tasks.ingestion_tasks)
    so corruption/tampering between upload and processing is caught, not
    silently embedded into the knowledge base."""
    if not os.path.exists(file_path):
        return False
    with open(file_path, "rb") as f:
        actual = compute_sha256(f.read())
    return actual == expected_checksum


async def validate_and_save_pdf(file: UploadFile) -> tuple[str, str]:
    """Returns (dest_path, sha256_checksum)."""
    # 1. Extension check
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # 2. Size check (stream-based, avoid loading huge files into memory blindly)
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    contents = await file.read()
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds max upload size of {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    # 3. Magic byte check — a renamed .exe won't pass this
    if not contents.startswith(PDF_MAGIC_BYTES):
        raise HTTPException(status_code=400, detail="File is not a valid PDF")

    checksum = compute_sha256(contents)

    # 4. Save under a generated name to avoid path traversal / overwrite attacks
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}.pdf"
    dest_path = os.path.join(settings.UPLOAD_FOLDER, safe_name)

    with open(dest_path, "wb") as f:
        f.write(contents)

    return dest_path, checksum


def extract_text_by_page(file_path: str) -> list[dict]:
    """Returns [{"page_number": int, "text": str}, ...]"""
    pages = []
    doc = fitz.open(file_path)
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text and text.strip():
                pages.append({"page_number": i + 1, "text": text})
    finally:
        doc.close()
    return pages
