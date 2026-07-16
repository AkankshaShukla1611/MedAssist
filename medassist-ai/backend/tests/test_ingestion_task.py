"""
Direct tests of ingest_document_task's internal logic — previously the only
coverage of this task was test_upload_integrity.py's fully-mocked
`.delay()` call, which never actually executes the task body. These tests
exercise the real function.

Fixture note: the task calls `SessionLocal()` internally (module-level
import from app.database.database), which is bound to a DIFFERENT engine
than the `db_session` fixture's own in-memory SQLite engine — two separate
`sqlite:///:memory:` engines are two independent empty databases, not a
shared one. We monkeypatch the task module's SessionLocal to a sessionmaker
bound to the SAME engine as `db_session` (via StaticPool, which keeps one
real connection alive), so the task sees the data the test set up.
"""
import pytest
from sqlalchemy.orm import sessionmaker

from app.database import models
from app.tasks.ingestion_tasks import ingest_document_task
from app.core.security import hash_password


@pytest.fixture
def patched_session_local(monkeypatch, db_session):
    """Points the task's internal SessionLocal() calls at the test's own in-memory DB."""
    test_sessionmaker = sessionmaker(bind=db_session.bind)
    monkeypatch.setattr("app.tasks.ingestion_tasks.SessionLocal", test_sessionmaker)
    return test_sessionmaker


def _make_uploader(db_session) -> models.User:
    user = models.User(
        name="Admin", email="admin@example.com",
        password_hash=hash_password("AdminPass123"), role=models.UserRole.ADMIN,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def test_task_returns_failed_status_for_nonexistent_document(patched_session_local):
    result = ingest_document_task(document_id=999999)
    assert result["status"] == "failed"
    assert result["reason"] == "document_not_found"


def test_task_returns_failed_status_and_does_not_raise_on_checksum_mismatch(db_session, patched_session_local, tmp_path):
    """
    Regression test for the retry-policy fix: a checksum mismatch must be a
    clean terminal failure (returned, not raised), so it does NOT trigger
    Celery's autoretry_for=(Exception,) — retrying a permanently-corrupted
    file 3 times would be pure wasted work. Prior to the fix, this path
    raised DocumentProcessingError, which WOULD have been retried.
    """
    uploader = _make_uploader(db_session)
    file_path = tmp_path / "doc.pdf"
    file_path.write_bytes(b"%PDF-1.4 fake content for checksum test")

    document = models.Document(
        title="Test Doc", file_path=str(file_path),
        checksum_sha256="0" * 64,  # deliberately wrong — real content hashes to something else
        uploaded_by=uploader.id, embedding_status="pending",
    )
    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)

    # Calling the task function directly (not .delay()) means no actual
    # retry machinery runs regardless — what we're verifying is that the
    # function RETURNS rather than RAISES, which is the behavior autoretry_for
    # keys off of.
    result = ingest_document_task(document_id=document.id)

    assert result["status"] == "failed"
    assert result["reason"] == "checksum_mismatch"

    db_session.refresh(document)
    assert document.embedding_status == "failed"


def test_task_processes_document_with_matching_checksum(db_session, patched_session_local, tmp_path, monkeypatch):
    """
    Happy path: correct checksum should proceed past the integrity gate.
    The actual embedding/FAISS/BM25 work is mocked out here since those are
    already covered by their own dedicated unit tests (test_embedder.py-
    equivalent coverage lives closer to those modules) — this test's job is
    only to verify the task's OWN control flow reaches that point.
    """
    from app.services.pdf_service import compute_sha256

    uploader = _make_uploader(db_session)
    content = b"%PDF-1.4 real content for a passing checksum test"
    file_path = tmp_path / "doc.pdf"
    file_path.write_bytes(content)
    correct_checksum = compute_sha256(content)

    document = models.Document(
        title="Test Doc", file_path=str(file_path),
        checksum_sha256=correct_checksum,
        uploaded_by=uploader.id, embedding_status="pending",
    )
    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)

    monkeypatch.setattr("app.services.embedding_service.process_document", lambda db, doc: setattr(doc, "embedding_status", "complete"))
    monkeypatch.setattr("app.services.embedding_service.rebuild_keyword_index", lambda db: None)

    result = ingest_document_task(document_id=document.id)

    assert result["status"] == "complete"
