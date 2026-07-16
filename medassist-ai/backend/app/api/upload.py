from fastapi import APIRouter, Depends, UploadFile, File, Form, Request
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database import models
from app.core.security import require_roles
from app.core.exceptions import MedAssistError
from app.services.pdf_service import validate_and_save_pdf
from app.services.audit_service import record_audit_event
from app.schemas.document import DocumentResponse
from app.tasks.ingestion_tasks import ingest_document_task

router = APIRouter(prefix="/documents", tags=["Documents"])


class DuplicateDocumentError(MedAssistError):
    status_code = 409
    public_message = "A document with identical content has already been uploaded."


@router.post("/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(
    request: Request,
    title: str = Form(...),
    category: str | None = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles("admin")),
):
    file_path, checksum = await validate_and_save_pdf(file)

    # Duplicate detection: identical file content (by SHA-256, independent
    # of filename/title) has already been ingested — reject rather than
    # silently double-embedding the same content into the vector store.
    existing = db.query(models.Document).filter(models.Document.checksum_sha256 == checksum).first()
    if existing is not None:
        record_audit_event(
            db, action="document.upload", success=False, request=request,
            user_id=current_user.id, resource_type="Document", resource_id=existing.id,
            details={"reason": "duplicate_checksum", "existing_document_title": existing.title},
        )
        raise DuplicateDocumentError(
            f"Identical content already exists as document #{existing.id} ('{existing.title}').",
            existing_document_id=existing.id,
        )

    document = models.Document(
        title=title,
        category=category,
        file_path=file_path,
        checksum_sha256=checksum,
        uploaded_by=current_user.id,
        embedding_status="pending",
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    record_audit_event(
        db, action="document.upload", success=True, request=request,
        user_id=current_user.id, resource_type="Document", resource_id=document.id,
        details={"title": title, "category": category},
    )

    # Ingestion (extract/chunk/embed/BM25) runs as a Celery task — durable
    # across worker restarts, unlike the FastAPI BackgroundTask this
    # previously used (see app.tasks.ingestion_tasks for why that matters).
    task = ingest_document_task.delay(document.id)
    document.celery_task_id = task.id
    db.commit()
    db.refresh(document)

    return document
