from fastapi import APIRouter, Depends, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database import models
from app.core.security import require_roles
from app.services.pdf_service import validate_and_save_pdf
from app.services.embedding_service import process_document_by_id
from app.schemas.document import DocumentResponse

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    category: str | None = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles("admin")),
):
    file_path = await validate_and_save_pdf(file)

    document = models.Document(
        title=title,
        category=category,
        file_path=file_path,
        uploaded_by=current_user.id,
        embedding_status="pending",
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # Ingestion (extract/chunk/embed) runs in the background so the upload
    # request returns immediately instead of blocking on model inference.
    background_tasks.add_task(process_document_by_id, document.id)

    return document
