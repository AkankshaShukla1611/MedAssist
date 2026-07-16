import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database import models
from app.core.security import get_current_user, require_roles
from app.schemas.document import DocumentResponse
from app.rag import retriever

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),  # any authenticated role can browse
):
    return db.query(models.Document).order_by(models.Document.created_at.desc()).all()


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    document = db.query(models.Document).filter(models.Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles("admin")),
):
    document = db.query(models.Document).filter(models.Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    chunk_ids = [c.id for c in document.chunks]
    if chunk_ids:
        retriever.remove_embeddings(chunk_ids)  # keep FAISS in sync with Postgres

    if os.path.exists(document.file_path):
        os.remove(document.file_path)

    db.delete(document)  # cascades to chunks via relationship config
    db.commit()
