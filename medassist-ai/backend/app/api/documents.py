import os
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database import models
from app.core.security import get_current_user, require_roles
from app.core.dependencies import get_hybrid_retriever
from app.core.config import settings
from app.schemas.document import DocumentResponse
from app.rag import retriever
from app.services.embedding_service import rebuild_keyword_index
from app.services.audit_service import record_audit_event

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("/search")
async def search_documents(
    q: str = Query(..., min_length=2, max_length=500, description="Search query"),
    category: str | None = Query(None, description="Filter by specialty/category, e.g. Cardiology"),
    top_k: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Metadata-aware hybrid search (dense + keyword + RRF fusion) WITHOUT LLM
    generation — useful for browsing/verifying source material directly,
    and for the "keyword search / semantic search + specialty filters"
    capability from the original spec.
    """
    hybrid = get_hybrid_retriever()
    candidates, timing = hybrid.retrieve(db, q, top_k=top_k, category=category)
    return {
        "query": q,
        "category_filter": category,
        "results": [
            {
                "chunk_id": c["chunk_id"],
                "document": c["document_title"],
                "page": c.get("page_number"),
                "section": c.get("section"),
                "score": round(c["similarity_score"], 4),
                "retrieval_source": c.get("retrieval_source"),
                "snippet": c["chunk_text"][:300],
            }
            for c in candidates
        ],
        "timing_ms": timing,
    }


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
    request: Request,
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

    title = document.title
    db.delete(document)  # cascades to chunks via relationship config
    db.commit()
    rebuild_keyword_index(db)

    record_audit_event(
        db, action="document.delete", success=True, request=request,
        user_id=current_user.id, resource_type="Document", resource_id=document_id,
        details={"title": title},
    )
