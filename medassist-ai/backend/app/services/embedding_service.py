from sqlalchemy.orm import Session

from app.database import models
from app.database.database import SessionLocal
from app.rag.loader import load_document_pages
from app.rag.chunker import chunk_pages
from app.rag.embedder import embed_texts
from app.rag import retriever


def process_document_by_id(document_id: int) -> None:
    """
    Entry point safe to call from a FastAPI BackgroundTask.
    Opens its OWN db session rather than reusing the request's session, which
    FastAPI closes as soon as the request finishes (a common source of
    "background task silently fails" bugs).
    """
    db = SessionLocal()
    try:
        document = db.query(models.Document).filter(models.Document.id == document_id).first()
        if document:
            process_document(db, document)
    finally:
        db.close()


def process_document(db: Session, document: models.Document) -> None:
    """
    Full ingestion pipeline for one uploaded document:
    extract -> chunk -> persist chunk rows -> embed -> store vectors in FAISS.
    Runs synchronously here for simplicity; swap for a background task/queue
    (Celery/RQ) once upload volume grows.
    """
    document.embedding_status = "processing"
    db.commit()

    try:
        pages = load_document_pages(document.file_path)
        raw_chunks = chunk_pages(pages)

        if not raw_chunks:
            document.embedding_status = "failed"
            db.commit()
            return

        # 1. Persist chunk rows first so we have DB ids to use as FAISS ids.
        chunk_rows = []
        for rc in raw_chunks:
            chunk = models.Chunk(
                document_id=document.id,
                page_number=rc["page_number"],
                section=rc["section"],
                chunk_text=rc["chunk_text"],
            )
            db.add(chunk)
            chunk_rows.append(chunk)
        db.commit()
        for chunk in chunk_rows:
            db.refresh(chunk)

        # 2. Embed and store in FAISS, keyed by the chunk's DB id.
        texts = [c.chunk_text for c in chunk_rows]
        embeddings = embed_texts(texts)
        chunk_ids = [c.id for c in chunk_rows]
        retriever.add_embeddings(chunk_ids, embeddings)

        for chunk in chunk_rows:
            chunk.embedding_id = chunk.id  # 1:1 mapping since we use IndexIDMap
        document.embedding_status = "complete"
        db.commit()

    except Exception:
        document.embedding_status = "failed"
        db.commit()
        raise
