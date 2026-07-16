"""
Document ingestion, as a Celery task. Replaces the FastAPI BackgroundTask
that previously ran this — the underlying pipeline logic is UNCHANGED, only
how it's invoked has moved, so ingestion behavior is identical to before
this migration.

Why Celery over BackgroundTasks here specifically: ingestion involves CPU-
bound embedding + FAISS/BM25 index rewrites that can take real time for a
large PDF. A BackgroundTask dies with its worker process (a redeploy mid-
ingestion silently loses the work); a Celery task is durable — it's re-picked
up by another worker (task_acks_late=True in celery_app.py).
"""
import time

from app.core.celery_app import celery_app
from app.core.logging import get_logger
from app.core.observability import INGESTION_STAGE_LATENCY
from app.database.database import SessionLocal
from app.database import models

log = get_logger(__name__)


@celery_app.task(
    bind=True,
    name="app.tasks.ingestion_tasks.ingest_document_task",
    autoretry_for=(Exception,),
    retry_backoff=True,       # exponential backoff between retries
    retry_backoff_max=300,    # cap backoff at 5 minutes
    retry_jitter=True,        # avoid thundering-herd retries if many documents fail at once (e.g. Ollama restart)
    max_retries=3,
)
def ingest_document_task(self, document_id: int) -> dict:
    """
    Full ingestion pipeline for one document: verify checksum -> extract ->
    chunk -> embed -> FAISS index -> BM25 rebuild. Delegates the actual
    pipeline steps to app.services.embedding_service (unchanged logic) so
    this task is a thin, retryable wrapper — the pipeline itself stays
    testable without Celery.
    """
    from app.services.embedding_service import process_document, rebuild_keyword_index
    from app.services.pdf_service import verify_file_integrity

    db = SessionLocal()
    task_start = time.perf_counter()
    try:
        document = db.query(models.Document).filter(models.Document.id == document_id).first()
        if not document:
            log.error("ingest_document_task_document_not_found", document_id=document_id)
            return {"document_id": document_id, "status": "failed", "reason": "document_not_found"}

        t0 = time.perf_counter()
        if document.checksum_sha256 and not verify_file_integrity(document.file_path, document.checksum_sha256):
            document.embedding_status = "failed"
            db.commit()
            log.error("ingest_document_checksum_mismatch", document_id=document_id)
            # Deliberately a RETURN, not a raise: a checksum mismatch is a
            # deterministic, permanent failure (the file on disk won't
            # change between retries), unlike the transient infra failures
            # autoretry_for=(Exception,) exists to handle. Raising here
            # would trigger 3 pointless retries with exponential backoff,
            # delaying the correct "failed" status for no benefit — same
            # reasoning as the document_not_found case above, which is also
            # a return, not a raise.
            return {"document_id": document_id, "status": "failed", "reason": "checksum_mismatch"}
        INGESTION_STAGE_LATENCY.labels("integrity_check").observe(time.perf_counter() - t0)

        t0 = time.perf_counter()
        process_document(db, document)  # extract -> chunk -> embed -> FAISS (unchanged pipeline)
        INGESTION_STAGE_LATENCY.labels("extract_chunk_embed_faiss").observe(time.perf_counter() - t0)

        t0 = time.perf_counter()
        rebuild_keyword_index(db)  # BM25
        INGESTION_STAGE_LATENCY.labels("bm25_rebuild").observe(time.perf_counter() - t0)

        total = time.perf_counter() - task_start
        INGESTION_STAGE_LATENCY.labels("total").observe(total)
        log.info("ingest_document_task_completed", document_id=document_id, total_seconds=round(total, 2))

        return {"document_id": document_id, "status": document.embedding_status}
    except Exception as e:
        log.error("ingest_document_task_failed", document_id=document_id, error=str(e))
        raise
    finally:
        db.close()
