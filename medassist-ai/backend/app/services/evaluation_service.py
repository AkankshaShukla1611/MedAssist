"""
Bridges the (already-built, API-independent) evaluation pipeline in
app.evaluation.pipeline to persistence and the admin API. This is the piece
that was missing: the pipeline could compute metrics, but nothing saved a
run or made it queryable.

`create_evaluation_run` and `execute_evaluation_run` are split deliberately:
the former is called synchronously from the API route (fast — just an
INSERT) and returns immediately with a "queued" row; the latter does the
actual (potentially slow) work and is what the Celery task invokes. This
split is also exactly what makes both halves unit-testable independently.
"""
import json

from sqlalchemy.orm import Session

from app.database import models
from app.core.config import settings
from app.core.logging import get_logger
from app.evaluation.schemas import EvaluationConfig
from app.evaluation.pipeline import run_evaluation
from app.evaluation.benchmark_loader import dataset_version as compute_dataset_version

log = get_logger(__name__)


def create_evaluation_run(db: Session, request, user_id: int | None) -> models.EvaluationRun:
    """Creates the DB row in QUEUED state. Does NOT run the pipeline —
    that happens in execute_evaluation_run, invoked async via Celery."""
    run = models.EvaluationRun(
        triggered_by_user_id=user_id,
        status=models.EvaluationRunStatus.QUEUED,
        dataset_path=request.dataset_path,
        embedding_model=settings.EMBEDDING_MODEL,
        llm_model=settings.LLM_MODEL,
        reranker_model=settings.RERANKER_MODEL,
        retrievers_compared=json.dumps(request.retrievers_to_compare),
        top_k_values=json.dumps(request.top_k_values),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def execute_evaluation_run(db: Session, evaluation_run_id: int) -> models.EvaluationRun:
    """
    Does the actual work: builds an EvaluationConfig from what was persisted
    at creation time (so a run's config is reproducible even if global
    settings change before it executes), calls the independent pipeline,
    and persists results. Called by the Celery task OR directly/synchronously
    in tests (no Celery required to exercise this function).
    """
    from datetime import datetime, timezone

    run = db.query(models.EvaluationRun).filter(models.EvaluationRun.id == evaluation_run_id).first()
    if run is None:
        raise ValueError(f"EvaluationRun {evaluation_run_id} not found")

    run.status = models.EvaluationRunStatus.RUNNING
    run.started_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()

    try:
        config = EvaluationConfig(
            dataset_path=run.dataset_path or "",
            dataset_version="",  # resolved by the pipeline itself from the actual file
            embedding_model=run.embedding_model or settings.EMBEDDING_MODEL,
            llm_model=run.llm_model or settings.LLM_MODEL,
            reranker_model=run.reranker_model or settings.RERANKER_MODEL,
            retrievers_to_compare=json.loads(run.retrievers_compared) if run.retrievers_compared else ["dense", "bm25", "hybrid"],
            top_k_values=json.loads(run.top_k_values) if run.top_k_values else [5, 10],
        )

        result = run_evaluation(db, config)

        run.dataset_version = result["dataset_version"]
        run.num_questions = result["num_questions"]
        run.retrieval_comparison = json.dumps(result["retrieval_comparison"])
        run.generation_metrics = json.dumps(result["generation_metrics"])
        run.hallucination_summary = json.dumps(result["hallucination_summary"])
        run.latency_summary = json.dumps(result["latency_summary"])
        run.errors = json.dumps(result["errors"])
        run.status = models.EvaluationRunStatus.COMPLETED
        run.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.commit()
        log.info("evaluation_run_completed", evaluation_run_id=evaluation_run_id, num_questions=run.num_questions)

    except Exception as e:
        run.status = models.EvaluationRunStatus.FAILED
        run.error_message = str(e)
        run.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.commit()
        log.error("evaluation_run_failed", evaluation_run_id=evaluation_run_id, error=str(e))
        raise

    db.refresh(run)
    return run


def get_evaluation_run(db: Session, run_id: int) -> models.EvaluationRun | None:
    return db.query(models.EvaluationRun).filter(models.EvaluationRun.id == run_id).first()


def list_evaluation_runs(db: Session, limit: int = 50, offset: int = 0) -> list[models.EvaluationRun]:
    return (
        db.query(models.EvaluationRun)
        .order_by(models.EvaluationRun.created_at.desc())
        .offset(offset)
        .limit(min(limit, 200))
        .all()
    )
