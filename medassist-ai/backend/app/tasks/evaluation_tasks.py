"""
Runs the evaluation pipeline (app.evaluation.pipeline) as a Celery task, so
POST /admin/evaluate returns immediately with a run id instead of blocking
on a full benchmark pass (which can take minutes once generation/hallucination
evaluation against a live LLM is included).

Same retry posture as ingestion: transient failures (Ollama momentarily
unreachable) shouldn't permanently fail an evaluation run.
"""
import time
from datetime import datetime, timezone

from app.core.celery_app import celery_app
from app.core.logging import get_logger
from app.core.observability import EVALUATION_LATENCY
from app.database.database import SessionLocal
from app.database import models

log = get_logger(__name__)


@celery_app.task(
    bind=True,
    name="app.tasks.evaluation_tasks.run_evaluation_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=2,  # evaluation runs are expensive; fewer retries than ingestion
)
def run_evaluation_task(self, evaluation_run_id: int) -> dict:
    """
    Thin, retryable wrapper: loads the EvaluationRun row, executes
    app.services.evaluation_service.execute_evaluation_run (unchanged
    business logic — testable without Celery), and lets that function own
    all status transitions/persistence.
    """
    from app.services.evaluation_service import execute_evaluation_run

    db = SessionLocal()
    task_start = time.perf_counter()
    try:
        run = db.query(models.EvaluationRun).filter(models.EvaluationRun.id == evaluation_run_id).first()
        if not run:
            log.error("evaluation_task_run_not_found", evaluation_run_id=evaluation_run_id)
            return {"evaluation_run_id": evaluation_run_id, "status": "failed", "reason": "run_not_found"}

        run.celery_task_id = self.request.id
        db.commit()

        execute_evaluation_run(db, evaluation_run_id)

        duration = time.perf_counter() - task_start
        EVALUATION_LATENCY.labels("full_run").observe(duration)
        log.info("evaluation_task_completed", evaluation_run_id=evaluation_run_id, duration_seconds=round(duration, 2))

        db.refresh(run)
        return {"evaluation_run_id": evaluation_run_id, "status": run.status.value}
    except Exception as e:
        log.error("evaluation_task_failed", evaluation_run_id=evaluation_run_id, error=str(e))
        raise
    finally:
        db.close()
