from sqlalchemy import func
from fastapi import APIRouter, Depends, HTTPException, Query, Response, Request
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database import models
from app.core.security import require_roles
from app.schemas.evaluation import EvaluateRequest, EvaluationRunSummary, EvaluationRunDetail, EvaluationRunCreatedResponse
from app.services import evaluation_service
from app.services.audit_service import record_audit_event, query_audit_logs
from app.evaluation import report as evaluation_report
from app.tasks.evaluation_tasks import run_evaluation_task

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/dashboard")
async def dashboard(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles("admin")),
):
    total_users = db.query(func.count(models.User.id)).scalar()
    total_documents = db.query(func.count(models.Document.id)).scalar()
    total_questions = db.query(func.count(models.Conversation.id)).scalar()

    embedding_status_counts = dict(
        db.query(models.Document.embedding_status, func.count(models.Document.id))
        .group_by(models.Document.embedding_status)
        .all()
    )

    return {
        "total_users": total_users,
        "total_documents": total_documents,
        "total_questions": total_questions,
        "embedding_status": embedding_status_counts,
    }


@router.get("/users")
async def list_users(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles("admin")),
):
    users = db.query(models.User).all()
    return [
        {"id": u.id, "name": u.name, "email": u.email, "role": u.role.value, "is_active": u.is_active}
        for u in users
    ]


@router.get("/analytics")
async def analytics(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles("admin")),
):
    # Most-asked questions is naive text grouping for the skeleton — swap for
    # semantic clustering later if you want real topic analysis.
    top_questions = (
        db.query(models.Conversation.question, func.count(models.Conversation.id).label("count"))
        .group_by(models.Conversation.question)
        .order_by(func.count(models.Conversation.id).desc())
        .limit(10)
        .all()
    )
    return {"most_searched_topics": [{"question": q, "count": c} for q, c in top_questions]}


@router.get("/analytics/retrieval")
async def retrieval_analytics(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles("admin")),
):
    """
    Latency breakdown per RAG pipeline stage — averages over the most recent
    500 requests. For live/streaming dashboards, scrape /metrics (Prometheus)
    instead; this endpoint is for a quick point-in-time view in the admin UI.
    """
    logs = (
        db.query(models.RetrievalLog)
        .order_by(models.RetrievalLog.created_at.desc())
        .limit(500)
        .all()
    )
    if not logs:
        return {"sample_size": 0, "message": "No retrieval logs yet."}

    def avg(field: str) -> float | None:
        values = [getattr(l, field) for l in logs if getattr(l, field) is not None]
        return round(sum(values) / len(values), 2) if values else None

    return {
        "sample_size": len(logs),
        "avg_latency_ms": {
            "query_expansion": avg("query_expansion_ms"),
            "dense_search": avg("dense_search_ms"),
            "keyword_search": avg("keyword_search_ms"),
            "fusion": avg("fusion_ms"),
            "rerank": avg("rerank_ms"),
            "llm_generation": avg("llm_ms"),
            "total": avg("total_ms"),
        },
        "avg_candidates_retrieved": avg("num_candidates_retrieved"),
        "avg_chunks_used": avg("num_chunks_used"),
    }


# --- Evaluation runs ---
# Design note: only mutating/sensitive admin actions are audit-logged
# (evaluate-trigger, uploads, deletes, auth) — read-only dashboard/analytics
# endpoints above are not, to avoid audit-log noise from routine browsing.
# This mirrors common audit-logging practice (log actions, not views).

@router.post("/evaluate", response_model=EvaluationRunCreatedResponse, status_code=202)
async def trigger_evaluation(
    request: Request,
    payload: EvaluateRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles("admin")),
):
    """
    Creates the EvaluationRun row (status=queued) and enqueues the Celery
    task. Returns immediately (202 Accepted) — poll GET /admin/evaluations/{id}
    for status/results, since a full benchmark run (retrieval + generation +
    hallucination eval across every question) can take minutes.
    """
    run = evaluation_service.create_evaluation_run(db, payload, current_user.id)

    task = run_evaluation_task.delay(run.id)
    run.celery_task_id = task.id
    db.commit()
    db.refresh(run)

    record_audit_event(
        db, action="evaluation.trigger", success=True, request=request,
        user_id=current_user.id, resource_type="EvaluationRun", resource_id=run.id,
        details={"dataset_path": payload.dataset_path, "max_questions": payload.max_questions},
    )

    return EvaluationRunCreatedResponse(
        id=run.id, status=run.status.value, celery_task_id=run.celery_task_id,
        message="Evaluation run queued. Poll GET /admin/evaluations/{id} for status and results.",
    )


@router.get("/evaluations", response_model=list[EvaluationRunSummary])
async def list_evaluations(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles("admin")),
):
    runs = evaluation_service.list_evaluation_runs(db, limit=limit, offset=offset)
    return [
        EvaluationRunSummary(
            id=r.id, status=r.status.value, dataset_version=r.dataset_version,
            num_questions=r.num_questions, created_at=r.created_at,
            started_at=r.started_at, finished_at=r.finished_at, celery_task_id=r.celery_task_id,
        )
        for r in runs
    ]


@router.get("/evaluations/{run_id}")
async def get_evaluation(
    run_id: int,
    format: str = Query("json", pattern="^(json|markdown)$", description="'json' or 'markdown'"),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles("admin")),
):
    run = evaluation_service.get_evaluation_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    if format == "markdown":
        return Response(content=evaluation_report.to_markdown_report(run), media_type="text/markdown")
    return evaluation_report.to_json_report(run)


# --- Celery task status ---

@router.get("/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    current_user: models.User = Depends(require_roles("admin")),
):
    """
    Generic Celery task status lookup — works for ingestion and evaluation
    tasks alike, since both are Celery tasks with a result backend (Redis).
    """
    from app.core.celery_app import celery_app

    try:
        result = celery_app.AsyncResult(task_id)
        return {
            "task_id": task_id,
            "status": result.status,   # PENDING | STARTED | RETRY | SUCCESS | FAILURE
            "result": result.result if result.successful() else None,
            "error": str(result.result) if result.failed() else None,
        }
    except Exception as e:
        # The result backend (Redis) being briefly unreachable shouldn't
        # surface as a raw 500 — report it as an explicit status instead,
        # same "fail open, report clearly" posture as app.core.cache.
        return {"task_id": task_id, "status": "UNKNOWN", "result": None, "error": f"Result backend unavailable: {e}"}


# --- Audit log query API ---

@router.get("/audit-logs")
async def list_audit_logs(
    user_id: int | None = Query(None),
    action: str | None = Query(None),
    success: bool | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles("admin")),
):
    logs = query_audit_logs(db, user_id=user_id, action=action, success=success, limit=limit, offset=offset)
    return [
        {
            "id": l.id, "user_id": l.user_id, "ip_address": l.ip_address,
            "endpoint": l.endpoint, "action": l.action, "resource_type": l.resource_type,
            "resource_id": l.resource_id, "success": l.success,
            "details": l.details, "created_at": l.created_at.isoformat(),
        }
        for l in logs
    ]
