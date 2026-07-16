from unittest.mock import patch

from app.database import models
from app.schemas.evaluation import EvaluateRequest
from app.services import evaluation_service

FAKE_PIPELINE_RESULT = {
    "started_at": "2026-01-01T00:00:00+00:00",
    "finished_at": "2026-01-01T00:05:00+00:00",
    "duration_seconds": 300.0,
    "dataset_version": "abc123def456",
    "num_questions": 5,
    "config": {"embedding_model": "test-model", "llm_model": "test-llm", "reranker_model": "test-reranker",
               "retrievers_compared": ["dense", "bm25", "hybrid"], "top_k_values": [5, 10]},
    "retrieval_comparison": {"dense": {"recall_at_5": 0.6}, "hybrid": {"recall_at_5": 0.8}},
    "generation_metrics": {"faithfulness": 0.9, "is_fallback": True},
    "hallucination_summary": {"total_questions": 5, "low_confidence_count": 1},
    "latency_summary": {"total_ms": {"p50": 100.0, "p95": 200.0, "p99": 250.0, "mean": 120.0, "count": 5}},
    "errors": [],
}


def test_create_evaluation_run_starts_queued(db_session):
    run = evaluation_service.create_evaluation_run(db_session, EvaluateRequest(), user_id=None)
    assert run.status == models.EvaluationRunStatus.QUEUED
    assert run.id is not None
    assert run.started_at is None


def test_create_evaluation_run_persists_requested_config(db_session):
    request = EvaluateRequest(retrievers_to_compare=["dense", "hybrid"], top_k_values=[3])
    run = evaluation_service.create_evaluation_run(db_session, request, user_id=1)
    import json
    assert json.loads(run.retrievers_compared) == ["dense", "hybrid"]
    assert json.loads(run.top_k_values) == [3]
    assert run.triggered_by_user_id == 1


def test_execute_evaluation_run_marks_completed_on_success(db_session):
    run = evaluation_service.create_evaluation_run(db_session, EvaluateRequest(), user_id=None)

    with patch("app.services.evaluation_service.run_evaluation", return_value=FAKE_PIPELINE_RESULT):
        result = evaluation_service.execute_evaluation_run(db_session, run.id)

    assert result.status == models.EvaluationRunStatus.COMPLETED
    assert result.num_questions == 5
    assert result.dataset_version == "abc123def456"
    assert result.started_at is not None
    assert result.finished_at is not None


def test_execute_evaluation_run_marks_failed_on_exception(db_session):
    run = evaluation_service.create_evaluation_run(db_session, EvaluateRequest(), user_id=None)

    with patch("app.services.evaluation_service.run_evaluation", side_effect=RuntimeError("pipeline exploded")):
        try:
            evaluation_service.execute_evaluation_run(db_session, run.id)
        except RuntimeError:
            pass

    db_session.refresh(run)
    assert run.status == models.EvaluationRunStatus.FAILED
    assert "pipeline exploded" in run.error_message


def test_execute_evaluation_run_raises_for_missing_run(db_session):
    import pytest
    with pytest.raises(ValueError):
        evaluation_service.execute_evaluation_run(db_session, 999999)


def test_list_evaluation_runs_orders_most_recent_first(db_session):
    run1 = evaluation_service.create_evaluation_run(db_session, EvaluateRequest(), user_id=None)
    run2 = evaluation_service.create_evaluation_run(db_session, EvaluateRequest(), user_id=None)

    runs = evaluation_service.list_evaluation_runs(db_session)
    assert runs[0].id == run2.id
    assert runs[1].id == run1.id


def test_get_evaluation_run_returns_none_for_missing_id(db_session):
    assert evaluation_service.get_evaluation_run(db_session, 999999) is None
