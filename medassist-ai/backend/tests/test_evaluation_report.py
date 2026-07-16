import json
from datetime import datetime

from app.database import models
from app.evaluation import report as evaluation_report


def _make_run(**overrides) -> models.EvaluationRun:
    defaults = dict(
        id=1, status=models.EvaluationRunStatus.COMPLETED,
        dataset_version="abc123", num_questions=5,
        created_at=datetime(2026, 1, 1), started_at=datetime(2026, 1, 1), finished_at=datetime(2026, 1, 1, 0, 5),
        embedding_model="test-model", llm_model="test-llm", reranker_model="test-reranker",
        retrievers_compared=json.dumps(["dense", "hybrid"]), top_k_values=json.dumps([5, 10]),
        retrieval_comparison=json.dumps({"dense": {"recall_at_5": 0.5}, "hybrid": {"recall_at_5": 0.8}}),
        generation_metrics=json.dumps({"faithfulness": 0.9}),
        hallucination_summary=json.dumps({"low_confidence_count": 1}),
        latency_summary=json.dumps({"total_ms": {"p50": 100, "p95": 200, "p99": 250, "mean": 120, "count": 5}}),
        errors=json.dumps([]), error_message=None,
    )
    defaults.update(overrides)
    run = models.EvaluationRun()
    for k, v in defaults.items():
        setattr(run, k, v)
    return run


def test_to_json_report_parses_all_json_fields():
    run = _make_run()
    report = evaluation_report.to_json_report(run)
    assert report["retrieval_comparison"]["hybrid"]["recall_at_5"] == 0.8
    assert report["generation_metrics"]["faithfulness"] == 0.9
    assert report["config"]["embedding_model"] == "test-model"


def test_to_json_report_handles_none_fields_gracefully():
    run = _make_run(retrieval_comparison=None, generation_metrics=None, hallucination_summary=None, latency_summary=None, errors=None)
    report = evaluation_report.to_json_report(run)
    assert report["retrieval_comparison"] is None
    assert report["errors"] == []


def test_to_markdown_report_includes_run_id_and_status():
    run = _make_run()
    md = evaluation_report.to_markdown_report(run)
    assert "Evaluation Run #1" in md
    assert "completed" in md.lower()


def test_to_markdown_report_includes_retriever_comparison_table():
    run = _make_run()
    md = evaluation_report.to_markdown_report(run)
    assert "dense" in md
    assert "hybrid" in md
    assert "recall_at_5" in md


def test_to_markdown_report_includes_error_section_when_failed():
    run = _make_run(status=models.EvaluationRunStatus.FAILED, error_message="Ollama unreachable")
    md = evaluation_report.to_markdown_report(run)
    assert "## Error" in md
    assert "Ollama unreachable" in md
