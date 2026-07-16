"""
Renders a persisted EvaluationRun as a report, in either shape:
  - to_json_report(run): plain dict, already what GET /admin/evaluations/{id} returns
  - to_markdown_report(run): human-readable Markdown, for sharing/archiving
    outside the API (e.g. attaching to a PR description or a portfolio writeup)
"""
import json

from app.database import models


def to_json_report(run: models.EvaluationRun) -> dict:
    return {
        "id": run.id,
        "status": run.status.value if run.status else None,
        "dataset_version": run.dataset_version,
        "num_questions": run.num_questions,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "config": {
            "embedding_model": run.embedding_model,
            "llm_model": run.llm_model,
            "reranker_model": run.reranker_model,
            "retrievers_compared": json.loads(run.retrievers_compared) if run.retrievers_compared else None,
            "top_k_values": json.loads(run.top_k_values) if run.top_k_values else None,
        },
        "retrieval_comparison": json.loads(run.retrieval_comparison) if run.retrieval_comparison else None,
        "generation_metrics": json.loads(run.generation_metrics) if run.generation_metrics else None,
        "hallucination_summary": json.loads(run.hallucination_summary) if run.hallucination_summary else None,
        "latency_summary": json.loads(run.latency_summary) if run.latency_summary else None,
        "errors": json.loads(run.errors) if run.errors else [],
        "error_message": run.error_message,
    }


def _markdown_table(rows: dict, headers: tuple[str, str] = ("Metric", "Value")) -> str:
    if not rows:
        return "_no data_\n"
    lines = [f"| {headers[0]} | {headers[1]} |", "|---|---|"]
    for k, v in rows.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"


def to_markdown_report(run: models.EvaluationRun) -> str:
    report = to_json_report(run)
    md = [
        f"# Evaluation Run #{report['id']}",
        "",
        f"**Status:** {report['status']}  ",
        f"**Dataset version:** {report['dataset_version']}  ",
        f"**Questions evaluated:** {report['num_questions']}  ",
        f"**Started:** {report['started_at']}  ",
        f"**Finished:** {report['finished_at']}  ",
        "",
        "## Configuration",
        "",
        _markdown_table(report["config"] or {}),
        "## Retrieval Comparison (Dense vs BM25 vs Hybrid)",
        "",
    ]

    retrieval = report.get("retrieval_comparison") or {}
    if retrieval:
        all_metrics = sorted({m for metrics in retrieval.values() for m in metrics})
        header = "| Retriever | " + " | ".join(all_metrics) + " |"
        separator = "|---|" + "|".join(["---"] * len(all_metrics)) + "|"
        md += [header, separator]
        for retriever_name, metrics in retrieval.items():
            row = [str(metrics.get(m, "-")) for m in all_metrics]
            md.append(f"| {retriever_name} | " + " | ".join(row) + " |")
        md.append("")
    else:
        md.append("_no retrieval comparison data_\n")

    md += ["## Generation Metrics (RAGAS or fallback)", ""]
    md.append(_markdown_table(report.get("generation_metrics") or {}))

    md += ["## Hallucination Summary", ""]
    md.append(_markdown_table(report.get("hallucination_summary") or {}))

    md += ["## Latency Summary (P50 / P95 / P99, ms)", ""]
    latency = report.get("latency_summary") or {}
    if latency:
        md += ["| Stage | P50 | P95 | P99 | Mean | Count |", "|---|---|---|---|---|---|"]
        for stage, stats in latency.items():
            md.append(f"| {stage} | {stats.get('p50')} | {stats.get('p95')} | {stats.get('p99')} | {stats.get('mean')} | {stats.get('count')} |")
        md.append("")
    else:
        md.append("_no latency data_\n")

    if report.get("error_message"):
        md += ["## Error", "", f"```\n{report['error_message']}\n```"]

    return "\n".join(md)
