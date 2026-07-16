"""
The evaluation pipeline is DELIBERATELY independent of app/services/rag_service.py
and the /chat API. It calls the retrieval/rerank/generation building blocks
directly, so it can be run:

  - as a CLI script (python -m app.evaluation.run_evaluation)
  - from the admin API (POST /admin/evaluate)
  - in CI, on a schedule, or ad hoc after uploading new documents, or after
    changing EMBEDDING_MODEL / LLM_MODEL / reranker / retriever config

...without ever touching conversation history, sessions, or user auth. It
only needs a DB session (to hydrate chunk/document metadata and to read the
current corpus) — it does not go through the chat endpoint's HTTP layer at all.

Methodology note on the retriever comparison: "dense" and "bm25" below are
each retriever run in ISOLATION (no query expansion) as clean baselines.
"hybrid" is the actual production configuration: query expansion + dense +
BM25 + RRF fusion. This directly answers "how much does the production
pipeline's hybrid approach improve over either baseline alone?"
"""
import time
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.database import models
from app.core.config import settings
from app.core.logging import get_logger
from app.rag.retriever import DenseRetriever
from app.rag.keyword_retriever import get_bm25_index
from app.rag.hybrid import HybridRetriever
from app.rag.embedder import embed_query
from app.rag.reranker import rerank
from app.rag.generator import generate_answer
from app.rag.hallucination import detect_unsupported_claims
from app.rag.confidence import calibrate_confidence
from app.evaluation.schemas import BenchmarkQuestion, EvaluationConfig, PerQuestionResult
from app.evaluation.benchmark_loader import load_benchmark, dataset_version
from app.evaluation.retrieval_metrics import evaluate_ranked_list, aggregate_metrics
from app.evaluation.generation_metrics import evaluate_generation
from app.evaluation.hallucination_eval import summarize_hallucination_risk
from app.evaluation.latency_benchmark import summarize_latencies

log = get_logger(__name__)


def _hydrate_chunks(db: Session, chunk_ids: list[int]) -> dict[int, tuple[str, int | None]]:
    """chunk_id -> (document_title, page_number)"""
    if not chunk_ids:
        return {}
    rows = (
        db.query(models.Chunk.id, models.Chunk.page_number, models.Document.title)
        .join(models.Document, models.Chunk.document_id == models.Document.id)
        .filter(models.Chunk.id.in_(chunk_ids))
        .all()
    )
    return {chunk_id: (title, page) for chunk_id, page, title in rows}


def _evaluate_single_retriever(
    db: Session, question: BenchmarkQuestion, hits: list[tuple[int, float]], k_values: list[int]
) -> dict:
    hydration = _hydrate_chunks(db, [chunk_id for chunk_id, _ in hits])
    titles = [hydration.get(cid, ("", None))[0] for cid, _ in hits]
    pages = [hydration.get(cid, ("", None))[1] for cid, _ in hits]
    return evaluate_ranked_list(titles, pages, question.expected_documents, question.expected_pages, k_values)


def run_retrieval_comparison(db: Session, questions: list[BenchmarkQuestion], config: EvaluationConfig) -> dict:
    """Returns {"dense": {aggregated metrics}, "bm25": {...}, "hybrid": {...}}"""
    dense_retriever = DenseRetriever()
    bm25_index = get_bm25_index()
    hybrid_retriever = HybridRetriever(dense_retriever, bm25_index)
    max_k = max(config.top_k_values)

    per_retriever_results: dict[str, list[dict]] = {name: [] for name in config.retrievers_to_compare}

    for q in questions:
        if "dense" in config.retrievers_to_compare:
            embedding = embed_query(q.question)
            dense_hits = dense_retriever.search(embedding, max_k)
            per_retriever_results["dense"].append(_evaluate_single_retriever(db, q, dense_hits, config.top_k_values))

        if "bm25" in config.retrievers_to_compare:
            bm25_hits = bm25_index.search(q.question, max_k)
            per_retriever_results["bm25"].append(_evaluate_single_retriever(db, q, bm25_hits, config.top_k_values))

        if "hybrid" in config.retrievers_to_compare:
            candidates, _timing = hybrid_retriever.retrieve(db, q.question, max_k, category=None)
            titles = [c["document_title"] for c in candidates]
            pages = [c["page_number"] for c in candidates]
            per_retriever_results["hybrid"].append(
                evaluate_ranked_list(titles, pages, q.expected_documents, q.expected_pages, config.top_k_values)
            )

    return {name: aggregate_metrics(results) for name, results in per_retriever_results.items()}


def run_full_pipeline_eval(db: Session, questions: list[BenchmarkQuestion], config: EvaluationConfig) -> tuple[list[PerQuestionResult], dict]:
    """
    Runs the PRODUCTION pipeline (hybrid retrieve -> rerank -> generate) for
    every benchmark question, capturing per-stage latency, and returns
    per-question results plus a latency summary (P50/P95/P99 per stage).
    """
    dense_retriever = DenseRetriever()
    bm25_index = get_bm25_index()
    hybrid_retriever = HybridRetriever(dense_retriever, bm25_index)

    stage_latencies: dict[str, list[float]] = {
        "embedding_ms": [], "retrieval_ms": [], "reranking_ms": [],
        "prompt_construction_ms": [], "llm_generation_ms": [], "total_ms": [],
    }
    results: list[PerQuestionResult] = []

    for q in questions:
        total_start = time.perf_counter()
        try:
            t0 = time.perf_counter()
            _ = embed_query(q.question)
            stage_latencies["embedding_ms"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            candidates, _timing = hybrid_retriever.retrieve(db, q.question, settings.TOP_K_RETRIEVE, category=None)
            stage_latencies["retrieval_ms"].append((time.perf_counter() - t0) * 1000)

            hit_chunk_ids = [c["chunk_id"] for c in candidates]

            t0 = time.perf_counter()
            top_chunks = rerank(q.question, candidates, settings.TOP_K_RERANK) if candidates else []
            stage_latencies["reranking_ms"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            from app.rag.generator import build_prompt
            _prompt = build_prompt(q.question, top_chunks)
            stage_latencies["prompt_construction_ms"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            gen_result = _run_sync_generate(q.question, top_chunks)
            stage_latencies["llm_generation_ms"].append((time.perf_counter() - t0) * 1000)

            context_texts = [c["chunk_text"] for c in top_chunks]
            hallucination_result = detect_unsupported_claims(
                gen_result["answer"], context_texts, settings.HALLUCINATION_SUPPORT_THRESHOLD
            )
            confidence_breakdown = calibrate_confidence(
                top_chunks, gen_result["confidence"], hallucination_result["support_ratio"]
            )

            from app.services.citation_service import build_citations
            citations = build_citations(q.question, top_chunks)

            total_ms = (time.perf_counter() - total_start) * 1000
            stage_latencies["total_ms"].append(total_ms)

            results.append(PerQuestionResult(
                question_id=q.id,
                specialty=q.specialty,
                difficulty=q.difficulty,
                retriever_hits={"hybrid": hit_chunk_ids},
                generated_answer=gen_result["answer"],
                confidence=confidence_breakdown.calibrated_confidence,
                citations=citations,
                unsupported_claims=hallucination_result["unsupported_sentences"],
                support_ratio=hallucination_result["support_ratio"],
                latencies_ms={
                    "retrieval_ms": stage_latencies["retrieval_ms"][-1],
                    "llm_generation_ms": stage_latencies["llm_generation_ms"][-1],
                    "total_ms": total_ms,
                },
            ))
        except Exception as e:
            log.error("eval_question_failed", question_id=q.id, error=str(e))
            results.append(PerQuestionResult(
                question_id=q.id, specialty=q.specialty, difficulty=q.difficulty,
                retriever_hits={}, error=str(e),
            ))

    return results, summarize_latencies(stage_latencies)


def _run_sync_generate(question: str, chunks: list[dict]) -> dict:
    """generate_answer is async (awaits the LLM HTTP call); the evaluation
    pipeline itself is synchronous (easy to run from a script or a
    BackgroundTask), so we run the coroutine to completion here."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're already inside an event loop (e.g. called from a FastAPI
            # BackgroundTask) — use a fresh loop in a thread to avoid conflicts.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(lambda: asyncio.run(generate_answer(question, chunks))).result()
        return loop.run_until_complete(generate_answer(question, chunks))
    except RuntimeError:
        return asyncio.run(generate_answer(question, chunks))


def run_evaluation(db: Session, config: EvaluationConfig) -> dict:
    """
    Top-level entry point. Returns the full evaluation result as a plain
    dict, ready to be persisted (app.database.models.EvaluationRun) and
    rendered as a report (app.evaluation.report).
    """
    started_at = datetime.now(timezone.utc)
    questions = load_benchmark(config.dataset_path or None, config.max_questions)
    version = dataset_version(config.dataset_path or None)

    log.info("evaluation_started", num_questions=len(questions), dataset_version=version)

    retrieval_comparison = run_retrieval_comparison(db, questions, config)

    generation_metrics = {}
    hallucination_summary = {}
    latency_summary = {}
    per_question_results = []

    if config.run_generation_eval or config.run_hallucination_eval:
        per_question_results, latency_summary = run_full_pipeline_eval(db, questions, config)

        if config.run_generation_eval:
            records = [
                {
                    "question": next(q.question for q in questions if q.id == r.question_id),
                    "answer": r.generated_answer or "",
                    "contexts": [c.get("evidence_snippet", "") for c in r.citations] if r.citations else [],
                    "reference_answer": next(q.reference_answer for q in questions if q.id == r.question_id),
                    "support_ratio": r.support_ratio,
                }
                for r in per_question_results if r.error is None
            ]
            generation_metrics = evaluate_generation(records)

        if config.run_hallucination_eval:
            hallucination_summary = summarize_hallucination_risk(questions, per_question_results).to_dict()

    finished_at = datetime.now(timezone.utc)

    return {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "dataset_version": version,
        "num_questions": len(questions),
        "config": {
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_model": settings.LLM_MODEL,
            "reranker_model": settings.RERANKER_MODEL,
            "retrievers_compared": config.retrievers_to_compare,
            "top_k_values": config.top_k_values,
        },
        "retrieval_comparison": retrieval_comparison,
        "generation_metrics": generation_metrics,
        "hallucination_summary": hallucination_summary,
        "latency_summary": latency_summary,
        "errors": [{"question_id": r.question_id, "error": r.error} for r in per_question_results if r.error],
    }
