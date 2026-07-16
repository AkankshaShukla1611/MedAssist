import asyncio
import json
import time

from sqlalchemy.orm import Session

from app.database import models
from app.core.config import settings
from app.core.logging import get_logger
from app.core.observability import RAG_STAGE_LATENCY, RAG_REQUESTS
from app.core.dependencies import get_hybrid_retriever
from app.rag.reranker import rerank
from app.rag.generator import generate_answer
from app.rag.hallucination import detect_unsupported_claims
from app.rag.confidence import calibrate_confidence
from app.services.citation_service import build_citations
from app.services import session_service

log = get_logger(__name__)


async def answer_question(
    db: Session,
    user: models.User,
    question: str,
    session_id: str | None = None,
    category: str | None = None,
) -> dict:
    """
    Full RAG query flow:
    expand -> {dense, keyword} retrieve -> RRF fuse -> metadata-filter ->
    cross-encoder rerank -> (session history +) generate -> hallucination
    check -> calibrate confidence -> rich citations -> persist + log.

    `session_id` and `category` are optional and default to None, so
    existing callers (and the original API contract) keep working exactly
    as before.
    """
    total_start = time.perf_counter()
    retriever = get_hybrid_retriever()

    # 1. Session memory: pull prior turns in this session (empty if none)
    prior_turns = session_service.get_recent_turns(db, user.id, session_id)
    conversation_history = session_service.format_history_for_prompt(prior_turns)

    # 2. Hybrid retrieval (CPU-bound FAISS/BM25 work off the event loop)
    candidates, retrieval_timing = await asyncio.to_thread(
        retriever.retrieve, db, question, settings.TOP_K_RETRIEVE, category
    )

    if not candidates:
        result = await generate_answer(question, [], conversation_history)
        conversation = _store_conversation(db, user, session_id, question, result, [], {}, [])
        _log_retrieval(db, conversation.id, retrieval_timing, 0, 0, (time.perf_counter() - total_start) * 1000)
        RAG_REQUESTS.labels("no_candidates").inc()
        return _build_response(result, [], conversation, {}, [])

    # 3. Cross-encoder rerank (also CPU-bound)
    t0 = time.perf_counter()
    top_chunks = await asyncio.to_thread(rerank, question, candidates, settings.TOP_K_RERANK)
    rerank_ms = (time.perf_counter() - t0) * 1000
    RAG_STAGE_LATENCY.labels("rerank").observe(rerank_ms / 1000)

    # 4. Generation (with session-aware history in the prompt)
    t0 = time.perf_counter()
    result = await generate_answer(question, top_chunks, conversation_history)
    llm_ms = (time.perf_counter() - t0) * 1000
    RAG_STAGE_LATENCY.labels("llm_generation").observe(llm_ms / 1000)

    # 5. Hallucination / unsupported-claim detection
    context_texts = [c["chunk_text"] for c in top_chunks]
    hallucination_result = detect_unsupported_claims(
        result["answer"], context_texts, settings.HALLUCINATION_SUPPORT_THRESHOLD
    )

    # 6. Confidence calibration (retrieval + rerank + generation + support signals)
    confidence_breakdown = calibrate_confidence(
        top_chunks, result["confidence"], hallucination_result["support_ratio"]
    )
    result["confidence"] = confidence_breakdown.calibrated_confidence

    # 7. Rich citations with evidence snippets
    citations = build_citations(question, top_chunks)

    # 8. Persist + analytics log
    conversation = _store_conversation(
        db, user, session_id, question, result, top_chunks,
        confidence_breakdown.to_dict(), hallucination_result["unsupported_sentences"],
    )

    total_ms = (time.perf_counter() - total_start) * 1000
    full_timing = {**retrieval_timing, "rerank_ms": rerank_ms, "llm_ms": llm_ms}
    _log_retrieval(db, conversation.id, full_timing, len(candidates), len(top_chunks), total_ms)
    RAG_REQUESTS.labels("success").inc()

    log.info(
        "chat_answered",
        user_id=user.id,
        session_id=session_id,
        num_candidates=len(candidates),
        num_chunks_used=len(top_chunks),
        confidence=result["confidence"],
        num_unsupported_claims=len(hallucination_result["unsupported_sentences"]),
        total_ms=round(total_ms, 2),
    )

    return _build_response(result, citations, conversation, confidence_breakdown.to_dict(), hallucination_result["unsupported_sentences"])


def _store_conversation(
    db: Session,
    user: models.User,
    session_id: str | None,
    question: str,
    result: dict,
    chunks: list[dict],
    confidence_breakdown: dict,
    unsupported_claims: list[str],
) -> models.Conversation:
    conversation = models.Conversation(
        user_id=user.id,
        session_id=session_id,
        question=question,
        answer=result["answer"],
        confidence=result["confidence"],
        confidence_breakdown=json.dumps(confidence_breakdown) if confidence_breakdown else None,
        unsupported_claims=json.dumps(unsupported_claims) if unsupported_claims else None,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    for c in chunks:
        db.add(models.RetrievedSource(
            conversation_id=conversation.id,
            chunk_id=c["chunk_id"],
            similarity_score=c.get("rerank_score", c.get("similarity_score", 0.0)),
        ))
    db.commit()
    return conversation


def _log_retrieval(db: Session, conversation_id, timing: dict, num_candidates: int, num_used: int, total_ms: float) -> None:
    try:
        log_row = models.RetrievalLog(
            conversation_id=conversation_id,
            query_expansion_ms=timing.get("query_expansion_ms"),
            dense_search_ms=timing.get("dense_search_ms"),
            keyword_search_ms=timing.get("keyword_search_ms"),
            fusion_ms=timing.get("fusion_ms"),
            rerank_ms=timing.get("rerank_ms"),
            llm_ms=timing.get("llm_ms"),
            total_ms=total_ms,
            num_candidates_retrieved=num_candidates,
            num_chunks_used=num_used,
        )
        db.add(log_row)
        db.commit()
    except Exception as e:
        # Analytics logging must never break the user-facing request.
        log.error("retrieval_log_failed", error=str(e))
        db.rollback()


def _build_response(result: dict, citations: list, conversation: models.Conversation, confidence_breakdown: dict, unsupported_claims: list) -> dict:
    related_documents = sorted({c["document"] for c in citations}) if citations else []

    return {
        "answer": result["answer"],
        "confidence": round(result["confidence"], 4),
        "citations": citations,
        "related_documents": related_documents,
        # New, additive fields — old clients that only read the fields above
        # keep working unchanged.
        "session_id": conversation.session_id,
        "conversation_id": conversation.id,
        "confidence_breakdown": confidence_breakdown or None,
        "unsupported_claims": unsupported_claims or [],
    }
