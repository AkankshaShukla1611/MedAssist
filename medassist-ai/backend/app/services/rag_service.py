from sqlalchemy.orm import Session

from app.database import models
from app.core.config import settings
from app.rag.embedder import embed_query
from app.rag import retriever
from app.rag.reranker import rerank
from app.rag.generator import generate_answer


async def answer_question(db: Session, user: models.User, question: str) -> dict:
    # 1. Embed the question
    query_embedding = embed_query(question)

    # 2. Vector search -> top N chunk_ids + similarity scores
    raw_hits = retriever.search(query_embedding, settings.TOP_K_RETRIEVE)

    if not raw_hits:
        result = await generate_answer(question, [])
        conversation = _store_conversation(db, user, question, result, [])
        return _build_response(result, [], conversation)

    chunk_ids = [chunk_id for chunk_id, _ in raw_hits]
    score_by_id = {chunk_id: score for chunk_id, score in raw_hits}

    # 3. Load chunk + document metadata from Postgres
    chunks = (
        db.query(models.Chunk)
        .filter(models.Chunk.id.in_(chunk_ids))
        .all()
    )
    documents_by_id = {
        d.id: d for d in db.query(models.Document).filter(
            models.Document.id.in_({c.document_id for c in chunks})
        ).all()
    }

    candidates = []
    for c in chunks:
        doc = documents_by_id.get(c.document_id)
        if not doc:
            continue
        candidates.append({
            "chunk_id": c.id,
            "chunk_text": c.chunk_text,
            "page_number": c.page_number,
            "section": c.section,
            "document_title": doc.title,
            "similarity_score": score_by_id.get(c.id, 0.0),
        })

    # 4. Cross-encoder rerank -> top K final chunks
    top_chunks = rerank(question, candidates, settings.TOP_K_RERANK)

    # 5. Generate the answer (LLM call, JSON-parsed, fail-safe)
    result = await generate_answer(question, top_chunks)

    # 6. Persist conversation + sources
    conversation = _store_conversation(db, user, question, result, top_chunks)

    return _build_response(result, top_chunks, conversation)


def _store_conversation(db: Session, user: models.User, question: str, result: dict, chunks: list[dict]) -> models.Conversation:
    conversation = models.Conversation(
        user_id=user.id,
        question=question,
        answer=result["answer"],
        confidence=result["confidence"],
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


def _build_response(result: dict, chunks: list[dict], conversation: models.Conversation) -> dict:
    citations = [
        {
            "document": c["document_title"],
            "page": c.get("page_number"),
            "section": c.get("section"),
            "similarity_score": round(c.get("rerank_score", c.get("similarity_score", 0.0)), 4),
        }
        for c in chunks
    ]
    related_documents = sorted({c["document_title"] for c in chunks})

    return {
        "answer": result["answer"],
        "confidence": round(result["confidence"], 4),
        "citations": citations,
        "related_documents": related_documents,
    }
