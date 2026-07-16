from functools import lru_cache
from sentence_transformers import CrossEncoder

from app.core.config import settings


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.RERANKER_MODEL)


def rerank(question: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    candidates: [{"chunk_id": int, "chunk_text": str, "similarity_score": float, ...}, ...]
    Returns the top_k candidates re-scored by a cross-encoder, which is far more
    accurate than raw vector similarity for picking the *best* few passages.
    """
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(question, c["chunk_text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k]
