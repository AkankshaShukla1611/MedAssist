"""
Orchestrates the full retrieval stage: query expansion -> {dense, keyword}
search across all query variants -> metadata filtering -> RRF fusion.
Reranking (cross-encoder) happens AFTER this, in reranker.py, since it needs
the fused candidate set first.

This class takes its dependencies through the constructor (dependency
injection) so it can be unit-tested with fake retrievers instead of a real
FAISS index + BM25 index on disk.
"""
import time
from sqlalchemy.orm import Session

from app.database import models
from app.rag.base import BaseDenseRetriever, BaseKeywordRetriever, RetrievedChunk
from app.rag.embedder import embed_query
from app.rag.query_expansion import expand_query
from app.rag.fusion import reciprocal_rank_fusion
from app.core.config import settings
from app.core.logging import get_logger
from app.core.observability import RAG_STAGE_LATENCY
from app.core.cache import retrieval_cache, cache_key_from_text

log = get_logger(__name__)


class HybridRetriever:
    def __init__(self, dense_retriever: BaseDenseRetriever, keyword_retriever: BaseKeywordRetriever):
        self.dense_retriever = dense_retriever
        self.keyword_retriever = keyword_retriever

    def _resolve_allowed_document_ids(self, db: Session, category: str | None) -> set[int] | None:
        if not category:
            return None
        doc_ids = {
            d.id for d in db.query(models.Document.id)
            .filter(models.Document.category == category)
            .all()
        }
        return doc_ids

    def retrieve(
        self,
        db: Session,
        question: str,
        top_k: int | None = None,
        category: str | None = None,
    ) -> tuple[list[dict], dict]:
        """
        Returns (candidates, timing_info).
        candidates: [{"chunk_id", "chunk_text", "document_id", "document_title",
                       "page_number", "section", "similarity_score", "retrieval_source"}, ...]
        """
        top_k = top_k or settings.TOP_K_RETRIEVE
        timing: dict[str, float] = {}

        cache_key = cache_key_from_text(question, str(category), str(top_k), settings.EMBEDDING_MODEL)
        if settings.CACHE_ENABLED:
            cached = retrieval_cache.get(cache_key)
            if cached is not None:
                timing["cache_hit"] = True
                return cached, timing

        t0 = time.perf_counter()
        allowed_document_ids = self._resolve_allowed_document_ids(db, category)
        timing["metadata_filter_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        query_variants = expand_query(question)
        timing["query_expansion_ms"] = (time.perf_counter() - t0) * 1000

        dense_ranked_lists = []
        keyword_ranked_lists = []

        t0 = time.perf_counter()
        for variant in query_variants:
            query_embedding = embed_query(variant)
            dense_hits = self.dense_retriever.search(query_embedding, top_k, allowed_document_ids)
            if allowed_document_ids is not None:
                dense_hits = self._filter_by_document(db, dense_hits, allowed_document_ids)
            dense_ranked_lists.append(dense_hits)
        timing["dense_search_ms"] = (time.perf_counter() - t0) * 1000
        RAG_STAGE_LATENCY.labels("dense_search").observe(timing["dense_search_ms"] / 1000)

        t0 = time.perf_counter()
        if settings.HYBRID_RETRIEVAL_ENABLED:
            for variant in query_variants:
                keyword_hits = self.keyword_retriever.search(variant, top_k, allowed_document_ids)
                keyword_ranked_lists.append(keyword_hits)
        timing["keyword_search_ms"] = (time.perf_counter() - t0) * 1000
        RAG_STAGE_LATENCY.labels("keyword_search").observe(timing["keyword_search_ms"] / 1000)

        t0 = time.perf_counter()
        all_ranked_lists = dense_ranked_lists + keyword_ranked_lists
        fused = reciprocal_rank_fusion(all_ranked_lists, k=settings.RRF_K)
        top_fused = fused[: top_k * 2]  # keep a generous pool for the reranker to work with
        timing["fusion_ms"] = (time.perf_counter() - t0) * 1000

        if not top_fused:
            return [], timing

        # Hydrate chunk_id -> full metadata from Postgres in one query.
        chunk_ids = [chunk_id for chunk_id, _ in top_fused]
        fused_score_by_id = {chunk_id: score for chunk_id, score in top_fused}

        chunks = db.query(models.Chunk).filter(models.Chunk.id.in_(chunk_ids)).all()
        documents_by_id = {
            d.id: d for d in db.query(models.Document)
            .filter(models.Document.id.in_({c.document_id for c in chunks}))
            .all()
        }

        candidates = []
        for c in chunks:
            doc = documents_by_id.get(c.document_id)
            if not doc:
                continue
            candidates.append({
                "chunk_id": c.id,
                "chunk_text": c.chunk_text,
                "document_id": c.document_id,
                "document_title": doc.title,
                "page_number": c.page_number,
                "section": c.section,
                "similarity_score": fused_score_by_id.get(c.id, 0.0),
                "retrieval_source": "hybrid" if keyword_ranked_lists else "dense",
            })

        candidates.sort(key=lambda c: c["similarity_score"], reverse=True)

        if settings.CACHE_ENABLED:
            retrieval_cache.set(cache_key, candidates, settings.CACHE_TTL_RETRIEVAL_SECONDS)

        return candidates, timing

    @staticmethod
    def _filter_by_document(db: Session, hits: list[tuple[int, float]], allowed_document_ids: set[int]) -> list[tuple[int, float]]:
        if not hits:
            return hits
        chunk_ids = [chunk_id for chunk_id, _ in hits]
        rows = (
            db.query(models.Chunk.id, models.Chunk.document_id)
            .filter(models.Chunk.id.in_(chunk_ids))
            .all()
        )
        doc_id_by_chunk = {chunk_id: doc_id for chunk_id, doc_id in rows}
        return [
            (chunk_id, score) for chunk_id, score in hits
            if doc_id_by_chunk.get(chunk_id) in allowed_document_ids
        ]
