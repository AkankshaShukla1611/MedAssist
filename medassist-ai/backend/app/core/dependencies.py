"""
Composition root: this is the ONE place concrete retrieval implementations
get instantiated and wired together. Routes and services depend on the
abstractions (HybridRetriever, BaseReranker) — swapping FAISS for pgvector,
or BM25 for OpenSearch, means changing code here only.
"""
from functools import lru_cache

from app.rag.retriever import DenseRetriever
from app.rag.keyword_retriever import get_bm25_index
from app.rag.hybrid import HybridRetriever


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(
        dense_retriever=DenseRetriever(),
        keyword_retriever=get_bm25_index(),
    )
