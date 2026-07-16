"""
Protocol/ABC definitions for the retrieval stack.

Why this exists: previously `rag_service.py` called module-level functions
in `retriever.py` directly, which is simple but means you can't unit-test
`rag_service` without a real FAISS index on disk, and you can't swap FAISS
for pgvector/Qdrant later without touching every call site.

These interfaces let `rag_service` depend on an abstraction (DIP), with
concrete implementations wired together in `app/core/dependencies.py`.
Existing module-level functions in retriever.py are UNCHANGED and still
work standalone — DenseRetriever below just wraps them.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    chunk_id: int
    chunk_text: str
    document_id: int
    document_title: str
    page_number: int | None
    section: str | None
    score: float                       # raw score from this retriever (cosine sim or BM25)
    source: str = "dense"              # "dense" | "keyword" | "fused"
    rerank_score: float | None = None
    evidence_snippet: str | None = None


class BaseDenseRetriever(ABC):
    @abstractmethod
    def search(self, query_embedding, top_k: int, allowed_document_ids: set[int] | None = None) -> list[tuple[int, float]]:
        """Returns [(chunk_id, similarity_score), ...]"""
        raise NotImplementedError


class BaseKeywordRetriever(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int, allowed_document_ids: set[int] | None = None) -> list[tuple[int, float]]:
        """Returns [(chunk_id, bm25_score), ...]"""
        raise NotImplementedError

    @abstractmethod
    def rebuild(self, chunk_rows: list[tuple[int, int, str]]) -> None:
        """chunk_rows: [(chunk_id, document_id, chunk_text), ...] — full corpus rebuild."""
        raise NotImplementedError


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, question: str, candidates: list[dict], top_k: int) -> list[dict]:
        raise NotImplementedError


class BaseLLMClient(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError
