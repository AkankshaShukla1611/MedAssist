"""
FAISS-backed vector store.

Security/robustness notes:
- A cross-process FILE lock (filelock.FileLock, backed by an OS-level advisory
  lock on FAISS_PATH + ".lock") guards read-modify-write of the index.
  IMPORTANT: this used to be a threading.Lock, which only synchronizes
  threads within one Python process. That was correct when ingestion ran as
  a FastAPI BackgroundTask inside the same process as the API. Once
  ingestion moved to Celery — separate OS processes, potentially several
  worker processes at once — a threading.Lock provided ZERO protection
  against two workers concurrently read-modify-writing the same on-disk
  index (a real corruption risk: interleaved read_index/write_index calls
  can produce a truncated or inconsistent file). filelock.FileLock uses the
  OS's file locking (fcntl on Linux), which works correctly across processes.
- The index is persisted to disk after every write so a crash doesn't
  lose embeddings (this is a simple RAG app, not a distributed system —
  simplicity here beats a fancier but fragile setup).
- We use IndexIDMap so FAISS vector IDs map 1:1 to our `chunks.id` in
  Postgres — no separate ID-translation table required.
"""
import os

import faiss
import numpy as np
from filelock import FileLock

from app.core.config import settings

_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension; override if you swap models


def _lock_path() -> str:
    return settings.FAISS_PATH + ".lock"


def _file_lock() -> FileLock:
    os.makedirs(os.path.dirname(settings.FAISS_PATH) or ".", exist_ok=True)
    # timeout=30s: a stuck lock (e.g. a crashed worker that never released it)
    # should surface as a clear error, not hang every future request forever.
    return FileLock(_lock_path(), timeout=30)


def _new_index() -> faiss.Index:
    base = faiss.IndexFlatIP(_EMBEDDING_DIM)  # inner product on normalized vectors == cosine similarity
    return faiss.IndexIDMap(base)


def load_index() -> faiss.Index:
    if os.path.exists(settings.FAISS_PATH):
        return faiss.read_index(settings.FAISS_PATH)
    return _new_index()


def save_index(index: faiss.Index) -> None:
    os.makedirs(os.path.dirname(settings.FAISS_PATH), exist_ok=True)
    faiss.write_index(index, settings.FAISS_PATH)


def add_embeddings(chunk_ids: list[int], embeddings: np.ndarray) -> None:
    with _file_lock():
        index = load_index()
        ids = np.array(chunk_ids, dtype="int64")
        index.add_with_ids(embeddings, ids)
        save_index(index)


def remove_embeddings(chunk_ids: list[int]) -> None:
    with _file_lock():
        index = load_index()
        ids = np.array(chunk_ids, dtype="int64")
        index.remove_ids(ids)
        save_index(index)


def search(query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Returns [(chunk_id, similarity_score), ...] sorted by descending similarity."""
    with _file_lock():
        index = load_index()
        if index.ntotal == 0:
            return []
        query = np.expand_dims(query_embedding, axis=0)
        scores, ids = index.search(query, min(top_k, index.ntotal))

    results = []
    for chunk_id, score in zip(ids[0], scores[0]):
        if chunk_id == -1:
            continue
        results.append((int(chunk_id), float(score)))
    return results


class DenseRetriever:
    """
    Thin class-based wrapper around the module-level FAISS functions above,
    implementing app.rag.base.BaseDenseRetriever. Kept separate from the
    functions themselves so existing call sites (embedding_service.py,
    documents.py) don't need to change at all.

    Metadata filtering note: FAISS's IndexFlatIP has no native filter-by-id
    support at query time, so when `allowed_document_ids` is provided we
    simply over-fetch (more candidates than requested) and let the caller
    (HybridRetriever / rag_service, which has the chunk->document mapping
    from Postgres) do the actual filtering. This is a standard, pragmatic
    approach for corpora of this scale; a migration to pgvector/Qdrant
    (both support native filtered ANN search) is the documented next step
    if the corpus grows large enough for over-fetching to hurt latency.
    """

    def search(self, query_embedding: np.ndarray, top_k: int, allowed_document_ids: set[int] | None = None) -> list[tuple[int, float]]:
        fetch_k = top_k * 5 if allowed_document_ids else top_k
        return search(query_embedding, fetch_k)
