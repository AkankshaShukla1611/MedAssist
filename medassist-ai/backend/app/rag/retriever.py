"""
FAISS-backed vector store.

Security/robustness notes:
- A threading.Lock guards read-modify-write of the index so concurrent
  uploads/queries can't corrupt it.
- The index is persisted to disk after every write so a crash doesn't
  lose embeddings (this is a simple RAG app, not a distributed system —
  simplicity here beats a fancier but fragile setup).
- We use IndexIDMap so FAISS vector IDs map 1:1 to our `chunks.id` in
  Postgres — no separate ID-translation table required.
"""
import os
import threading

import faiss
import numpy as np

from app.core.config import settings

_lock = threading.Lock()
_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension; override if you swap models


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
    with _lock:
        index = load_index()
        ids = np.array(chunk_ids, dtype="int64")
        index.add_with_ids(embeddings, ids)
        save_index(index)


def remove_embeddings(chunk_ids: list[int]) -> None:
    with _lock:
        index = load_index()
        ids = np.array(chunk_ids, dtype="int64")
        index.remove_ids(ids)
        save_index(index)


def search(query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Returns [(chunk_id, similarity_score), ...] sorted by descending similarity."""
    with _lock:
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
