"""
Keyword (lexical) retrieval via BM25 — the complement to dense/semantic
search. Dense embeddings are great at "meaning" but can miss exact terms
that matter a lot in medicine: drug names, dosages ("500mg"), ICD-10 codes,
abbreviations. BM25 catches those reliably.

Cross-process correctness notes (both fixed together, since they're the
same root cause — this module was written when ingestion and querying
shared one process, and neither assumption survived the move to Celery):

1. LOCKING: uses filelock.FileLock (an OS-level advisory lock), not
   threading.Lock. A threading.Lock only synchronizes threads within one
   process; ingestion (which rebuilds this index) now runs in Celery worker
   processes, separate from the API process — a threading.Lock gave zero
   protection against two processes writing the pickle file concurrently.

2. STALENESS: this index is loaded into memory once and cached
   per-process (see get_bm25_index below). When a Celery worker rebuilds
   the index after an upload, that update only lands in the worker's own
   memory + on disk — the API process's cached copy has no way to know.
   Fixed by checking the on-disk file's mtime on every search and
   transparently reloading if it changed, mirroring what FAISS already
   does "for free" by not caching at all (see retriever.py).

3. RELEVANCE FILTERING: previously used `score <= 0` to decide whether a
   chunk was a "real" match. BM25's IDF term can be zero or negative when a
   query term appears in more than half the corpus (normal for the classic
   Okapi formula on small/topically-concentrated corpora — e.g. "diabetes"
   appearing in most chunks of a diabetes-guideline knowledge base), which
   silently dropped genuinely relevant chunks. Fixed to check actual query/
   chunk token overlap instead of the score's sign.

Scale note: this rebuilds the whole BM25 index in-process on every document
add/remove, and holds the corpus in memory. That's fine up to a few thousand
documents (typical portfolio/demo/small-hospital scale). Past that, migrate
to a real search engine (OpenSearch/Elasticsearch) — flagged in the README
"remaining tasks" list.
"""
import os
import pickle
import re
import threading

from filelock import FileLock
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory BM25 index with disk persistence (pickle) and cross-process
    lock + staleness detection (see module docstring)."""

    def __init__(self, path: str | None = None):
        self.path = path or settings.BM25_INDEX_PATH
        self.chunk_ids: list[int] = []
        self.document_ids: list[int] = []
        self.tokenized_corpus: list[list[str]] = []  # rank_bm25 discards this internally after building IDF; we need it for relevance filtering (see search())
        self._bm25: BM25Okapi | None = None
        self._loaded_mtime: float | None = None
        self._load()

    def _lock(self) -> FileLock:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        return FileLock(self.path + ".lock", timeout=30)

    def _current_mtime(self) -> float | None:
        return os.path.getmtime(self.path) if os.path.exists(self.path) else None

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    data = pickle.load(f)
                self.chunk_ids = data["chunk_ids"]
                self.document_ids = data["document_ids"]
                self.tokenized_corpus = data.get("tokenized_corpus", [])
                self._bm25 = data["bm25"]
                self._loaded_mtime = self._current_mtime()

                if self.chunk_ids and not self.tokenized_corpus:
                    # Index was written before tokenized_corpus was persisted
                    # (pre-fix pickle format). Every search would silently
                    # return zero results (empty overlap against an empty
                    # corpus) until a real rebuild happens — surface this
                    # loudly instead of failing silently.
                    log.error(
                        "bm25_index_stale_pickle_format",
                        path=self.path,
                        action_required="Call rebuild_keyword_index(db) once (e.g. re-save any document) to regenerate this index in the current format.",
                    )
            except Exception as e:
                log.error("bm25_index_load_failed", error=str(e))
                self.chunk_ids, self.document_ids, self.tokenized_corpus, self._bm25 = [], [], [], None

    def _reload_if_stale(self) -> None:
        """Called before every search. Cheap (one stat() call) in the common
        case where nothing changed; reloads the full pickle only when the
        on-disk file's mtime has moved since we last loaded it — i.e. some
        other process (a Celery worker) rebuilt it."""
        current = self._current_mtime()
        if current is not None and current != self._loaded_mtime:
            log.info("bm25_index_reloading_due_to_external_update", path=self.path)
            self._load()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(
                {
                    "chunk_ids": self.chunk_ids,
                    "document_ids": self.document_ids,
                    "tokenized_corpus": self.tokenized_corpus,
                    "bm25": self._bm25,
                },
                f,
            )
        self._loaded_mtime = self._current_mtime()

    def rebuild(self, chunk_rows: list[tuple[int, int, str]]) -> None:
        """chunk_rows: [(chunk_id, document_id, chunk_text), ...] for the FULL corpus."""
        with self._lock():
            if not chunk_rows:
                self.chunk_ids, self.document_ids, self.tokenized_corpus, self._bm25 = [], [], [], None
                self._save()
                return

            self.chunk_ids = [r[0] for r in chunk_rows]
            self.document_ids = [r[1] for r in chunk_rows]
            self.tokenized_corpus = [_tokenize(r[2]) for r in chunk_rows]
            self._bm25 = BM25Okapi(self.tokenized_corpus)
            self._save()

    def search(self, query: str, top_k: int, allowed_document_ids: set[int] | None = None) -> list[tuple[int, float]]:
        query_tokens = _tokenize(query)
        with self._lock():
            self._reload_if_stale()
            if self._bm25 is None or not self.chunk_ids:
                return []
            scores = self._bm25.get_scores(query_tokens)
            corpus = self.tokenized_corpus  # same order as self.chunk_ids

        ranked = sorted(
            zip(self.chunk_ids, self.document_ids, scores, corpus),
            key=lambda x: x[2],
            reverse=True,
        )

        query_token_set = set(query_tokens)
        results = []
        for chunk_id, doc_id, score, chunk_tokens in ranked:
            if allowed_document_ids is not None and doc_id not in allowed_document_ids:
                continue
            # A chunk is "relevant" if it actually shares at least one token
            # with the query — NOT if score > 0. BM25's IDF term goes to
            # zero or negative when a query term appears in more than half
            # the corpus (a normal property of the Okapi formula on small or
            # topically-concentrated corpora), which would otherwise silently
            # discard genuinely relevant chunks. See module-level note.
            if not (query_token_set & set(chunk_tokens)):
                continue
            results.append((chunk_id, float(score)))
            if len(results) >= top_k:
                break
        return results


_index_singleton: BM25Index | None = None
_singleton_lock = threading.Lock()  # protects the Python-level singleton pointer only; safe as threading.Lock


def get_bm25_index() -> BM25Index:
    global _index_singleton
    with _singleton_lock:
        if _index_singleton is None:
            _index_singleton = BM25Index()
        return _index_singleton


def reload_bm25_index() -> None:
    """Call after a rebuild from a different process/thread to force a fresh read from disk."""
    global _index_singleton
    with _singleton_lock:
        _index_singleton = BM25Index()
