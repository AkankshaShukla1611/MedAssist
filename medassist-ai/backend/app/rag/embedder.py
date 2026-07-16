from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.cache import embedding_cache, cache_key_from_text


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    # Loaded once per process — embedding models are expensive to initialize.
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")


def embed_query(text: str) -> np.ndarray:
    """
    Cached at the single-query level: identical question text (very common —
    "what is the first-line treatment for X" gets asked repeatedly) skips
    re-encoding entirely. Bulk embedding (embed_texts, used during document
    ingestion) is NOT cached — each chunk is normally unique, so there's
    nothing to reuse there.
    """
    if not settings.CACHE_ENABLED:
        return embed_texts([text])[0]

    key = cache_key_from_text(settings.EMBEDDING_MODEL, text)
    cached = embedding_cache.get(key)
    if cached is not None:
        return np.array(cached, dtype="float32")

    embedding = embed_texts([text])[0]
    embedding_cache.set(key, embedding.tolist(), settings.CACHE_TTL_EMBEDDING_SECONDS)
    return embedding
