"""
Thin Redis cache wrapper used for:
  - embedding cache (identical question text -> same embedding, skip re-encoding)
  - retrieval result cache (identical question+filters -> same ranked chunks,
    within a short TTL since the corpus can change)
  - evaluation result cache (re-running the same config right after -> reuse)
  - document metadata cache (list/get endpoints hit repeatedly by the UI)

Design choice: cache failures NEVER break the request. If Redis is down,
every get() returns a miss and every set() is a no-op — the app degrades to
"no caching", not "500 error". This matters a lot for a clinical tool: a
cache outage should never be the reason a doctor can't get an answer.

Hit ratio is tracked via Prometheus counters (see app.core.observability)
and exposed at /metrics as `medassist_cache_hits_total` /
`medassist_cache_misses_total`, ratio computed in Grafana as
hits / (hits + misses).
"""
import json
from functools import lru_cache

import redis

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis | None:
    if not settings.CACHE_ENABLED:
        return None
    try:
        client = redis.from_url(settings.REDIS_URL, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        return client
    except Exception as e:
        log.warning("redis_unavailable_caching_disabled", error=str(e))
        return None


class Cache:
    """Namespaced cache helper: Cache("embedding").get(key) etc."""

    def __init__(self, namespace: str):
        self.namespace = namespace

    def _key(self, key: str) -> str:
        return f"medassist:{self.namespace}:{key}"

    def get(self, key: str):
        from app.core.observability import CACHE_HITS, CACHE_MISSES
        client = get_redis_client()
        if client is None:
            CACHE_MISSES.labels(self.namespace).inc()
            return None
        try:
            raw = client.get(self._key(key))
        except Exception as e:
            log.warning("cache_get_failed", namespace=self.namespace, error=str(e))
            CACHE_MISSES.labels(self.namespace).inc()
            return None

        if raw is None:
            CACHE_MISSES.labels(self.namespace).inc()
            return None
        CACHE_HITS.labels(self.namespace).inc()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set(self, key: str, value, ttl_seconds: int) -> None:
        client = get_redis_client()
        if client is None:
            return
        try:
            client.setex(self._key(key), ttl_seconds, json.dumps(value))
        except (TypeError, Exception) as e:
            log.warning("cache_set_failed", namespace=self.namespace, error=str(e))

    def delete(self, key: str) -> None:
        client = get_redis_client()
        if client is None:
            return
        try:
            client.delete(self._key(key))
        except Exception as e:
            log.warning("cache_delete_failed", namespace=self.namespace, error=str(e))

    def clear_namespace(self) -> int:
        """Deletes all keys in this namespace. Use sparingly (SCAN-based, safe for production)."""
        client = get_redis_client()
        if client is None:
            return 0
        deleted = 0
        try:
            for key in client.scan_iter(match=self._key("*")):
                client.delete(key)
                deleted += 1
        except Exception as e:
            log.warning("cache_clear_failed", namespace=self.namespace, error=str(e))
        return deleted


embedding_cache = Cache("embedding")
retrieval_cache = Cache("retrieval")
evaluation_cache = Cache("evaluation")
document_metadata_cache = Cache("document_metadata")


def cache_key_from_text(*parts: str) -> str:
    """Deterministic, collision-resistant key from arbitrary text parts (e.g. question + filters)."""
    import hashlib
    joined = "||".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
