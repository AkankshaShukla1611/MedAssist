"""
Liveness vs readiness is a deliberate distinction:
- /health/live: is the process itself running? (used by orchestrators to
  decide whether to restart the container)
- /health/ready: can it actually serve traffic right now? (used by load
  balancers/orchestrators to decide whether to route traffic to it)
Conflating the two means a slow dependency (e.g. Ollama warming up) gets
your container killed and restarted in a loop instead of just excluded
from the load balancer until it's ready.
"""
import os

import httpx
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


def check_database(db: Session) -> dict:
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        log.error("health_check_database_failed", error=str(e))
        return {"status": "failed", "detail": str(e)}


def check_faiss() -> dict:
    try:
        index_dir = os.path.dirname(settings.FAISS_PATH) or "."
        writable = os.access(index_dir, os.W_OK) if os.path.isdir(index_dir) else False
        return {"status": "ok" if writable or not os.path.isdir(index_dir) else "degraded"}
    except Exception as e:
        log.error("health_check_faiss_failed", error=str(e))
        return {"status": "failed", "detail": str(e)}


async def check_llm() -> dict:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{settings.LLM_BASE_URL}/api/tags")
            if response.status_code == 200:
                return {"status": "ok"}
            return {"status": "degraded", "detail": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "failed", "detail": str(e)}


def check_redis() -> dict:
    try:
        from app.core.cache import get_redis_client
        client = get_redis_client()
        if client is None:
            return {"status": "failed", "detail": "Redis client unavailable or CACHE_ENABLED=false"}
        client.ping()
        return {"status": "ok"}
    except Exception as e:
        log.error("health_check_redis_failed", error=str(e))
        return {"status": "failed", "detail": str(e)}


async def readiness_report(db: Session) -> dict:
    db_check = check_database(db)
    faiss_check = check_faiss()
    llm_check = await check_llm()
    redis_check = check_redis()

    # Redis is intentionally NOT part of the pass/fail gate: caching is a
    # performance optimization, not a correctness dependency (see cache.py —
    # every cache operation degrades gracefully to a no-op when Redis is
    # down). A readiness probe that fails because the cache is unavailable
    # would take the whole app out of the load balancer for a non-fatal issue.
    checks = {"database": db_check, "vector_store": faiss_check, "llm": llm_check, "redis": redis_check}
    critical_checks = {"database": db_check, "vector_store": faiss_check, "llm": llm_check}
    overall_ok = all(c["status"] == "ok" for c in critical_checks.values())
    return {"status": "ready" if overall_ok else "not_ready", "checks": checks}
