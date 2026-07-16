"""
Approximates "active users" (for the Grafana "active users" panel) using
short-TTL Redis keys rather than trying to derive it from in-process state,
which wouldn't be accurate across multiple backend replicas.

Each authenticated request refreshes a `medassist:active_user:{user_id}` key
with a 5-minute TTL; the count of such keys IS the active-user estimate.
Degrades gracefully to "unknown" (metric simply doesn't update) if Redis is
unavailable — never blocks the request.
"""
from app.core.cache import get_redis_client
from app.core.observability import ACTIVE_USERS

ACTIVE_WINDOW_SECONDS = 300


def track_active_user(user_id: int) -> None:
    client = get_redis_client()
    if client is None:
        return
    try:
        client.setex(f"medassist:active_user:{user_id}", ACTIVE_WINDOW_SECONDS, "1")
        count = sum(1 for _ in client.scan_iter(match="medassist:active_user:*"))
        ACTIVE_USERS.set(count)
    except Exception:
        pass  # best-effort metric; never affects the request
