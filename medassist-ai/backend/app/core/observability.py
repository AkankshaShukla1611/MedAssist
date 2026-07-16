"""
Cross-cutting observability:
- Every request gets a UUID correlation ID, bound into structlog's contextvars
  so every log line during that request (across services) carries it, and
  it's echoed back as a response header for client-side correlation.
- Every request's latency and status are recorded as Prometheus metrics,
  exposed at /metrics for scraping.
"""
import time
import uuid

import structlog
from fastapi import Request
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

REQUEST_COUNT = Counter(
    "medassist_http_requests_total", "Total HTTP requests", ["method", "path", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "medassist_http_request_duration_seconds", "HTTP request latency", ["method", "path"]
)

# RAG-pipeline-specific metrics, incremented from rag_service.
RAG_STAGE_LATENCY = Histogram(
    "medassist_rag_stage_duration_seconds", "Latency of each RAG pipeline stage", ["stage"]
)
RAG_REQUESTS = Counter(
    "medassist_rag_requests_total", "Total RAG chat requests", ["outcome"]
)

# Cache metrics — ratio computed downstream (Grafana/PromQL) as
# hits / (hits + misses) per namespace.
CACHE_HITS = Counter("medassist_cache_hits_total", "Cache hits", ["namespace"])
CACHE_MISSES = Counter("medassist_cache_misses_total", "Cache misses", ["namespace"])

# Document ingestion pipeline (Celery task) latency.
INGESTION_STAGE_LATENCY = Histogram(
    "medassist_ingestion_stage_duration_seconds", "Latency of each document ingestion stage", ["stage"]
)

# Evaluation run latency (full benchmark run, and per-question within it).
EVALUATION_LATENCY = Histogram(
    "medassist_evaluation_duration_seconds", "Latency of evaluation runs", ["scope"]  # scope: "full_run" | "per_question"
)

# Approximate concurrent/active users — incremented on login, sampled via
# unique request_ids isn't meaningful across processes, so we track
# distinct authenticated user IDs seen in a rolling window at the app layer
# (see app.core.dependencies.track_active_user) instead of trying to derive
# it from HTTP metrics alone.
ACTIVE_USERS = Gauge("medassist_active_users", "Distinct authenticated users seen in the last 5 minutes")


async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)

    response = await call_next(request)

    duration = time.perf_counter() - start
    # Use route path template where available to avoid high-cardinality metrics
    # from path params (e.g. /documents/123 vs /documents/{id}).
    route = request.scope.get("route")
    path_label = route.path if route else request.url.path

    REQUEST_COUNT.labels(request.method, path_label, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, path_label).observe(duration)

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-ms"] = f"{duration * 1000:.2f}"
    return response


def metrics_endpoint() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
