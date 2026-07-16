from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from slowapi.errors import RateLimitExceeded
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.limiter import limiter
from app.core.logging import configure_logging, get_logger
from app.core.exceptions import MedAssistError
from app.core.exception_handlers import medassist_error_handler, unhandled_exception_handler
from app.core.observability import request_context_middleware, metrics_endpoint
from app.core import health
from app.database.database import Base, engine, get_db
from app.api import auth, upload, documents, chat, admin

configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Alembic migrations now exist (backend/alembic/, initial migration
    # verified to apply and roll back cleanly against a real schema — see
    # backend/alembic/versions/). create_all() is kept ONLY as a dev/test
    # convenience for a from-scratch SQLite/local Postgres with no
    # migration history yet; it is a no-op against tables that already
    # exist, so it's harmless in dev, but it must NOT be relied on in
    # production — it only creates missing tables, it never alters existing
    # ones, which is exactly how schema drift happens silently. Production
    # deployments run `alembic upgrade head` explicitly before the app
    # starts (see docker-compose.yml's backend command).
    if settings.ENV != "production":
        Base.metadata.create_all(bind=engine)
    log.info("app_startup", env=settings.ENV, llm_model=settings.LLM_MODEL)
    yield
    # Shutdown: nothing to flush yet (no in-process queues/connections held
    # open outside SQLAlchemy's own pool, which it cleans up itself).
    log.info("app_shutdown")


app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered Clinical Decision Support Assistant using RAG",
    version="0.2.0",
    # Hide interactive docs in production — don't advertise your API surface.
    docs_url="/docs" if settings.ENV != "production" else None,
    redoc_url="/redoc" if settings.ENV != "production" else None,
    lifespan=lifespan,
)

# --- Rate limiting ---
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Too many requests. Please slow down."})


# --- Domain + catch-all exception handlers (single source of truth for error shape) ---
app.add_exception_handler(MedAssistError, medassist_error_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)


# --- CORS: explicit allow-list only, never "*" once credentials are involved ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["Authorization", "Content-Type"],
)

# --- Request correlation ID + latency + Prometheus metrics ---
app.middleware("http")(request_context_middleware)


# --- Security headers on every response ---
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = settings.CSP_POLICY
    if settings.ENV == "production":
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return response


# --- Routers ---
# Mounted TWICE: once under the versioned prefix (canonical, matches the
# documented API contract: /api/v1/chat, /api/v1/admin/evaluate, etc.), and
# once at the legacy root path for backward compatibility with anything
# built against the original unversioned URLs. Same router objects, same
# endpoint functions — this is a routing-layer addition, not duplicated logic.
for _router in (auth.router, upload.router, documents.router, chat.router, admin.router):
    app.include_router(_router, prefix=settings.API_V1_PREFIX)
    app.include_router(_router)  # legacy alias, see deprecation_headers middleware below


@app.middleware("http")
async def deprecation_headers(request: Request, call_next):
    """Marks legacy (non-versioned) API paths as deprecated without breaking them."""
    response = await call_next(request)
    path = request.url.path
    if not path.startswith(settings.API_V1_PREFIX) and not path.startswith(("/health", "/metrics", "/docs", "/redoc", "/openapi")):
        response.headers["Deprecation"] = "true"
        response.headers["Link"] = f'<{settings.API_V1_PREFIX}{path}>; rel="successor-version"'
    return response


@app.get("/health")
async def health_check():
    """Kept for backward compatibility with the original skeleton."""
    return {"status": "ok", "app": settings.APP_NAME}


@app.get("/health/live")
async def liveness():
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness(db: Session = Depends(get_db)):
    report = await health.readiness_report(db)
    status_code = 200 if report["status"] == "ready" else 503
    return JSONResponse(status_code=status_code, content=report)


# --- Per-dependency health endpoints ---
# Useful for debugging ("which dependency is actually down?") separately
# from /health/ready's aggregate pass/fail used by orchestrators.
@app.get("/health/database")
async def health_database(db: Session = Depends(get_db)):
    result = health.check_database(db)
    return JSONResponse(status_code=200 if result["status"] == "ok" else 503, content=result)


@app.get("/health/faiss")
async def health_faiss():
    result = health.check_faiss()
    return JSONResponse(status_code=200 if result["status"] == "ok" else 503, content=result)


@app.get("/health/ollama")
async def health_ollama():
    result = await health.check_llm()
    return JSONResponse(status_code=200 if result["status"] == "ok" else 503, content=result)


@app.get("/health/redis")
async def health_redis():
    result = health.check_redis()
    # Always 200: Redis is a performance optimization, not a hard
    # dependency (see the comment in health.readiness_report). This endpoint
    # reports status without implying the app is unhealthy without it.
    return JSONResponse(status_code=200, content=result)


@app.get("/metrics")
async def metrics():
    return metrics_endpoint()
