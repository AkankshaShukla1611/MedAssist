from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.limiter import limiter
from app.database.database import Base, engine
from app.api import auth, upload, documents, chat, admin

app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered Clinical Decision Support Assistant using RAG",
    version="0.1.0",
    # Hide interactive docs in production — don't advertise your API surface.
    docs_url="/docs" if settings.ENV != "production" else None,
    redoc_url="/redoc" if settings.ENV != "production" else None,
)

# --- Rate limiting ---
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Too many requests. Please slow down."})


# --- CORS: explicit allow-list only, never "*" once credentials are involved ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["Authorization", "Content-Type"],
)


# --- Security headers on every response ---
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    if settings.ENV == "production":
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return response


# --- Routers ---
app.include_router(auth.router)
app.include_router(upload.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(admin.router)


@app.on_event("startup")
def on_startup():
    # For the skeleton: auto-create tables if they don't exist.
    # Once this goes to production, switch fully to Alembic migrations
    # (`alembic upgrade head`) and remove this line — mixing the two causes
    # drift between what Alembic thinks the schema is and what's really there.
    Base.metadata.create_all(bind=engine)


@app.get("/health")
async def health_check():
    return {"status": "ok", "app": settings.APP_NAME}
