"""
Central application configuration.
Every tunable value comes from the environment — nothing sensitive is hardcoded.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    # --- App ---
    APP_NAME: str = "MedAssist AI"
    API_PORT: int = 8000
    ENV: str = "development"  # development | production

    # --- Database ---
    DATABASE_URL: str = "postgresql+psycopg2://medassist:medassist@localhost:5432/medassist"

    # --- JWT / Auth ---
    JWT_SECRET: str  # REQUIRED — no default. App will refuse to start without it.
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # --- Embedding / LLM (swappable) ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_PROVIDER: str = "ollama"          # ollama | openai-compatible
    LLM_MODEL: str = "mistral"            # e.g. "mistral", "gemma", "gemma2" — whatever `ollama pull` name
    LLM_BASE_URL: str = "http://localhost:11434"
    LLM_TIMEOUT_SECONDS: int = 120

    # --- RAG tuning ---
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    TOP_K_RETRIEVE: int = 10
    TOP_K_RERANK: int = 5
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Storage ---
    FAISS_PATH: str = "../vector_store/index.faiss"
    FAISS_METADATA_PATH: str = "../vector_store/metadata.json"
    BM25_INDEX_PATH: str = "../vector_store/bm25_index.pkl"
    UPLOAD_FOLDER: str = "../uploads"
    MAX_UPLOAD_SIZE_MB: int = 25
    ALLOWED_UPLOAD_EXTENSIONS: List[str] = [".pdf"]

    # --- Hybrid retrieval ---
    HYBRID_RETRIEVAL_ENABLED: bool = True
    RRF_K: int = 60                       # standard reciprocal-rank-fusion constant
    QUERY_EXPANSION_ENABLED: bool = True
    MAX_QUERY_EXPANSION_TERMS: int = 5

    # --- Session memory ---
    MAX_SESSION_HISTORY_TURNS: int = 5

    # --- Confidence calibration weights (must sum to ~1.0) ---
    CONFIDENCE_WEIGHT_RETRIEVAL: float = 0.35
    CONFIDENCE_WEIGHT_RERANK: float = 0.25
    CONFIDENCE_WEIGHT_GENERATION: float = 0.25
    CONFIDENCE_WEIGHT_SUPPORT: float = 0.15

    # --- Hallucination detection ---
    HALLUCINATION_SUPPORT_THRESHOLD: float = 0.35  # below this, a claim is "unsupported"

    # --- Security ---
    RATE_LIMIT_DEFAULT: str = "60/minute"
    RATE_LIMIT_AUTH: str = "10/minute"
    CORS_ORIGINS: List[str] = ["http://localhost:5173"]
    FRONTEND_URL: str = "http://localhost:5173"
    CSP_POLICY: str = "default-src 'self'; frame-ancestors 'none'; object-src 'none'"
    ENABLE_PDF_MALWARE_SCAN: bool = True

    # --- Redis (cache + Celery broker/backend) ---
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_EMBEDDING_SECONDS: int = 86400       # embeddings for identical text never change -> cache a full day
    CACHE_TTL_RETRIEVAL_SECONDS: int = 600          # retrieval results shift as documents are added -> shorter TTL
    CACHE_TTL_EVALUATION_SECONDS: int = 3600
    CACHE_TTL_DOCUMENT_METADATA_SECONDS: int = 300
    CACHE_ENABLED: bool = True

    # --- Celery ---
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    CELERY_TASK_ALWAYS_EAGER: bool = False  # set True in tests to run tasks synchronously, no broker needed

    # --- API versioning ---
    API_V1_PREFIX: str = "/api/v1"

    # NOTE: OpenTelemetry distributed tracing was scoped in an earlier phase
    # but never actually implemented (no instrumentation code exists
    # anywhere in app/). Settings for it were previously defined here but
    # read by nothing — removed rather than left as misleading dead config
    # that implies tracing works when it doesn't. Tracked as remaining work
    # (see README) if/when it's actually built.

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
