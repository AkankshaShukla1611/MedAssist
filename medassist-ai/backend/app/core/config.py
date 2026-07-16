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
    UPLOAD_FOLDER: str = "../uploads"
    MAX_UPLOAD_SIZE_MB: int = 25
    ALLOWED_UPLOAD_EXTENSIONS: List[str] = [".pdf"]

    # --- Security ---
    RATE_LIMIT_DEFAULT: str = "60/minute"
    RATE_LIMIT_AUTH: str = "10/minute"
    CORS_ORIGINS: List[str] = ["http://localhost:5173"]
    FRONTEND_URL: str = "http://localhost:5173"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
