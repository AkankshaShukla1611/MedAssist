import enum
from datetime import datetime, timezone


def utc_now_naive() -> datetime:
    """
    Equivalent to the deprecated datetime.utcnow(), kept naive (no tzinfo)
    because these columns are Column(DateTime) without timezone=True — mixing
    naive and aware datetimes in comparisons/ORM defaults would be a bigger,
    riskier change than this migration warrants right now. Migrating every
    DateTime column to timezone=True with aware datetimes throughout is a
    reasonable future improvement, but touches every date comparison in the
    codebase — tracked as optional cleanup, not done here to avoid an
    unrelated behavioral change alongside this deprecation fix.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)

from sqlalchemy import (
    Column, Integer, String, Text, Boolean, Float, ForeignKey, DateTime, Enum
)
from sqlalchemy.orm import relationship

from app.database.database import Base


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    DOCTOR = "doctor"
    MEDICAL_STUDENT = "medical_student"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.MEDICAL_STUDENT)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utc_now_naive)

    documents = relationship("Document", back_populates="uploader")
    conversations = relationship("Conversation", back_populates="user")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    category = Column(String(100), nullable=True)  # e.g. Cardiology, Diabetes
    file_path = Column(String(500), nullable=False)
    checksum_sha256 = Column(String(64), nullable=True, index=True)  # detects duplicates + verifies integrity before ingestion
    celery_task_id = Column(String(64), nullable=True)  # ingestion task id, for GET /admin/tasks/{id} status lookups
    uploaded_by = Column(Integer, ForeignKey("users.id"))
    embedding_status = Column(String(50), default="pending")  # pending|processing|complete|failed
    created_at = Column(DateTime, default=utc_now_naive)

    uploader = relationship("User", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    page_number = Column(Integer, nullable=True)
    section = Column(String(255), nullable=True)
    chunk_text = Column(Text, nullable=False)
    embedding_id = Column(Integer, nullable=True)  # position in FAISS index

    document = relationship("Document", back_populates="chunks")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(String(64), nullable=True, index=True)  # groups multi-turn conversations
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    # Calibrated confidence breakdown, stored as JSON-serialized text for
    # transparency/debugging without needing a schema migration per new signal.
    confidence_breakdown = Column(Text, nullable=True)
    unsupported_claims = Column(Text, nullable=True)  # JSON list of flagged sentences
    created_at = Column(DateTime, default=utc_now_naive)

    user = relationship("User", back_populates="conversations")
    sources = relationship("RetrievedSource", back_populates="conversation", cascade="all, delete-orphan")


class RetrievalLog(Base):
    """Per-request retrieval/generation analytics for observability and
    latency tracking (see /admin/analytics/retrieval)."""
    __tablename__ = "retrieval_logs"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    query_expansion_ms = Column(Float, nullable=True)
    dense_search_ms = Column(Float, nullable=True)
    keyword_search_ms = Column(Float, nullable=True)
    fusion_ms = Column(Float, nullable=True)
    rerank_ms = Column(Float, nullable=True)
    llm_ms = Column(Float, nullable=True)
    total_ms = Column(Float, nullable=True)
    num_candidates_retrieved = Column(Integer, nullable=True)
    num_chunks_used = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=utc_now_naive)


class RetrievedSource(Base):
    __tablename__ = "retrieved_sources"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    chunk_id = Column(Integer, ForeignKey("chunks.id"))
    similarity_score = Column(Float, nullable=False)

    conversation = relationship("Conversation", back_populates="sources")


class EvaluationRunStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationRun(Base):
    """
    Persists one execution of the evaluation pipeline (app.evaluation.pipeline).
    Stored as JSON-serialized text fields (not a fully normalized schema) —
    deliberate choice: evaluation output shape evolves as metrics are added
    (see app.evaluation's extensibility design), and a normalized schema
    would need a migration every time a new metric is added. JSON columns
    trade some queryability for that flexibility, which is the right
    tradeoff for an evaluation log that's read as a whole report, not
    queried column-by-column.
    """
    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, index=True)
    triggered_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    status = Column(Enum(EvaluationRunStatus), nullable=False, default=EvaluationRunStatus.QUEUED)
    celery_task_id = Column(String(64), nullable=True, index=True)

    # Configuration used for this run (persisted verbatim for reproducibility)
    dataset_path = Column(String(500), nullable=True)
    dataset_version = Column(String(32), nullable=True)
    embedding_model = Column(String(255), nullable=True)
    llm_model = Column(String(255), nullable=True)
    reranker_model = Column(String(255), nullable=True)
    retrievers_compared = Column(Text, nullable=True)   # JSON list, e.g. ["dense","bm25","hybrid"]
    top_k_values = Column(Text, nullable=True)           # JSON list, e.g. [5,10]

    # Results (JSON-serialized dicts — see app.evaluation.report for shape)
    retrieval_comparison = Column(Text, nullable=True)
    generation_metrics = Column(Text, nullable=True)
    hallucination_summary = Column(Text, nullable=True)
    latency_summary = Column(Text, nullable=True)
    num_questions = Column(Integer, nullable=True)
    errors = Column(Text, nullable=True)

    error_message = Column(Text, nullable=True)  # populated only if status == FAILED
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utc_now_naive)


class AuditLog(Base):
    """
    Append-only audit trail. Deliberately NOT linked via a cascading FK to
    User (a deleted user's audit history should still exist) — user_id is
    stored as a plain nullable integer, not a ForeignKey with ON DELETE
    CASCADE, so audit history outlives the account it describes.
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    ip_address = Column(String(64), nullable=True)
    endpoint = Column(String(255), nullable=False)
    action = Column(String(100), nullable=False, index=True)  # e.g. "document.upload", "auth.login"
    resource_type = Column(String(100), nullable=True)         # e.g. "Document", "User"
    resource_id = Column(String(64), nullable=True)
    success = Column(Boolean, nullable=False, default=True)
    details = Column(Text, nullable=True)  # JSON blob for action-specific context
    created_at = Column(DateTime, default=utc_now_naive, index=True)
