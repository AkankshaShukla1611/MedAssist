from typing import List, Optional
from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None       # groups multi-turn conversations; omit for a standalone question
    category: Optional[str] = None         # e.g. "Cardiology" — filters retrieval to that specialty

    @field_validator("question")
    @classmethod
    def question_bounds(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Question is too short")
        if len(v) > 2000:
            raise ValueError("Question is too long (max 2000 characters)")
        return v

    @field_validator("session_id")
    @classmethod
    def session_id_bounds(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) > 64:
            raise ValueError("session_id is too long (max 64 characters)")
        return v


class Citation(BaseModel):
    document: str
    page: Optional[int] = None
    section: Optional[str] = None
    similarity_score: float
    evidence_snippet: Optional[str] = None
    retrieval_source: Optional[str] = None  # "dense" | "hybrid"


class ConfidenceBreakdown(BaseModel):
    retrieval_strength: float
    rerank_agreement: float
    generation_confidence: float
    support_ratio: float
    calibrated_confidence: float


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    citations: List[Citation]
    related_documents: List[str]
    # Additive fields — existing clients reading only the fields above are unaffected.
    session_id: Optional[str] = None
    conversation_id: Optional[int] = None
    confidence_breakdown: Optional[ConfidenceBreakdown] = None
    unsupported_claims: List[str] = []


class HistoryItem(BaseModel):
    id: int
    session_id: Optional[str] = None
    question: str
    answer: str
    confidence: Optional[float]
    created_at: str
