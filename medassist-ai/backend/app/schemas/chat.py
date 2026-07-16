from typing import List, Optional
from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_bounds(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Question is too short")
        if len(v) > 2000:
            raise ValueError("Question is too long (max 2000 characters)")
        return v


class Citation(BaseModel):
    document: str
    page: Optional[int] = None
    section: Optional[str] = None
    similarity_score: float


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    citations: List[Citation]
    related_documents: List[str]


class HistoryItem(BaseModel):
    id: int
    question: str
    answer: str
    confidence: Optional[float]
    created_at: str
