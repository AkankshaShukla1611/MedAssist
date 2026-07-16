from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class EvaluateRequest(BaseModel):
    """
    All fields optional with sensible defaults, so `POST /admin/evaluate {}`
    runs a full evaluation against the default benchmark dataset — but every
    knob from app.evaluation.schemas.EvaluationConfig is overridable, which
    is what "support multiple benchmark datasets" means in practice: point
    dataset_path at a different .jsonl file to evaluate against it.
    """
    dataset_path: Optional[str] = Field(None, description="Path to a benchmark .jsonl file; defaults to the built-in dataset")
    max_questions: Optional[int] = Field(None, ge=1, description="Cap the number of questions, for a fast smoke-test run")
    retrievers_to_compare: list[str] = Field(default_factory=lambda: ["dense", "bm25", "hybrid"])
    top_k_values: list[int] = Field(default_factory=lambda: [5, 10])
    run_generation_eval: bool = True
    run_hallucination_eval: bool = True


class EvaluationRunSummary(BaseModel):
    id: int
    status: str
    dataset_version: Optional[str] = None
    num_questions: Optional[int] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    celery_task_id: Optional[str] = None

    class Config:
        from_attributes = True


class EvaluationRunDetail(EvaluationRunSummary):
    embedding_model: Optional[str] = None
    llm_model: Optional[str] = None
    reranker_model: Optional[str] = None
    retrievers_compared: Optional[list[str]] = None
    top_k_values: Optional[list[int]] = None
    retrieval_comparison: Optional[dict] = None
    generation_metrics: Optional[dict] = None
    hallucination_summary: Optional[dict] = None
    latency_summary: Optional[dict] = None
    errors: Optional[list] = None
    error_message: Optional[str] = None


class EvaluationRunCreatedResponse(BaseModel):
    id: int
    status: str
    celery_task_id: Optional[str] = None
    message: str
