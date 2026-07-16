"""
Typed domain models for the evaluation framework. Kept separate from
app/schemas (which are API request/response contracts) since these are
internal to the evaluation pipeline, not part of the HTTP API surface.
"""
from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel


class BenchmarkQuestion(BaseModel):
    id: str
    question: str
    reference_answer: str
    expected_documents: list[str]
    expected_pages: list[int] = []
    specialty: str
    difficulty: str  # "easy" | "medium" | "hard"
    expected_keywords: list[str] = []


@dataclass
class RetrieverConfig:
    name: str                      # "dense" | "bm25" | "hybrid"
    embedding_model: str
    top_k: int


@dataclass
class EvaluationConfig:
    dataset_path: str
    dataset_version: str
    embedding_model: str
    llm_model: str
    reranker_model: str
    retrievers_to_compare: list[str] = field(default_factory=lambda: ["dense", "bm25", "hybrid"])
    top_k_values: list[int] = field(default_factory=lambda: [5, 10])
    run_generation_eval: bool = True
    run_hallucination_eval: bool = True
    max_questions: Optional[int] = None  # cap for fast smoke-test runs


@dataclass
class PerQuestionResult:
    question_id: str
    specialty: str
    difficulty: str
    retriever_hits: dict          # {"dense": [chunk_ids...], "bm25": [...], "hybrid": [...]}
    generated_answer: Optional[str] = None
    confidence: Optional[float] = None
    citations: list = field(default_factory=list)
    unsupported_claims: list = field(default_factory=list)
    support_ratio: Optional[float] = None
    latencies_ms: dict = field(default_factory=dict)
    error: Optional[str] = None
