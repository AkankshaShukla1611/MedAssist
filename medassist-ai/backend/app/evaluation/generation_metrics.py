"""
Generation-quality evaluation: Faithfulness, Answer Relevancy, Context
Precision, Context Recall.

Primary path: RAGAS, configured to use the SAME local LLM (via Ollama) as
the rest of the app as its judge, and the same Sentence-Transformers model
for embeddings — so evaluation doesn't require an OpenAI key or any
external API call, consistent with this app's "everything stays local"
privacy posture.

Fallback path: RAGAS (a) may not be installed, or (b) its LLM-judge calls
may fail (Ollama not reachable, model not pulled, etc). Rather than crash
the whole evaluation run, we fall back to lightweight heuristic proxy
metrics, clearly labeled `"is_fallback": true` in the output so results are
never mistaken for real RAGAS scores. This keeps the evaluation pipeline
runnable in CI/tests without a live LLM.

Extensibility: add a new metric by adding one function + one registry entry
in FALLBACK_METRIC_REGISTRY (and the equivalent RAGAS metric import in
_run_ragas) — no other code changes needed.
"""
from typing import Callable

from app.rag.embedder import embed_texts
from app.rag.hallucination import _tokenize
from app.core.logging import get_logger

log = get_logger(__name__)

try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def _cosine_similarity(a, b) -> float:
    import numpy as np
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# --- Fallback proxy metrics (no LLM judge required) ---

def _proxy_faithfulness(answer: str, contexts: list[str], support_ratio: float | None) -> float:
    # Reuses the same lexical-overlap support ratio computed by hallucination.py
    return support_ratio if support_ratio is not None else 0.0


def _proxy_answer_relevancy(question: str, answer: str) -> float:
    if not answer.strip():
        return 0.0
    q_emb, a_emb = embed_texts([question, answer])
    return max(0.0, _cosine_similarity(q_emb, a_emb))


def _proxy_context_precision(question: str, contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 0.0
    relevant = sum(1 for c in contexts if len(_tokenize(c) & q_tokens) > 0)
    return relevant / len(contexts)


def _proxy_context_recall(reference_answer: str, contexts: list[str]) -> float:
    ref_tokens = _tokenize(reference_answer)
    if not ref_tokens:
        return 0.0
    context_tokens: set[str] = set()
    for c in contexts:
        context_tokens |= _tokenize(c)
    covered = len(ref_tokens & context_tokens)
    return covered / len(ref_tokens)


FALLBACK_METRIC_REGISTRY: dict[str, Callable] = {
    "faithfulness": _proxy_faithfulness,
    "answer_relevancy": _proxy_answer_relevancy,
    "context_precision": _proxy_context_precision,
    "context_recall": _proxy_context_recall,
}


def _run_fallback(records: list[dict]) -> dict:
    scores: dict[str, list[float]] = {name: [] for name in FALLBACK_METRIC_REGISTRY}
    for r in records:
        scores["faithfulness"].append(_proxy_faithfulness(r["answer"], r["contexts"], r.get("support_ratio")))
        scores["answer_relevancy"].append(_proxy_answer_relevancy(r["question"], r["answer"]))
        scores["context_precision"].append(_proxy_context_precision(r["question"], r["contexts"]))
        scores["context_recall"].append(_proxy_context_recall(r["reference_answer"], r["contexts"]))

    result = {
        name: round(sum(vals) / len(vals), 4) if vals else 0.0
        for name, vals in scores.items()
    }
    result["is_fallback"] = True
    result["fallback_reason"] = "ragas not installed or LLM judge unavailable"
    return result


def _run_ragas(records: list[dict]) -> dict | None:
    if not RAGAS_AVAILABLE:
        return None
    try:
        from app.core.config import settings
        from langchain_community.llms import Ollama
        from langchain_community.embeddings import HuggingFaceEmbeddings

        judge_llm = Ollama(base_url=settings.LLM_BASE_URL, model=settings.LLM_MODEL)
        judge_embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

        dataset = Dataset.from_dict({
            "question": [r["question"] for r in records],
            "answer": [r["answer"] for r in records],
            "contexts": [r["contexts"] for r in records],
            "ground_truth": [r["reference_answer"] for r in records],
        })

        result = ragas_evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge_llm,
            embeddings=judge_embeddings,
        )
        scores = dict(result)
        scores["is_fallback"] = False
        return scores
    except Exception as e:
        log.warning("ragas_evaluation_failed_falling_back", error=str(e))
        return None


def evaluate_generation(records: list[dict]) -> dict:
    """
    records: [{"question", "answer", "contexts": [...], "reference_answer",
               "support_ratio"}, ...]
    Returns a metric dict with an `is_fallback` flag indicating which path ran.
    """
    if not records:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0,
                 "context_recall": 0.0, "is_fallback": True, "fallback_reason": "no records to evaluate"}

    ragas_result = _run_ragas(records)
    if ragas_result is not None:
        return ragas_result

    return _run_fallback(records)
