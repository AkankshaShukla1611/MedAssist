"""
Confidence calibration.

Problem this solves: previously the ONLY confidence signal was the LLM's own
self-reported number in its JSON output — which is exactly the number a
hallucinating model is least equipped to get right. This module instead
combines four independently-measurable signals:

1. Retrieval strength   — how well did the top fused candidates score
                           (RRF/dense scores are hard to interpret directly,
                           so we use the *cross-encoder rerank score*, which
                           is a calibrated relevance measure).
2. Rerank agreement      — how much do the top chunks agree with each other
                           (low variance among the top few = converging
                           evidence from multiple sources = more trustworthy).
3. Generation confidence — the LLM's own self-reported estimate (kept as
                           ONE signal among several, not the only one).
4. Support ratio         — fraction of generated claims backed by retrieved
                           context (from hallucination.py) — the most direct
                           check against fabrication.

Final confidence is a weighted sum, each weight configurable via env vars
(app/core/config.py) so this can be tuned against eval data without a
code change.
"""
from dataclasses import dataclass, asdict

from app.core.config import settings


@dataclass
class ConfidenceBreakdown:
    retrieval_strength: float
    rerank_agreement: float
    generation_confidence: float
    support_ratio: float
    calibrated_confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


def _retrieval_strength(top_chunks: list[dict]) -> float:
    if not top_chunks:
        return 0.0
    # Cross-encoder rerank scores are logits, roughly in [-10, 10] in practice;
    # squash to [0, 1] with a sigmoid so it combines cleanly with the other signals.
    best_score = max(c.get("rerank_score", 0.0) for c in top_chunks)
    return 1 / (1 + pow(2.71828, -best_score))


def _rerank_agreement(top_chunks: list[dict]) -> float:
    scores = [c.get("rerank_score", 0.0) for c in top_chunks]
    if len(scores) < 2:
        return 0.5  # can't measure agreement with 0-1 chunks; stay neutral
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    # Low variance (chunks agree) -> higher agreement score.
    return max(0.0, 1.0 - min(variance, 1.0))


def calibrate_confidence(
    top_chunks: list[dict],
    generation_confidence: float,
    support_ratio: float,
) -> ConfidenceBreakdown:
    retrieval_strength = _retrieval_strength(top_chunks)
    rerank_agreement = _rerank_agreement(top_chunks)

    calibrated = (
        settings.CONFIDENCE_WEIGHT_RETRIEVAL * retrieval_strength
        + settings.CONFIDENCE_WEIGHT_RERANK * rerank_agreement
        + settings.CONFIDENCE_WEIGHT_GENERATION * generation_confidence
        + settings.CONFIDENCE_WEIGHT_SUPPORT * support_ratio
    )
    calibrated = max(0.0, min(1.0, calibrated))

    return ConfidenceBreakdown(
        retrieval_strength=round(retrieval_strength, 4),
        rerank_agreement=round(rerank_agreement, 4),
        generation_confidence=round(generation_confidence, 4),
        support_ratio=round(support_ratio, 4),
        calibrated_confidence=round(calibrated, 4),
    )
