import os
os.environ.setdefault("JWT_SECRET", "test-secret-key-for-pytest-only-do-not-use-in-prod")

from app.rag.confidence import calibrate_confidence


def test_no_chunks_gives_low_confidence():
    result = calibrate_confidence([], generation_confidence=0.9, support_ratio=0.0)
    assert result.calibrated_confidence < 0.5


def test_strong_signals_give_high_confidence():
    top_chunks = [
        {"rerank_score": 8.0},
        {"rerank_score": 7.8},
        {"rerank_score": 7.9},
    ]
    result = calibrate_confidence(top_chunks, generation_confidence=0.95, support_ratio=1.0)
    assert result.calibrated_confidence > 0.7


def test_low_support_ratio_pulls_confidence_down():
    top_chunks = [{"rerank_score": 8.0}, {"rerank_score": 7.9}]
    high_support = calibrate_confidence(top_chunks, generation_confidence=0.9, support_ratio=1.0)
    low_support = calibrate_confidence(top_chunks, generation_confidence=0.9, support_ratio=0.0)
    assert high_support.calibrated_confidence > low_support.calibrated_confidence


def test_confidence_bounded_between_zero_and_one():
    result = calibrate_confidence([{"rerank_score": 100.0}], generation_confidence=1.0, support_ratio=1.0)
    assert 0.0 <= result.calibrated_confidence <= 1.0


def test_disagreeing_chunks_lower_rerank_agreement():
    agreeing = [{"rerank_score": 5.0}, {"rerank_score": 5.1}, {"rerank_score": 4.9}]
    disagreeing = [{"rerank_score": 10.0}, {"rerank_score": -5.0}, {"rerank_score": 2.0}]
    r1 = calibrate_confidence(agreeing, generation_confidence=0.8, support_ratio=0.8)
    r2 = calibrate_confidence(disagreeing, generation_confidence=0.8, support_ratio=0.8)
    assert r1.rerank_agreement > r2.rerank_agreement
