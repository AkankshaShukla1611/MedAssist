from unittest.mock import patch
import numpy as np

from app.evaluation.generation_metrics import (
    _proxy_faithfulness, _proxy_context_precision, _proxy_context_recall,
    evaluate_generation, RAGAS_AVAILABLE,
)


def test_proxy_faithfulness_returns_support_ratio_directly():
    assert _proxy_faithfulness("some answer", ["context"], support_ratio=0.75) == 0.75


def test_proxy_faithfulness_defaults_to_zero_when_no_support_ratio_given():
    assert _proxy_faithfulness("some answer", ["context"], support_ratio=None) == 0.0


def test_proxy_context_precision_all_contexts_relevant():
    score = _proxy_context_precision("what is metformin", ["metformin is a diabetes drug", "metformin dosage guidance"])
    assert score == 1.0


def test_proxy_context_precision_no_relevant_contexts():
    score = _proxy_context_precision("what is metformin", ["completely unrelated text about weather"])
    assert score == 0.0


def test_proxy_context_precision_empty_contexts_is_zero():
    assert _proxy_context_precision("question", []) == 0.0


def test_proxy_context_recall_full_coverage():
    score = _proxy_context_recall("metformin treats diabetes", ["metformin is used to treat type 2 diabetes"])
    assert score > 0.5


def test_proxy_context_recall_no_coverage():
    score = _proxy_context_recall("metformin treats diabetes", ["completely unrelated content about weather patterns"])
    assert score == 0.0


def test_evaluate_generation_empty_records_returns_fallback_shape():
    result = evaluate_generation([])
    assert result["is_fallback"] is True
    assert "faithfulness" in result


def test_evaluate_generation_falls_back_when_ragas_unavailable():
    records = [{
        "question": "What is metformin used for?",
        "answer": "Metformin is used to treat type 2 diabetes.",
        "contexts": ["Metformin is a first-line medication for type 2 diabetes."],
        "reference_answer": "Metformin treats type 2 diabetes.",
        "support_ratio": 0.9,
    }]
    # answer_relevancy uses real sentence embeddings, which would otherwise
    # require downloading a model from HuggingFace on every test run — a
    # unit test of the FALLBACK ROUTING logic shouldn't depend on network
    # access. We fake a plausible embedding pair instead (cosine similarity
    # of two near-identical unit vectors ~= high relevance).
    fake_embeddings = np.array([[1.0, 0.0], [0.98, 0.02]])
    with patch("app.evaluation.generation_metrics.RAGAS_AVAILABLE", False), \
         patch("app.evaluation.generation_metrics.embed_texts", return_value=fake_embeddings):
        result = evaluate_generation(records)

    assert result["is_fallback"] is True
    assert result["faithfulness"] == 0.9
    assert 0.0 <= result["answer_relevancy"] <= 1.0
    assert 0.0 <= result["context_precision"] <= 1.0
    assert 0.0 <= result["context_recall"] <= 1.0
