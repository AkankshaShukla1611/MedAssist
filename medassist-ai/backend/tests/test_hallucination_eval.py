from app.evaluation.schemas import BenchmarkQuestion, PerQuestionResult
from app.evaluation.hallucination_eval import summarize_hallucination_risk


def _question(id_="q1", expected_documents=None):
    return BenchmarkQuestion(
        id=id_, question="What is X?", reference_answer="X is Y.",
        expected_documents=expected_documents or ["Doc A"], expected_pages=[1],
        specialty="Cardiology", difficulty="easy", expected_keywords=["x"],
    )


def test_clean_result_has_zero_risk_counts():
    question = _question()
    result = PerQuestionResult(
        question_id="q1", specialty="Cardiology", difficulty="easy",
        retriever_hits={"hybrid": [1, 2]}, generated_answer="X is Y.",
        confidence=0.9, citations=[{"document": "Doc A"}], unsupported_claims=[],
    )
    summary = summarize_hallucination_risk([question], [result]).to_dict()
    assert summary["questions_with_unsupported_claims"] == 0
    assert summary["missing_citation_count"] == 0
    assert summary["citation_mismatch_count"] == 0
    assert summary["low_confidence_count"] == 0
    assert summary["empty_retrieval_count"] == 0


def test_missing_citation_flagged_when_answer_is_not_insufficient_evidence():
    question = _question()
    result = PerQuestionResult(
        question_id="q1", specialty="Cardiology", difficulty="easy",
        retriever_hits={"hybrid": [1]}, generated_answer="X is Y.",
        confidence=0.8, citations=[], unsupported_claims=[],
    )
    summary = summarize_hallucination_risk([question], [result]).to_dict()
    assert summary["missing_citation_count"] == 1


def test_insufficient_evidence_answer_does_not_count_as_missing_citation():
    question = _question()
    result = PerQuestionResult(
        question_id="q1", specialty="Cardiology", difficulty="easy",
        retriever_hits={}, generated_answer="I could not find sufficient evidence in the supplied medical documents.",
        confidence=0.0, citations=[], unsupported_claims=[],
    )
    summary = summarize_hallucination_risk([question], [result]).to_dict()
    assert summary["missing_citation_count"] == 0
    assert summary["empty_retrieval_count"] == 1


def test_citation_mismatch_flagged_when_document_not_expected():
    question = _question(expected_documents=["Doc A"])
    result = PerQuestionResult(
        question_id="q1", specialty="Cardiology", difficulty="easy",
        retriever_hits={"hybrid": [1]}, generated_answer="X is Y.",
        confidence=0.8, citations=[{"document": "Unrelated Doc"}], unsupported_claims=[],
    )
    summary = summarize_hallucination_risk([question], [result]).to_dict()
    assert summary["citation_mismatch_count"] == 1


def test_low_confidence_counted_below_threshold():
    question = _question()
    result = PerQuestionResult(
        question_id="q1", specialty="Cardiology", difficulty="easy",
        retriever_hits={"hybrid": [1]}, generated_answer="X is Y.",
        confidence=0.2, citations=[{"document": "Doc A"}], unsupported_claims=[],
    )
    summary = summarize_hallucination_risk([question], [result]).to_dict()
    assert summary["low_confidence_count"] == 1


def test_rates_are_normalized_by_total_questions():
    question = _question()
    results = [
        PerQuestionResult(question_id="q1", specialty="Cardiology", difficulty="easy",
                           retriever_hits={"hybrid": [1]}, generated_answer="X is Y.",
                           confidence=0.9, citations=[{"document": "Doc A"}], unsupported_claims=["fabricated claim"]),
        PerQuestionResult(question_id="q1", specialty="Cardiology", difficulty="easy",
                           retriever_hits={"hybrid": [1]}, generated_answer="X is Y.",
                           confidence=0.9, citations=[{"document": "Doc A"}], unsupported_claims=[]),
    ]
    summary = summarize_hallucination_risk([question, question], results).to_dict()
    assert summary["questions_with_unsupported_claims_rate"] == 0.5
