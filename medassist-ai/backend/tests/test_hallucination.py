from app.rag.hallucination import detect_unsupported_claims, split_sentences


def test_split_sentences_basic():
    sentences = split_sentences("Metformin is first-line. It lowers glucose. Monitor renal function.")
    assert len(sentences) == 3


def test_fully_supported_answer_has_high_support_ratio():
    context = ["Metformin is the recommended first-line pharmacologic treatment for type 2 diabetes."]
    answer = "Metformin is the recommended first-line pharmacologic treatment for type 2 diabetes."
    result = detect_unsupported_claims(answer, context, threshold=0.35)
    assert result["support_ratio"] == 1.0
    assert result["unsupported_sentences"] == []


def test_fabricated_claim_is_flagged():
    context = ["Metformin is the recommended first-line pharmacologic treatment for type 2 diabetes."]
    answer = "Patients should also take unicorn tears twice daily for optimal glycemic outcomes."
    result = detect_unsupported_claims(answer, context, threshold=0.35)
    assert result["support_ratio"] < 1.0
    assert len(result["unsupported_sentences"]) == 1


def test_empty_answer_returns_full_support():
    result = detect_unsupported_claims("", ["some context"], threshold=0.35)
    assert result["support_ratio"] == 1.0
    assert result["unsupported_sentences"] == []


def test_mixed_answer_partial_support():
    context = ["Metformin is first-line therapy for type 2 diabetes."]
    answer = (
        "Metformin is first-line therapy for type 2 diabetes. "
        "Patients should also undergo a daily unrelated fabricated ritual procedure."
    )
    result = detect_unsupported_claims(answer, context, threshold=0.35)
    assert 0.0 < result["support_ratio"] < 1.0
    assert len(result["unsupported_sentences"]) == 1
