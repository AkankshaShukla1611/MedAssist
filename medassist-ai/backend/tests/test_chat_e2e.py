from unittest.mock import patch, MagicMock


FAKE_CANDIDATES = [
    {
        "chunk_id": 1,
        "chunk_text": "Metformin is the recommended first-line pharmacologic treatment for type 2 diabetes.",
        "document_id": 1,
        "document_title": "ADA Guidelines 2025",
        "page_number": 34,
        "section": "Initial Therapy",
        "similarity_score": 0.91,
        "retrieval_source": "hybrid",
    }
]

FAKE_RERANKED = [{**FAKE_CANDIDATES[0], "rerank_score": 6.2}]

FAKE_LLM_JSON = (
    '{"summary": "Metformin is first-line therapy for type 2 diabetes.", '
    '"detailed_explanation": "It improves insulin sensitivity.", '
    '"clinical_notes": "Monitor renal function.", '
    '"limitations": "Individual response varies.", '
    '"confidence": 0.9}'
)


def test_chat_requires_auth(client):
    r = client.post("/chat", json={"question": "What treats type 2 diabetes?"})
    assert r.status_code == 401


def test_chat_end_to_end_with_mocked_pipeline(client, auth_headers):
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = (FAKE_CANDIDATES, {
        "query_expansion_ms": 0.1, "dense_search_ms": 1.0, "keyword_search_ms": 0.5, "fusion_ms": 0.1,
    })

    with patch("app.services.rag_service.get_hybrid_retriever", return_value=mock_retriever), \
         patch("app.services.rag_service.rerank", return_value=FAKE_RERANKED), \
         patch("app.rag.generator.llm_generate", return_value=FAKE_LLM_JSON):

        r = client.post("/chat", json={"question": "What is first-line treatment for type 2 diabetes?"}, headers=auth_headers)

    assert r.status_code == 200
    body = r.json()
    assert "Metformin" in body["answer"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["citations"][0]["document"] == "ADA Guidelines 2025"
    assert body["citations"][0]["page"] == 34
    assert "evidence_snippet" in body["citations"][0]
    assert body["confidence_breakdown"] is not None
    assert "conversation_id" in body


def test_chat_no_retrieval_results_gives_zero_confidence(client, auth_headers):
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = ([], {})

    with patch("app.services.rag_service.get_hybrid_retriever", return_value=mock_retriever), \
         patch("app.rag.generator.llm_generate", return_value='{"summary": "no evidence", "confidence": 0.9}'):
        r = client.post("/chat", json={"question": "some obscure question with no matches"}, headers=auth_headers)

    assert r.status_code == 200
    body = r.json()
    assert body["confidence"] == 0.0
    assert body["citations"] == []


def test_chat_rejects_too_short_question(client, auth_headers):
    r = client.post("/chat", json={"question": "hi"}, headers=auth_headers)
    assert r.status_code == 422


def test_session_id_persisted_and_returned(client, auth_headers):
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = (FAKE_CANDIDATES, {})

    with patch("app.services.rag_service.get_hybrid_retriever", return_value=mock_retriever), \
         patch("app.services.rag_service.rerank", return_value=FAKE_RERANKED), \
         patch("app.rag.generator.llm_generate", return_value=FAKE_LLM_JSON):

        r = client.post(
            "/chat",
            json={"question": "What is first-line treatment for type 2 diabetes?", "session_id": "session-abc-123"},
            headers=auth_headers,
        )

    assert r.status_code == 200
    assert r.json()["session_id"] == "session-abc-123"
