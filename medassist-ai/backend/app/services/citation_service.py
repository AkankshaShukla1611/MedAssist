"""
Builds rich citation objects: not just "which document/page", but the
specific sentence within that chunk that best supports the answer — so a
clinician can verify a claim without reading the whole page.
"""
import re

from app.rag.hallucination import _tokenize


def _best_evidence_sentence(chunk_text: str, question: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", chunk_text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return chunk_text[:200]

    question_tokens = _tokenize(question)
    if not question_tokens:
        return sentences[0]

    best_sentence = sentences[0]
    best_score = -1.0
    for sentence in sentences:
        overlap = len(_tokenize(sentence) & question_tokens)
        if overlap > best_score:
            best_score = overlap
            best_sentence = sentence
    return best_sentence


def build_citations(question: str, top_chunks: list[dict]) -> list[dict]:
    citations = []
    for c in top_chunks:
        evidence = _best_evidence_sentence(c["chunk_text"], question)
        citations.append({
            "document": c["document_title"],
            "page": c.get("page_number"),
            "section": c.get("section"),
            "similarity_score": round(c.get("rerank_score", c.get("similarity_score", 0.0)), 4),
            "evidence_snippet": evidence,
            "retrieval_source": c.get("retrieval_source", "dense"),
        })
    return citations
