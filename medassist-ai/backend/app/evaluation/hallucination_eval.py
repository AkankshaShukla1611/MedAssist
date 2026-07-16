"""
Aggregates hallucination/quality-risk signals across an entire benchmark
run into summary statistics an ML/eng team can track over time.

Note on "incorrect citations": we don't have ground-truth-verified citation
correctness (that would require a human-annotated citation-accuracy label
per question). As a practical proxy, a citation is flagged as a possible
mismatch if its source document isn't among the question's
`expected_documents` — this is a heuristic, not a certainty (a good answer
CAN legitimately cite an additional relevant document we didn't anticipate),
and is documented as such in the report.
"""
from dataclasses import dataclass, field

from app.evaluation.schemas import BenchmarkQuestion, PerQuestionResult

LOW_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class HallucinationSummary:
    total_questions: int
    unsupported_claims_total: int
    questions_with_unsupported_claims: int
    missing_citation_count: int
    citation_mismatch_count: int
    low_confidence_count: int
    empty_retrieval_count: int

    def to_dict(self) -> dict:
        n = max(self.total_questions, 1)
        return {
            "total_questions": self.total_questions,
            "unsupported_claims_total": self.unsupported_claims_total,
            "questions_with_unsupported_claims": self.questions_with_unsupported_claims,
            "questions_with_unsupported_claims_rate": round(self.questions_with_unsupported_claims / n, 4),
            "missing_citation_count": self.missing_citation_count,
            "missing_citation_rate": round(self.missing_citation_count / n, 4),
            "citation_mismatch_count": self.citation_mismatch_count,
            "low_confidence_count": self.low_confidence_count,
            "low_confidence_rate": round(self.low_confidence_count / n, 4),
            "empty_retrieval_count": self.empty_retrieval_count,
            "empty_retrieval_rate": round(self.empty_retrieval_count / n, 4),
        }


def summarize_hallucination_risk(
    questions: list[BenchmarkQuestion],
    results: list[PerQuestionResult],
) -> HallucinationSummary:
    by_id = {q.id: q for q in questions}

    unsupported_total = 0
    questions_with_unsupported = 0
    missing_citations = 0
    citation_mismatches = 0
    low_confidence = 0
    empty_retrieval = 0

    for r in results:
        question = by_id.get(r.question_id)
        if question is None:
            continue

        if r.unsupported_claims:
            unsupported_total += len(r.unsupported_claims)
            questions_with_unsupported += 1

        is_insufficient_evidence_answer = (
            r.generated_answer is not None
            and "could not find sufficient evidence" in r.generated_answer.lower()
        )
        if not r.citations and not is_insufficient_evidence_answer:
            missing_citations += 1

        for citation in r.citations:
            cited_doc = citation.get("document") if isinstance(citation, dict) else getattr(citation, "document", None)
            if cited_doc and cited_doc not in question.expected_documents:
                citation_mismatches += 1

        if r.confidence is not None and r.confidence < LOW_CONFIDENCE_THRESHOLD:
            low_confidence += 1

        if not r.retriever_hits.get("hybrid") and not r.retriever_hits.get("dense"):
            empty_retrieval += 1

    return HallucinationSummary(
        total_questions=len(results),
        unsupported_claims_total=unsupported_total,
        questions_with_unsupported_claims=questions_with_unsupported,
        missing_citation_count=missing_citations,
        citation_mismatch_count=citation_mismatches,
        low_confidence_count=low_confidence,
        empty_retrieval_count=empty_retrieval,
    )
