"""
Standard IR metrics, implemented from scratch (no heavy IR library
dependency) so they're easy to audit and test.

Relevance grading: a retrieved chunk is graded relevant to a benchmark
question based on whether its source document matches `expected_documents`
(grade 1) and, if `expected_pages` is also given, whether the page matches
too (grade 2 = a more precise match). This graded relevance feeds NDCG;
the binary versions (grade > 0) feed Recall/Precision/MRR.
"""
import math


def grade_relevance(document_title: str, page_number: int | None, expected_documents: list[str], expected_pages: list[int]) -> int:
    if document_title not in expected_documents:
        return 0
    if expected_pages and page_number in expected_pages:
        return 2
    return 1


def recall_at_k(graded_hits: list[int], total_relevant: int, k: int) -> float:
    """graded_hits: relevance grades (0/1/2) for the top-K retrieved items, in rank order."""
    if total_relevant == 0:
        return 0.0
    relevant_found = sum(1 for g in graded_hits[:k] if g > 0)
    return relevant_found / total_relevant


def precision_at_k(graded_hits: list[int], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = graded_hits[:k]
    relevant_found = sum(1 for g in top_k if g > 0)
    return relevant_found / min(k, len(top_k)) if top_k else 0.0


def reciprocal_rank(graded_hits: list[int]) -> float:
    for rank, grade in enumerate(graded_hits, start=1):
        if grade > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(graded_hits: list[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain with graded relevance (0/1/2)."""
    def dcg(grades: list[int]) -> float:
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))

    actual = dcg(graded_hits[:k])
    ideal_order = sorted(graded_hits, reverse=True)[:k]
    ideal = dcg(ideal_order)
    if ideal == 0:
        return 0.0
    return actual / ideal


def evaluate_ranked_list(
    document_titles_in_rank_order: list[str],
    page_numbers_in_rank_order: list[int | None],
    expected_documents: list[str],
    expected_pages: list[int],
    k_values: list[int] = [5, 10],
) -> dict:
    """
    Computes all metrics for one question's ranked retrieval results.
    Returns a flat dict, e.g. {"recall_at_5": ..., "recall_at_10": ...,
    "precision_at_5": ..., "mrr": ..., "ndcg_at_5": ..., "ndcg_at_10": ...}
    """
    graded = [
        grade_relevance(title, page, expected_documents, expected_pages)
        for title, page in zip(document_titles_in_rank_order, page_numbers_in_rank_order)
    ]
    # total_relevant: for recall, count how many expected documents exist at all
    # (a document can satisfy the expectation via any one of its matching chunks).
    total_relevant = len(expected_documents) if expected_documents else 0

    result = {"mrr": reciprocal_rank(graded)}
    for k in k_values:
        result[f"recall_at_{k}"] = recall_at_k(graded, total_relevant, k)
        result[f"precision_at_{k}"] = precision_at_k(graded, k)
        result[f"ndcg_at_{k}"] = ndcg_at_k(graded, k)
    return result


def aggregate_metrics(per_question_metrics: list[dict]) -> dict:
    """Averages metrics across all evaluated questions."""
    if not per_question_metrics:
        return {}
    keys = per_question_metrics[0].keys()
    return {
        key: round(sum(m[key] for m in per_question_metrics) / len(per_question_metrics), 4)
        for key in keys
    }
