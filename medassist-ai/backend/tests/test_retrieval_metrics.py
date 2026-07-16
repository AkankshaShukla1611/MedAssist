import pytest

from app.evaluation.retrieval_metrics import (
    grade_relevance, recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k,
    evaluate_ranked_list, aggregate_metrics,
)


def test_grade_relevance_no_match_is_zero():
    assert grade_relevance("Other Doc", 5, ["ADA Guidelines"], [5]) == 0


def test_grade_relevance_document_match_only_is_one():
    assert grade_relevance("ADA Guidelines", 99, ["ADA Guidelines"], [5]) == 1


def test_grade_relevance_document_and_page_match_is_two():
    assert grade_relevance("ADA Guidelines", 5, ["ADA Guidelines"], [5]) == 2


def test_grade_relevance_no_expected_pages_falls_back_to_document_match():
    assert grade_relevance("ADA Guidelines", 5, ["ADA Guidelines"], []) == 1


def test_recall_at_k_perfect():
    # All 3 relevant docs found within top 3
    assert recall_at_k([1, 1, 1], total_relevant=3, k=3) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k([1, 0, 0], total_relevant=3, k=3) == pytest.approx(1 / 3)


def test_recall_at_k_no_relevant_docs_expected():
    assert recall_at_k([1, 1, 1], total_relevant=0, k=3) == 0.0


def test_precision_at_k():
    assert precision_at_k([1, 0, 1], k=3) == pytest.approx(2 / 3)


def test_precision_at_k_empty_hits():
    assert precision_at_k([], k=3) == 0.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank([1, 0, 0]) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank([0, 0, 1]) == pytest.approx(1 / 3)


def test_reciprocal_rank_not_found():
    assert reciprocal_rank([0, 0, 0]) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([2, 1], k=2) == 1.0


def test_ndcg_worse_ranking_scores_lower():
    perfect = ndcg_at_k([2, 1], k=2)
    worse = ndcg_at_k([1, 2], k=2)
    assert worse < perfect


def test_ndcg_no_relevant_items_is_zero():
    assert ndcg_at_k([0, 0], k=2) == 0.0


def test_evaluate_ranked_list_returns_all_requested_k_values():
    result = evaluate_ranked_list(
        document_titles_in_rank_order=["ADA Guidelines", "Other", "WHO Guidelines"],
        page_numbers_in_rank_order=[34, None, 12],
        expected_documents=["ADA Guidelines", "WHO Guidelines"],
        expected_pages=[34],
        k_values=[1, 3],
    )
    assert "recall_at_1" in result
    assert "recall_at_3" in result
    assert "precision_at_1" in result
    assert "ndcg_at_1" in result
    assert "mrr" in result
    # Both expected documents appear within top 3 -> full recall@3
    assert result["recall_at_3"] == 1.0
    # First result matches document AND page -> best possible MRR
    assert result["mrr"] == 1.0


def test_aggregate_metrics_averages_across_questions():
    per_question = [
        {"recall_at_5": 1.0, "mrr": 1.0},
        {"recall_at_5": 0.0, "mrr": 0.0},
    ]
    result = aggregate_metrics(per_question)
    assert result["recall_at_5"] == 0.5
    assert result["mrr"] == 0.5


def test_aggregate_metrics_empty_input_returns_empty_dict():
    assert aggregate_metrics([]) == {}
