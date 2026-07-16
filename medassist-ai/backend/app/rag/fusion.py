"""
Reciprocal Rank Fusion (RRF) — combines multiple ranked lists (dense,
keyword, and query-expansion variants) into one ranking, without needing
to normalize incomparable score scales (cosine similarity vs BM25 score).

RRF score for an item = sum over each list it appears in of 1 / (k + rank),
where rank is 1-indexed position in that list. This is the standard
formulation from Cormack et al., 2009 ("Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods").
"""
from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    ranked_lists: list of ranked lists, each [(item_id, original_score), ...]
                  already sorted best-first.
    Returns: [(item_id, fused_score), ...] sorted best-first.
    """
    fused_scores: dict[int, float] = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, (item_id, _original_score) in enumerate(ranked_list, start=1):
            fused_scores[item_id] += 1.0 / (k + rank)

    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
