from app.rag.fusion import reciprocal_rank_fusion


def test_single_list_preserves_order():
    ranked = [(1, 0.9), (2, 0.8), (3, 0.7)]
    fused = reciprocal_rank_fusion([ranked], k=60)
    fused_ids = [item_id for item_id, _ in fused]
    assert fused_ids == [1, 2, 3]


def test_item_top_in_both_lists_wins():
    dense = [(10, 0.9), (20, 0.8), (30, 0.7)]
    keyword = [(10, 5.0), (30, 4.0), (20, 3.0)]
    # item 10 is rank 1 in BOTH lists -> unambiguously the strongest RRF candidate.
    fused = reciprocal_rank_fusion([dense, keyword], k=60)
    fused_ids = [item_id for item_id, _ in fused]
    assert set(fused_ids) == {10, 20, 30}
    assert fused_ids[0] == 10


def test_item_only_in_one_list_ranks_below_items_in_both():
    list_a = [(1, 0.9), (2, 0.8)]
    list_b = [(1, 5.0), (2, 4.0)]
    list_c_only = [(3, 9.0)]  # item 3 is top of a THIRD list but absent from the other two
    fused = reciprocal_rank_fusion([list_a, list_b, list_c_only], k=60)
    fused_ids = [item_id for item_id, _ in fused]
    # item 1 (present + top-ranked in 2 of 3 lists) should outrank item 3 (top of only 1 list).
    assert fused_ids.index(1) < fused_ids.index(3)


def test_empty_lists_return_empty():
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_disjoint_lists_all_present():
    list_a = [(1, 0.9)]
    list_b = [(2, 0.9)]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    assert {i for i, _ in fused} == {1, 2}
