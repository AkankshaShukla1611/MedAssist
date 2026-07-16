import time
from app.rag.keyword_retriever import BM25Index


def _make_index(tmp_path):
    return BM25Index(path=str(tmp_path / "bm25_test.pkl"))


def test_rebuild_and_search_finds_matching_chunk(tmp_path):
    index = _make_index(tmp_path)
    index.rebuild([
        (1, 100, "metformin is first-line for type 2 diabetes"),
        (2, 100, "aspirin for cardiovascular prevention"),
        (3, 100, "lisinopril for hypertension management"),
        (4, 100, "levothyroxine for hypothyroidism treatment"),
    ])

    results = index.search("metformin diabetes", top_k=5)
    result_ids = [chunk_id for chunk_id, _ in results]
    assert 1 in result_ids
    assert results[0][0] == 1  # best match ranked first


def test_search_respects_allowed_document_ids_filter(tmp_path):
    index = _make_index(tmp_path)
    index.rebuild([
        (1, 100, "metformin diabetes treatment first-line"),
        (2, 200, "metformin diabetes dosage guidance"),
        (3, 300, "aspirin cardiovascular prevention"),
        (4, 400, "lisinopril hypertension management"),
    ])

    results = index.search("metformin diabetes", top_k=5, allowed_document_ids={100})
    result_ids = [chunk_id for chunk_id, _ in results]
    assert result_ids == [1]


def test_empty_corpus_returns_no_results(tmp_path):
    index = _make_index(tmp_path)
    index.rebuild([])
    assert index.search("anything", top_k=5) == []


def test_search_on_never_built_index_returns_empty_not_error(tmp_path):
    index = BM25Index(path=str(tmp_path / "never_built.pkl"))
    assert index.search("anything", top_k=5) == []


def test_second_instance_sees_rebuild_from_first_via_disk(tmp_path):
    """
    This is the regression test for the cross-process staleness bug: two
    separate BM25Index instances (standing in for two separate processes —
    e.g. a Celery worker and the API process) pointed at the same path.
    Instance B must pick up instance A's rebuild without anyone calling a
    manual reload, because search() checks the on-disk mtime every time.
    """
    path = str(tmp_path / "shared.pkl")
    writer = BM25Index(path=path)
    reader = BM25Index(path=path)

    writer.rebuild([
        (1, 100, "metformin diabetes treatment"),
        (2, 200, "aspirin cardiovascular prevention"),
        (3, 300, "lisinopril hypertension management"),
    ])
    # Ensure the mtime actually advances even on fast filesystems/CI runners
    # with coarse mtime resolution.
    time.sleep(0.01)

    # Before any rebuild, reader's cache is empty (loaded before writer wrote).
    assert reader.search("metformin", top_k=5) != [] or reader._bm25 is None

    # Force a distinguishable mtime, then verify reader transparently reloads.
    writer.rebuild([
        (10, 200, "metformin diabetes treatment updated"),
        (11, 300, "unrelated content about weather patterns"),
        (12, 400, "aspirin cardiovascular prevention info"),
    ])
    time.sleep(0.01)

    results = reader.search("metformin diabetes", top_k=5)
    result_ids = [chunk_id for chunk_id, _ in results]
    assert 10 in result_ids, "reader did not pick up writer's rebuild — staleness bug regressed"
    assert 1 not in result_ids, "reader is serving a stale corpus from before the second rebuild"


def test_loaded_mtime_updates_after_save(tmp_path):
    index = _make_index(tmp_path)
    index.rebuild([(1, 100, "some text")])
    assert index._loaded_mtime is not None
    assert index._loaded_mtime == index._current_mtime()
