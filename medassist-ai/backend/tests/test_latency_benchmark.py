import pytest

from app.evaluation.latency_benchmark import percentile, summarize_latencies


def test_percentile_empty_list_returns_zero():
    assert percentile([], 50) == 0.0


def test_percentile_single_value():
    assert percentile([42.0], 95) == 42.0


def test_percentile_p50_of_sorted_range():
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 50) == 30


def test_percentile_p100_is_max():
    values = [10, 20, 30]
    assert percentile(values, 100) == 30


def test_percentile_p0_is_min():
    values = [10, 20, 30]
    assert percentile(values, 0) == 10


def test_summarize_latencies_computes_all_stats():
    summary = summarize_latencies({"llm_ms": [100.0, 200.0, 300.0]})
    assert summary["llm_ms"]["count"] == 3
    assert summary["llm_ms"]["mean"] == 200.0
    assert summary["llm_ms"]["p50"] == 200.0


def test_summarize_latencies_handles_empty_stage():
    summary = summarize_latencies({"empty_stage": []})
    assert summary["empty_stage"]["count"] == 0
    assert summary["empty_stage"]["p50"] == 0.0
