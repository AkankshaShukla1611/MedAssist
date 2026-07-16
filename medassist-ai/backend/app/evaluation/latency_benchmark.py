"""
Percentile statistics over collected per-stage latencies. Pure math, no
dependencies beyond the standard library, so it's trivially unit-testable.
"""
import math


def percentile(values: list[float], pct: float) -> float:
    """pct in [0, 100]. Uses linear interpolation (same convention as numpy's default)."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (pct / 100) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize_latencies(stage_latencies: dict[str, list[float]]) -> dict:
    """
    stage_latencies: {"embedding_ms": [...], "retrieval_ms": [...], ...}
    Returns: {"embedding_ms": {"p50": ..., "p95": ..., "p99": ..., "mean": ..., "count": ...}, ...}
    """
    summary = {}
    for stage, values in stage_latencies.items():
        if not values:
            summary[stage] = {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "count": 0}
            continue
        summary[stage] = {
            "p50": round(percentile(values, 50), 2),
            "p95": round(percentile(values, 95), 2),
            "p99": round(percentile(values, 99), 2),
            "mean": round(sum(values) / len(values), 2),
            "count": len(values),
        }
    return summary
