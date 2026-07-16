"""
Loads and validates the benchmark dataset (JSONL). Kept as a pure function
with no DB/network dependency so it can be unit-tested trivially and reused
by both the CLI evaluation runner and the admin API.
"""
import hashlib
import json
from pathlib import Path

from app.evaluation.schemas import BenchmarkQuestion

DEFAULT_DATASET_PATH = Path(__file__).parent.parent / "data" / "benchmark_dataset.jsonl"


def load_benchmark(path: str | Path | None = None, max_questions: int | None = None) -> list[BenchmarkQuestion]:
    path = Path(path) if path else DEFAULT_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at {path}")

    questions = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                questions.append(BenchmarkQuestion(**data))
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                raise ValueError(f"Invalid benchmark entry at line {line_num} of {path}: {e}")

    if not questions:
        raise ValueError(f"Benchmark dataset at {path} is empty")

    if max_questions:
        questions = questions[:max_questions]

    return questions


def dataset_version(path: str | Path | None = None) -> str:
    """
    Content-hash based version identifier — changes automatically whenever
    the dataset file changes, no manual version bumping required, and it's
    reproducible (same content -> same version string) which matters for
    comparing evaluation runs over time.
    """
    path = Path(path) if path else DEFAULT_DATASET_PATH
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:12]
