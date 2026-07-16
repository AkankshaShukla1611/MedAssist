import json
import pytest

from app.evaluation.benchmark_loader import load_benchmark, dataset_version, DEFAULT_DATASET_PATH
from app.evaluation.schemas import BenchmarkQuestion


def test_default_dataset_loads_at_least_50_questions():
    questions = load_benchmark()
    assert len(questions) >= 50
    assert all(isinstance(q, BenchmarkQuestion) for q in questions)


def test_default_dataset_has_no_duplicate_ids():
    questions = load_benchmark()
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids)), "benchmark dataset contains duplicate question ids"


def test_default_dataset_covers_multiple_specialties():
    questions = load_benchmark()
    specialties = {q.specialty for q in questions}
    assert len(specialties) >= 5


def test_default_dataset_difficulty_values_are_valid():
    questions = load_benchmark()
    assert all(q.difficulty in {"easy", "medium", "hard"} for q in questions)


def test_max_questions_caps_result():
    questions = load_benchmark(max_questions=5)
    assert len(questions) == 5


def test_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_benchmark(path="/nonexistent/path/dataset.jsonl")


def test_malformed_line_raises_value_error(tmp_path):
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text('{"id": "x"}\n')  # missing required fields
    with pytest.raises(ValueError):
        load_benchmark(path=bad_file)


def test_empty_file_raises_value_error(tmp_path):
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("")
    with pytest.raises(ValueError):
        load_benchmark(path=empty_file)


def test_dataset_version_is_deterministic():
    v1 = dataset_version()
    v2 = dataset_version()
    assert v1 == v2
    assert len(v1) == 12


def test_dataset_version_changes_with_content(tmp_path):
    file_a = tmp_path / "a.jsonl"
    file_b = tmp_path / "b.jsonl"
    file_a.write_text('{"a": 1}\n')
    file_b.write_text('{"a": 2}\n')
    assert dataset_version(file_a) != dataset_version(file_b)
