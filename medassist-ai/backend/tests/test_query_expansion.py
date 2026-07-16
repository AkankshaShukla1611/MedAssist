import os
os.environ.setdefault("JWT_SECRET", "test-secret-key-for-pytest-only-do-not-use-in-prod")

from app.rag.query_expansion import expand_query


def test_expands_known_abbreviation():
    variants = expand_query("first-line treatment for T2DM")
    assert "first-line treatment for t2dm" in [v.lower() for v in variants] or len(variants) > 1
    assert any("type 2 diabetes" in v.lower() for v in variants)


def test_original_query_always_first():
    question = "What is HTN management?"
    variants = expand_query(question)
    assert variants[0] == question


def test_no_match_returns_only_original():
    variants = expand_query("xyzzy nonsense query with no medical terms")
    assert variants == ["xyzzy nonsense query with no medical terms"]


def test_word_boundary_prevents_false_match():
    # "ra" (rheumatoid arthritis) should NOT match inside "administration"
    variants = expand_query("medication administration guidelines")
    assert len(variants) == 1


def test_respects_max_expansion_limit():
    from app.core.config import settings
    # A query hitting many abbreviations should still be capped.
    variants = expand_query("bp hr rr temp sob cp iv im po")
    assert len(variants) <= settings.MAX_QUERY_EXPANSION_TERMS
