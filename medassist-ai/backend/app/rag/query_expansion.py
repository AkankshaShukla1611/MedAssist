"""
Expands a clinical query with synonyms/abbreviations before retrieval, so
"first-line treatment for T2DM" also retrieves passages that say "type 2
diabetes mellitus" and vice versa.

Approach: dictionary-based expansion (curated JSON, not an LLM call) —
deterministic, fast (no extra network/model round trip), and auditable,
which matters more than cleverness for a clinical tool. An LLM-based
paraphrase-expansion is a documented future upgrade if the dictionary proves
too narrow in practice.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

from app.core.config import settings

_DICT_PATH = Path(__file__).parent.parent / "data" / "medical_abbreviations.json"


@lru_cache(maxsize=1)
def _load_dictionary() -> dict[str, list[str]]:
    with open(_DICT_PATH, "r") as f:
        return json.load(f)


def expand_query(question: str) -> list[str]:
    """
    Returns a list of query variants: [original, expansion_1, expansion_2, ...]
    capped at settings.MAX_QUERY_EXPANSION_TERMS variants total (including original).
    """
    variants = [question]
    if not settings.QUERY_EXPANSION_ENABLED:
        return variants

    dictionary = _load_dictionary()
    lower_q = question.lower()
    matched_terms: set[str] = set()

    for term, expansions in dictionary.items():
        # Word-boundary match so "ra" doesn't match inside "administra..." etc.
        pattern = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
        if re.search(pattern, lower_q):
            for expansion in expansions:
                if expansion in matched_terms:
                    continue
                matched_terms.add(expansion)
                expanded = re.sub(pattern, expansion, lower_q, flags=re.IGNORECASE)
                variants.append(expanded)
                if len(variants) >= settings.MAX_QUERY_EXPANSION_TERMS:
                    return variants

    return variants
