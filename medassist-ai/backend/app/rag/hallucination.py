"""
Unsupported-claim detection: splits the generated answer into sentences and
checks each one against the retrieved context for lexical support. Any
sentence that doesn't overlap enough with ANY retrieved chunk is flagged as
"unsupported" and surfaced to the user rather than silently trusted.

Method: token-overlap (Jaccard-style) similarity between each generated
sentence and each context chunk, keeping the best match per sentence. This
is deliberately lightweight (no extra model/network call) so it runs on
every request cheaply. An NLI-based entailment model (e.g.
cross-encoder/nli-deberta) is a documented upgrade path if lexical overlap
proves too coarse in practice — swap the `_sentence_support` implementation
without touching the calling code.
"""
import re

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in",
    "on", "for", "and", "or", "with", "as", "by", "at", "this", "that", "it", "its", "their",
    "these", "those", "from", "which", "may", "can", "should", "will", "not",
}


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


def split_sentences(text: str) -> list[str]:
    # Simple, dependency-free sentence splitter — good enough for clinical
    # prose which is mostly declarative sentences ending in '.', '?', '!'.
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 5]


def _sentence_support(sentence: str, context_texts: list[str]) -> float:
    sentence_tokens = _tokenize(sentence)
    if not sentence_tokens:
        return 1.0  # nothing substantive to fact-check (e.g. a heading fragment)

    best = 0.0
    for context in context_texts:
        context_tokens = _tokenize(context)
        if not context_tokens:
            continue
        overlap = len(sentence_tokens & context_tokens)
        support = overlap / len(sentence_tokens)
        best = max(best, support)
    return best


def detect_unsupported_claims(answer: str, context_texts: list[str], threshold: float) -> dict:
    """
    Returns {"unsupported_sentences": [...], "support_ratio": float}
    support_ratio = fraction of sentences that ARE adequately supported
    (used directly as one signal in confidence calibration).
    """
    sentences = split_sentences(answer)
    if not sentences:
        return {"unsupported_sentences": [], "support_ratio": 1.0}

    unsupported = []
    supported_count = 0
    for sentence in sentences:
        score = _sentence_support(sentence, context_texts)
        if score >= threshold:
            supported_count += 1
        else:
            unsupported.append(sentence)

    support_ratio = supported_count / len(sentences)
    return {"unsupported_sentences": unsupported, "support_ratio": support_ratio}
