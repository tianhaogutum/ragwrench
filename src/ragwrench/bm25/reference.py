"""Independent token-space BM25 oracle (no hashing or Qdrant scoring)."""

from __future__ import annotations

import math
from collections import Counter

from .identity import id_key
from .tokenizers import Tokenizer


def bm25_search(
    documents: list[dict],
    query: str,
    *,
    tokenizer: str = "cjk-bigram",
    k1: float = 1.2,
    b: float = 0.75,
    avgdl: float | None = None,
    stopwords: tuple[str, ...] = (),
    top_k: int = 10,
) -> list[dict]:
    tokenize = Tokenizer(tokenizer, stopwords)
    counts = [(doc, Counter(tokenize(doc["text"]))) for doc in documents]
    counts = [(doc, terms) for doc, terms in counts if terms]
    if not counts:
        return []
    average = avgdl if avgdl is not None else sum(sum(t.values()) for _, t in counts) / len(counts)
    terms = set(tokenize(query))
    df = Counter(term for _, frequencies in counts for term in frequencies)
    result = []
    for doc, frequencies in counts:
        score = 0.0
        length = sum(frequencies.values())
        for term in sorted(terms):
            tf = frequencies[term]
            if tf:
                idf = math.log(1 + (len(counts) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average))
        if score > 0:
            result.append({"id": doc["id"], "score": score})
    return sorted(result, key=lambda hit: (-hit["score"], id_key(hit["id"])))[:top_k]
