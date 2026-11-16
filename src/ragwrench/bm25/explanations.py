"""BM25 explanations grouped by the actual sparse scoring dimensions."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from qdrant_client import models

from .encoder import Encoder, token_hash
from .identity import point_id


def explain_scores(
    encoder: Encoder,
    query: str,
    hits: list[dict],
    document_count: int,
    frequencies: Counter[int],
    vectors: dict[str, models.SparseVector],
) -> dict:
    tokens = encoder.tokenizer(query)
    query_terms: dict[int, list[str]] = defaultdict(list)
    for token in sorted(set(tokens)):
        query_terms[token_hash(token)].append(token)
    dimensions = [
        {
            "index": index,
            "tokens": query_terms[index],
            "query_weight": 1.0,
            "document_frequency": frequencies[index],
            "idf": math.log(
                1 + (document_count - frequencies[index] + 0.5) / (frequencies[index] + 0.5)
            ),
        }
        for index in sorted(query_terms)
    ]
    results = []
    for hit in hits:
        doc_tokens = encoder.tokenizer(hit["text"])
        counts = Counter(doc_tokens)
        terms: dict[int, list[dict]] = defaultdict(list)
        for token, frequency in sorted(counts.items()):
            index = token_hash(token)
            if index in query_terms:
                tf = (
                    frequency
                    * (encoder.k1 + 1)
                    / (
                        frequency
                        + encoder.k1 * (1 - encoder.b + encoder.b * len(doc_tokens) / encoder.avgdl)
                    )
                )
                terms[index].append({"token": token, "frequency": frequency, "tf_weight": tf})
        vector = vectors[point_id(hit["id"])]
        weights = dict(zip(vector.indices, vector.values))
        contributions = [
            {
                "index": dimension["index"],
                "document_terms": terms[dimension["index"]],
                "query_weight": dimension["query_weight"],
                "idf": dimension["idf"],
                "tf_weight": weights.get(dimension["index"], 0.0),
                "contribution": weights.get(dimension["index"], 0.0) * dimension["idf"],
            }
            for dimension in dimensions
        ]
        score_sum = math.fsum(d["contribution"] for d in contributions)
        results.append(
            {
                **hit,
                "explanation": {
                    "tokens": doc_tokens,
                    "term_frequencies": dict(sorted(counts.items())),
                    "document_length": len(doc_tokens),
                    "dimensions": contributions,
                    "score_sum": score_sum,
                    "score_delta": score_sum - hit["score"],
                },
            }
        )
    return {
        "schema": 1,
        "config": encoder.config,
        "corpus": {"document_count": document_count},
        "query": {
            "text": query,
            "tokens": tokens,
            "term_frequencies": dict(sorted(Counter(tokens).items())),
            "dimensions": dimensions,
        },
        "results": results,
    }
