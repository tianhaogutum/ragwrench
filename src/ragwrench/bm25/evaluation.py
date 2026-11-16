"""Reproducible offline tokenizer comparison using actual Qdrant local retrieval."""

from __future__ import annotations

import hashlib
import json
from contextlib import closing
from importlib.metadata import version
from importlib.resources import files

from qdrant_client import QdrantClient

from .reference import bm25_search
from .store import build_collection, read_corpus, search_collection


def metrics(ranked: list[str], relevant: list[str]) -> dict:
    wanted = set(relevant)
    recall = len(set(ranked) & wanted) / len(wanted)
    reciprocal_rank = next((1 / rank for rank, doc in enumerate(ranked, 1) if doc in wanted), 0.0)
    return {"recall": recall, "mrr": reciprocal_rank}


def evaluate(*, top_k: int = 3, include_jieba: bool = False) -> dict:
    if type(top_k) is not int or not 1 <= top_k <= 1000:
        raise ValueError("top-k must be between 1 and 1000")
    data = files("ragwrench.bm25").joinpath("data")
    corpus_bytes = data.joinpath("corpus.jsonl").read_bytes()
    documents = read_corpus(str(data.joinpath("corpus.jsonl")))
    queries_bytes = data.joinpath("queries.json").read_bytes()
    queries = json.loads(queries_bytes)
    report = {
        "schema": 1,
        "fixture_sha256": hashlib.sha256(corpus_bytes + queries_bytes).hexdigest(),
        "qdrant_client_version": version("qdrant-client"),
        "documents": len(documents),
        "queries": len(queries),
        "top_k": top_k,
        "tokenizers": {},
    }
    names = ["whitespace", "unicode", "cjk-bigram"] + (["jieba"] if include_jieba else [])
    for name in names:
        with closing(QdrantClient(":memory:")) as client:
            manifest = build_collection(client, "evaluation", documents, tokenizer=name)
            runs = []
            max_error = 0.0
            for query in queries:
                # Fetch all matches before truncating for deterministic evaluation ties.
                hits = search_collection(client, "evaluation", query["text"], top_k=len(documents))
                reference = bm25_search(
                    documents, query["text"], tokenizer=name, top_k=len(documents)
                )
                actual = {hit["id"]: hit["score"] for hit in hits}
                expected = {hit["id"]: hit["score"] for hit in reference}
                if actual.keys() != expected.keys():
                    raise RuntimeError(f"reference match-set mismatch for {name}/{query['id']}")
                error = max((abs(actual[key] - expected[key]) for key in actual), default=0.0)
                max_error = max(max_error, error)
                if error > 1e-5:
                    raise RuntimeError(
                        f"reference score mismatch for {name}/{query['id']}: {error}"
                    )
                ranked = [hit["id"] for hit in hits[:top_k]]
                runs.append(
                    {
                        "query_id": query["id"],
                        "lang": query["lang"],
                        "ranked": ranked,
                        **metrics(ranked, query["relevant"]),
                    }
                )

            def average(rows):
                return {
                    f"recall@{top_k}": sum(row["recall"] for row in rows) / len(rows),
                    "mrr": sum(row["mrr"] for row in rows) / len(rows),
                }

            report["tokenizers"][name] = {
                "config": manifest["encoder"],
                "overall": average(runs),
                "by_language": {
                    lang: average([run for run in runs if run["lang"] == lang])
                    for lang in ("zh", "ja", "mixed")
                },
                "reference_max_abs_score_error": max_error,
                "collision_stats": manifest["collision_stats"],
                "runs": runs,
            }
    return report
