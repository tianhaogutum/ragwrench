"""Replayable synthetic cell retrieval; all scores computed, never hard-coded."""
from collections import Counter
import hashlib
import json
import math
import re
from .splitter import split_document
from .tokens import TOKEN_RE, count_tokens
from .models import TableChunkError


def terms(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold(), re.UNICODE)


class BM25:
    """Okapi BM25, k1=1.5, b=0.75, positive Robertson IDF, stable ties."""
    def __init__(self, documents: list[str]):
        self.frequencies = [Counter(terms(doc)) for doc in documents]
        self.lengths = [sum(freq.values()) for freq in self.frequencies]
        self.average = sum(self.lengths) / len(documents) if documents else 1
        self.df = Counter(term for freq in self.frequencies for term in freq)

    def rank(self, query: str) -> list[int]:
        scores = []
        query_terms = sorted(set(terms(query)))
        for index, freq in enumerate(self.frequencies):
            score = 0.0
            for term in query_terms:
                tf = freq.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (len(self.frequencies) - self.df[term] + .5) / (self.df[term] + .5))
                norm = 1.5 * (1 - .75 + .75 * self.lengths[index] / (self.average or 1))
                score += idf * tf * 2.5 / (tf + norm)
            scores.append((index, score))
        return [index for index, _ in sorted(scores, key=lambda item: (-item[1], item[0]))]


def synthetic_fixture(rows: int = 60, tables: int = 4) -> tuple[str, list[dict]]:
    parts, queries = [], []
    for table in range(tables):
        parts.append(f"## Region {table}\n\nPublished subscription plans.\n\n"
                     "| Plan | Seats | Annual cost | Storage GB |\n|---|---|---|---|")
        for row in range(rows):
            plan = f"PLAN{table:02d}{row:04d}"
            seats, cost, storage = str(5 + row), str(1000 + table * rows + row), str(500 + row)
            parts.append(f"| {plan} | {seats} | {cost} | {storage} |")
            for column, answer in (("Seats", seats), ("Annual cost", cost), ("Storage GB", storage)):
                queries.append({"query": f"{plan} {column}", "entity": plan,
                                "column": column, "answer": answer})
        parts.append("")
    return "\n".join(parts).rstrip() + "\n", queries


def naive_chunks(text: str, budget: int) -> list[str]:
    matches = list(TOKEN_RE.finditer(text))
    return [text[matches[i].start():matches[min(i + budget, len(matches)) - 1].end()]
            for i in range(0, len(matches), budget)]


def evaluate(documents: list[str], queries: list[dict], k: int) -> dict:
    index = BM25(documents)
    normalized = [set(terms(doc)) for doc in documents]
    hits, reciprocal, row_hits, answerable = 0, 0.0, 0, 0
    for query in queries:
        # Evidence relevance requires entity, column label and answer in one chunk.
        # This measures self-contained retrieval, not just matching an entity name.
        required = set(terms(query["entity"] + " " + query["column"] + " " + query["answer"]))
        entity = set(terms(query["entity"]))
        relevant = {i for i, tokens in enumerate(normalized) if required <= tokens}
        ranking = index.rank(query["query"])
        answerable += bool(relevant)
        position = next((rank for rank, doc in enumerate(ranking, 1) if doc in relevant), None)
        hits += position is not None and position <= k
        reciprocal += 1 / position if position else 0
        row_hits += any(entity <= normalized[doc] for doc in ranking[:k])
    n = len(queries)
    return {"chunks": len(documents), "queries": n, f"hit@{k}": hits / n,
            "mrr": reciprocal / n, "answerable_fraction": answerable / n,
            f"entity_hit@{k}": row_hits / n,
            "total_chunk_tokens": sum(map(count_tokens, documents)),
            "max_chunk_tokens": max(map(count_tokens, documents), default=0)}


def run_benchmark(*, rows: int = 60, tables: int = 4, max_tokens: int = 120,
                  k: int = 3, corpus: str | None = None, queries: list[dict] | None = None) -> dict:
    if not 1 <= rows <= 500 or not 1 <= tables <= 20 or not 1 <= k <= 100:
        raise TableChunkError("bench requires rows=1..500, tables=1..20, k=1..100")
    if not 40 <= max_tokens <= 10000:
        raise TableChunkError("bench max_tokens must be 40..10000")
    if (corpus is None) != (queries is None):
        raise TableChunkError("supply both corpus and queries for replay")
    if corpus is None:
        corpus, queries = synthetic_fixture(rows, tables)
    if not queries or len(queries) > 30000:
        raise TableChunkError("bench requires 1..30000 queries")
    for query in queries:
        if not isinstance(query, dict) or any(not isinstance(query.get(key), str) or not query[key]
                                             for key in ("query", "entity", "column", "answer")):
            raise TableChunkError("each query needs nonempty query, entity, column, answer strings")
    methods = {"naive_fixed": naive_chunks(corpus, max_tokens)}
    for mode in ("markdown", "kv"):
        methods["table_" + mode] = [chunk.text for chunk in split_document(corpus, max_tokens=max_tokens, mode=mode)]
    return {"schema_version": 1, "corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
            "queries_sha256": hashlib.sha256(json.dumps(queries, sort_keys=True).encode()).hexdigest(),
            "configuration": {"max_tokens": max_tokens, "k": k, "counter": "regex-v1",
                              "bm25": {"k1": 1.5, "b": .75}},
            "relevance": "one chunk contains entity, all column-label terms, and answer; token co-occurrence, not semantic proof",
            "methods": {name: evaluate(docs, queries, k) for name, docs in methods.items()}}
