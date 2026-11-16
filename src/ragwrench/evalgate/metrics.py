"""Metric definitions on the complete, sorted qrels query universe."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .formats import InputError, Qrels, Run, ranked

TIE_RULE = "score descending, document ID lexicographically descending; input ranks ignored"


@dataclass(frozen=True)
class Metric:
    name: str
    k: int

    def __str__(self):
        return f"{self.name}@{self.k}"


def parse_metrics(text: str) -> list[Metric]:
    result = []
    for token in text.split(","):
        match = re.fullmatch(r"(ndcg|mrr|recall|precision|map|hit)@([1-9][0-9]{0,6})", token.strip())
        if not match or int(match[2]) > 1_000_000:
            raise InputError(f"invalid metric {token!r}; use ndcg/mrr/recall/precision/map/hit@k, 1 <= k <= 1000000")
        metric = Metric(match[1], int(match[2]))
        if metric in result:
            raise InputError(f"duplicate metric: {metric}")
        result.append(metric)
    return result


def validate_metrics(metrics):
    """Require one named score/hypothesis per metric for API callers too."""
    if not metrics:
        raise InputError("at least one metric is required")
    seen = set()
    for metric in metrics:
        key = str(metric)
        if key in seen:
            raise InputError(f"duplicate metric: {key}")
        seen.add(key)


def query_metric(judgments: dict[str, int], ranking: list[str], metric: Metric, gain="linear") -> float:
    rels = [judgments.get(docid, 0) for docid in ranking[:metric.k]]
    positives = [i for i, rel in enumerate(rels, 1) if rel > 0]
    total = sum(rel > 0 for rel in judgments.values())
    if metric.name == "precision":
        return len(positives) / metric.k
    if metric.name == "recall":
        return len(positives) / total if total else 0.0
    if metric.name == "hit":
        return float(bool(positives))
    if metric.name == "mrr":
        return 1.0 / positives[0] if positives else 0.0
    if metric.name == "map":
        return math.fsum(n / rank for n, rank in enumerate(positives, 1)) / total if total else 0.0
    if metric.name == "ndcg":
        transform = (lambda rel: float(rel)) if gain == "linear" else (lambda rel: float(2 ** rel - 1))
        dcg = math.fsum(transform(rel) / math.log2(i + 2) for i, rel in enumerate(rels))
        ideal = math.fsum(transform(rel) / math.log2(i + 2) for i, rel in enumerate(sorted(judgments.values(), reverse=True)[:metric.k]))
        return dcg / ideal if ideal else 0.0
    raise InputError(f"unknown metric: {metric.name}")


def evaluate(qrels: Qrels, run: Run, metrics: list[Metric], gain="linear") -> dict:
    validate_metrics(metrics)
    if gain not in {"linear", "exponential"}:
        raise InputError("gain must be linear or exponential")
    if not qrels or not run or not (qrels.keys() & run.keys()):
        raise InputError("qrels/run must be nonempty and share at least one query ID")
    per_query = {}
    for qid in sorted(qrels):
        ranking = ranked(run.get(qid, {}))
        per_query[qid] = {str(m): query_metric(qrels[qid], ranking, m, gain) for m in metrics}
    means = {str(m): math.fsum(row[str(m)] for row in per_query.values()) / len(qrels) for m in metrics}
    return {
        "query_count": len(qrels),
        "gain": gain,
        "tie_rule": TIE_RULE,
        "means": means,
        "per_query": per_query,
        "missing_queries": sorted(qrels.keys() - run.keys()),
        "ignored_queries": sorted(run.keys() - qrels.keys()),
    }
