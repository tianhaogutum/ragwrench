"""Reproducible regression experiment: stable ranking vs input-order ties."""
import json
import random

from ragwrench.evalgate.formats import ranked
from ragwrench.evalgate.metrics import evaluate, parse_metrics, query_metric


def main():
    judgments = {"doc-a": 0, "doc-z": 2, "doc-b": 1}
    rows = [(doc, 1.0) for doc in judgments]
    metrics = parse_metrics("ndcg@2,mrr@2,recall@2,precision@2,map@2,hit@2")
    rng = random.Random(42)
    stable, input_order = set(), set()
    for _ in range(100):
        rng.shuffle(rows)
        run = {"q": dict(rows)}
        result = evaluate({"q": judgments}, run, metrics)
        stable.add(json.dumps(result["means"], sort_keys=True))
        # Explicitly model the naive stable-score sort whose ties use file order.
        naive = [doc for doc, _ in sorted(rows, key=lambda row: row[1], reverse=True)]
        input_order.add(round(query_metric(judgments, naive, metrics[0]), 12))
    assert ranked(dict(rows)) == ["doc-z", "doc-b", "doc-a"]
    assert len(stable) == 1 and len(input_order) > 1
    print(json.dumps({"seed": 42, "shuffles": 100, "retgate_distinct_metric_vectors": len(stable),
                      "input_order_distinct_ndcg_values": len(input_order),
                      "retgate_metrics": json.loads(next(iter(stable))),
                      "input_order_ndcg_values": sorted(input_order)}, indent=2))


if __name__ == "__main__":
    main()
