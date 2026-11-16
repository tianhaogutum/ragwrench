"""Controlled offline ranking improvement, measured on 40 synthetic queries."""
import json

from ragwrench.evalgate.compare import compare
from ragwrench.evalgate.metrics import parse_metrics


def main():
    qrels, baseline, candidate = {}, {}, {}
    for i in range(40):
        qid = f"q{i:02d}"
        qrels[qid] = {"relevant": 1, "noise": 0}
        baseline[qid] = {"noise": 2.0, "relevant": 1.0}
        candidate[qid] = {"relevant": 2.0, "noise": 1.0}
    report = compare(qrels, baseline, candidate, parse_metrics("ndcg@10,mrr@10"),
                     seed=42, samples=5000, thresholds={"ndcg@10": 0.0})
    assert report["passed"]
    for row in report["metrics"].values():
        assert row["candidate"] == 1.0 and row["wins"] == 40 and row["p_value"] < 0.01
    print(json.dumps({"queries": 40, "change": "move the relevant document from rank 2 to rank 1",
                      "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
