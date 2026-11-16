import math
import random

import pytest

from ragwrench.evalgate.formats import InputError, ranked
from ragwrench.evalgate.metrics import evaluate, parse_metrics, query_metric


@pytest.mark.parametrize("metric,expected", [
    ("mrr@4", 1 / 2), ("recall@4", 2 / 3), ("precision@4", 2 / 4),
    ("map@4", (1 / 2 + 2 / 3) / 3), ("hit@4", 1),
    ("map@2", (1 / 2) / 3), ("precision@100", 2 / 100),
    ("recall@100", 2 / 3), ("hit@1", 0), ("mrr@1", 0),
    ("ndcg@4", (2 / math.log2(3) + 1 / 2) / (3 + 2 / math.log2(3) + 1 / 2)),
])
def test_hand_computed(metric, expected):
    judgments = {"a": 3, "b": 2, "c": 1, "d": 0}
    assert query_metric(judgments, ["unjudged", "b", "c", "d"], parse_metrics(metric)[0]) == pytest.approx(expected)


def test_exponential_gain():
    judgments = {"a": 3, "b": 2, "c": 1}
    actual = query_metric(judgments, ["c", "b", "a"], parse_metrics("ndcg@3")[0], "exponential")
    assert actual == pytest.approx((1 + 3 / math.log2(3) + 7 / 2) / (7 + 3 / math.log2(3) + 1 / 2))


@pytest.mark.parametrize("name", ["ndcg", "mrr", "recall", "precision", "map", "hit"])
def test_no_relevant_or_no_results(name):
    metric = parse_metrics(name + "@10")[0]
    assert query_metric({"d": 0}, ["d"], metric) == 0
    assert query_metric({"d": 2}, [], metric) == 0
    assert query_metric({}, ["d"], metric) == 0


def test_trec_eval_tie_rule_and_100_shuffles():
    # form_res_rels.c comp_sim_docno: descending score, descending docno.
    # doc10 sorts below doc2: this is lexical, not numeric or input-rank order.
    rows = [("doc1", 1.0), ("doc10", 1.0), ("doc2", 1.0), ("top", 2.0)]
    qrels = {"q": {"doc1": 0, "doc10": 1, "doc2": 2, "top": 0}}
    metrics = parse_metrics("ndcg@2,mrr@2,recall@2,precision@2,map@2,hit@2")
    expected = evaluate(qrels, {"q": dict(rows)}, metrics)
    rng = random.Random(42)
    for _ in range(100):
        rng.shuffle(rows)
        assert ranked(dict(rows)) == ["top", "doc2", "doc10", "doc1"]
        assert evaluate(qrels, {"q": dict(rows)}, metrics) == expected
    assert expected["means"]["ndcg@2"] == pytest.approx((2 / math.log2(3)) / (2 + 1 / math.log2(3)))


def test_query_pairing_and_missing_extra_queries():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}, "missing": {"c": 1}}
    run = {"q2": {"b": 1.0}, "extra": {"a": 2.0}, "q1": {"a": 1.0}}
    metrics = parse_metrics("ndcg@10")
    report = evaluate(qrels, run, metrics)
    assert report["means"]["ndcg@10"] == pytest.approx(2 / 3)
    assert report["missing_queries"] == ["missing"]
    assert report["ignored_queries"] == ["extra"]
    assert report == evaluate(dict(reversed(list(qrels.items()))), dict(reversed(list(run.items()))), metrics)


@pytest.mark.parametrize("text", ["bpref@10", "ndcg", "ndcg@0", "ndcg@-1", "ndcg@1.0", "mrr@1000001", "", "hit@2,hit@2"])
def test_bad_metrics(text):
    with pytest.raises(InputError):
        parse_metrics(text)


@pytest.mark.parametrize("qrels,run", [({}, {"q": {"a": 1.0}}), ({"q": {"a": 1}}, {}), ({"q": {"a": 1}}, {"other": {"a": 1.0}})])
def test_empty_and_no_shared_queries(qrels, run):
    with pytest.raises(InputError, match="share at least one"):
        evaluate(qrels, run, parse_metrics("ndcg@10"))
