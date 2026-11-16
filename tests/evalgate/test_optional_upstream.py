"""Optional independent cross-check; not a runtime or default dev dependency."""
import pytest

from ragwrench.evalgate.metrics import evaluate, parse_metrics


def test_pytrec_eval_when_installed():
    pytrec_eval = pytest.importorskip("pytrec_eval")
    qrels = {"q1": {"a": 2, "b": 1, "c": 0}, "q2": {"x": 1, "y": 0}}
    # q1 explicitly exercises tie breaking, with relevant b ahead of relevant a.
    run = {"q2": {"y": 2.0, "x": 1.0}, "q1": {"a": 1.0, "b": 1.0, "c": 2.0}}
    measures = {"ndcg_cut_2", "map_cut_2", "recall_2", "P_2", "recip_rank"}
    reference = pytrec_eval.RelevanceEvaluator(qrels, measures).evaluate(run)
    actual = evaluate(qrels, run, parse_metrics("ndcg@2,map@2,recall@2,precision@2,mrr@100"))["per_query"]
    mapping = {"ndcg@2": "ndcg_cut_2", "map@2": "map_cut_2", "recall@2": "recall_2", "precision@2": "P_2", "mrr@100": "recip_rank"}
    for qid in qrels:
        for metric, upstream in mapping.items():
            assert actual[qid][metric] == pytest.approx(reference[qid][upstream], abs=1e-12)


def test_ranx_when_installed():
    ranx = pytest.importorskip("ranx")
    # Deliberately no tied scores; tie semantics differ across ranx versions.
    qrels = {"q": {"a": 2, "b": 1, "c": 0}}
    run = {"q": {"a": 3.0, "c": 2.0, "b": 1.0}}
    ref = ranx.evaluate(ranx.Qrels(qrels), ranx.Run(run), ["ndcg@3", "recall@3", "mrr@3"])
    actual = evaluate(qrels, run, parse_metrics("ndcg@3,recall@3,mrr@3"))["means"]
    for metric in actual:
        assert actual[metric] == pytest.approx(ref[metric])
