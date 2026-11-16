"""Metric names must identify exactly one hypothesis, including for API callers."""
import copy
import json
import xml.etree.ElementTree as ET

import pytest

from ragwrench.evalgate.compare import compare, compare_many, compare_many_saved
from ragwrench.evalgate.formats import InputError
from ragwrench.evalgate.metrics import evaluate, parse_metrics
from ragwrench.evalgate.report import render
from ragwrench.evalgate.saved import saved_evaluation


@pytest.fixture
def inputs():
    qrels = {"q1": {"d": 1}, "q2": {"d": 1}}
    base = {q: {"d": 1.0} for q in qrels}
    cand = {"q2": {"d": 1.0}}
    return qrels, base, cand


@pytest.mark.parametrize("entry", ["evaluate", "saved_evaluation", "compare", "compare_many"])
@pytest.mark.parametrize("kind", ["empty", "same_instance", "equal_definition"])
def test_api_metric_sequence_rejected_before_scoring(inputs, entry, kind, monkeypatch):
    qrels, base, cand = inputs
    metrics = parse_metrics("hit@1,recall@1")
    if kind == "empty":
        metrics = []
        error = "at least one metric is required"
    else:
        metrics.append(metrics[0] if kind == "same_instance" else parse_metrics("hit@1")[0])
        error = "duplicate metric: hit@1"
    before = list(metrics)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid metric sequence must reject before scoring or inference")

    monkeypatch.setattr("ragwrench.evalgate.metrics.query_metric", forbidden)
    monkeypatch.setattr("ragwrench.evalgate.compare.paired_test", forbidden)
    with pytest.raises(InputError, match=error):
        if entry == "evaluate":
            evaluate(qrels, base, metrics)
        elif entry == "saved_evaluation":
            saved_evaluation(qrels, base, metrics)
        elif entry == "compare":
            compare(qrels, base, cand, metrics, method="ttest")
        else:
            compare_many(qrels, base, {"same": base, "off": cand}, metrics,
                         correction="holm", method="ttest")
    assert metrics == before


@pytest.mark.parametrize("method", ["ttest", "bootstrap", "randomization"])
@pytest.mark.parametrize("limits", [{}, {"max_family_pairs": 3},
    {"max_family_pairs": 4, "max_family_resamples": 1},
    {"max_family_pairs": 6, "max_family_resamples": 1200}])
def test_duplicate_metrics_reject_before_family_evaluation_or_limits(inputs, method, limits, monkeypatch):
    qrels, base, cand = inputs
    metrics = parse_metrics("hit@1,recall@1") + parse_metrics("hit@1")

    def forbidden(*args, **kwargs):
        pytest.fail("duplicate metrics must reject before evaluating family inputs")

    monkeypatch.setattr("ragwrench.evalgate.compare.evaluate", forbidden)
    monkeypatch.setattr("ragwrench.evalgate.compare.paired_test", forbidden)
    with pytest.raises(InputError, match="duplicate metric: hit@1"):
        compare_many(qrels, base, {"same": base, "off": cand}, metrics,
                     correction="holm", method=method, samples=100,
                     thresholds={"hit@1": 0}, candidate_thresholds={"off": {"hit@1": None}}, **limits)


@pytest.mark.parametrize("member", ["baseline", "z_bad"])
@pytest.mark.parametrize("limit", [None, 3])
def test_saved_duplicate_metrics_keep_member_diagnostic(inputs, member, limit, monkeypatch):
    qrels, base, cand = inputs
    artifact = saved_evaluation(qrels, base, parse_metrics("hit@1,recall@1"))
    invalid = copy.deepcopy(artifact)
    invalid["manifest"]["parameters"]["metrics"].append("hit@1")

    def forbidden(*args, **kwargs):
        pytest.fail("invalid saved metrics must reject before inference")

    monkeypatch.setattr("ragwrench.evalgate.compare.paired_test", forbidden)
    with pytest.raises(InputError, match=f"{member if member == 'baseline' else repr(member)}: duplicate metric: hit@1"):
        compare_many_saved(invalid if member == "baseline" else artifact,
                           {"a_valid": artifact, "z_bad": invalid if member == "z_bad" else artifact},
                           correction="holm", method="ttest", max_family_pairs=limit)


@pytest.mark.parametrize("method", ["ttest", "bootstrap", "randomization"])
def test_unique_metric_cardinality_matches_saved_json_and_junit(inputs, method):
    qrels, base, cand = inputs
    metrics = parse_metrics("recall@1,hit@1")
    before = list(metrics)
    artifacts = [saved_evaluation(qrels, run, metrics) for run in (base, cand)]
    options = dict(correction="holm", method=method, samples=100,
                   thresholds={"hit@1": .49}, candidate_thresholds={"off": {"hit@1": None}})
    for limit in (None, 4):
        opts = dict(options, max_family_pairs=limit, max_family_resamples=1 if method == "ttest" else 800)
        direct = compare_many(qrels, base, {"same": base, "off": cand}, metrics, **opts)
        saved = compare_many_saved(artifacts[0], {"same": artifacts[0], "off": artifacts[1]}, **opts)
        assert saved["policy"] == direct["policy"]
        assert ("max_family_pairs" in direct["policy"]) == (limit is not None)
        for report in (direct, saved):
            exported = json.loads(render(report, "json"))
            actual = sum(len(c["metrics"]) for c in exported["candidates"].values())
            suite = ET.fromstring(render(report, "junit"))
            assert report["policy"]["family_size"] == actual == len(suite.findall("testcase")) == 4
            assert suite.attrib["tests"] == "4"
            assert report["policy"]["estimated_family_resamples"] == (0 if method == "ttest" else 800)
        expected = copy.deepcopy(saved)
        expected["command"] = "compare-many"
        for row in expected["candidates"].values():
            row.pop("evaluation_manifests")
            row["command"] = "compare"
        assert expected == direct
    assert metrics == before
