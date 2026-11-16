"""Explicit candidate gate opt-outs retain complete family inference and audit policy."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from ragwrench.evalgate.compare import compare_many, compare_many_saved
from ragwrench.evalgate.formats import InputError
from ragwrench.evalgate.metrics import parse_metrics
from ragwrench.evalgate.report import render
from ragwrench.evalgate.saved import saved_evaluation

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def family(tmp_path):
    qrels = {"q1": {"d": 1}, "q2": {"d": 1}}
    base = {"q1": {"d": 1}, "q2": {"d": 1}}
    cand = {"q2": {"d": 1}}
    metrics = parse_metrics("recall@1,hit@1")
    paths = {name: tmp_path / name for name in ("qrels", "base", "cand", "base.json", "cand.json")}
    paths["qrels"].write_text("q1 0 d 1\nq2 0 d 1\n")
    paths["base"].write_text("q1 Q0 d 1 1 base\nq2 Q0 d 1 1 base\n")
    paths["cand"].write_text("q2 Q0 d 1 1 cand\n")
    artifacts = [saved_evaluation(qrels, run, metrics) for run in (base, cand)]
    for name, artifact in zip(("base.json", "cand.json"), artifacts):
        paths[name].write_text(json.dumps(artifact))
    return qrels, base, cand, metrics, artifacts, paths


def cli(family, saved, *options):
    paths = family[-1]
    args = ["compare-many-saved" if saved else "compare-many"]
    suffix = ".json" if saved else ""
    args += ["--baseline", str(paths["base" + suffix]),
             "--candidate", f"strict={paths['cand' + suffix]}",
             "--candidate", f"relaxed={paths['cand' + suffix]}",
             "--test", "ttest", "--correction", "holm", "--format", "json"]
    if not saved:
        args += ["--qrels", str(paths["qrels"]), "--metrics", "hit@1,recall@1"]
    return subprocess.run([sys.executable, "-m", "ragwrench.evalgate", *args, *map(str, options)],
                          cwd=ROOT, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("saved", [False, True])
def test_opt_out_end_to_end_json_junit(family, saved):
    options = ["--max-drop", "hit@1=.49", "--candidate-max-drop", "relaxed:hit@1=off"]
    result = cli(family, saved, *options)
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["policy"]["candidate_thresholds"] == {"relaxed": {"hit@1": None}}
    assert report["policy"]["effective_thresholds"] == {"relaxed": {}, "strict": {"hit@1": .49}}
    assert report["policy"]["thresholds"] == {"hit@1": .49}
    assert report["policy"]["family_size"] == 4
    assert list(report["candidates"]) == ["relaxed", "strict"]
    assert report["violations"] == [{"candidate": "strict", "metric": "hit@1", "drop": .5, "max_drop": .49}]
    for name, candidate in report["candidates"].items():
        assert candidate["candidate_missing_queries"] == ["q1"]
        assert candidate["per_query"]["q1"]["hit@1"] == {
            "baseline": 1, "candidate": 0, "delta": -1, "outcome": "loss"}
        assert candidate["metrics"]["hit@1"]["passed"] is (name == "relaxed")
        assert candidate["metrics"]["hit@1"]["max_drop"] == (None if name == "relaxed" else .49)
        if saved:
            assert candidate["evaluation_manifests"] == dict(zip(("baseline", "candidate"),
                                                                (a["manifest"] for a in family[-2])))
    junit = cli(family, saved, *options, "--format", "junit")
    assert junit.returncode == 1, junit.stderr
    suite = ET.fromstring(junit.stdout)
    assert suite.attrib["tests"] == "4" and suite.attrib["failures"] == "1"
    for case in suite.findall("testcase"):
        name, metric = case.attrib["name"].split("/")
        details = json.loads(case.find("system-out").text)
        assert details["policy"] == report["policy"]
        assert details["max_drop"] == report["candidates"][name]["metrics"][metric]["max_drop"]
        if saved:
            assert details["evaluation_manifests"] == report["candidates"][name]["evaluation_manifests"]


@pytest.mark.parametrize("method", ["ttest", "randomization", "bootstrap"])
@pytest.mark.parametrize("correction", ["none", "bonferroni", "holm"])
def test_opt_out_inference_equivalence_nonmutation_and_order(family, method, correction):
    qrels, base, cand, metrics, artifacts, _ = family
    options = dict(method=method, correction=correction, samples=100, seed=73)
    shared = {"hit@1": .49, "recall@1": .5}
    overrides = {"relaxed": {"hit@1": None, "recall@1": 0}, "strict": {"recall@1": .5}}
    before = copy.deepcopy([shared, overrides, artifacts])
    direct = compare_many(qrels, base, {"strict": cand, "relaxed": cand}, metrics,
                          thresholds=shared, candidate_thresholds=overrides, **options)
    saved = compare_many_saved(artifacts[0], {"strict": artifacts[1], "relaxed": artifacts[1]},
                               thresholds=shared, candidate_thresholds=overrides, **options)
    assert [shared, overrides, artifacts] == before
    normalized = copy.deepcopy(saved)
    normalized["command"] = "compare-many"
    for candidate in normalized["candidates"].values():
        candidate["command"] = "compare"
        candidate.pop("evaluation_manifests")
    assert normalized == direct
    ungated = compare_many(qrels, base, {"strict": cand, "relaxed": cand}, metrics, **options)
    assert direct["policy"]["family_size"] == ungated["policy"]["family_size"] == 4
    assert direct["policy"]["effective_thresholds"] == {
        "relaxed": {"recall@1": 0}, "strict": {"hit@1": .49, "recall@1": .5}}
    assert [(v["candidate"], v["metric"]) for v in direct["violations"]] == [
        ("relaxed", "recall@1"), ("strict", "hit@1")]
    for name in direct["candidates"]:
        for metric, row in direct["candidates"][name]["metrics"].items():
            assert {k: v for k, v in row.items() if k not in {"passed", "max_drop"}} == {
                k: v for k, v in ungated["candidates"][name]["metrics"][metric].items()
                if k not in {"passed", "max_drop"}}
    reordered = compare_many_saved(artifacts[0], {"relaxed": artifacts[1], "strict": artifacts[1]},
                                   thresholds=dict(reversed(list(shared.items()))),
                                   candidate_thresholds=dict(reversed(list(overrides.items()))), **options)
    for fmt in ("json", "junit", "text", "markdown", "csv"):
        assert render(saved, fmt) == render(reordered, fmt)


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_opt_out_with_or_without_shared_defaults_can_pass(family, saved, shared):
    options = ["--candidate-max-drop", "strict:hit@1=off", "--candidate-max-drop", "relaxed:hit@1=off"]
    if shared:
        options += ["--max-drop", "hit@1=0"]
    result = cli(family, saved, *options)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["passed"] and not report["violations"]
    assert report["policy"]["effective_thresholds"] == {"relaxed": {}, "strict": {}}
    assert report["policy"]["candidate_thresholds"] == {"relaxed": {"hit@1": None}, "strict": {"hit@1": None}}


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("options,error", [
    (["--candidate-max-drop", "unknown:hit@1=off"], "candidate"),
    (["--candidate-max-drop", "relaxed:hit@2=off"], "metric"),
    (["--candidate-max-drop", "relaxed:hit@1=off", "--candidate-max-drop", "relaxed:hit@1=0"], "duplicate"),
    (["--candidate-max-drop", "relaxed:hit@1=0", "--candidate-max-drop", "relaxed:hit@1=off"], "duplicate"),
    (["--candidate-max-drop", "relaxed:hit@1=off", "--candidate-max-drop", "relaxed:hit@1=off"], "duplicate"),
    (["--candidate-max-drop", "relaxed:hit@1=OFF"], "expected"),
    (["--candidate-max-drop", "relaxed:hit@1= off"], "expected"),
    (["--candidate-max-drop", "relaxed:hit@1=off=0"], "expected"),
    (["--candidate-max-drop", "relaxed:hit@1=null"], "expected"),
    (["--candidate-max-drop", "relaxed:hit@1=nan"], "finite"),
    (["--candidate-max-drop", "relaxed:hit@1=inf"], "finite"),
    (["--max-drop", "hit@1=off"], "expected"),
])
def test_opt_out_invalid_policy_is_atomic(family, saved, tmp_path, options, error):
    output = tmp_path / "report.json"
    output.write_text("KEEP")
    result = cli(family, saved, *options, "--output", output, "--force")
    assert result.returncode == 2 and error in result.stderr, result.stderr
    assert not result.stdout and "Traceback" not in result.stderr
    assert output.read_text() == "KEEP" and not list(tmp_path.glob(".retgate-*"))


@pytest.mark.parametrize("overrides,shared,error", [
    ({"unknown": {"hit@1": None}}, {}, "candidate"),
    ({"relaxed": {"hit@2": None}}, {}, "metric"),
    ({"relaxed": {"hit@1": "off"}}, {}, "finite"),
    ({}, {"hit@1": None}, "finite"),
])
def test_opt_out_api_rejects_unknown_or_unsupported_policy(family, overrides, shared, error):
    qrels, base, cand, metrics, artifacts, _ = family
    with pytest.raises(InputError, match=error):
        compare_many(qrels, base, {"relaxed": cand}, metrics, correction="holm",
                     thresholds=shared, candidate_thresholds=overrides)
    with pytest.raises(InputError, match=error):
        compare_many_saved(artifacts[0], {"relaxed": artifacts[1]}, correction="holm",
                           thresholds=shared, candidate_thresholds=overrides)


def test_opt_out_does_not_bypass_late_saved_artifact_validation(family, tmp_path):
    bad = copy.deepcopy(family[-2][1])
    bad["per_query"].pop("q1")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    output = tmp_path / "report.json"
    output.write_text("KEEP")
    result = cli(family, True, "--candidate", f"z_bad={path}",
                 "--candidate-max-drop", "z_bad:hit@1=off", "--output", output, "--force")
    assert result.returncode == 2 and "candidate 'z_bad'" in result.stderr
    assert "per-query" in result.stderr and not result.stdout
    assert output.read_text() == "KEEP"
