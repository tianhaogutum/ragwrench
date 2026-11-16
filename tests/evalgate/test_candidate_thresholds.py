"""Candidate drop policies: precedence, audit exports, and atomic rejection."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from ragwrench.evalgate.compare import compare_many
from ragwrench.evalgate.formats import InputError
from ragwrench.evalgate.metrics import parse_metrics
from ragwrench.evalgate.report import render

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def inputs(tmp_path):
    qrels, baseline, candidate = (tmp_path / name for name in ("qrels", "base", "cand"))
    qrels.write_text("q1 0 d 1\nq2 0 d 1\n")
    baseline.write_text("q1 Q0 d 1 1 base\nq2 Q0 d 1 1 base\n")
    candidate.write_text("q2 Q0 d 1 1 cand\n")
    return qrels, baseline, candidate


def family_cli(inputs, *extra):
    qrels, baseline, candidate = inputs
    return subprocess.run(
        [sys.executable, "-m", "ragwrench.evalgate", "compare-many", "--qrels", str(qrels),
         "--baseline", str(baseline), "--candidate", f"strict={candidate}",
         "--candidate", f"relaxed={candidate}", "--metrics", "hit@1,recall@1",
         "--test", "ttest", "--correction", "holm", "--format", "json", *map(str, extra)],
        capture_output=True, text=True, cwd=ROOT, timeout=30)


def test_candidate_policy_precedence_and_json_junit_exports(inputs):
    options = ["--max-drop", "hit@1=0.49", "--candidate-max-drop", "relaxed:hit@1=0.5",
               "--candidate-max-drop", "strict:recall@1=0.1"]
    result = family_cli(inputs, *options)
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["policy"] == {
        "correction": "holm", "alpha": 0.05, "family": "all candidate-metric pairs",
        "family_size": 4, "ci_adjustment": "none", "gate": "absolute-mean-drop",
        "significance_affects_gate": False, "thresholds": {"hit@1": .49},
        "candidate_thresholds": {"relaxed": {"hit@1": .5}, "strict": {"recall@1": .1}},
        "threshold_precedence": "candidate override > shared metric default > no gate",
        "effective_thresholds": {"relaxed": {"hit@1": .5}, "strict": {"hit@1": .49, "recall@1": .1}},
    }
    assert report["candidates"]["relaxed"]["passed"]
    assert report["candidates"]["relaxed"]["metrics"]["recall@1"]["max_drop"] is None
    assert report["violations"] == [
        {"candidate": "strict", "metric": "hit@1", "drop": .5, "max_drop": .49},
        {"candidate": "strict", "metric": "recall@1", "drop": .5, "max_drop": .1},
    ]
    junit = family_cli(inputs, *options, "--format", "junit")
    assert junit.returncode == 1, junit.stderr
    suite = ET.fromstring(junit.stdout)
    assert suite.attrib["tests"] == "4" and suite.attrib["failures"] == "2"
    for case in suite.findall("testcase"):
        name, metric = case.attrib["name"].split("/")
        details = json.loads(case.find("system-out").text)
        assert details["policy"] == report["policy"]
        assert details["max_drop"] == report["candidates"][name]["metrics"][metric]["max_drop"]
        assert (case.find("failure") is not None) is (name == "strict")


@pytest.mark.parametrize("shared,override,passed", [
    (.49, .5, True), (.5, .49, False), (.5, 0, False), (0, .5, True),
    (None, .49, False), (None, .5, True),
])
def test_candidate_policy_boundaries_and_override_only(inputs, shared, override, passed):
    options = ["--candidate-max-drop", f"strict:hit@1={override}"]
    if shared is not None:
        options += ["--max-drop", f"hit@1={shared}"]
    result = family_cli(inputs, *options)
    assert result.returncode in {0, 1}, result.stderr
    report = json.loads(result.stdout)
    assert report["candidates"]["strict"]["passed"] is passed
    assert report["candidates"]["strict"]["metrics"]["hit@1"]["max_drop"] == override
    assert report["candidates"]["relaxed"]["metrics"]["hit@1"]["max_drop"] == shared
    assert report["passed"] is (passed and (shared is None or shared >= .5))
    assert (result.returncode == 0) is report["passed"]


@pytest.mark.parametrize("value,error", [
    ("unknown:hit@1=0.1", "candidate"), ("Strict:hit@1=0.1", "candidate"),
    ("strict:hit@2=0.1", "metric"), ("strict:unknown@1=0.1", "metric"),
    ("strict:hit@1=NaN", "finite"), ("strict:hit@1=inf", "finite"),
    ("strict:hit@1=1e999", "finite"), ("strict:hit@1=-0.1", "nonnegative"),
    ("strict:hit@1=oops", "expected"), ("strict:hit@1", "expected"),
    ("strict=0.1", "expected"), (":hit@1=0.1", "candidate"),
    ("strict:=0.1", "metric"), ("strict:hit@1=", "expected"),
    ("strict:hit@1=0.1=0.2", "expected"),
])
def test_invalid_candidate_policy_is_atomic(inputs, tmp_path, value, error):
    output = tmp_path / "report.json"
    output.write_text("KEEP")
    result = family_cli(inputs, "--candidate-max-drop", value, "--output", output, "--force")
    assert result.returncode == 2
    assert result.stdout == "" and "Traceback" not in result.stderr
    assert error in result.stderr, result.stderr
    assert output.read_text() == "KEEP"
    assert not list(tmp_path.glob(".retgate-*"))


def test_duplicate_candidate_metric_is_rejected_before_creating_output(inputs, tmp_path):
    output = tmp_path / "absent.json"
    result = family_cli(inputs, "--candidate-max-drop", "strict:hit@1=.49",
                        "--candidate-max-drop", "strict:hit@1=.5", "--output", output)
    assert result.returncode == 2 and "duplicate" in result.stderr
    assert result.stdout == "" and not output.exists()


@pytest.mark.parametrize("method", ["randomization", "bootstrap", "ttest"])
def test_api_policy_does_not_mutate_inputs_or_change_inference(method):
    qrels = {"q1": {"d": 1}, "q2": {"d": 1}}
    baseline = {qid: {"d": 1} for qid in qrels}
    candidate = {"q2": {"d": 1}}
    metrics = parse_metrics("recall@1,hit@1")
    candidates = {"strict": candidate, "relaxed": candidate}
    opts = dict(correction="holm", method=method, samples=100, seed=73)
    shared = {"hit@1": .49, "recall@1": .5}
    overrides = {"strict": {"recall@1": .1, "hit@1": 0}, "relaxed": {"hit@1": .5}}
    before = json.dumps([shared, overrides])
    report = compare_many(qrels, baseline, candidates, metrics, thresholds=shared,
                          candidate_thresholds=overrides, **opts)
    assert json.dumps([shared, overrides]) == before
    reordered = compare_many(qrels, baseline, dict(reversed(list(candidates.items()))),
                             list(reversed(metrics)), thresholds=dict(reversed(list(shared.items()))),
                             candidate_thresholds={k: dict(reversed(list(v.items())))
                                                   for k, v in reversed(list(overrides.items()))}, **opts)
    for fmt in ("json", "junit", "text", "markdown", "csv"):
        assert render(report, fmt) == render(reordered, fmt)
    ungated = compare_many(qrels, baseline, candidates, metrics, **opts)
    for name in candidates:
        for metric, row in report["candidates"][name]["metrics"].items():
            assert {k: v for k, v in row.items() if k not in {"passed", "max_drop"}} == {
                k: v for k, v in ungated["candidates"][name]["metrics"][metric].items()
                if k not in {"passed", "max_drop"}}


@pytest.mark.parametrize("overrides,error", [
    ({"unknown": {}}, "candidate"), ({"strict": {"hit@2": .1}}, "metric"),
    ({"strict": {"hit@1": -.1}}, "nonnegative"),
    ({"strict": {"hit@1": float("nan")}}, "finite"),
    ({"strict": {"hit@1": float("inf")}}, "finite"),
])
def test_api_rejects_invalid_candidate_policy(overrides, error):
    # Invalid policies must be rejected before run evaluation even with empty inputs.
    with pytest.raises(InputError, match=error):
        compare_many({}, {}, {"strict": {}}, parse_metrics("hit@1"),
                     correction="none", candidate_thresholds=overrides)
