"""Opt-in family work limits include ungated and explicitly opted-out pairs."""
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
    base = {q: {"d": 1.0} for q in qrels}
    cand = {"q2": {"d": 1.0}}
    metrics = parse_metrics("hit@1,recall@1")
    artifacts = [saved_evaluation(qrels, run, metrics) for run in (base, cand)]
    paths = [tmp_path / name for name in ("qrels", "base", "cand", "base.json", "cand.json")]
    for path, text in zip(paths, ["q1 0 d 1\nq2 0 d 1\n", "q1 Q0 d 1 1 b\nq2 Q0 d 1 1 b\n",
                                  "q2 Q0 d 1 1 c\n", *(json.dumps(a) for a in artifacts)]):
        path.write_text(text)
    return qrels, base, cand, metrics, artifacts, paths


def cli(family, saved, *extra):
    qrels, base, cand, bs, cs = family[-1]
    args = ["compare-many-saved" if saved else "compare-many", "--baseline", str(bs if saved else base)]
    args += ["--candidate", f"strict={cs if saved else cand}", "--candidate", f"relaxed={cs if saved else cand}"]
    if not saved:
        args += ["--qrels", str(qrels), "--metrics", "hit@1,recall@1"]
    return subprocess.run([sys.executable, "-m", "ragwrench.evalgate", *args, "--test", "bootstrap", "--samples", "100",
                           "--correction", "holm", "--max-drop", "hit@1=.49",
                           "--candidate-max-drop", "relaxed:hit@1=off", "--format", "json", *map(str, extra)],
                          capture_output=True, text=True, cwd=ROOT, timeout=30)


def api(family, saved, **extra):
    qrels, base, cand, metrics, artifacts, _ = family
    options = dict(method="bootstrap", samples=100, seed=73, correction="holm",
                   thresholds={"hit@1": .49}, candidate_thresholds={"relaxed": {"hit@1": None}})
    options.update(extra)
    if saved:
        return compare_many_saved(artifacts[0], {"strict": artifacts[1], "relaxed": artifacts[1]}, **options)
    return compare_many(qrels, base, {"strict": cand, "relaxed": cand}, metrics, **options)


@pytest.mark.parametrize("saved", [False, True])
def test_family_budget_cli_end_to_end_and_exports(family, saved, tmp_path):
    # Two queries * 100 samples * two candidates * two metrics, even though
    # relaxed/hit@1 is opted out and recall@1 has no gate for either candidate.
    result = cli(family, saved, "--max-family-resamples", "800")
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["policy"]["family_size"] == 4
    assert report["policy"]["estimated_family_resamples"] == 800
    assert report["policy"]["max_family_resamples"] == 800
    assert report["policy"]["candidate_thresholds"] == {"relaxed": {"hit@1": None}}
    assert report["policy"]["effective_thresholds"] == {"relaxed": {}, "strict": {"hit@1": .49}}
    assert report["violations"] == [{"candidate": "strict", "metric": "hit@1", "drop": .5, "max_drop": .49}]
    suite_result = cli(family, saved, "--max-family-resamples", "800", "--format", "junit")
    assert suite_result.returncode == 1, suite_result.stderr
    suite = ET.fromstring(suite_result.stdout)
    assert suite.attrib["tests"] == "4" and suite.attrib["failures"] == "1"
    for case in suite.findall("testcase"):
        name, metric = case.attrib["name"].split("/")
        details = json.loads(case.find("system-out").text)
        assert details["policy"] == report["policy"]
        assert details["max_drop"] == report["candidates"][name]["metrics"][metric]["max_drop"]
        if saved:
            assert details["evaluation_manifests"] == report["candidates"][name]["evaluation_manifests"]
    output = tmp_path / "report"
    output.write_text("KEEP")
    rejected = cli(family, saved, "--max-family-resamples", "799", "--output", output, "--force")
    assert rejected.returncode == 2, rejected.stderr
    assert "required=800" in rejected.stderr and "allowed=799" in rejected.stderr
    assert not rejected.stdout and output.read_text() == "KEEP"
    assert not list(tmp_path.glob(".retgate-*"))


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("method", ["bootstrap", "randomization"])
def test_family_budget_rejects_before_any_inference(family, saved, method, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("over-budget families must reject before any paired inference")

    monkeypatch.setattr("ragwrench.evalgate.compare.paired_test", forbidden)
    with pytest.raises(InputError, match="required=800.*allowed=799"):
        api(family, saved, method=method, max_family_resamples=799)


@pytest.mark.parametrize("method", ["bootstrap", "randomization", "ttest"])
@pytest.mark.parametrize("correction", ["none", "bonferroni", "holm"])
def test_allowed_budget_preserves_inference_policy_and_inputs(family, method, correction):
    before = copy.deepcopy(family[:-1])
    budget = 1 if method == "ttest" else 800
    opts = dict(method=method, correction=correction)
    direct = api(family, False, **opts, max_family_resamples=budget)
    saved = api(family, True, **opts, max_family_resamples=budget)
    normalized = copy.deepcopy(saved)
    normalized["command"] = "compare-many"
    for report in normalized["candidates"].values():
        report["command"] = "compare"
        report.pop("evaluation_manifests")
    assert normalized == direct and family[:-1] == before
    assert direct["policy"]["estimated_family_resamples"] == (0 if method == "ttest" else 800)
    for is_saved, budgeted in ((False, direct), (True, saved)):
        unbudgeted = api(family, is_saved, **opts)
        assert "estimated_family_resamples" not in unbudgeted["policy"]
        stripped = copy.deepcopy(budgeted)
        stripped["policy"].pop("estimated_family_resamples")
        stripped["policy"].pop("max_family_resamples")
        for fmt in ("json", "junit", "text", "markdown", "csv"):
            assert render(stripped, fmt) == render(unbudgeted, fmt)


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("value", ["0", "-1", "1.0", "NaN", "inf", "1e999", "off", "", "800=900"])
def test_invalid_cli_budget_preserves_output(family, saved, tmp_path, value):
    output = tmp_path / "report"
    output.write_text("KEEP")
    result = cli(family, saved, "--max-family-resamples", value, "--output", output, "--force")
    assert result.returncode == 2 and "--max-family-resamples" in result.stderr
    assert not result.stdout and "Traceback" not in result.stderr
    assert output.read_text() == "KEEP" and not list(tmp_path.glob(".retgate-*"))


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("limit", [0, -1, 800.0, True, False, "800", float("nan"), float("inf"), {}, []])
def test_invalid_api_budget_rejects_before_inference(family, saved, limit, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid budget must reject before any paired inference")

    monkeypatch.setattr("ragwrench.evalgate.compare.paired_test", forbidden)
    with pytest.raises(InputError, match="max-family-resamples.*positive integer"):
        api(family, saved, max_family_resamples=limit)


def test_budget_does_not_bypass_saved_validation(family, monkeypatch):
    artifacts = family[-2]
    invalid = copy.deepcopy(artifacts[1])
    invalid["per_query"].pop("q1")

    def forbidden(*args, **kwargs):
        pytest.fail("invalid artifact must reject before any paired inference")

    monkeypatch.setattr("ragwrench.evalgate.compare.paired_test", forbidden)
    with pytest.raises(InputError, match="candidate 'z_bad'.*per-query"):
        compare_many_saved(artifacts[0], {"a_valid": artifacts[1], "z_bad": invalid},
                           correction="holm", max_family_resamples=1)


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("extra,error", [
    ({"method": "unknown"}, "unknown statistical test"),
    ({"samples": 99}, "samples must be an integer"),
    ({"samples": float("inf")}, "samples must be an integer"),
])
def test_budget_requires_valid_inference_options(family, saved, extra, error, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid inference options must reject before paired inference")

    monkeypatch.setattr("ragwrench.evalgate.compare.paired_test", forbidden)
    with pytest.raises(InputError, match=error):
        api(family, saved, max_family_resamples=800, **extra)


@pytest.mark.parametrize("saved", [False, True])
def test_family_budget_does_not_relax_existing_per_pair_limit(family, saved):
    # Actual inference stops at the existing per-pair cap before doing draws.
    qrels = {f"q{i}": {"d": 1} for i in range(51)}
    base = {q: {"d": 1.0} for q in qrels}
    metrics = parse_metrics("hit@1")
    options = dict(correction="holm", samples=1_000_000, max_family_resamples=51_000_000)
    with pytest.raises(InputError, match="resampling budget exceeds 50000000"):
        if saved:
            artifact = saved_evaluation(qrels, base, metrics)
            compare_many_saved(artifact, {"same": artifact}, **options)
        else:
            compare_many(qrels, base, {"same": base}, metrics, **options)


@pytest.mark.parametrize("saved", [False, True])
def test_pair_limit_cli_end_to_end_and_exports(family, saved, tmp_path):
    # Four hypotheses still count under analytic inference and a gate opt-out.
    options = ("--test", "ttest", "--max-family-resamples", "1", "--max-family-pairs", "4")
    result = cli(family, saved, *options)
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["policy"]["family_size"] == report["policy"]["max_family_pairs"] == 4
    assert report["policy"]["estimated_family_resamples"] == 0
    assert report["violations"] == [{"candidate": "strict", "metric": "hit@1", "drop": .5, "max_drop": .49}]
    junit = cli(family, saved, *options, "--format", "junit")
    assert junit.returncode == 1, junit.stderr
    suite = ET.fromstring(junit.stdout)
    assert suite.attrib["tests"] == "4" and suite.attrib["failures"] == "1"
    for case in suite.findall("testcase"):
        name = case.attrib["name"].split("/")[0]
        details = json.loads(case.find("system-out").text)
        assert details["policy"] == report["policy"]
        if saved:
            assert details["evaluation_manifests"] == report["candidates"][name]["evaluation_manifests"]
    output = tmp_path / "report"
    output.write_text("KEEP")
    rejected = cli(family, saved, *options, "--max-family-pairs", "3", "--output", output, "--force")
    assert rejected.returncode == 2, rejected.stderr
    assert "family pair limit exceeded: required=4, allowed=3" in rejected.stderr
    assert not rejected.stdout and output.read_text() == "KEEP"
    assert not list(tmp_path.glob(".retgate-*"))


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("method", ["bootstrap", "randomization", "ttest"])
def test_pair_limit_rejects_before_any_inference(family, saved, method, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("pair count must reject before any paired inference")

    monkeypatch.setattr("ragwrench.evalgate.compare.paired_test", forbidden)
    with pytest.raises(InputError, match="family pair limit exceeded: required=4, allowed=3"):
        api(family, saved, method=method, max_family_pairs=3)


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("method", ["bootstrap", "randomization", "ttest"])
@pytest.mark.parametrize("correction", ["none", "bonferroni", "holm"])
def test_pair_limit_preserves_allowed_family(family, saved, method, correction):
    before = copy.deepcopy(family[:-1])
    # The limits can be used independently or together.
    for work_limit in (None, 1 if method == "ttest" else 800):
        opts = dict(method=method, correction=correction, max_family_resamples=work_limit)
        unbounded = api(family, saved, **opts)
        assert "max_family_pairs" not in unbounded["policy"]
        bounded = api(family, saved, **opts, max_family_pairs=4)
        assert bounded["policy"].pop("max_family_pairs") == 4
        assert bounded == unbounded and family[:-1] == before
        for fmt in ("json", "junit", "text", "markdown", "csv"):
            assert render(bounded, fmt) == render(unbounded, fmt)


@pytest.mark.parametrize("saved", [False, True])
def test_pair_limit_cannot_relax_resampling_limit(family, saved, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("work budget must still reject before inference")

    monkeypatch.setattr("ragwrench.evalgate.compare.paired_test", forbidden)
    with pytest.raises(InputError, match="resampling budget exceeded: required=800, allowed=799"):
        api(family, saved, max_family_pairs=4, max_family_resamples=799)


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("value", ["0", "-1", "1.0", "NaN", "inf", "1e999", "off", "", "4=5"])
def test_invalid_cli_pair_limit_preserves_output(family, saved, tmp_path, value):
    output = tmp_path / "report"
    output.write_text("KEEP")
    result = cli(family, saved, "--max-family-pairs", value, "--output", output, "--force")
    assert result.returncode == 2 and "--max-family-pairs" in result.stderr
    assert not result.stdout and "Traceback" not in result.stderr
    assert output.read_text() == "KEEP" and not list(tmp_path.glob(".retgate-*"))


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("limit", [0, -1, 4.0, True, False, "4", float("nan"), float("inf"), {}, []])
def test_invalid_api_pair_limit_rejects_before_inference(family, saved, limit, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid pair limit must reject before any paired inference")

    monkeypatch.setattr("ragwrench.evalgate.compare.paired_test", forbidden)
    with pytest.raises(InputError, match="max-family-pairs.*positive integer"):
        api(family, saved, max_family_pairs=limit)


@pytest.mark.parametrize("saved", [False, True])
def test_pair_limit_counts_unchanged_and_fully_disabled_candidates(family, saved, monkeypatch):
    qrels, base, cand, metrics, artifacts, _ = family
    options = dict(method="ttest", correction="holm", max_family_pairs=5,
                   thresholds={"hit@1": 0}, candidate_thresholds={"off": {"hit@1": None}})

    def forbidden(*args, **kwargs):
        pytest.fail("all six hypotheses must count before any paired inference")

    monkeypatch.setattr("ragwrench.evalgate.compare.paired_test", forbidden)
    with pytest.raises(InputError, match="required=6, allowed=5"):
        if saved:
            compare_many_saved(artifacts[0], {"same": artifacts[0], "off": artifacts[1], "strict": artifacts[1]}, **options)
        else:
            compare_many(qrels, base, {"same": base, "off": cand, "strict": cand}, metrics, **options)


@pytest.mark.parametrize("limit", [0, 3])
def test_pair_limit_does_not_bypass_saved_validation(family, limit, monkeypatch):
    artifacts = family[-2]
    invalid = copy.deepcopy(artifacts[1])
    invalid["per_query"].pop("q1")

    def forbidden(*args, **kwargs):
        pytest.fail("invalid artifact must reject before any paired inference")

    monkeypatch.setattr("ragwrench.evalgate.compare.paired_test", forbidden)
    with pytest.raises(InputError, match="candidate 'z_bad'.*per-query"):
        compare_many_saved(artifacts[0], {"a_valid": artifacts[1], "z_bad": invalid},
                           correction="holm", max_family_pairs=limit)
