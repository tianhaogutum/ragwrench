"""Saved candidate families use the same fixed-baseline policy as direct runs."""
import copy
import json
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[2]


def cli(*args):
    return subprocess.run([sys.executable, "-m", "ragwrench.evalgate", *map(str, args)],
                          capture_output=True, text=True, cwd=ROOT, timeout=30)


@pytest.fixture
def family(tmp_path):
    from ragwrench.evalgate.metrics import parse_metrics
    from ragwrench.evalgate.saved import saved_evaluation

    qrels = {"q1": {"d": 1}, "q2": {"d": 1}}
    base = {q: {"d": 1.0} for q in qrels}
    cand = {"q2": {"d": 1.0}, "extra": {"x": 1.0}}
    metrics = parse_metrics("hit@1,recall@1")
    artifacts = {"base": saved_evaluation(qrels, base, metrics),
                 "regressed": saved_evaluation(qrels, cand, metrics)}
    paths = {}
    for name, artifact in artifacts.items():
        paths[name] = tmp_path / f"{name}=eval.json"
        paths[name].write_text(json.dumps(artifact))
    return qrels, base, cand, metrics, artifacts, paths


def many(family, *extra):
    paths = family[-1]
    return cli("compare-many-saved", "--baseline", paths["base"],
               "--candidate", f"same={paths['base']}",
               "--candidate", f"strict={paths['regressed']}",
               "--candidate", f"relaxed={paths['regressed']}",
               "--test", "ttest", "--correction", "holm", "--max-drop", "hit@1=0.49",
               "--candidate-max-drop", "relaxed:hit@1=0.5", "--format", "json", *extra)


def test_saved_family_end_to_end_and_manifest_exports(family):
    result = many(family)
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["command"] == "compare-many-saved"
    assert report["query_count"] == 2 and report["policy"]["family_size"] == 6
    assert list(report["candidates"]) == ["relaxed", "same", "strict"]
    assert report["policy"]["effective_thresholds"] == {
        "relaxed": {"hit@1": .5}, "same": {"hit@1": .49}, "strict": {"hit@1": .49}}
    assert report["violations"] == [
        {"candidate": "strict", "metric": "hit@1", "drop": .5, "max_drop": .49}]
    for name, candidate in report["candidates"].items():
        same = name == "same"
        assert candidate["passed"] is (name != "strict")
        assert candidate["metrics"]["hit@1"]["delta"] == (0 if same else -.5)
        assert candidate["metrics"]["recall@1"]["max_drop"] is None
        assert candidate["candidate_missing_queries"] == ([] if same else ["q1"])
        assert candidate["candidate_ignored_queries"] == ([] if same else ["extra"])
        assert candidate["evaluation_manifests"] == {
            "baseline": family[-2]["base"]["manifest"],
            "candidate": family[-2]["base" if same else "regressed"]["manifest"]}
    suite = ET.fromstring(many(family, "--format", "junit").stdout)
    assert suite.attrib["tests"] == "6" and suite.attrib["failures"] == "1"
    assert suite.attrib["name"] == "retgate.compare-many-saved"
    for case in suite:
        name, metric = case.attrib["name"].split("/")
        details = json.loads(case.find("system-out").text)
        assert details["policy"] == report["policy"]
        assert details["evaluation_manifests"] == report["candidates"][name]["evaluation_manifests"]
        assert details["candidate_missing_queries"] == report["candidates"][name]["candidate_missing_queries"]
        assert details["max_drop"] == report["candidates"][name]["metrics"][metric]["max_drop"]


@pytest.mark.parametrize("method", ["randomization", "bootstrap", "ttest"])
@pytest.mark.parametrize("correction", ["none", "bonferroni", "holm"])
def test_saved_matches_direct_inference_gates_and_order(method, correction):
    from ragwrench.evalgate.compare import compare_many, compare_many_saved
    from ragwrench.evalgate.metrics import parse_metrics
    from ragwrench.evalgate.report import render
    from ragwrench.evalgate.saved import saved_evaluation

    qrels = {f"q{i}": {"d": 1} for i in range(5)} | {"empty": {}}
    baseline = {f"q{i}": {"d": 1, "x": 2} for i in range(4)}
    candidate = {f"q{i}": {"d": 2, "x": i} for i in range(3)} | {"extra": {"x": 1}}
    metrics = parse_metrics("mrr@2,ndcg@2")
    opts = dict(method=method, correction=correction, seed=73, samples=300,
                thresholds={"mrr@2": 0}, candidate_thresholds={"mixed": {"ndcg@2": .1}})
    direct = compare_many(qrels, baseline, {"same": baseline, "mixed": candidate}, metrics, **opts)
    base = saved_evaluation(qrels, baseline, metrics)
    mixed = saved_evaluation(qrels, candidate, list(reversed(metrics)))
    before = copy.deepcopy([base, mixed, opts])
    saved = compare_many_saved(base, {"same": base, "mixed": mixed}, **opts)
    assert [base, mixed, opts] == before
    normalized = copy.deepcopy(saved)
    normalized["command"] = "compare-many"
    for report in normalized["candidates"].values():
        report["command"] = "compare"
        report.pop("evaluation_manifests")
    assert normalized == direct
    base["per_query"] = dict(reversed(list(base["per_query"].items())))
    mixed["per_query"] = dict(reversed(list(mixed["per_query"].items())))
    reordered = compare_many_saved(base, {"mixed": mixed, "same": base}, **opts)
    for fmt in ("json", "junit", "text", "markdown", "csv"):
        assert render(saved, fmt) == render(reordered, fmt)
    assert saved["candidates"]["mixed"]["baseline_missing_queries"] == ["empty", "q4"]
    assert saved["candidates"]["mixed"]["per_query"]["q4"]["mrr@2"]["candidate"] == 0


@pytest.mark.parametrize("kind,message", [
    ("queries", "judged query sets differ"), ("judgments", "dataset hashes differ"),
    ("metrics", "parameters differ"), ("gain", "parameters differ"),
    ("legacy", "manifest required"), ("row", "missing=['q1']"),
    ("metric-value", "per_query['q2'] fields"), ("checksum", "checksum mismatch"),
    ("version", "unsupported evaluator/version"), ("json", "invalid JSON"),
    ("duplicate-key", "duplicate JSON key"), ("missing-file", "No such file"),
])
def test_invalid_late_member_is_atomic_and_named(family, tmp_path, kind, message):
    from ragwrench.evalgate.saved import saved_evaluation

    qrels, base, cand, metrics, artifacts, paths = family
    bad = copy.deepcopy(artifacts["regressed"])
    if kind == "queries":
        bad = saved_evaluation({"q2": {"d": 1}, "q3": {"d": 1}}, cand, metrics)
    elif kind == "judgments":
        bad = saved_evaluation({"q1": {"d": 0}, "q2": {"d": 1}}, cand, metrics)
    elif kind == "metrics":
        bad = saved_evaluation(qrels, cand, metrics[:1])
    elif kind == "gain":
        bad = saved_evaluation(qrels, cand, metrics, "exponential")
    elif kind == "legacy":
        bad.pop("manifest")
    elif kind == "row":
        bad["per_query"].pop("q1")
    elif kind == "metric-value":
        bad["per_query"]["q2"].pop("hit@1")
    elif kind == "checksum":
        bad["manifest"]["run"]["sha256"] = "0" * 64
    elif kind == "version":
        bad["manifest"]["evaluator"]["version"] = "unknown"
    path = tmp_path / "bad.json"
    path.write_text("{" if kind == "json" else '{"x":1,"x":2}' if kind == "duplicate-key" else json.dumps(bad))
    if kind == "missing-file":
        path.unlink()
    output = tmp_path / "report"
    output.write_text("KEEP")
    result = many(family, "--candidate", f"z_bad={path}", "--output", output, "--force")
    assert result.returncode == 2, result.stderr
    assert "candidate 'z_bad'" in result.stderr and message in result.stderr
    assert not result.stdout and "Traceback" not in result.stderr
    assert output.read_text() == "KEEP" and not list(tmp_path.glob(".retgate-*"))
    # Invalid input also never creates a new output file.
    output.unlink()
    result = many(family, "--candidate", f"z_bad={path}", "--output", output)
    assert result.returncode == 2 and not output.exists() and not result.stdout


def test_api_validates_every_member_before_any_inference(family, monkeypatch):
    from ragwrench.evalgate.compare import compare_many_saved
    from ragwrench.evalgate.formats import InputError

    artifacts = family[-2]
    invalid = copy.deepcopy(artifacts["regressed"])
    invalid["per_query"].pop("q1")

    def forbidden(*args, **kwargs):
        pytest.fail("inference must not start before all saved artifacts pass validation")

    monkeypatch.setattr("ragwrench.evalgate.compare.paired_test", forbidden)
    with pytest.raises(InputError, match="candidate 'z_bad'.*per-query"):
        compare_many_saved(artifacts["base"], {"a_valid": artifacts["base"], "z_bad": invalid}, correction="holm")
    with pytest.raises(InputError, match="baseline.*per-query"):
        compare_many_saved(invalid, {"valid": artifacts["base"]}, correction="holm")


@pytest.mark.parametrize("extra", [
    ["--candidate", "same=missing"], ["--candidate", "bad/name=missing"],
    ["--candidate", "missing-equals"], ["--candidate", "empty="],
    ["--alpha", "NaN"], ["--alpha", "0"],
    ["--candidate-max-drop", "unknown:hit@1=0.1"],
    ["--candidate-max-drop", "strict:hit@2=0.1"],
    ["--candidate-max-drop", "relaxed:hit@1=0.3"],
    ["--candidate-max-drop", "strict:hit@1=nan"],
    ["--max-drop", "hit@2=0.1"], ["--max-drop", "recall@1=-1"],
    ["--samples", "0"], ["--top", "-1"], ["--metrics", "hit@2"], ["--gain", "linear"],
])
def test_bad_family_options_preserve_output(family, tmp_path, extra):
    output = tmp_path / "report"
    output.write_text("KEEP")
    result = many(family, *extra, "--output", output, "--force")
    assert result.returncode == 2 and "error:" in result.stderr
    assert not result.stdout and "Traceback" not in result.stderr
    assert output.read_text() == "KEEP"


@pytest.mark.parametrize("fmt", ["json", "junit", "text", "markdown", "csv"])
def test_cli_reproducible_reports_without_original_sources(family, fmt):
    result = many(family, "--format", fmt)
    assert result.returncode == 1, result.stderr
    assert result.stdout == many(family, "--format", fmt).stdout
    if fmt == "csv":
        import csv
        import io
        assert len(list(csv.DictReader(io.StringIO(result.stdout)))) == 12
    elif fmt in {"text", "markdown"}:
        assert "holm" in result.stdout and "adjusted" in result.stdout
        assert "Missing baseline/candidate: 0/1" in result.stdout


def test_saved_family_baseline_and_all_candidates_protected(family):
    for path in family[-1].values():
        before = path.read_bytes()
        result = many(family, "--output", path, "--force")
        assert result.returncode == 2 and path.read_bytes() == before
    family[-1]["base"].write_text("{}")
    result = many(family)
    assert result.returncode == 2 and "baseline:" in result.stderr and not result.stdout


def test_saved_family_requires_explicit_correction_and_accepts_no_gates(family):
    paths = family[-1]
    args = ("compare-many-saved", "--baseline", paths["base"], "--candidate", f"r={paths['regressed']}",
            "--test", "ttest", "--format", "json")
    result = cli(*args)
    assert result.returncode == 2 and "--correction" in result.stderr
    for gates in ([], ["--max-drop", "hit@1=0.5"]):
        result = cli(*args, "--correction", "bonferroni", *gates)
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["passed"] and report["policy"]["family_size"] == 2
