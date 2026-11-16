"""Synthetic TREC/JSONL integrations for one-baseline candidate families."""
import csv
import io
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
def inputs(tmp_path):
    qrels, baseline, candidate = (tmp_path / name for name in ("qrels", "base", "cand=run.jsonl"))
    qrels.write_text("q1 0 d 1\nq2 0 d 1\n")
    baseline.write_text("q1 Q0 d 1 1 base\nq2 Q0 d 1 1 base\n")
    # q1 is absent, so its candidate score is zero. Extra run queries are ignored.
    candidate.write_text('{"query_id":"q2","doc_id":"d","score":1}\n'
                         '{"query_id":"extra","doc_id":"d","score":1}\n')
    return qrels, baseline, candidate


def many(inputs, *extra, correction="holm"):
    qrels, baseline, candidate = inputs
    return cli("compare-many", "--qrels", qrels, "--baseline", baseline,
               "--candidate", f"same={baseline}", "--candidate", f"regressed={candidate}",
               "--metrics", "hit@1", "--test", "ttest", "--correction", correction,
               "--max-drop", "hit@1=0.49", "--format", "json", *extra)


def test_fixed_baseline_family_and_per_query_export(inputs):
    result = many(inputs)
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["command"] == "compare-many" and not report["passed"]
    assert report["policy"] == {
        "correction": "holm", "alpha": 0.05, "family": "all candidate-metric pairs",
        "family_size": 2, "ci_adjustment": "none", "gate": "absolute-mean-drop",
        "significance_affects_gate": False, "thresholds": {"hit@1": 0.49},
        "candidate_thresholds": {},
        "threshold_precedence": "candidate override > shared metric default > no gate",
        "effective_thresholds": {"regressed": {"hit@1": .49}, "same": {"hit@1": .49}},
    }
    assert list(report["candidates"]) == ["regressed", "same"]
    regressed = report["candidates"]["regressed"]
    assert regressed["candidate_missing_queries"] == ["q1"]
    assert regressed["candidate_ignored_queries"] == ["extra"]
    row = regressed["metrics"]["hit@1"]
    assert (row["baseline"], row["candidate"], row["delta"]) == (1, .5, -.5)
    assert (row["wins"], row["ties"], row["losses"]) == (0, 1, 1)
    assert row["p_value"] == pytest.approx(.5)
    assert row["adjusted_p_value"] == 1 and row["significant"] is False
    assert not row["passed"]  # Nonsignificance never suppresses an absolute-drop failure.
    assert regressed["per_query"]["q1"]["hit@1"] == {
        "baseline": 1, "candidate": 0, "delta": -1, "outcome": "loss"}
    same = report["candidates"]["same"]
    assert same["passed"] and same["metrics"]["hit@1"]["ties"] == 2
    assert report["violations"] == [
        {"candidate": "regressed", "metric": "hit@1", "drop": .5, "max_drop": .49}]


def test_csv_per_query_export_and_boundary(inputs):
    result = many(inputs, "--format", "csv")
    assert result.returncode == 1, result.stderr
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert rows == [
        {"candidate_name": name, "query_id": qid, "metric": "hit@1", "baseline": "1.0",
         "candidate": str(score), "delta": str(delta), "outcome": outcome}
        for name, qid, score, delta, outcome in [
            ("regressed", "q1", 0.0, -1.0, "loss"), ("regressed", "q2", 1.0, 0.0, "tie"),
            ("same", "q1", 1.0, 0.0, "tie"), ("same", "q2", 1.0, 0.0, "tie")]]
    # Exactly 0.5 is permitted; no thresholds also means no failures.
    qrels, baseline, candidate = inputs
    for thresholds in ([], ["--max-drop", "hit@1=0.5"]):
        result = cli("compare-many", "--qrels", qrels, "--baseline", baseline,
                     "--candidate", f"r={candidate}", "--metrics", "hit@1", "--test", "ttest",
                     "--correction", "none", "--format", "json", *thresholds)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["passed"]


@pytest.mark.parametrize("fmt", ["text", "markdown", "junit"])
def test_family_reports(inputs, fmt):
    result = many(inputs, "--format", fmt)
    assert result.returncode == 1, result.stderr
    if fmt == "junit":
        suite = ET.fromstring(result.stdout)
        assert suite.attrib["tests"] == "2" and suite.attrib["failures"] == "1"
        assert {case.attrib["name"] for case in suite} == {"regressed/hit@1", "same/hit@1"}
        assert len(suite.findall("testcase/failure")) == 1
        assert '"correction": "holm"' in suite.find("testcase/system-out").text
    else:
        assert "regressed" in result.stdout and "same" in result.stdout
        assert "holm" in result.stdout and "adjusted" in result.stdout
        assert "significance does not affect gates" in result.stdout


@pytest.mark.parametrize("extra", [
    ["--candidate", "same=missing"], ["--candidate", "no-equals"],
    ["--candidate", "=missing"], ["--candidate", "bad/name=missing"], ["--candidate", "empty="],
    ["--alpha", "NaN"], ["--alpha", "0"], ["--alpha", "1"],
    ["--max-drop", "hit@1=0.1"], ["--max-drop", "ndcg@10=0.1"],
])
def test_invalid_family_options_preserve_existing_output(inputs, tmp_path, extra):
    output = tmp_path / "report"
    output.write_text("KEEP")
    result = many(inputs, *extra, "--output", output, "--force")
    assert result.returncode == 2, result.stderr
    assert result.stdout == "" and "error:" in result.stderr and "Traceback" not in result.stderr
    assert output.read_text() == "KEEP"
    assert not list(tmp_path.glob(".retgate-*"))


def test_explicit_correction_required_and_bad_candidate_is_atomic(inputs, tmp_path):
    qrels, baseline, candidate = inputs
    result = cli("compare-many", "--qrels", qrels, "--baseline", baseline,
                 "--candidate", f"same={baseline}")
    assert result.returncode == 2 and "--correction" in result.stderr
    candidate.write_text("broken\n")
    output = tmp_path / "report"
    result = many(inputs, "--output", output)
    assert result.returncode == 2 and not output.exists()
    assert result.stdout == "" and "Traceback" not in result.stderr


def test_all_candidate_inputs_protected_from_overwrite(inputs):
    qrels, baseline, candidate = inputs
    for path in inputs:
        before = path.read_bytes()
        result = many(inputs, "--output", path, "--force")
        assert result.returncode == 2 and path.read_bytes() == before


@pytest.mark.parametrize("correction", ["none", "bonferroni", "holm"])
def test_correction_family_spans_candidates_and_metrics(correction):
    from ragwrench.evalgate.compare import compare_many
    from ragwrench.evalgate.metrics import parse_metrics

    # Precision@10 paired deltas are [.1, .2, .3, .4, .5]; @20 halves them.
    # The Student-t p-value is invariant to that scaling, with df=4.
    qrels = {f"q{i}": {f"d{j}": 1 for j in range(1, 6)} for i in range(1, 6)}
    baseline = {qid: {"unjudged": 1} for qid in qrels}
    better = {f"q{i}": {f"d{j}": float(j) for j in range(1, i + 1)} for i in range(1, 6)}
    report = compare_many(qrels, baseline, {"better": better, "same": baseline},
                          parse_metrics("precision@10,precision@20"), correction=correction, method="ttest")
    assert report["policy"]["family_size"] == 4
    for row in report["candidates"]["better"]["metrics"].values():
        raw = 0.013235599563682695  # Independent Student-t reference, also used in test_statistics.
        assert row["p_value"] == pytest.approx(raw)
        assert row["adjusted_p_value"] == pytest.approx(raw * (1 if correction == "none" else 4))
        assert row["significant"] is (correction == "none")
        assert (row["wins"], row["ties"], row["losses"]) == (5, 0, 0)
    assert report["passed"]


@pytest.mark.parametrize("method", ["randomization", "bootstrap", "ttest"])
def test_seeded_family_order_invariance_and_single_comparison_equivalence(method):
    from ragwrench.evalgate.compare import compare, compare_many
    from ragwrench.evalgate.metrics import parse_metrics
    from ragwrench.evalgate.report import render

    qrels = {f"q{i}": {"d": 1} for i in range(7)}
    baseline = {qid: {"d": 1, "x": 2} for qid in qrels}
    candidate = {qid: {"d": 2, "x": i} for i, qid in enumerate(qrels)}
    metrics = parse_metrics("mrr@2,ndcg@2")
    opts = dict(method=method, seed=73, samples=300)
    report = compare_many(qrels, baseline, {"mixed": candidate, "same": baseline}, metrics,
                          correction="holm", **opts)
    reversed_report = compare_many(dict(reversed(list(qrels.items()))), dict(reversed(list(baseline.items()))),
                                   {"same": baseline, "mixed": dict(reversed(list(candidate.items())))},
                                   list(reversed(metrics)), correction="holm", **opts)
    for fmt in ("json", "text", "markdown", "junit", "csv"):
        assert render(report, fmt) == render(reversed_report, fmt)
    single = compare(qrels, baseline, candidate, metrics, **opts)
    assert single["per_query"] == report["candidates"]["mixed"]["per_query"]
    alone = compare_many(qrels, baseline, {"mixed": candidate}, metrics, correction="holm", **opts)
    for key, row in report["candidates"]["mixed"]["metrics"].items():
        assert {k: v for k, v in row.items() if k not in {"adjusted_p_value", "significant"}} == single["metrics"][key]
        assert (row["p_value"], row["ci"]) == (
            alone["candidates"]["mixed"]["metrics"][key]["p_value"],
            alone["candidates"]["mixed"]["metrics"][key]["ci"])
    if method != "ttest":
        different = compare_many(qrels, baseline, {"mixed": candidate, "same": baseline}, metrics,
                                 correction="holm", **{**opts, "seed": 99})
        a, b = report["candidates"]["mixed"]["metrics"], different["candidates"]["mixed"]["metrics"]
        assert any((a[k]["p_value"], a[k]["ci"]) != (b[k]["p_value"], b[k]["ci"]) for k in a)


def test_csv_quotes_query_ids(inputs):
    qrels, baseline, candidate = inputs
    for path in (qrels, baseline):
        path.write_text(path.read_text().replace("q1", 'q,"1"'))
    result = many(inputs, "--format", "csv")
    assert result.returncode == 1, result.stderr
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert rows[0]["query_id"] == 'q,"1"'
