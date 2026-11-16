"""Saved evaluations use synthetic local TREC/JSONL inputs, never a retriever."""
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
def saved(tmp_path):
    qrels, base, cand = [tmp_path / name for name in ("qrels", "base", "cand.jsonl")]
    qrels.write_text("q1 0 d 1\nq2 0 d 1\n")
    base.write_text("q1 Q0 d 1 1 tag\nq2 Q0 d 1 1 tag\n")
    cand.write_text('{"query_id":"q2","doc_id":"d","score":1}\n'
                    '{"query_id":"extra","doc_id":"d","score":1}\n')
    outputs = [tmp_path / name for name in ("baseline.json", "candidate.json")]
    for run, output in zip((base, cand), outputs):
        result = cli("run", "--qrels", qrels, "--run", run, "--metrics", "hit@1", "--output", output)
        assert result.returncode == 0, result.stderr
    return qrels, base, cand, *outputs


def comparison(saved, *extra):
    return cli("compare-saved", "--baseline", saved[3], "--candidate", saved[4],
               "--test", "ttest", "--max-drop", "hit@1=0.49", "--format", "json", *extra)


def test_saved_manifest_and_missing_query_gate(saved):
    base, cand = [json.loads(path.read_text()) for path in saved[3:]]
    assert "manifest" in base, "eval JSON must bind its inputs and parameters"
    assert base["manifest"]["dataset"] == cand["manifest"]["dataset"]
    assert base["manifest"]["run"] != cand["manifest"]["run"]
    assert base["manifest"]["dataset"]["query_ids"] == ["q1", "q2"]
    assert base["manifest"]["parameters"]["metrics"] == ["hit@1"]
    result = comparison(saved)
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["command"] == "compare-saved"
    assert report["candidate_missing_queries"] == ["q1"]
    assert report["candidate_ignored_queries"] == ["extra"]
    assert report["query_count"] == 2
    row = report["metrics"]["hit@1"]
    assert (row["baseline"], row["candidate"], row["delta"]) == (1, .5, -.5)
    assert report["per_query"]["q1"]["hit@1"]["candidate"] == 0
    assert report["evaluation_manifests"]["candidate"] == cand["manifest"]


@pytest.mark.parametrize("qrels,metrics,gain,message", [
    ("q2 0 d 1\nq3 0 d 1\n", "hit@1", "linear", "judged query sets differ"),
    ("q1 0 d 0\nq2 0 d 1\n", "hit@1", "linear", "dataset hashes differ"),
    ("q1 0 d 1\nq2 0 d 1\n", "hit@2", "linear", "parameters differ"),
    ("q1 0 d 1\nq2 0 d 1\n", "hit@1", "exponential", "parameters differ"),
])
def test_incompatible_saved_evaluations_preserve_output(saved, tmp_path, qrels, metrics, gain, message):
    saved[0].write_text(qrels)
    result = cli("run", "--qrels", saved[0], "--run", saved[2], "--metrics", metrics,
                 "--gain", gain, "--output", saved[4], "--force")
    assert result.returncode == 0, result.stderr
    output = tmp_path / "report"
    output.write_text("KEEP")
    result = comparison(saved, "--output", output, "--force")
    assert result.returncode == 2 and message in result.stderr
    assert result.stdout == "" and "Traceback" not in result.stderr
    assert output.read_text() == "KEEP"
    if "query sets" in message:
        assert "q1" in result.stderr and "q3" in result.stderr


def test_canonical_hashes_and_format_conversion(saved, tmp_path):
    from ragwrench.evalgate.formats import read_qrels, read_run, write_qrels, write_run

    before = json.loads(saved[3].read_text())
    saved[0].write_text(write_qrels(read_qrels(saved[0]), "beir"))
    saved[1].write_text(write_run(read_run(saved[1]), "jsonl"))
    # Changes to interchange format, rank, tag and record order do not change inputs to scoring.
    saved[1].write_text("\n".join(reversed(saved[1].read_text().splitlines())) + "\n")
    result = cli("run", "--qrels", saved[0], "--run", saved[1], "--metrics", "hit@1")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == before
    saved[1].write_text(saved[1].read_text().replace('"score": 1.0', '"score": 2.0'))
    changed = json.loads(cli("run", "--qrels", saved[0], "--run", saved[1], "--metrics", "hit@1").stdout)
    assert before["means"] == changed["means"]
    assert before["manifest"]["run"] != changed["manifest"]["run"]
    assert before["manifest"]["artifact_sha256"] != changed["manifest"]["artifact_sha256"]


@pytest.mark.parametrize("method", ["randomization", "bootstrap", "ttest"])
def test_saved_equals_direct_with_missing_from_both_and_zero_judgments(method):
    from ragwrench.evalgate.compare import compare, compare_saved
    from ragwrench.evalgate.metrics import parse_metrics
    from ragwrench.evalgate.saved import saved_evaluation

    qrels = {"q1": {"d": 1}, "q2": {"d": 1}, "q3": {"d": 0}, "empty": {}}
    base = {"q1": {"d": 1.0}, "q2": {"d": 1.0}}
    cand = {"q2": {"d": 1.0}, "extra": {"x": 1.0}}
    metrics = parse_metrics("recall@10,hit@1")
    a = saved_evaluation(qrels, base, metrics)
    b = saved_evaluation(qrels, cand, list(reversed(metrics)))
    opts = dict(method=method, seed=73, samples=300, thresholds={"hit@1": .1})
    result = compare_saved(a, b, **opts)
    direct = compare(qrels, base, cand, sorted(metrics, key=str), **opts)
    assert {k: v for k, v in result.items() if k not in {"command", "evaluation_manifests"}} == {
        k: v for k, v in direct.items() if k != "command"}
    assert result["query_count"] == 4
    assert result["baseline_missing_queries"] == ["empty", "q3"]
    assert result["candidate_missing_queries"] == ["empty", "q1", "q3"]
    row = result["metrics"]["hit@1"]
    assert (row["baseline"], row["candidate"], row["delta"]) == (.5, .25, -.25)
    # JSON object ordering is outside the canonical checksum and must not change
    # which query receives each seeded resampling draw.
    a["per_query"] = dict(reversed(list(a["per_query"].items())))
    b["per_query"] = dict(reversed(list(b["per_query"].items())))
    assert compare_saved(a, b, **opts) == result


@pytest.mark.parametrize("mutation,message", [
    (lambda r: r.pop("manifest"), "manifest required"),
    (lambda r: r["per_query"].pop("q1"), "missing=['q1']"),
    (lambda r: r["per_query"]["q2"].pop("hit@1"), "per_query['q2'] fields"),
    (lambda r: r["per_query"]["q2"].update({"hit@1": True}), "finite numbers"),
    (lambda r: r["per_query"]["q2"].update({"hit@1": 1e300}), "finite numbers"),
    (lambda r: r["per_query"]["q1"].update({"hit@1": 1}), "missing query 'q1' must score zero"),
    (lambda r: r["means"].update({"hit@1": 1}), "inconsistent with complete"),
    (lambda r: r.update({"missing_queries": ["q2", "q1"]}), "sorted and unique"),
    (lambda r: r.update({"ignored_queries": ["q1"]}), "inconsistent with judged"),
    (lambda r: r.update({"query_count": True}), "query count differs"),
    (lambda r: r["manifest"].update({"dataset": []}), "dataset fields"),
    (lambda r: r["manifest"]["parameters"].update({"metrics": [None]}), "nonempty array"),
    (lambda r: r["manifest"]["parameters"].update({"query_policy": "intersection"}), "unsupported ranking/query policy"),
    (lambda r: r["manifest"].update({"schema_version": 2}), "unsupported manifest schema"),
    (lambda r: r["manifest"]["evaluator"].update({"version": "unknown"}), "unsupported evaluator/version"),
    (lambda r: r["manifest"]["run"].update({"sha256": "invalid"}), "SHA-256 hex digest"),
    (lambda r: r["manifest"]["run"].update({"sha256": "0" * 64}), "checksum mismatch"),
])
def test_corrupt_saved_data_is_never_dropped_or_trusted(saved, tmp_path, mutation, message):
    report = json.loads(saved[4].read_text())
    mutation(report)
    saved[4].write_text(json.dumps(report))
    output = tmp_path / "report"
    output.write_text("KEEP")
    result = comparison(saved, "--output", output, "--force")
    assert result.returncode == 2 and message in result.stderr
    assert result.stdout == "" and "Traceback" not in result.stderr
    assert output.read_text() == "KEEP"
    assert not list(tmp_path.glob(".retgate-*"))


@pytest.mark.parametrize("content", ["[]", "null", '{"command":"eval","command":"eval"}',
                                    '{"means":{"hit@1":NaN}}', '{"means":{"hit@1":Infinity}}'])
def test_invalid_saved_json_is_a_cli_error(saved, content):
    saved[4].write_text(content)
    result = comparison(saved)
    assert result.returncode == 2 and "error:" in result.stderr
    assert result.stdout == "" and "Traceback" not in result.stderr


@pytest.mark.parametrize("fmt", ["text", "json", "markdown", "junit"])
def test_saved_reports_after_source_removal_and_input_protection(saved, fmt):
    for path in saved[:3]:
        path.unlink()
    result = comparison(saved, "--format", fmt)
    assert result.returncode == 1, result.stderr
    if fmt == "junit":
        suite = ET.fromstring(result.stdout)
        assert suite.attrib["failures"] == "1"
        assert suite.find("testcase/failure") is not None
        details = json.loads(suite.find("testcase/system-out").text)
        assert details["candidate_missing_queries"] == ["q1"]
        assert details["evaluation_manifests"]["candidate"]["dataset"]["query_ids"] == ["q1", "q2"]
    elif fmt != "json":
        assert "Missing baseline/candidate: 0/1" in result.stdout
    for path in saved[3:]:
        before = path.read_bytes()
        result = comparison(saved, "--output", path, "--force")
        assert result.returncode == 2 and path.read_bytes() == before


def test_saved_thresholds_and_inference_errors(saved):
    for args in (("--max-drop", "recall@10=1"), ("--samples", "0"), ("--top", "-1")):
        result = comparison(saved, *args)
        assert result.returncode == 2 and "error:" in result.stderr
        assert "Traceback" not in result.stderr and result.stdout == ""
    for args in ([], ["--max-drop", "hit@1=0.5"]):
        result = cli("compare-saved", "--baseline", saved[3], "--candidate", saved[4],
                     "--test", "ttest", *args)
        assert result.returncode == 0, result.stderr
