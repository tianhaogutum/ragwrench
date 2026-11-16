import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from ragwrench.evalgate.compare import compare
from ragwrench.evalgate.formats import read_qrels, read_run
from ragwrench.evalgate.metrics import parse_metrics

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "eval"
FIXTURES = ROOT / "tests" / "evalgate" / "fixtures"


def cli(*args):
    return subprocess.run([sys.executable, "-m", "ragwrench.evalgate", *map(str, args)], capture_output=True, text=True, cwd=ROOT, timeout=30)


def comparison(*args):
    return cli("compare", "--qrels", EXAMPLES / "qrels.tsv", "--baseline", EXAMPLES / "base.trec", "--candidate", EXAMPLES / "cand.trec", "--metrics", "ndcg@10", "--samples", "500", *args)


def test_eval_real_beir_trec_e2e():
    result = cli("run", "--qrels", EXAMPLES / "qrels.tsv", "--run", EXAMPLES / "base.trec")
    assert result.returncode == 0, result.stderr
    obj = json.loads(result.stdout)
    assert obj["means"] == {"ndcg@10": 1, "mrr@10": 1, "recall@100": 1}
    assert len(obj["per_query"]) == 3


def test_legacy_eval_alias_matches_run_and_stays_hidden():
    args = ("--qrels", EXAMPLES / "qrels.tsv", "--run", EXAMPLES / "base.trec")
    legacy, current = cli("eval", *args), cli("run", *args)
    assert legacy.returncode == current.returncode == 0, legacy.stderr
    assert legacy.stdout == current.stdout
    usage = cli("--help").stdout
    assert "run" in usage and "{run," in usage and "eval," not in usage


def test_llama_jsonl_e2e():
    result = cli("run", "--qrels", FIXTURES / "llamaindex.json", "--run", FIXTURES / "results.jsonl")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["means"]["recall@100"] == 1


@pytest.mark.parametrize("fmt", ["text", "json", "markdown", "junit"])
def test_compare_gate_and_reports(fmt):
    result = comparison("--max-drop", "ndcg@10=0.01", "--format", fmt)
    assert result.returncode == 1, result.stderr
    if fmt == "json":
        obj = json.loads(result.stdout)
        assert obj["passed"] is False
        assert obj["metrics"]["ndcg@10"]["losses"] == 2
        assert obj["metrics"]["ndcg@10"]["top_regressed"][0]["query_id"] == "q1"
    elif fmt == "junit":
        root = ET.fromstring(result.stdout)
        assert root.attrib["failures"] == "1"
        assert root.find("testcase/failure") is not None
    else:
        assert "FAIL: ndcg@10 dropped" in result.stdout
        assert "q1:" in result.stdout


def test_gate_disabled_or_allowed():
    for args in ([], ["--max-drop", "ndcg@10=0.9"]):
        result = comparison(*args)
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("PASS")


@pytest.mark.parametrize("args", [
    ["--metrics", "foo@10"], ["--max-drop", "recall@100=0.1"],
    ["--max-drop", "ndcg@10=NaN"], ["--max-drop", "ndcg@10=-0.1"],
    ["--max-drop", "ndcg@10=1", "--max-drop", "ndcg@10=2"],
    ["--max-drop", "bad"], ["--top", "-1"], ["--samples", "-1"], ["--confidence", "NaN"],
])
def test_cli_invalid_options(args):
    result = comparison(*args)
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


def test_no_shared_empty_and_malformed_cli(tmp_path):
    path = tmp_path / "run"
    for content in ("q-unrelated Q0 d 1 1 t\n", "", "broken\n"):
        path.write_text(content)
        result = cli("run", "--qrels", EXAMPLES / "qrels.tsv", "--run", path)
        assert result.returncode == 2
        assert "error:" in result.stderr and "Traceback" not in result.stderr


def test_compare_pairing_order_and_metric_order():
    qrels = read_qrels(EXAMPLES / "qrels.tsv")
    base, cand = read_run(EXAMPLES / "base.trec"), read_run(EXAMPLES / "cand.trec")
    a = compare(qrels, base, cand, parse_metrics("ndcg@10,mrr@10"), samples=500)
    b = compare(dict(reversed(list(qrels.items()))), dict(reversed(list(base.items()))), cand,
                parse_metrics("mrr@10,ndcg@10"), samples=500)
    assert a == b


def test_threshold_exact_boundary():
    qrels = {"q1": {"d": 1}, "q2": {"d": 1}}
    baseline = {"q1": {"d": 1}, "q2": {"d": 1}}
    candidate = {"q1": {"x": 1}, "q2": {"d": 1}}
    m = parse_metrics("hit@1")
    assert compare(qrels, baseline, candidate, m, thresholds={"hit@1": .5}, samples=100)["passed"]
    assert not compare(qrels, baseline, candidate, m, thresholds={"hit@1": .49}, samples=100)["passed"]


def test_convert_round_trip_cli(tmp_path):
    trec, beir = tmp_path / "qrels.trec", tmp_path / "qrels.tsv"
    a = cli("convert", "--input", EXAMPLES / "qrels.tsv", "--from", "beir-qrels", "--to", "trec-qrels", "--output", trec)
    assert a.returncode == 0, a.stderr
    b = cli("convert", "--input", trec, "--from", "trec-qrels", "--to", "beir-qrels", "--output", beir)
    assert b.returncode == 0, b.stderr
    assert read_qrels(beir) == read_qrels(EXAMPLES / "qrels.tsv")
    run = tmp_path / "run.trec"
    c = cli("convert", "--input", FIXTURES / "results.jsonl", "--from", "jsonl-run", "--to", "trec-run", "--output", run)
    assert c.returncode == 0, c.stderr
    assert read_run(run) == read_run(FIXTURES / "results.jsonl")
    d = cli("convert", "--input", FIXTURES / "llamaindex.json", "--from", "llamaindex", "--to", "trec-qrels")
    assert d.returncode == 0, d.stderr
    assert "q1 0 d1 1" in d.stdout


def test_output_no_clobber_force_and_no_input_overwrite(tmp_path):
    output = tmp_path / "report.json"
    output.write_text("KEEP")
    result = comparison("--format", "json", "--output", output)
    assert result.returncode == 2 and output.read_text() == "KEEP"
    result = comparison("--format", "json", "--output", output, "--force")
    assert result.returncode == 0 and json.loads(output.read_text())["passed"]
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text((EXAMPLES / "qrels.tsv").read_text())
    result = cli("convert", "--input", qrels, "--from", "beir-qrels", "--to", "trec-qrels", "--output", qrels, "--force")
    assert result.returncode == 2
    assert qrels.read_text().startswith("query-id")
    assert not list(tmp_path.glob(".retgate-*"))


def test_symlink_output_rejected_and_cross_kind_conversion(tmp_path):
    target = tmp_path / "target"
    target.write_text("KEEP")
    link = tmp_path / "link"
    link.symlink_to(target)
    assert comparison("--output", link, "--force").returncode == 2
    assert target.read_text() == "KEEP"
    result = cli("convert", "--input", EXAMPLES / "base.trec", "--from", "trec-run", "--to", "beir-qrels")
    assert result.returncode == 2 and "cannot convert" in result.stderr


def test_reports_xml_special_ids_and_eval_formats(tmp_path):
    for fmt in ("text", "markdown", "junit"):
        result = cli("run", "--qrels", EXAMPLES / "qrels.tsv", "--run", EXAMPLES / "base.trec", "--format", fmt)
        assert result.returncode == 0
        if fmt == "junit":
            assert ET.fromstring(result.stdout).attrib["failures"] == "0"
    qrels = tmp_path / "qrels"
    run = tmp_path / "run"
    qrels.write_text('q<& 0 d 1\nq2 0 d 1\n')
    run.write_text('q<& Q0 d 1 1 t\nq2 Q0 d 1 1 t\n')
    result = cli("compare", "--qrels", qrels, "--baseline", run, "--candidate", run, "--format", "junit", "--samples", "100")
    assert result.returncode == 0, result.stderr
    assert ET.fromstring(result.stdout).attrib["failures"] == "0"


def test_reproducible_example_scripts():
    for script in ("tie_invariance_demo.py", "paired_experiment.py"):
        result = subprocess.run([sys.executable, str(EXAMPLES / script)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert isinstance(json.loads(result.stdout), dict)


def test_markdown_query_ids_cannot_inject_html(tmp_path):
    qrels, base, cand = (tmp_path / name for name in ("qrels", "base", "cand"))
    qid = "<script>alert(1)</script>|[x]"
    qrels.write_text(f"{qid} 0 d 1\nq2 0 d 1\n")
    base.write_text(f"{qid} Q0 d 1 1 t\nq2 Q0 d 1 1 t\n")
    cand.write_text(f"{qid} Q0 x 1 1 t\nq2 Q0 d 1 1 t\n")
    result = cli("compare", "--qrels", qrels, "--baseline", base, "--candidate", cand,
                 "--metrics", "hit@1", "--format", "markdown", "--samples", "100")
    assert result.returncode == 0, result.stderr
    assert "<script>" not in result.stdout
    assert "&lt;script" in result.stdout
    assert r"\|\[x\]" in result.stdout


@pytest.mark.parametrize("method", ["bootstrap", "randomization", "ttest"])
def test_all_inference_methods_through_cli(method):
    result = comparison("--test", method, "--format", "json", "--max-drop", "ndcg@10=0.01")
    assert result.returncode == 1, result.stderr
    row = json.loads(result.stdout)["metrics"]["ndcg@10"]
    assert row["test"] == method
    assert 0 <= row["p_value"] <= 1
    assert len(row["ci"]) == 2


def test_data_ids_are_not_filesystem_paths(tmp_path):
    source, output = tmp_path / "source.jsonl", tmp_path / "run.trec"
    source.write_text(json.dumps({"query_id": "../../query", "doc_id": "../../escape", "score": 1}) + "\n")
    result = cli("convert", "--input", source, "--from", "jsonl-run", "--to", "trec-run", "--output", output)
    assert result.returncode == 0, result.stderr
    assert read_run(output) == {"../../query": {"../../escape": 1}}
    assert {p.name for p in tmp_path.iterdir()} == {"source.jsonl", "run.trec"}


def test_invalid_input_never_leaves_output(tmp_path):
    source, output = tmp_path / "bad.trec", tmp_path / "must-not-exist"
    source.write_text("q Q0 d 1 1 tag\nmalformed\n")
    result = cli("convert", "--input", source, "--from", "trec-run", "--to", "jsonl-run", "--output", output)
    assert result.returncode == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".retgate-*"))
