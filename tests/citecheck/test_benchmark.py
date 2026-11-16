import json

import pytest

from ragwrench.citecheck.benchmark import metrics, run_benchmark
from ragwrench.citecheck.cli import main
from ragwrench.citecheck.model import InputError


def test_real_replay_is_deterministic_and_better_than_marker_baseline():
    a, b = run_benchmark(), run_benchmark()
    assert a["case_count"] >= 40
    assert a["dataset_sha256"] == b["dataset_sha256"]
    assert a["cases"] == b["cases"]
    assert a["metrics"] == b["metrics"]
    assert a["metrics"]["micro"]["f1"] > a["baseline_metrics"]["micro"]["f1"]
    assert all(a["metrics"][f"CC00{i}"]["tp"] >= 5 for i in range(1, 9))
    # Do not compare elapsed time or claim zero false positives on other data.
    assert a["elapsed_ms"]["checker"] > 0


def test_metrics_do_not_hide_false_positives_or_false_negatives():
    m = metrics([{"CC004"}, {"CC004"}, set()], [{"CC004"}, set(), {"CC004"}])["CC004"]
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 0)
    assert m["precision"] == m["recall"] == m["f1"] == 0.5


def test_benchmark_does_not_derive_labels_from_predictions(tmp_path):
    path = tmp_path / "wrong-label.jsonl"
    path.write_text(json.dumps({"id": "deliberate-label-error", "expected": ["CC004"], "record": {
        "answer": "The rover uses solar panels.", "sources": []}}) + "\n")
    report = run_benchmark(str(path))
    assert report["metrics"]["CC004"]["fn"] == 1
    assert report["metrics"]["CC002"]["fp"] == 1
    assert report["cases"][0]["false_negatives"] == ["CC004"]


def test_bench_cli_writes_measured_report(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["bench"]) == 0
    report = json.loads((tmp_path / "citecheck-bench.json").read_text())
    assert report["kind"] == "citecheck-benchmark"
    assert report["case_count"] >= 40 and "baseline_metrics" in report
    assert "marker-exists baseline" in capsys.readouterr().out
    assert main(["bench"]) == 2  # Never silently overwrite an existing file.
    assert main(["bench", "--force"]) == 0


@pytest.mark.parametrize("case", [{}, {"id": "x", "expected": ["CC999"]}, {"id": "x", "expected": ["CC001", "CC001"]}])
def test_bad_corpus_rejected(tmp_path, case):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(case))
    with pytest.raises(InputError, match="benchmark line"):
        run_benchmark(str(path))
