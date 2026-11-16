import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from ragwrench.tablechunk.benchmark import BM25, evaluate, naive_chunks, run_benchmark, synthetic_fixture
from ragwrench.tablechunk.tokens import count_tokens

ROOT = Path(__file__).parents[2]


def cli(*args, input=None):
    environment = {**os.environ, "NO_PROXY": "*", "HTTP_PROXY": "http://127.0.0.1:1", "HTTPS_PROXY": "http://127.0.0.1:1"}
    return subprocess.run([sys.executable, "-m", "ragwrench.tablechunk", *map(str, args)],
                          input=input, text=True, capture_output=True, env=environment)


def test_cli_fixture_e2e(tmp_path):
    output = tmp_path / "chunks.jsonl"
    run = cli("split", ROOT / "examples/table-chunk/report.md", "--max-tokens", "120", "-o", output)
    assert run.returncode == 0, run.stderr
    chunks = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(chunks) > 3
    assert all(c["metadata"]["token_count"] <= 120 for c in chunks)
    tables = [c for c in chunks if "table_index" in c["metadata"]]
    assert sum(c["metadata"]["row_end"] - c["metadata"]["row_start"] for c in tables) == 12


@pytest.mark.parametrize("adapter,key", [("langchain", "page_content"), ("llamaindex", "text")])
def test_cli_html_adapters(adapter, key):
    run = cli("split", ROOT / "tests/tablechunk/fixtures/export.html", "--adapter", adapter, "--mode", "kv")
    assert run.returncode == 0, run.stderr
    chunks = [json.loads(line) for line in run.stdout.splitlines()]
    assert all(key in c for c in chunks)
    assert any("Capacity / Used: 20" in c[key] for c in chunks)


def test_stdin_warning(tmp_path):
    run = cli("split", "-", input="| A | B |\n|---|---|\n| x |")
    assert run.returncode == 0
    assert "ragged_rows_padded" in run.stderr
    assert json.loads(run.stdout)["metadata"]["source"] == "-"


def test_cli_failures_leave_existing_output_intact(tmp_path):
    bad = tmp_path / "bad.html"
    bad.write_text("<table><tr><td>missing closes")
    output = tmp_path / "output.jsonl"
    output.write_text("existing")
    run = cli("split", bad, "-o", output)
    assert run.returncode == 2 and "unclosed" in run.stderr
    assert output.read_text() == "existing"
    assert not list(tmp_path.glob(".tablechunk-*"))


@pytest.mark.parametrize("args,expected", [
    (("split", "missing-file.md"), "No such file"),
    (("split", "-", "--max-tokens", "0"), "positive integer"),
    (("split", "-", "--mode", "sql"), "invalid choice"),
    (("bench", "--rows", "0"), "rows=1..500"),
    (("bench", "--corpus", "x.md"), "used together"),
])
def test_cli_errors(args, expected):
    run = cli(*args, input="content")
    assert run.returncode == 2
    assert expected in run.stderr


def test_invalid_utf8_and_input_limit(tmp_path):
    path = tmp_path / "invalid.md"
    path.write_bytes(b"\xff")
    assert cli("split", path).returncode == 2
    path.write_bytes(b"x" * (10 * 1024 * 1024 + 1))
    run = cli("split", path)
    assert run.returncode == 2 and "10 MiB" in run.stderr


def test_bm25_known_ranking_and_metrics():
    documents = ["apple apple pear", "banana banana", "apple banana"]
    assert BM25(documents).rank("apple")[0] == 0
    assert BM25(documents).rank("banana")[0] == 1
    assert BM25(documents).rank("unknown") == [0, 1, 2]
    metrics = evaluate(["PLAN Seats 25", "nothing"],
                       [{"query": "PLAN Seats", "entity": "PLAN", "column": "Seats", "answer": "25"}], 1)
    assert metrics["hit@1"] == 1 and metrics["mrr"] == 1
    missing = evaluate(["PLAN 25"], [{"query": "PLAN Seats", "entity": "PLAN", "column": "Seats", "answer": "25"}], 1)
    assert missing["hit@1"] == 0 and missing["entity_hit@1"] == 1


def test_benchmark_deterministic_and_improves_fixture():
    first = run_benchmark(rows=20, tables=2)
    assert first == run_benchmark(rows=20, tables=2)
    methods = first["methods"]
    for name in ("table_markdown", "table_kv"):
        assert methods[name]["hit@3"] >= methods["naive_fixed"]["hit@3"]
        assert methods[name]["mrr"] >= methods["naive_fixed"]["mrr"]
        assert methods[name]["answerable_fraction"] == 1
    assert methods["table_markdown"]["hit@3"] > methods["naive_fixed"]["hit@3"]
    assert all(m["max_chunk_tokens"] <= 120 for m in methods.values())


def test_cli_benchmark_replay(tmp_path):
    directory = tmp_path / "fixture"
    generated = cli("bench", "--rows", 8, "--tables", 2, "--fixture-dir", directory)
    assert generated.returncode == 0, generated.stderr
    replay = cli("bench", "--corpus", directory / "corpus.md", "--queries", directory / "queries.json")
    assert replay.returncode == 0, replay.stderr
    left, right = json.loads(generated.stdout), json.loads(replay.stdout)
    assert left == right
    rerun = cli("bench", "--rows", 8, "--tables", 2, "--fixture-dir", directory)
    assert rerun.returncode == 2


def test_naive_fixed_token_budget_and_content():
    corpus, _ = synthetic_fixture(3, 1)
    chunks = naive_chunks(corpus, 12)
    assert all(count_tokens(chunk) <= 12 for chunk in chunks)
    assert "PLAN000002" in "".join(chunks)
