import json
import os
import subprocess
import sys

import pytest

from ragwrench.bm25.cli import main
from ragwrench.bm25.evaluation import evaluate, metrics


def invoke(*args):
    return subprocess.run(
        [sys.executable, "-m", "ragwrench.bm25", *map(str, args)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "10"},
    )


def test_cli_end_to_end(tmp_path):
    store = tmp_path / "qd"
    indexed = invoke(
        "index",
        "--path",
        store,
        "--collection",
        "kb",
        "examples/bm25/corpus_zh.jsonl",
        "--tokenizer",
        "cjk-bigram",
    )
    assert indexed.returncode == 0, indexed.stderr
    assert json.loads(indexed.stdout)["indexed"] == 4
    result = invoke(
        "search", "--path", store, "--collection", "kb", "向量数据库 备份", "--top-k", "3"
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)[0]["id"] == "d1"
    mismatch = invoke(
        "search", "--path", store, "--collection", "kb", "向量", "--tokenizer", "unicode"
    )
    assert mismatch.returncode == 2
    assert "differs" in mismatch.stderr and mismatch.stdout == ""
    encoded = invoke("encode", "Qdrant向量数据库", "--document", "--avgdl", "4")
    assert encoded.returncode == 0, encoded.stderr
    payload = json.loads(encoded.stdout)
    assert payload["tokens"] == ["qdrant", "向量", "量数", "数据", "据库"]
    assert payload["collision_stats"]["collision_count"] == 0


@pytest.mark.parametrize(
    "args,match",
    [
        (["encode", "abc", "--k1", "nan"], "finite"),
        (["encode", "abc", "--b", "2"], "require"),
        (["search", "--path", "nonexistent-store", "--collection", "kb", "abc"], "does not exist"),
    ],
)
def test_cli_errors(args, match, capsys):
    assert main(args) == 2
    output = capsys.readouterr()
    assert match in output.err and output.out == ""


def test_invalid_input_does_not_create_store(tmp_path, capsys):
    source = tmp_path / "bad.jsonl"
    source.write_text('{"id":"d1","text":"ok"}\ninvalid')
    store = tmp_path / "never-created"
    assert main(["index", "--path", str(store), "--collection", "kb", str(source)]) == 2
    assert not store.exists()
    assert "line 2" in capsys.readouterr().err
    source.write_text('{"id":"d1","text":"ok"}')
    assert main(["index", "--path", str(store), "--collection", "../escape", str(source)]) == 2
    assert not store.exists()
    assert "collection must" in capsys.readouterr().err


def test_empty_cli_query(tmp_path, capsys):
    store = tmp_path / "store"
    assert (
        main(["index", "--path", str(store), "--collection", "kb", "examples/bm25/corpus_zh.jsonl"]) == 0
    )
    capsys.readouterr()
    with pytest.warns(UserWarning, match="empty"):
        assert main(["search", "--path", str(store), "--collection", "kb", ""]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_evaluation_deterministic():
    first = evaluate()
    assert first == evaluate()
    bigram = first["tokenizers"]["cjk-bigram"]
    whitespace = first["tokenizers"]["whitespace"]
    assert bigram["by_language"]["zh"]["recall@3"] > whitespace["by_language"]["zh"]["recall@3"]
    assert bigram["overall"]["recall@3"] == 1
    for result in first["tokenizers"].values():
        assert result["reference_max_abs_score_error"] < 1e-5
        assert result["collision_stats"]["collision_count"] == 0


def test_eval_cli_writes_results(tmp_path, capsys):
    output = tmp_path / "result.json"
    assert main(["eval", "--output", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert json.loads(output.read_text()) == report
    assert report["documents"] == 12 and report["queries"] == 9
    assert list(tmp_path.iterdir()) == [output]


def test_metrics_not_mocked():
    assert metrics(["bad", "r1", "r2"], ["r1", "r2", "r3"]) == {"recall": 2 / 3, "mrr": 0.5}
    assert metrics([], ["r1"]) == {"recall": 0, "mrr": 0}
    with pytest.raises(ValueError, match="top-k"):
        evaluate(top_k=0)


def test_encoder_and_reference_need_no_qdrant():
    source = """
import builtins
original = builtins.__import__
def block(name, *args, **kwargs):
    if name.startswith(('qdrant_client', 'numpy')):
        raise RuntimeError('unexpected integration dependency')
    return original(name, *args, **kwargs)
builtins.__import__ = block
from ragwrench.bm25 import Encoder
from ragwrench.bm25.reference import bm25_search
assert Encoder().encode_query('数据库').values == [1.0, 1.0]
assert bm25_search([{'id': 'a', 'text': '数据库支持备份'}], '数据库')[0]['id'] == 'a'
"""
    result = subprocess.run([sys.executable, "-c", source], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_cli_rejects_remote_path_before_writing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["search", "--path", "https://example.com", "--collection", "kb", "text"]) == 2
    assert "local directory" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_cli_incremental_update_and_rebuild(tmp_path):
    store = tmp_path / "qd"
    source, addition, deletion, final = [
        tmp_path / name for name in ("initial.jsonl", "add.jsonl", "delete.json", "final.jsonl")
    ]
    initial = [{"id": 7, "text": "数据库 数据库"}, {"id": "7", "text": "数据库 备份"}]
    new = {"id": "c", "text": "数据库 快照 恢复 备份"}
    for path, docs in ((source, initial), (addition, [new]), (final, [initial[1], new])):
        path.write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in docs), encoding="utf-8"
        )
    deletion.write_text("[7]")
    indexed = invoke("index", "--path", store, "--collection", "kb", source)
    assert indexed.returncode == 0, indexed.stderr
    updated = invoke(
        "update", "--path", store, "--collection", "kb", "--add", addition, "--delete-ids", deletion
    )
    assert updated.returncode == 0, updated.stderr
    report = json.loads(updated.stdout)
    assert report["indexed"] == 2 and report["added"] == 1 and report["deleted"] == 1
    assert report["encoder"]["avgdl"] == 4
    assert report["corpus"]["total_tokens"] == 8
    assert invoke("index", "--path", store, "--collection", "rebuilt", final).returncode == 0
    actual = invoke("search", "--path", store, "--collection", "kb", "数据库 备份")
    expected = invoke("search", "--path", store, "--collection", "rebuilt", "数据库 备份")
    assert actual.returncode == expected.returncode == 0
    assert json.loads(actual.stdout) == json.loads(expected.stdout)
    assert [h["id"] for h in json.loads(actual.stdout)] == ["7", "c"]
    # Unknown and malformed deletions fail without changing the index.
    for content in ("[7]", "[true]", '["7", "7"]', "{}", "[NaN]", "[7,"):
        deletion.write_text(content)
        failed = invoke("update", "--path", store, "--collection", "kb", "--delete-ids", deletion)
        assert failed.returncode == 2 and failed.stdout == ""
        unchanged = invoke("search", "--path", store, "--collection", "kb", "数据库 备份")
        assert json.loads(unchanged.stdout) == json.loads(actual.stdout)


def test_invalid_update_input_does_not_open_store(tmp_path, capsys):
    missing = tmp_path / "missing"
    deletion = tmp_path / "delete.json"
    deletion.write_text("[false]")
    assert (
        main(
            ["update", "--path", str(missing), "--collection", "kb", "--delete-ids", str(deletion)]
        )
        == 2
    )
    assert "id must" in capsys.readouterr().err
    assert not missing.exists()
    assert main(["update", "--path", str(missing), "--collection", "kb"]) == 2
    assert "at least one" in capsys.readouterr().err
    assert not missing.exists()
