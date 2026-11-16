"""A caller-saved config is explicit, compatible and fixed-average index input."""

from contextlib import closing
import json

import pytest

from ragwrench.bm25 import cli, store
from ragwrench.bm25.encoder import Encoder


DOCS = [
    {"id": 7, "text": "原文 ＡＴＬＡＳ 北京\n서울", "metadata": {"n": 1, "a": [True, None, 1.0]}},
    {"id": "7", "text": "東京 データ é", "metadata": {"n": True, "s": "𠮷"}},
]


def inputs(tmp_path, config=None):
    source, saved, path = tmp_path / "source.jsonl", tmp_path / "config.json", tmp_path / "qd"
    source.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in DOCS), encoding="utf-8")
    saved.write_text(json.dumps(config or Encoder().config, ensure_ascii=False), encoding="utf-8")
    return source, saved, path


def args(source, saved, path, *extra):
    return [
        "index",
        "--path",
        str(path),
        "--collection",
        "kb",
        str(source),
        "--encoder-config",
        str(saved),
        *extra,
    ]


def test_index_saved_config(tmp_path, capsys):
    config = Encoder("whitespace", k1=1.8, b=0.3, avgdl=19, stopwords=("atlas",)).config
    source, saved, path = inputs(tmp_path, config)
    # A valid existing corpus/config and ordinary index are positive controls.
    assert store.read_corpus(source) == DOCS
    assert Encoder.from_config(json.loads(saved.read_text())).config == config
    assert cli.main(["index", "--path", str(path), "--collection", "control", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["created"] is True
    assert cli.main(args(source, saved, path)) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["encoder"] == config
    assert result["corpus"] == {"schema": 1, "avgdl_mode": "fixed", "total_tokens": 6}
    assert result["created"] is True and result["indexed"] == 2 and result["skipped"] == 0
    with closing(store.open_local(path, must_exist=True)) as client:
        assert store.status_collection(client, "kb")["config"] == config
        assert store.export_documents(client, "kb", [7, "7"]) == DOCS


def no_access(*args, **kwargs):
    raise AssertionError("storage must not be opened")


# Every invalid config must fail as a CLI diagnostic, before any storage access.


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"{",
        b"[]",
        b"null",
        b"true",
        b"1",
        b'"config"',
        b"{} {}",
        b'{"schema":1,"schema":1}',
        b'{"extra":{"nested":1,"nested":2}}',
        b'{"extra":NaN}',
        b'{"extra":Infinity}',
        b'{"extra":-Infinity}',
        b'{"extra":1e999}',
        b'{"extra":"\\ud800"}',
        b"\xff",
        b"\xef\xbb\xbf{}",
        b"[" * 1100 + b"]" * 1100,
    ],
)
def test_malformed_before_storage(tmp_path, capsys, monkeypatch, raw):
    source, saved, path = inputs(tmp_path)
    saved.write_bytes(raw)
    monkeypatch.setattr(cli, "open_local", no_access)
    assert cli.main(args(source, saved, path)) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "encoder config" in captured.err
    assert not path.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", True),
        ("schema", 1.0),
        ("schema", "1"),
        ("schema", 2),
        ("tokenizer_version", True),
        ("tokenizer_version", 1.0),
        ("tokenizer_version", 2),
        ("unicode_version", "incompatible"),
        ("unicode_version", None),
        ("jieba_version", "incompatible"),
        ("hash", "unsupported"),
        ("tokenizer", "unknown"),
        ("tokenizer", []),
        ("tokenizer", None),
        ("stopwords", None),
        ("stopwords", {}),
        ("stopwords", "atlas"),
        ("stopwords", [1]),
        ("stopwords", [None]),
        ("stopwords", [["a"]]),
        ("stopwords", ["ＡＴＬＡＳ"]),
        ("stopwords", ["a", "a"]),
        ("stopwords", ["z", "a"]),
        ("k1", True),
        ("k1", "1.2"),
        ("k1", None),
        ("k1", 0),
        ("k1", -1),
        ("k1", 10**400),
        ("b", False),
        ("b", -0.1),
        ("b", 1.1),
        ("avgdl", True),
        ("avgdl", 0),
        ("avgdl", -1),
        ("avgdl", 10**400),
        ("extra", "unsupported"),
    ],
)
def test_incompatible_before_storage(tmp_path, capsys, monkeypatch, field, value):
    config = {**Encoder().config, field: value}
    source, saved, path = inputs(tmp_path, config)
    monkeypatch.setattr(cli, "open_local", no_access)
    assert cli.main(args(source, saved, path)) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "encoder config" in captured.err


@pytest.mark.parametrize("field", list(Encoder().config))
def test_missing_field_before_storage(tmp_path, capsys, monkeypatch, field):
    config = Encoder().config
    del config[field]
    source, saved, path = inputs(tmp_path, config)
    monkeypatch.setattr(cli, "open_local", no_access)
    assert cli.main(args(source, saved, path)) == 2
    assert capsys.readouterr().out == ""


def test_status_wrapper_rejected(tmp_path, capsys, monkeypatch):
    source, saved, path = inputs(tmp_path, {"schema": 1, "config": Encoder().config})
    monkeypatch.setattr(cli, "open_local", no_access)
    assert cli.main(args(source, saved, path)) == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("field", ["k1", "b", "avgdl"])
@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_nonfinite_parameter(tmp_path, capsys, monkeypatch, field, number):
    source, saved, path = inputs(tmp_path)
    config = Encoder().config
    del config[field]
    raw = json.dumps(config)[:-1] + ', "' + field + '":' + number + "}"
    saved.write_text(raw, encoding="utf-8")
    monkeypatch.setattr(cli, "open_local", no_access)
    assert cli.main(args(source, saved, path)) == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--tokenizer", "cjk-bigram"),
        ("--stopwords", "missing-stopwords"),
        ("--k1", "1.2"),
        ("--b", "0.75"),
        ("--avgdl", "100"),
        ("--b", "0"),
    ],
)
def test_conflict_even_if_equal_before_files(tmp_path, capsys, monkeypatch, flag, value):
    # No files exist; conflict must precede file parsing and storage.
    monkeypatch.setattr(cli, "open_local", no_access)
    monkeypatch.setattr(cli, "read_encoder_config", no_access)
    monkeypatch.setattr(cli, "read_corpus", no_access)
    monkeypatch.setattr(cli, "read_stopwords", no_access)
    assert (
        cli.main(args(tmp_path / "missing", tmp_path / "missing", tmp_path / "qd", flag, value))
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == "" and "cannot be combined" in captured.err


def test_file_byte_boundary(tmp_path, capsys, monkeypatch):
    source, saved, path = inputs(tmp_path)
    raw = json.dumps(Encoder(stopwords=("東京",)).config, ensure_ascii=False).encode("utf-8")
    raw += b" " * (store.MAX_ENCODER_CONFIG_BYTES - len(raw))
    saved.write_bytes(raw)
    assert store.read_encoder_config(saved).config == Encoder(stopwords=("東京",)).config
    saved.write_bytes(raw + b" ")
    monkeypatch.setattr(cli, "open_local", no_access)
    assert cli.main(args(source, saved, path)) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "exceeds 1 MiB" in captured.err


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_unreadable_file_before_storage(tmp_path, capsys, monkeypatch, kind):
    source, saved, path = inputs(tmp_path)
    saved.unlink()
    if kind == "directory":
        saved.mkdir()
    monkeypatch.setattr(cli, "open_local", no_access)
    assert cli.main(args(source, saved, path)) == 2
    assert capsys.readouterr().out == ""


def test_missing_jieba_before_storage(tmp_path, capsys, monkeypatch):
    from ragwrench.bm25 import tokenizers

    def unavailable(*args, **kwargs):
        raise ImportError("forced missing optional dependency")

    config = {**Encoder().config, "tokenizer": "jieba", "jieba_version": "0.42.1"}
    source, saved, path = inputs(tmp_path, config)
    monkeypatch.setattr(tokenizers.importlib, "import_module", unavailable)
    monkeypatch.setattr(cli, "open_local", no_access)
    assert cli.main(args(source, saved, path)) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "jieba is not installed" in captured.err


@pytest.mark.parametrize("tokenizer", ["whitespace", "unicode", "cjk-bigram"])
def test_saved_auto_average_becomes_fixed(tmp_path, capsys, tokenizer):
    # A config has a value, not the original corpus's averaging policy.
    with closing(store.open_local(tmp_path / "original")) as client:
        manifest = store.build_collection(
            client,
            "original",
            DOCS + [{"id": "long", "text": "x " * 30}],
            tokenizer=tokenizer,
            k1=2,
            b=0,
            stopwords=("atlas",),
        )
    source, saved, path = inputs(tmp_path, manifest["encoder"])
    assert manifest["corpus"]["avgdl_mode"] == "auto"
    assert cli.main(args(source, saved, path)) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["encoder"] == manifest["encoder"]
    assert result["corpus"]["avgdl_mode"] == "fixed"
    assert result["encoder"]["avgdl"] != result["corpus"]["total_tokens"] / result["indexed"]
    # Subsequent updates retain this fixed policy too.
    with closing(store.open_local(path)) as client:
        result = store.update_collection(client, "kb", additions=[{"id": "add", "text": "x " * 50}])
        assert result["encoder"] == manifest["encoder"]


@pytest.mark.parametrize("case", ["json", "duplicate", "line", "total", "documents", "text"])
def test_corpus_validation_reused(tmp_path, capsys, monkeypatch, case):
    source, saved, path = inputs(tmp_path)
    if case == "json":
        source.write_text('{"id":7,"text":"ok"}\ninvalid')
    elif case == "duplicate":
        source.write_text('{"id":7,"text":"ok"}\n{"id":7,"text":"other"}')
    elif case == "line":
        monkeypatch.setattr(store, "MAX_LINE_BYTES", 10)
    elif case == "total":
        monkeypatch.setattr(store, "MAX_CORPUS_BYTES", 10)
    elif case == "documents":
        monkeypatch.setattr(store, "MAX_DOCUMENTS", 1)
    else:
        monkeypatch.setattr(store, "MAX_TEXT_CHARS", 1)
    monkeypatch.setattr(cli, "open_local", no_access)
    assert cli.main(args(source, saved, path)) == 2
    assert capsys.readouterr().out == ""


def test_existing_collection_refused_unchanged(tmp_path, capsys):
    import hashlib

    source, saved, path = inputs(tmp_path)
    assert cli.main(args(source, saved, path)) == 0
    capsys.readouterr()

    def fingerprint():
        return {
            str(p.relative_to(path)): (
                hashlib.sha256(p.read_bytes()).hexdigest(),
                p.stat().st_mtime_ns,
            )
            for p in path.rglob("*")
            if p.is_file()
        }

    before = fingerprint()
    saved.write_text(json.dumps(Encoder(k1=2.5, avgdl=90).config))
    assert cli.main(args(source, saved, path)) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "collection already exists" in captured.err
    assert fingerprint() == before


def test_all_skipped_same_build_behavior(tmp_path, capsys):
    source, saved, path = inputs(tmp_path, Encoder(stopwords=("atlas",)).config)
    source.write_text('{"id":7,"text":"ＡＴＬＡＳ"}')
    with pytest.warns(UserWarning, match="all documents are empty"):
        assert cli.main(args(source, saved, path)) == 0
    assert json.loads(capsys.readouterr().out) == {"indexed": 0, "skipped": 1, "created": False}
    with closing(store.open_local(path)) as client:
        assert not client.collection_exists("kb")
