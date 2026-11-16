"""Offline encode uses a caller-saved compatible config without a collection."""

import json

import pytest

from ragwrench.bm25 import cli, store
from ragwrench.bm25.encoder import Encoder, token_hash


TEXT = "ＡＴＬＡＳ 北京 北京 東京 서울 é"


def test_encode_saved_config(tmp_path, capsys):
    config = Encoder("whitespace", k1=2, b=1, avgdl=4, stopwords=("atlas",)).config
    saved = tmp_path / "encoder.json"
    saved.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    # Valid compatible config and ordinary encode are positive controls.
    assert store.read_encoder_config(saved).config == config
    assert cli.main(["encode", TEXT]) == 0
    assert json.loads(capsys.readouterr().out)["config"] == Encoder().config
    for document in (False, True):
        assert (
            cli.main(
                [
                    "encode",
                    TEXT,
                    "--encoder-config",
                    str(saved),
                    *(["--document"] if document else []),
                ]
            )
            == 0
        )
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert captured.err == ""
        assert result["tokens"] == ["北京", "北京", "東京", "서울", "é"]
        assert result["config"] == config
        weights = {"北京": 4 / 3, "東京": 6 / 7, "서울": 6 / 7, "é": 6 / 7}
        assert result["indices"] == sorted(map(token_hash, weights))
        assert result["values"] == [
            weights[next(t for t in weights if token_hash(t) == i)] if document else 1.0
            for i in result["indices"]
        ]
        assert result["collision_stats"] == {
            "unique_tokens": 4,
            "unique_indices": 4,
            "collision_count": 0,
            "collisions": {},
        }


def forbidden(*args, **kwargs):
    raise AssertionError("unexpected collection/file/encoding access")


@pytest.fixture(autouse=True)
def no_storage(monkeypatch):
    monkeypatch.setattr(cli, "open_local", forbidden)
    monkeypatch.setattr(store, "open_local", forbidden)
    monkeypatch.setattr(store, "QdrantClient", forbidden)
    monkeypatch.setattr(cli, "read_corpus", forbidden)


def saved_file(tmp_path, config):
    path = tmp_path / "encoder.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return path


def invoke(saved, text=TEXT, document=False, *extra):
    return cli.main(
        [
            "encode",
            text,
            "--encoder-config",
            str(saved),
            *(["--document"] if document else []),
            *extra,
        ]
    )


@pytest.mark.parametrize("tokenizer", ["whitespace", "unicode", "cjk-bigram"])
@pytest.mark.parametrize("document", [False, True])
@pytest.mark.parametrize("numeric", [(2, 0, 4), (1.8, 0.3, 19.5)])
def test_from_config_parity(tmp_path, capsys, tokenizer, document, numeric):
    k1, b, avgdl = numeric
    config = Encoder(tokenizer, k1=k1, b=b, avgdl=avgdl, stopwords=("atlas",)).config
    # Preserve JSON integer parameter types at the input boundary.
    config.update(k1=k1, b=b, avgdl=avgdl)
    saved = saved_file(tmp_path, config)
    expected = Encoder.from_config(config)
    vector = expected.encode_documents([TEXT])[0] if document else expected.encode_query(TEXT)
    assert invoke(saved, document=document) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "tokens": expected.tokenizer(TEXT),
        **vector.as_dict(),
        "config": expected.config,
        "collision_stats": expected.collision_stats,
    }
    assert all(type(result["config"][name]) is float for name in ("k1", "b", "avgdl"))


@pytest.mark.parametrize("text", ["", "ＡＴＬＡＳ atlas"])
@pytest.mark.parametrize("document", [False, True])
def test_empty_retains_warning(tmp_path, capsys, text, document):
    config = Encoder(stopwords=("atlas",)).config
    saved = saved_file(tmp_path, config)
    expected = Encoder.from_config(config)
    with pytest.warns(UserWarning, match="empty or stopword-only"):
        vector = expected.encode_documents([text])[0] if document else expected.encode_query(text)
    with pytest.warns(UserWarning, match="empty or stopword-only"):
        assert invoke(saved, text, document) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "tokens": [],
        **vector.as_dict(),
        "config": expected.config,
        "collision_stats": expected.collision_stats,
    }


@pytest.mark.parametrize("document", [False, True])
def test_collision_warning_and_stats(tmp_path, capsys, document):
    saved = saved_file(tmp_path, Encoder("whitespace", b=0).config)
    with pytest.warns(UserWarning, match="hash collision"):
        assert invoke(saved, "collision402 collision169677", document) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["indices"] == [3793890753]
    assert result["values"] == ([2.0] if document else [1.0])
    assert result["collision_stats"]["collision_count"] == 1


@pytest.mark.parametrize(
    "raw",
    [
        b"{",
        b"[]",
        b"null",
        b"{} {}",
        b'"config"',
        b'{"schema":1,"schema":1}',
        b'{"a":{"b":1,"b":2}}',
        b'{"k1":NaN}',
        b'{"b":Infinity}',
        b'{"avgdl":-Infinity}',
        b'{"k1":1e999}',
        b'{"a":"\\ud800"}',
        b"\xff",
        b"\xef\xbb\xbf{}",
        b"[" * 1100 + b"]" * 1100,
    ],
)
def test_invalid_json_before_encoding(tmp_path, capsys, monkeypatch, raw):
    saved = tmp_path / "encoder.json"
    saved.write_bytes(raw)
    monkeypatch.setattr(Encoder, "_encode", forbidden)
    assert invoke(saved) == 2
    result = capsys.readouterr()
    assert result.out == "" and "encoder config" in result.err


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", True),
        ("schema", 1.0),
        ("schema", 2),
        ("tokenizer_version", True),
        ("tokenizer_version", 1.0),
        ("hash", "unsupported"),
        ("unicode_version", "incompatible"),
        ("jieba_version", "incompatible"),
        ("tokenizer", "unknown"),
        ("stopwords", "atlas"),
        ("stopwords", ["ＡＴＬＡＳ"]),
        ("stopwords", ["z", "a"]),
        ("stopwords", ["a", "a"]),
        ("k1", True),
        ("k1", "1.2"),
        ("k1", 0),
        ("k1", 10**400),
        ("b", False),
        ("b", 1.1),
        ("avgdl", 0),
        ("avgdl", 10**400),
        ("extra", "unsupported"),
    ],
)
def test_incompatible_before_encoding(tmp_path, capsys, monkeypatch, field, value):
    saved = saved_file(tmp_path, {**Encoder().config, field: value})
    monkeypatch.setattr(Encoder, "_encode", forbidden)
    assert invoke(saved) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "encoder config" in captured.err


@pytest.mark.parametrize("field", list(Encoder().config))
def test_missing_fields_before_encoding(tmp_path, capsys, monkeypatch, field):
    config = Encoder().config
    del config[field]
    saved = saved_file(tmp_path, config)
    monkeypatch.setattr(Encoder, "_encode", forbidden)
    assert invoke(saved) == 2
    assert capsys.readouterr().out == ""


def test_wrapper_rejected(tmp_path, capsys, monkeypatch):
    saved = saved_file(tmp_path, {"schema": 1, "config": Encoder().config})
    monkeypatch.setattr(Encoder, "_encode", forbidden)
    assert invoke(saved) == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--tokenizer", "cjk-bigram"),
        ("--stopwords", "missing"),
        ("--k1", "1.2"),
        ("--b", "0.75"),
        ("--avgdl", "100"),
        ("--b", "0"),
    ],
)
def test_conflicting_flags_before_files(tmp_path, capsys, monkeypatch, flag, value):
    monkeypatch.setattr(cli, "read_encoder_config", forbidden)
    monkeypatch.setattr(cli, "read_stopwords", forbidden)
    monkeypatch.setattr(Encoder, "_encode", forbidden)
    assert invoke(tmp_path / "missing", TEXT, False, flag, value) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "cannot be combined" in captured.err


def test_config_byte_boundary(tmp_path, capsys):
    config = Encoder(stopwords=("東京",)).config
    saved = saved_file(tmp_path, config)
    raw = saved.read_bytes()
    raw += b" " * (store.MAX_ENCODER_CONFIG_BYTES - len(raw))
    saved.write_bytes(raw)
    assert invoke(saved) == 0
    assert json.loads(capsys.readouterr().out)["config"] == config
    saved.write_bytes(raw + b" ")
    assert invoke(saved) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "exceeds 1 MiB" in captured.err


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_unreadable_file(tmp_path, capsys, kind):
    saved = tmp_path / "encoder.json"
    if kind == "directory":
        saved.mkdir()
    assert invoke(saved) == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("document", [False, True])
def test_text_limit_reused(tmp_path, capsys, document):
    from ragwrench.bm25 import tokenizers

    saved = saved_file(tmp_path, Encoder().config)
    # Test the real 1M boundary with a short-token input; CLI subprocess ARG_MAX
    # is a platform limit, so exercise the parser/main with an in-process argv.
    assert invoke(saved, " " * tokenizers.MAX_TEXT_CHARS + "x", document) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "text exceeds 1000000 characters" in captured.err
    assert invoke(saved, " " * (tokenizers.MAX_TEXT_CHARS - 1) + "x", document) == 0
    assert json.loads(capsys.readouterr().out)["tokens"] == ["x"]


def test_missing_jieba(tmp_path, capsys, monkeypatch):
    from ragwrench.bm25 import tokenizers

    def missing(*args, **kwargs):
        raise ImportError("forced missing optional dependency")

    saved = saved_file(
        tmp_path, {**Encoder().config, "tokenizer": "jieba", "jieba_version": "0.42.1"}
    )
    monkeypatch.setattr(tokenizers.importlib, "import_module", missing)
    monkeypatch.setattr(Encoder, "_encode", forbidden)
    assert invoke(saved) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "jieba is not installed" in captured.err


@pytest.mark.parametrize("params", [{"k1": 1e308}, {"avgdl": 1e-308}])
def test_numeric_encoding_error_retained(tmp_path, capsys, params):
    saved = saved_file(tmp_path, Encoder("whitespace", **params).config)
    assert invoke(saved, "a a a", True) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "TF weights" in captured.err


def test_ordinary_defaults_and_flags(tmp_path, capsys):
    stopwords = tmp_path / "words.txt"
    stopwords.write_text("ＡＴＬＡＳ\n", encoding="utf-8")
    assert (
        cli.main(
            [
                "encode",
                TEXT,
                "--document",
                "--tokenizer",
                "whitespace",
                "--stopwords",
                str(stopwords),
                "--k1",
                "2",
                "--b",
                "0",
                "--avgdl",
                "4",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert (
        result["config"] == Encoder("whitespace", k1=2, b=0, avgdl=4, stopwords=("atlas",)).config
    )
    assert cli.main(["encode", TEXT]) == 0
    assert json.loads(capsys.readouterr().out)["config"] == Encoder().config
