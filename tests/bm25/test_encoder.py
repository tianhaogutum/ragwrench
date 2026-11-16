import importlib.util
import json
import math
import os
import subprocess
import sys

import pytest

import ragwrench.bm25.encoder as encoding
from ragwrench.bm25 import Encoder, token_hash
from ragwrench.bm25.tokenizers import Tokenizer, read_stopwords


@pytest.mark.parametrize(
    "text,expected",
    [
        ("向量数据库", ["向量", "量数", "数据", "据库"]),
        ("日本語カタカナ", ["日本", "本語", "語カ", "カタ", "タカ", "カナ"]),
        ("서울 데이터베이스", ["서울", "데이", "이터", "터베", "베이", "이스"]),
        ("Qdrant向量DB42", ["qdrant", "向量", "db42"]),
        ("ＡＢＣ１２ 向量 ｶﾀｶﾅ", ["abc12", "向量", "カタ", "タカ", "カナ"]),
        ("中，文。a_b/猫", ["中", "文", "a", "b", "猫"]),
        ("𠀀𠀁𠀂", ["𠀀𠀁", "𠀁𠀂"]),
        ("Cafe\u0301 cafe\u0301", ["café", "café"]),
    ],
)
def test_cjk(text, expected):
    assert Tokenizer()(text) == expected


def test_baselines():
    assert Tokenizer("whitespace")("ＡＢＣ, 中文。") == ["abc,", "中文。"]
    assert Tokenizer("unicode")("ＡＢＣ, 中文。DB42_日本語") == ["abc", "中文", "db42", "日本語"]


def test_stopwords(tmp_path):
    path = tmp_path / "words.txt"
    path.write_text("ＡＢＣ\n数据\n\n数据\n", encoding="utf-8")
    assert read_stopwords(path) == ("abc", "数据")
    assert Tokenizer(stopwords=read_stopwords(path))("ABC 数据库") == ["据库"]
    path.write_text("two words")
    with pytest.raises(ValueError, match="single token"):
        read_stopwords(path)
    path.write_bytes(b"x" * 1_048_577)
    with pytest.raises(ValueError, match="1 MiB"):
        read_stopwords(path)


def test_weights_hand_computed():
    encoder = Encoder("whitespace", k1=1.2, b=0.75, avgdl=4)
    vector = encoder.encode_documents(["a a b"])[0]
    weights = dict(zip(vector.indices, vector.values))
    assert weights[token_hash("a")] == pytest.approx(2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / 4)))
    assert weights[token_hash("b")] == pytest.approx(2.2 / (1 + 1.2 * (0.25 + 0.75 * 3 / 4)))
    query = encoder.encode_query("a a b")
    assert query.indices == sorted(weights)
    assert query.values == [1.0, 1.0]


def test_hash_stability_in_processes():
    expected = [token_hash(token) for token in ("hello", "向量", "カタカナ", "서울")]
    assert expected[0] == 0x2CF24DBA
    for seed in ("1", "987"):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json; from ragwrench.bm25 import token_hash; "
                "print(json.dumps([token_hash(t) for t in ['hello','向量','カタカナ','서울']]))",
            ],
            text=True,
            capture_output=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        assert json.loads(result.stdout) == expected


def test_collision_reporting_and_aggregation(monkeypatch):
    monkeypatch.setattr(encoding, "token_hash", lambda _: 42)
    encoder = Encoder("whitespace", b=0)
    with pytest.warns(UserWarning, match="collision"):
        vector = encoder.encode_documents(["alpha beta"])[0]
    assert vector.indices == [42]
    assert vector.values == pytest.approx([2.0])
    assert encoder.collision_stats == {
        "unique_tokens": 2,
        "unique_indices": 1,
        "collision_count": 1,
        "collisions": {"42": ["alpha", "beta"]},
    }
    assert encoder.encode_query("alpha beta").values == [1.0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k1": 0},
        {"k1": -1},
        {"k1": math.inf},
        {"k1": True},
        {"b": -0.1},
        {"b": 1.1},
        {"b": math.nan},
        {"avgdl": 0},
        {"avgdl": math.inf},
        {"tokenizer": "unknown"},
    ],
)
def test_invalid_encoder(kwargs):
    with pytest.raises(ValueError):
        Encoder(**kwargs)


def test_empty_inputs_and_config():
    encoder = Encoder("unicode", stopwords=["the"])
    with pytest.warns(UserWarning, match="empty"):
        assert encoder.encode_query("the").indices == []
    with pytest.warns(UserWarning, match="empty"):
        assert encoder.encode_documents([""])[0].values == []
    assert Encoder.from_config(encoder.config).config == encoder.config
    config = {**encoder.config, "hash": "unrecognized"}
    with pytest.raises(ValueError, match="mismatch"):
        Encoder.from_config(config)
    with pytest.raises(ValueError, match="iterable"):
        encoder.encode_documents("abc")
    with pytest.raises(ValueError, match="string"):
        encoder.encode_query(1)
    with pytest.raises(ValueError, match="exceeds"):
        encoder.encode_query("a" * 1_000_001)


@pytest.mark.parametrize("field", ["schema", "tokenizer_version"])
@pytest.mark.parametrize("value", [True, 1.0])
def test_config_versions_require_integers(field, value):
    config = Encoder().config
    config[field] = value
    with pytest.raises(ValueError, match=field):
        Encoder.from_config(config)


def test_config_integer_versions_and_numeric_parameters():
    config = Encoder(k1=2, b=1, avgdl=4).config
    config.update(k1=2, b=1, avgdl=4)
    restored = Encoder.from_config(config).config
    assert type(restored["schema"]) is int
    assert type(restored["tokenizer_version"]) is int
    assert (restored["k1"], restored["b"], restored["avgdl"]) == (2.0, 1.0, 4.0)


def test_jieba_missing(monkeypatch):
    import ragwrench.bm25.tokenizers as tokenizers

    original = tokenizers.importlib.import_module

    def missing(name):
        if name == "jieba":
            raise ImportError("not installed")
        return original(name)

    monkeypatch.setattr(tokenizers.importlib, "import_module", missing)
    with pytest.raises(ValueError, match=r"ragwrench\[jieba\]"):
        Encoder("jieba")


@pytest.mark.skipif(
    importlib.util.find_spec("jieba") is None, reason="optional jieba extra not installed"
)
def test_real_jieba(client):
    encoder = Encoder("jieba")
    assert encoder.tokenizer("我来到北京清华大学。") == ["我", "来到", "北京", "清华大学"]
    assert encoder.config["jieba_version"] == "0.42.1"
    assert encoder.encode_query("数据库").values == [1.0]
    from ragwrench.bm25.store import build_collection, search_collection

    build_collection(
        client, "jieba", [{"id": "a", "text": "我来到北京清华大学。"}], tokenizer="jieba"
    )
    assert search_collection(client, "jieba", "清华大学")[0]["id"] == "a"


def test_real_sha256_collision():
    # Redistributable synthetic strings found by a deterministic birthday search.
    left, right = "collision402", "collision169677"
    assert left != right and token_hash(left) == token_hash(right) == 3793890753
    encoder = Encoder("whitespace", b=0)
    with pytest.warns(UserWarning, match="hash collision"):
        vector = encoder.encode_documents([f"{left} {right}"])[0]
    assert vector.indices == [3793890753] and vector.values == [2.0]
    assert encoder.collision_stats["collision_count"] == 1


@pytest.mark.parametrize("params", [{"k1": 1e308}, {"avgdl": 1e-308}])
def test_extreme_numeric_weights_refused(params):
    with pytest.raises(ValueError, match="TF weights"):
        Encoder("whitespace", **params).encode_documents(["a a a"])
