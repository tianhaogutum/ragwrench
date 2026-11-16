from contextlib import closing

import pytest
from qdrant_client import models

from ragwrench.bm25.reference import bm25_search
from ragwrench.bm25.store import (
    build_collection,
    load_encoder,
    open_local,
    point_id,
    read_corpus,
    search_collection,
    validate_documents,
)


def test_persistent_e2e(tmp_path, corpus):
    path = tmp_path / "store"
    with closing(open_local(path)) as client:
        report = build_collection(client, "kb", corpus)
        assert report["indexed"] == 4
        assert report["encoder"]["avgdl"] > 1
        assert report["collision_stats"]["collision_count"] == 0
    with closing(open_local(path, must_exist=True)) as client:
        hits = search_collection(client, "kb", "向量数据库 备份")
        assert hits[0]["id"] == "d1"
        assert hits[0]["metadata"] == {"tag": "backup"}
        assert hits[0]["text"] == corpus[0]["text"]
        assert point_id(7) != point_id("7")
        assert len(client.retrieve("kb", [point_id(7), point_id("7")])) == 2
        assert search_collection(client, "kb", "no_such_word") == []
        with pytest.raises(ValueError, match="already exists"):
            build_collection(client, "kb", corpus)


@pytest.mark.parametrize("tokenizer", ["whitespace", "unicode", "cjk-bigram"])
@pytest.mark.parametrize("params", [{}, {"b": 0}, {"k1": 2.0, "b": 1.0, "avgdl": 11}])
def test_reference_score_and_ranking_parity(client, corpus, tokenizer, params):
    build_collection(client, "kb", corpus, tokenizer=tokenizer, **params)
    for query in ("向量数据库 备份", "数据库", "Qdrant", "面包", "不存在"):
        actual = search_collection(client, "kb", query, top_k=10)
        expected = bm25_search(corpus, query, tokenizer=tokenizer, **params)
        assert [hit["id"] for hit in actual] == [hit["id"] for hit in expected]
        assert [hit["score"] for hit in actual] == pytest.approx(
            [hit["score"] for hit in expected], rel=1e-6
        )


@pytest.mark.parametrize(
    "override",
    [{"tokenizer": "unicode"}, {"k1": 2}, {"b": 0}, {"avgdl": 1000}, {"stopwords": ("向量",)}],
)
def test_config_mismatch(client, corpus, override):
    build_collection(client, "kb", corpus)
    with pytest.raises(ValueError, match="differs"):
        search_collection(client, "kb", "向量", **override)
    assert search_collection(client, "kb", "向量", tokenizer="cjk-bigram", k1=1.2)


def test_stopwords_are_persisted(client, corpus):
    build_collection(client, "kb", corpus, stopwords=("向量",))
    with pytest.warns(UserWarning, match="empty"):
        assert search_collection(client, "kb", "向量") == []
    assert load_encoder(client, "kb").config["stopwords"] == ["向量"]


def test_empty_documents_skipped(client):
    documents = [
        {"id": "empty", "text": ""},
        {"id": "stop", "text": "the"},
        {"id": "ok", "text": "alpha alpha"},
    ]
    with pytest.warns(UserWarning, match="empty"):
        report = build_collection(
            client, "kb", documents, tokenizer="whitespace", stopwords=("the",)
        )
    assert report["indexed"] == 1 and report["skipped"] == 2
    assert report["encoder"]["avgdl"] == 2
    score = search_collection(client, "kb", "alpha")[0]["score"]
    expected = bm25_search(documents, "alpha", tokenizer="whitespace", stopwords=("the",))[0][
        "score"
    ]
    assert score == pytest.approx(expected)
    with pytest.warns(UserWarning, match="empty"):
        result = build_collection(
            client, "nothing", documents[:2], tokenizer="whitespace", stopwords=("the",)
        )
    assert result == {"indexed": 0, "skipped": 2, "created": False}
    assert not client.collection_exists("nothing")


def test_incompatible_and_incomplete_collection(client, corpus):
    client.create_collection("foreign", vectors_config={})
    with pytest.raises(ValueError, match="sparse vector"):
        search_collection(client, "foreign", "abc")
    client.create_collection(
        "unmanaged",
        vectors_config={},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    with pytest.raises(ValueError, match="metadata"):
        search_collection(client, "unmanaged", "abc")
    build_collection(client, "kb", corpus)
    metadata = client.get_collection("kb").config.metadata
    metadata["qbm25"]["state"] = "building"
    client.update_collection("kb", metadata=metadata)
    with pytest.raises(ValueError, match="incomplete"):
        search_collection(client, "kb", "abc")


def test_external_edit_refused(client, corpus):
    build_collection(client, "kb", corpus)
    client.delete("kb", points_selector=models.PointIdsList(points=[point_id("d1")]))
    with pytest.raises(ValueError, match="point count"):
        search_collection(client, "kb", "向量")


def test_failed_upsert_never_ready(client, corpus, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("simulated storage failure")

    monkeypatch.setattr(client, "upsert", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        build_collection(client, "kb", corpus)
    with pytest.raises(ValueError, match="incomplete"):
        search_collection(client, "kb", "向量")


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "a/b", ".", "", "a" * 65, "中文"])
def test_collection_path_boundaries(client, corpus, name):
    with pytest.raises(ValueError, match="collection must"):
        build_collection(client, name, corpus)
    assert client.get_collections().collections == []


@pytest.mark.parametrize(
    "content,match",
    [
        ('{"id":"a","text":"ok"}\nnot json\n', "line 2"),
        ('{"id":"a","text":"ok","id":"b"}', "duplicate JSON key"),
        ('{"id":"a","text":"ok","metadata":{"v":NaN}}', "non-finite"),
        ('{"id":"a","text":"ok"}\n{"id":"a","text":"again"}', "duplicate id"),
        ('{"id":true,"text":"x"}', "id must"),
        ('{"id":1,"text":22}', "text must"),
        ('{"id":1,"text":"x","metadata":[]}', "metadata must"),
        ('{"id":1,"text":"x","unknown":0}', "expected object"),
        ("[]", "expected object"),
        ("\n", "no documents"),
    ],
)
def test_bad_jsonl(tmp_path, content, match):
    path = tmp_path / "bad.jsonl"
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        read_corpus(path)


def test_limits_and_unicode(tmp_path):
    path = tmp_path / "large.jsonl"
    path.write_bytes(b"x" * (4 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="4 MiB"):
        read_corpus(path)
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="line 1"):
        read_corpus(path)
    with pytest.raises(ValueError, match="10000"):
        validate_documents([{"id": i, "text": "a"} for i in range(10_001)])


def test_bad_search_arguments(client, corpus, tmp_path):
    build_collection(client, "kb", corpus)
    for top_k in (0, -1, 1001, True):
        with pytest.raises(ValueError, match="top-k"):
            search_collection(client, "kb", "query", top_k=top_k)
    with pytest.raises(ValueError, match="does not exist"):
        search_collection(client, "missing", "abc")
    with pytest.raises(ValueError, match="does not exist"):
        open_local(tmp_path / "missing", must_exist=True)
    assert not (tmp_path / "missing").exists()
    with pytest.raises(ValueError, match="local directory"):
        open_local("https://example.com")


def test_real_jsonl_fixture(client):
    documents = read_corpus("examples/bm25/corpus_zh.jsonl")
    build_collection(client, "fixture", documents)
    assert search_collection(client, "fixture", "向量数据库 备份", top_k=3)[0]["id"] == "d1"


def test_collision_manifest_and_sparse_merge(client):
    documents = [{"id": "a", "text": "collision402 collision169677"}]
    with pytest.warns(UserWarning, match="hash collision"):
        report = build_collection(client, "kb", documents, tokenizer="whitespace", b=0)
    assert report["collision_stats"]["collision_count"] == 1
    assert (
        client.get_collection("kb").config.metadata["qbm25"]["collision_stats"]
        == report["collision_stats"]
    )
    vector = client.retrieve("kb", [point_id("a")], with_vectors=True)[0].vector["bm25"]
    assert vector.indices == [3793890753] and vector.values == [2.0]


def test_malformed_encoder_metadata_is_refused(client, corpus):
    build_collection(client, "kb", corpus)
    metadata = client.get_collection("kb").config.metadata
    metadata["qbm25"]["encoder"] = []
    client.update_collection("kb", metadata=metadata)
    with pytest.raises(ValueError, match="invalid collection encoder"):
        search_collection(client, "kb", "query")


def test_failed_finalization_stays_building(client, corpus, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("simulated metadata failure")

    monkeypatch.setattr(client, "update_collection", fail)
    with pytest.raises(RuntimeError, match="metadata failure"):
        build_collection(client, "kb", corpus)
    assert client.get_collection("kb").config.metadata["qbm25"]["state"] == "building"
    with pytest.raises(ValueError, match="incomplete"):
        search_collection(client, "kb", "向量")
