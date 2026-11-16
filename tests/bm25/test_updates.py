"""Incremental mutations must rank exactly like an independent full build."""

from contextlib import closing
from copy import deepcopy

import pytest

from ragwrench.bm25 import store
from ragwrench.bm25.reference import bm25_search


def update(client, collection="kb", **changes):
    return store.update_collection(client, collection, **changes)


def manifest(client, collection="kb"):
    return deepcopy(client.get_collection(collection).config.metadata["qbm25"])


def assert_parity(client, documents, name, **options):
    store.build_collection(client, name, documents, **options)
    for query in ("数据库 备份", "快照", "서울", "alpha", "missing"):
        actual = store.search_collection(client, "kb", query)
        rebuilt = store.search_collection(client, name, query)
        oracle = bm25_search(documents, query, **options)
        assert [h["id"] for h in actual] == [h["id"] for h in rebuilt]
        assert [h["score"] for h in actual] == pytest.approx([h["score"] for h in rebuilt])
        assert [h["id"] for h in actual] == [h["id"] for h in oracle]
        assert [h["score"] for h in actual] == pytest.approx([h["score"] for h in oracle])


@pytest.mark.parametrize("tokenizer", ["whitespace", "unicode", "cjk-bigram"])
@pytest.mark.parametrize("avgdl", [None, 11])
def test_add_delete_replacement_matches_rebuild(client, tokenizer, avgdl):
    options = {"tokenizer": tokenizer, "stopwords": ("the",), "k1": 1.7, "b": 0.8}
    if avgdl is not None:
        options["avgdl"] = avgdl
    a = {"id": "a", "text": "数据库 数据库"}
    b = {"id": 7, "text": "数据库 备份", "metadata": {"lang": "zh"}}
    c = {"id": "7", "text": "数据库 快照 恢复 备份 alpha 서울"}
    store.build_collection(client, "kb", [a, b], **options)
    report = update(client, additions=[c])
    assert report["added"] == 1 and report["deleted"] == 0
    assert report["corpus"]["avgdl_mode"] == ("auto" if avgdl is None else "fixed")
    assert_parity(client, [a, b, c], "after_add", **options)
    report = update(client, delete_ids=["a"])
    assert report["indexed"] == 2 and report["deleted"] == 1
    assert_parity(client, [b, c], "after_delete", **options)
    replacement = {"id": 7, "text": "快照 alpha", "metadata": {"changed": True}}
    with pytest.warns(UserWarning, match="empty"):
        report = update(
            client,
            additions=[replacement, {"id": "skip", "text": "the"}],
            delete_ids=[7, "7"],
        )
    assert report["added"] == 1 and report["deleted"] == 2 and report["skipped"] == 1
    assert_parity(client, [replacement], "after_replace", **options)
    assert store.search_collection(client, "kb", "alpha")[0]["metadata"] == {"changed": True}


def test_explicit_example_and_persistent_metadata(tmp_path):
    path = tmp_path / "store"
    options = {"stopwords": ("THE",), "k1": 1.5, "b": 0.6}
    with closing(store.open_local(path)) as client:
        store.build_collection(
            client,
            "kb",
            [{"id": "a", "text": "数据库 数据库"}, {"id": "b", "text": "数据库 备份"}],
            **options,
        )
    with closing(store.open_local(path, must_exist=True)) as client:
        report = update(
            client,
            additions=[{"id": "c", "text": "数据库 快照 恢复 备份 the"}],
            delete_ids=["a"],
        )
        assert report["indexed"] == 2
        assert report["corpus"] == {"schema": 1, "avgdl_mode": "auto", "total_tokens": 8}
        assert report["encoder"]["avgdl"] == 4
        assert report["encoder"]["stopwords"] == ["the"]
    with closing(store.open_local(path, must_exist=True)) as client:
        assert manifest(client) == {
            k: v for k, v in report.items() if k not in {"added", "deleted"}
        }
        assert_parity(
            client,
            [{"id": "b", "text": "数据库 备份"}, {"id": "c", "text": "数据库 快照 恢复 备份 the"}],
            "rebuilt",
            **options,
        )


@pytest.mark.parametrize(
    "changes,match",
    [
        ({}, "at least one"),
        ({"additions": [{"id": "a", "text": "again"}]}, "already exists"),
        ({"additions": [{"id": "c", "text": "x"}] * 2}, "duplicate id"),
        ({"delete_ids": ["missing"]}, "does not exist"),
        ({"delete_ids": ["a", "a"]}, "duplicate id"),
        ({"delete_ids": [True]}, "id must"),
        ({"delete_ids": "a"}, "list"),
        ({"additions": [{"id": "x", "text": 1}]}, "text must"),
    ],
)
def test_invalid_mutation_never_writes(client, monkeypatch, changes, match):
    store.build_collection(client, "kb", [{"id": "a", "text": "alpha"}])
    before = manifest(client)

    def unexpected(*args, **kwargs):
        pytest.fail("invalid mutation must not write")

    monkeypatch.setattr(client, "update_collection", unexpected)
    monkeypatch.setattr(client, "upsert", unexpected)
    monkeypatch.setattr(client, "delete", unexpected)
    with pytest.raises(ValueError, match=match):
        update(client, **changes)
    assert manifest(client) == before
    assert store.search_collection(client, "kb", "alpha")[0]["id"] == "a"


def test_final_corpus_limit(client, monkeypatch):
    store.build_collection(client, "kb", [{"id": "a", "text": "alpha"}])
    monkeypatch.setattr(store, "MAX_DOCUMENTS", 1)
    with pytest.raises(ValueError, match="exceeds 1"):
        update(client, additions=[{"id": "b", "text": "beta"}])
    assert manifest(client)["state"] == "ready"


def test_delete_all_and_readd(client):
    store.build_collection(client, "kb", [{"id": 7, "text": "alpha"}], tokenizer="whitespace")
    report = update(client, delete_ids=[7])
    assert report["indexed"] == 0 and report["corpus"]["total_tokens"] == 0
    assert report["encoder"]["avgdl"] == 1
    assert store.search_collection(client, "kb", "alpha") == []
    update(client, additions=[{"id": 7, "text": "alpha alpha"}])
    assert_parity(client, [{"id": 7, "text": "alpha alpha"}], "fresh", tokenizer="whitespace")


def test_collision_statistics_drop_deleted_tokens(client):
    with pytest.warns(UserWarning, match="collision"):
        store.build_collection(
            client,
            "kb",
            [{"id": "a", "text": "collision402"}, {"id": "b", "text": "collision169677"}],
            tokenizer="whitespace",
        )
    report = update(client, delete_ids=["b"])
    assert report["collision_stats"] == {
        "unique_tokens": 1,
        "unique_indices": 1,
        "collision_count": 0,
        "collisions": {},
    }
    assert manifest(client)["collision_stats"] == report["collision_stats"]


@pytest.mark.parametrize("phase", ["delete", "upsert", "finalize"])
def test_failed_mutation_is_persistently_unsearchable(tmp_path, monkeypatch, phase):
    path = tmp_path / "qd"
    with closing(store.open_local(path)) as client:
        store.build_collection(client, "kb", [{"id": "a", "text": "alpha"}])
        original = client.update_collection

        def fail(*args, **kwargs):
            raise RuntimeError("injected storage failure")

        def fail_final(*args, **kwargs):
            if kwargs["metadata"]["qbm25"]["state"] == "ready":
                fail()
            return original(*args, **kwargs)

        monkeypatch.setattr(
            client,
            "update_collection" if phase == "finalize" else phase,
            fail_final if phase == "finalize" else fail,
        )
        with pytest.raises(RuntimeError, match="injected"):
            update(client, delete_ids=["a"], additions=[{"id": "b", "text": "alpha beta"}])
    with closing(store.open_local(path, must_exist=True)) as client:
        assert manifest(client)["state"] == "updating"
        with pytest.raises(ValueError, match="incomplete"):
            store.search_collection(client, "kb", "alpha")
        with pytest.raises(ValueError, match="incomplete"):
            update(client, additions=[{"id": "c", "text": "gamma"}])


def test_legacy_and_incompatible_metadata_refused_before_writes(client):
    store.build_collection(client, "kb", [{"id": "a", "text": "alpha"}])
    metadata = deepcopy(client.get_collection("kb").config.metadata)
    metadata["qbm25"]["encoder"]["unicode_version"] = "incompatible"
    client.update_collection("kb", metadata=metadata)
    with pytest.raises(ValueError, match="mismatch"):
        update(client, delete_ids=["a"])
    assert manifest(client)["state"] == "ready"
    # Simulate a pre-incremental collection: searching remains supported; updates cannot
    # infer whether avgdl was computed or explicitly pinned by the original builder.
    store.build_collection(client, "legacy", [{"id": "a", "text": "alpha"}])
    metadata = deepcopy(client.get_collection("legacy").config.metadata)
    del metadata["qbm25"]["corpus"]
    client.update_collection("legacy", metadata={"qbm25": None})
    client.update_collection("legacy", metadata=metadata)
    assert store.search_collection(client, "legacy", "alpha")
    with pytest.raises(ValueError, match="rebuild"):
        update(client, "legacy", delete_ids=["a"])


@pytest.mark.parametrize("avgdl", [None, 5])
def test_paged_corpus_and_vector_write_selection(client, monkeypatch, avgdl):
    documents = [{"id": i, "text": "alpha"} for i in range(130)]
    store.build_collection(client, "kb", documents, tokenizer="whitespace", avgdl=avgdl)
    written = []
    original = client.upsert

    def record(*args, **kwargs):
        written.extend(point.id for point in kwargs["points"])
        return original(*args, **kwargs)

    monkeypatch.setattr(client, "upsert", record)
    addition = {"id": "new", "text": "alpha beta beta"}
    update(client, additions=[addition], delete_ids=[0])
    assert len(written) == (130 if avgdl is None else 1)
    options = {"tokenizer": "whitespace"}
    if avgdl is not None:
        options["avgdl"] = avgdl
    remaining = documents[1:] + [addition]
    # Fetch all matching hits to resolve ties consistently across the full set.
    store.build_collection(client, "rebuilt", remaining, **options)
    actual = store.search_collection(client, "kb", "alpha beta", top_k=200)
    expected = store.search_collection(client, "rebuilt", "alpha beta", top_k=200)
    assert [h["id"] for h in actual] == [h["id"] for h in expected]
    assert [h["score"] for h in actual] == pytest.approx([h["score"] for h in expected])


def test_second_update_batch_failure_leaves_real_partial_write(tmp_path, monkeypatch):
    path = tmp_path / "qd"
    with closing(store.open_local(path)) as client:
        store.build_collection(client, "kb", [{"id": "a", "text": "alpha"}])
        original = client.upsert
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("second batch failed")
            return original(*args, **kwargs)

        monkeypatch.setattr(client, "upsert", fail_second)
        with pytest.raises(RuntimeError, match="second batch"):
            update(client, additions=[{"id": i, "text": "beta"} for i in range(129)])
    with closing(store.open_local(path, must_exist=True)) as client:
        assert client.get_collection("kb").points_count == 129
        with pytest.raises(ValueError, match="incomplete"):
            store.search_collection(client, "kb", "beta")


def test_delete_file_bound_and_typed_ids(tmp_path):
    path = tmp_path / "ids.json"
    path.write_text('[7, "7"]')
    assert store.read_delete_ids(path) == [7, "7"]
    path.write_bytes(b" " * (store.MAX_LINE_BYTES + 1))
    with pytest.raises(ValueError, match="4 MiB"):
        store.read_delete_ids(path)


def test_jieba_update_preserves_dictionary_metadata(client):
    pytest.importorskip("jieba")
    documents = [
        {"id": "a", "text": "我来到北京清华大学。"},
        {"id": "b", "text": "数据库支持备份。"},
    ]
    store.build_collection(client, "kb", documents, tokenizer="jieba", stopwords=("我",))
    version = manifest(client)["encoder"]["jieba_version"]
    addition = {"id": "c", "text": "清华大学数据库快照备份恢复。"}
    report = update(client, additions=[addition], delete_ids=["a"])
    assert report["encoder"]["jieba_version"] == version == "0.42.1"
    assert report["encoder"]["stopwords"] == ["我"]
    assert_parity(client, [documents[1], addition], "rebuilt", tokenizer="jieba", stopwords=("我",))
