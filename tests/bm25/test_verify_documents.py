"""Bounded selected comparisons return complete summaries or no result at all."""

from contextlib import closing
from copy import deepcopy
import json
import os
import warnings

import pytest
from qdrant_client import models

from ragwrench.bm25 import store
from ragwrench.bm25.cli import main
from ragwrench.bm25.encoder import Encoder
from .test_verify_document import files


DOCS = [
    {"id": 7, "text": "private ＡＴＬＡＳ 北京", "metadata": {"n": 1}},
    {"id": "7", "text": "other 数据库", "metadata": {"n": True}},
    {"id": "edited", "text": "private 北京 北京", "metadata": {"n": 1.0}},
]


def test_batch_mixed_same_count_edit(client):
    store.build_collection(client, "kb", DOCS, stopwords=("private",), avgdl=19, k1=1.8, b=0.3)
    before = store.status_collection(client, "kb", require_ready=True)
    client.set_payload("kb", {"text": "externally changed"}, points=[store.point_id("edited")])
    assert store.status_collection(client, "kb", require_ready=True) == before
    documents = [DOCS[1], {"id": "absent", "text": "北京"}, DOCS[2], DOCS[0]]
    report = store.verify_documents(client, "kb", documents)
    assert report["schema"] == 1 and report["collection"] == "kb"
    assert report["matches"] is False
    assert report["totals"] == {"documents": 4, "matched": 2, "mismatched": 2, "missing": 1}
    assert report["mismatch_counts"] == {
        "point_missing": 1,
        "source_id": 0,
        "text": 1,
        "metadata": 0,
        "sparse_indices": 0,
        "sparse_values": 0,
    }
    assert report["results"] == [store.verify_document(client, "kb", doc) for doc in documents]
    assert [type(r["id"]) for r in report["results"]] == [str, str, str, int]


def batch_args(path, source, collection="kb"):
    return ["verify-documents", "--path", str(path), "--collection", collection, str(source)]


def no_access(*args, **kwargs):
    pytest.fail("unexpected storage access, scan, query, mutation or fsync")


def guard(monkeypatch, target):
    for name in (
        "scroll",
        "query_points",
        "upsert",
        "delete",
        "create_collection",
        "delete_collection",
        "update_collection",
        "set_payload",
        "overwrite_payload",
        "delete_payload",
        "update_vectors",
        "delete_vectors",
    ):
        monkeypatch.setattr(target, name, no_access)
    monkeypatch.setattr(os, "fsync", no_access)


@pytest.mark.parametrize("mismatch", [False, True])
def test_cli_read_only_selected_saved_config(tmp_path, monkeypatch, capsys, mismatch):
    path, source = tmp_path / "qd", tmp_path / "sources.jsonl"
    documents = deepcopy(DOCS)
    if mismatch:
        documents[1]["metadata"] = {"n": 1}  # JSON bool and number differ.
        documents.append({"id": "absent", "text": "private"})
    source.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in documents) + "\n")
    with closing(store.open_local(path)) as client:
        store.build_collection(
            client,
            "kb",
            [*DOCS, {"id": "unselected", "text": "elsewhere"}],
            stopwords=("private",),
            avgdl=19,
            k1=1.8,
            b=0.3,
        )
        # Unselected damage must never cause a scan/comparison.
        client.delete_vectors("kb", vectors=["bm25"], points=[store.point_id("unselected")])
    before = files(tmp_path)
    original_docs = deepcopy(documents)
    calls = {"config": 0, "info": 0, "stats": 0, "retrieve": []}
    config = Encoder.from_config
    info = store.QdrantClient.get_collection
    retrieve = store.QdrantClient.retrieve
    stats = store._validate_ready_statistics
    encode = Encoder.encode_documents

    def from_config(value):
        calls["config"] += 1
        return config(value)

    def get_collection(self, *args, **kwargs):
        calls["info"] += 1
        return info(self, *args, **kwargs)

    def statistics(*args):
        calls["stats"] += 1
        return stats(*args)

    def selected(self, collection, **kwargs):
        position = len(calls["retrieve"])
        assert collection == "kb" and kwargs == {
            "ids": [store.point_id(documents[position]["id"])],
            "with_payload": True,
            "with_vectors": ["bm25"],
        }
        calls["retrieve"].append(kwargs["ids"][0])
        return retrieve(self, collection, **kwargs)

    def encoding(self, texts):
        position = len(calls["retrieve"])
        texts = list(texts)
        assert texts == [documents[position]["text"]]
        assert (self.avgdl, self.k1, self.b, self.config["stopwords"]) == (
            19,
            1.8,
            0.3,
            ["private"],
        )
        warnings.warn("private token must not escape", UserWarning)
        return encode(self, texts)

    with monkeypatch.context() as m:
        guard(m, store.QdrantClient)
        m.setattr(Encoder, "from_config", staticmethod(from_config))
        m.setattr(Encoder, "encode_documents", encoding)
        m.setattr(store.QdrantClient, "get_collection", get_collection)
        m.setattr(store.QdrantClient, "retrieve", selected)
        m.setattr(store, "_validate_ready_statistics", statistics)
        assert main(batch_args(path, source)) == int(mismatch)
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert output.err == ""
    assert report["matches"] is (not mismatch)
    assert report["totals"] == {
        "documents": len(documents),
        "matched": 2 if mismatch else 3,
        "mismatched": 2 if mismatch else 0,
        "missing": int(mismatch),
    }
    assert calls == {
        "config": 1,
        "info": 1,
        "stats": 1,
        "retrieve": [store.point_id(d["id"]) for d in documents],
    }
    assert files(tmp_path) == before and documents == original_docs
    assert all(
        secret not in output.out
        for secret in ("private", "ＡＴＬＡＳ", '"n":', '"values":', '"config":', '"tokens":')
    )


@pytest.mark.parametrize("damage", ["no-vector", "bad-vector", "wrong-id", "duplicates", "payload"])
@pytest.mark.parametrize("complete", [False, True])
def test_late_storage_error_no_partial_json(tmp_path, monkeypatch, capsys, damage, complete):
    path, source = tmp_path / "qd", tmp_path / "sources.jsonl"
    documents = [{"id": "absent", "text": "北京"}, *DOCS]
    if complete:
        documents.pop(1)  # Equal indexed count, with a missing point before the late error.
    source.write_text("\n".join(json.dumps(d) for d in documents))
    with closing(store.open_local(path)) as client:
        store.build_collection(client, "kb", DOCS)
        if damage == "no-vector":
            client.delete_vectors("kb", vectors=["bm25"], points=[store.point_id("edited")])
    original = store.QdrantClient.retrieve
    seen = []

    def damaged(self, collection, **kwargs):
        seen.extend(kwargs["ids"])
        records = original(self, collection, **kwargs)
        if kwargs["ids"] == [store.point_id("edited")]:
            if damage == "bad-vector":
                records[0].vector["bm25"].values[0] = float("nan")
            elif damage == "wrong-id":
                records[0].id = store.point_id(7)
            elif damage == "duplicates":
                records *= 2
            elif damage == "payload":
                records[0].payload["metadata"] = {"bad": float("nan")}
        return records

    before = files(tmp_path)
    guard(monkeypatch, store.QdrantClient)
    monkeypatch.setattr(store.QdrantClient, "retrieve", damaged)
    assert main([*batch_args(path, source), *(["--require-complete"] if complete else [])]) == 2
    output = capsys.readouterr()
    assert output.out == "" and "error:" in output.err and "Traceback" not in output.err
    assert seen == [store.point_id(d["id"]) for d in documents]
    assert files(tmp_path) == before


@pytest.mark.parametrize(
    "damage", ["missing", "foreign", "building", "updating", "count", "config", "corpus"]
)
@pytest.mark.parametrize("complete", [False, True])
def test_readiness_error_before_retrieval(tmp_path, monkeypatch, capsys, damage, complete):
    path, source = tmp_path / "qd", tmp_path / "sources.jsonl"
    source.write_text(json.dumps(DOCS[0]))
    with closing(store.open_local(path)) as client:
        if damage == "foreign":
            client.create_collection("kb", vectors_config={})
        elif damage != "missing":
            store.build_collection(client, "kb", DOCS)
            data = deepcopy(client.get_collection("kb").config.metadata["qbm25"])
            if damage in ("building", "updating"):
                data["state"] = damage
            elif damage == "config":
                data["encoder"]["schema"] = True
            elif damage == "corpus":
                data["corpus"]["total_tokens"] = -1
            elif damage == "count":
                client.delete("kb", points_selector=models.PointIdsList(points=[store.point_id(7)]))
            client.update_collection("kb", metadata={"qbm25": data})
    before = files(tmp_path)
    guard(monkeypatch, store.QdrantClient)
    monkeypatch.setattr(store.QdrantClient, "retrieve", no_access)
    assert main([*batch_args(path, source), *(["--require-complete"] if complete else [])]) == 2
    output = capsys.readouterr()
    assert output.out == "" and "error:" in output.err
    assert "document count mismatch" not in output.err
    assert files(tmp_path) == before


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\n \n",
        b"{}",
        b"[]",
        b"null",
        b'{"id":true,"text":"x"}',
        b'{"id":7.0,"text":"x"}',
        b'{"id":7,"text":null}',
        b'{"id":7,"text":"x","extra":0}',
        b'{"id":7,"text":"x","metadata":[]}',
        b'{"id":7,"text":"x","metadata":{"n":NaN}}',
        b'{"id":7,"text":"x","metadata":{"n":1e999}}',
        b'{"id":7,"text":"x","text":"y"}',
        b'{"id":7,"text":"x","metadata":{"n":1,"n":2}}',
        b"\xff",
        b'{"id":"7","text":"x"}\n{"id":"7","text":"y"}',
        b'{"id":7,"text":"x"}\n{"id":7,"text":"y"}',
        b'{"id":7,"text":"x"}\n{',
        b" " * (store.MAX_LINE_BYTES + 1),
        json.dumps({"id": 7, "text": "x" * (store.MAX_TEXT_CHARS + 1)}).encode(),
        b"\n".join(
            json.dumps({"id": i, "text": "x"}).encode() for i in range(store.MAX_DOCUMENTS + 1)
        ),
    ],
)
@pytest.mark.parametrize("complete", [False, True])
def test_entire_input_before_storage(tmp_path, monkeypatch, capsys, raw, complete):
    source = tmp_path / "sources.jsonl"
    source.write_bytes(raw)
    before = files(tmp_path)
    monkeypatch.setattr("ragwrench.bm25.cli.open_local", no_access)
    assert (
        main(
            [
                *batch_args(tmp_path / "absent", source),
                *(["--require-complete"] if complete else []),
            ]
        )
        == 2
    )
    output = capsys.readouterr()
    assert output.out == "" and "error:" in output.err and "Traceback" not in output.err
    assert files(tmp_path) == before


@pytest.mark.parametrize("limit", ["MAX_LINE_BYTES", "MAX_CORPUS_BYTES", "MAX_DOCUMENTS"])
def test_reader_bounds_inclusive(tmp_path, monkeypatch, capsys, limit):
    source = tmp_path / "sources.jsonl"
    raw = b'{"id":7,"text":"x"}\n'
    source.write_bytes(raw)
    threshold = 1 if limit == "MAX_DOCUMENTS" else len(raw)
    monkeypatch.setattr(store, limit, threshold)
    assert store.read_corpus(source) == [{"id": 7, "text": "x"}]
    monkeypatch.setattr(store, limit, threshold - 1)
    monkeypatch.setattr("ragwrench.bm25.cli.open_local", no_access)
    assert main(batch_args(tmp_path / "absent", source)) == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "documents",
    [
        [],
        [DOCS[0], DOCS[0]],
        [DOCS[0], {"id": True, "text": "x"}],
        [DOCS[0], {"id": "bad", "text": "x", "metadata": {"n": float("nan")}}],
    ],
)
@pytest.mark.parametrize("complete", [False, True])
def test_api_full_input_validation_before_client(documents, complete):
    class NoClient:
        def __getattr__(self, name):
            no_access()

    with pytest.raises(ValueError):
        store.verify_documents(NoClient(), "kb", documents, require_complete=complete)


@pytest.mark.parametrize("tokenizer", ["whitespace", "unicode", "cjk-bigram"])
@pytest.mark.parametrize("avgdl", [None, 23.0])
def test_saved_updated_avgdl_and_single_semantics(client, tokenizer, avgdl):
    store.build_collection(
        client, "kb", DOCS, tokenizer=tokenizer, avgdl=avgdl, stopwords=("PRIVATE",), k1=2, b=0.4
    )
    store.update_collection(
        client, "kb", additions=[{"id": "new", "text": "long long long 北京 北京"}]
    )
    documents = [*DOCS, {"id": "missing", "text": ""}]
    before = deepcopy(documents)
    report = store.verify_documents(client, "kb", documents)
    assert report["results"] == [store.verify_document(client, "kb", doc) for doc in documents]
    assert report["totals"] == {"documents": 4, "matched": 3, "mismatched": 1, "missing": 1}
    assert documents == before
