"""Selected source verification is a comparison, not a corpus integrity audit."""

from contextlib import closing
from copy import deepcopy
import hashlib
import json
import math
import os
import warnings

import pytest
from qdrant_client import models

from ragwrench.bm25 import store
from ragwrench.bm25.cli import main
from ragwrench.bm25.encoder import Encoder


DOC = {"id": 7, "text": "private ＡＴＬＡＳ 数据库 北京 北京", "metadata": {"n": 1, "ok": True}}


def files(path):
    return {
        str(p.relative_to(path)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in path.rglob("*")
        if p.is_file()
    }


def guard(monkeypatch, client=None):
    def unexpected(*args, **kwargs):
        pytest.fail("verification must not scan, query, write, fsync or encode other points")

    target = client if client is not None else store.QdrantClient
    for name in (
        "upsert",
        "delete",
        "create_collection",
        "delete_collection",
        "update_collection",
        "scroll",
        "set_payload",
        "overwrite_payload",
        "delete_payload",
        "update_vectors",
        "delete_vectors",
        "query_points",
    ):
        monkeypatch.setattr(target, name, unexpected)
    monkeypatch.setattr(os, "fsync", unexpected)
    original = store.QdrantClient.retrieve

    def retrieve(self, collection, **kwargs):
        assert kwargs == {
            "ids": [store.point_id(7)],
            "with_payload": True,
            "with_vectors": ["bm25"],
        }
        return original(self, collection, **kwargs)

    monkeypatch.setattr(store.QdrantClient, "retrieve", retrieve)
    encode = Encoder.encode_documents

    def selected(self, texts):
        texts = list(texts)
        assert texts == [DOC["text"]]
        return encode(self, texts)

    monkeypatch.setattr(Encoder, "encode_documents", selected)


def args(path, document, collection="kb"):
    return ["verify-document", "--path", str(path), "--collection", collection, str(document)]


def test_same_count_edit_missed_by_status(client):
    store.build_collection(client, "kb", [DOC, {"id": "7", "text": "other 北京"}])
    before = store.status_collection(client, "kb", require_ready=True)
    client.set_payload("kb", {"text": "externally changed"}, points=[store.point_id(7)])
    assert store.status_collection(client, "kb", require_ready=True) == before
    report = store.verify_document(client, "kb", DOC)
    assert report["mismatches"] == ["text"]
    assert report["payload"]["matches"] is False
    assert report["sparse_vector"]["matches"] is True
    assert report["matches"] is False


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "text",
        "metadata",
        "source_id",
        "indices",
        "values",
        "missing",
        "no-metadata",
        "extra-payload",
        "reorder-metadata",
    ],
)
def test_cli_point_comparison_read_only(tmp_path, monkeypatch, capsys, damage):
    path, document = tmp_path / "qd", tmp_path / "document.json"
    document.write_text(json.dumps(DOC, ensure_ascii=False, indent=2))
    with closing(store.open_local(path)) as client:
        store.build_collection(
            client,
            "kb",
            [DOC, {"id": "7", "text": "other 北京"}],
            stopwords=("private",),
            avgdl=19,
            k1=1.8,
            b=0.3,
        )
        if damage in ("text", "metadata", "source_id", "extra-payload", "reorder-metadata"):
            values = {"text": "changed"} if damage == "text" else {"metadata": {"n": True}}
            if damage == "source_id":
                values = {"source_id": "7"}
            if damage == "extra-payload":
                values = {"unmanaged": "ignored"}
            if damage == "reorder-metadata":
                values = {"metadata": {"ok": True, "n": 1}}
            client.set_payload("kb", values, points=[store.point_id(7)])
        elif damage == "no-metadata":
            client.delete_payload("kb", keys=["metadata"], points=[store.point_id(7)])
        elif damage in ("indices", "values"):
            record = client.retrieve("kb", ids=[store.point_id(7)], with_vectors=True)[0]
            vector = record.vector["bm25"]
            if damage == "indices":
                vector.indices[0] = 0
            else:
                vector.values[0] = math.nextafter(vector.values[0], math.inf)
            client.update_vectors(
                "kb", [models.PointVectors(id=store.point_id(7), vector={"bm25": vector})]
            )
        elif damage == "missing":
            # A same-count replacement hides the selected missing point from status.
            client.delete("kb", points_selector=models.PointIdsList(points=[store.point_id(7)]))
            encoder = Encoder.from_config(
                client.get_collection("kb").config.metadata["qbm25"]["encoder"]
            )
            client.upsert(
                "kb", store._encode_points([{"id": "replacement", "text": "北京"}], encoder)
            )
        assert store.status_collection(client, "kb", require_ready=True)["points"]["current"] == 2
    before = files(tmp_path)
    mismatch = damage not in (None, "extra-payload", "reorder-metadata")
    with monkeypatch.context() as m:
        guard(m)
        assert main(args(path, document)) == (1 if mismatch else 0)
        output = capsys.readouterr()
        assert output.err == ""
        report = json.loads(output.out)
        with closing(store.open_local(path)) as client:
            original = deepcopy(DOC)
            assert store.verify_document(client, "kb", DOC) == report
            assert DOC == original
    assert files(tmp_path) == before
    assert report["id"] == 7 and type(report["id"]) is int
    assert report["matches"] is (not mismatch)
    category = {
        "indices": ["sparse_indices", "sparse_values"],
        "values": ["sparse_values"],
        "missing": ["point_missing"],
        "no-metadata": ["metadata"],
    }.get(damage, [damage] if mismatch else [])
    assert report["mismatches"] == category
    if damage == "missing":
        assert report["point_present"] is False
        assert report["payload"] is None and report["sparse_vector"] is None
    else:
        assert report["point_present"] is True
        assert report["payload"] == {
            "source_id": damage != "source_id",
            "text": damage != "text",
            "metadata": damage not in ("metadata", "no-metadata"),
            "matches": damage not in ("source_id", "text", "metadata", "no-metadata"),
        }
        assert report["sparse_vector"] == {
            "indices_match": damage != "indices",
            "values_match": damage not in ("indices", "values"),
            "matches": damage not in ("indices", "values"),
        }
    assert not any(
        secret in output.out
        for secret in (DOC["text"], "private", '"values":', "tokens", "stopwords")
    )


@pytest.mark.parametrize("tokenizer", ["whitespace", "unicode", "cjk-bigram"])
@pytest.mark.parametrize("avgdl", [None, 23.0])
def test_saved_encoder_typed_ids_and_updated_avgdl(client, tokenizer, avgdl):
    docs = [DOC, {"id": "7", "text": "other 北京", "metadata": {}}]
    store.build_collection(
        client, "kb", docs, tokenizer=tokenizer, avgdl=avgdl, stopwords=("PRIVATE",), k1=2, b=0.4
    )
    store.update_collection(
        client, "kb", additions=[{"id": "new", "text": "more more more 数据库"}]
    )
    for doc in docs:
        report = store.verify_document(client, "kb", doc)
        assert report["matches"] is True and report["id"] == doc["id"]
        assert type(report["id"]) is type(doc["id"])
    assert store.verify_document(client, "kb", {"id": 8, "text": DOC["text"]})["mismatches"] == [
        "point_missing"
    ]
    # Same normalized text/vector is still a raw payload mismatch.
    changed = {**DOC, "text": DOC["text"].replace("ＡＴＬＡＳ", "atlas")}
    report = store.verify_document(client, "kb", changed)
    assert report["mismatches"] == ["text"] and report["sparse_vector"]["matches"] is True


@pytest.mark.parametrize("text", ["", "private", "\n\t"])
def test_empty_source_is_missing_not_authenticated(client, text):
    store.build_collection(client, "kb", [DOC], stopwords=("private",))
    report = store.verify_document(client, "kb", {"id": "skipped", "text": text})
    assert report["mismatches"] == ["point_missing"] and report["matches"] is False


@pytest.mark.parametrize(
    "damage",
    ["missing", "foreign", "building", "updating", "count", "config", "corpus", "no-vector"],
)
def test_invalid_collection_no_json(tmp_path, monkeypatch, capsys, damage):
    path, document = tmp_path / "qd", tmp_path / "document.json"
    document.write_text(json.dumps(DOC))
    with closing(store.open_local(path)) as client:
        if damage == "foreign":
            client.create_collection("kb", vectors_config={})
        elif damage != "missing":
            store.build_collection(client, "kb", [DOC])
            data = deepcopy(client.get_collection("kb").config.metadata["qbm25"])
            if damage in ("building", "updating"):
                data["state"] = damage
            elif damage == "config":
                data["encoder"]["schema"] = True
            elif damage == "corpus":
                data["corpus"]["total_tokens"] = -1
            elif damage == "count":
                client.delete("kb", points_selector=models.PointIdsList(points=[store.point_id(7)]))
            elif damage == "no-vector":
                client.delete_vectors("kb", vectors=["bm25"], points=[store.point_id(7)])
            client.update_collection("kb", metadata={"qbm25": data})
    before = files(tmp_path)
    guard(monkeypatch)
    assert main(args(path, document)) == 2
    output = capsys.readouterr()
    assert output.out == "" and "error:" in output.err and "Traceback" not in output.err
    assert DOC["text"] not in output.err
    assert files(tmp_path) == before


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"{}",
        b"[]",
        b"null",
        b'{"id":true,"text":"x"}',
        b'{"id":7.0,"text":"x"}',
        b'{"id":7,"text":null}',
        b'{"id":7,"text":"x","extra":0}',
        b'{"id":7,"text":"x","metadata":[]}',
        b'{"id":7,"text":"x","metadata":{"n":NaN}}',
        b'{"id":7,"text":"x","text":"y"}',
        b'{"id":7,"text":"x","metadata":{"n":1,"n":2}}',
        b'{"id":7,"text":"x"}\n{"id":8,"text":"y"}',
        b"\xff",
        b" " * (store.MAX_LINE_BYTES + 1),
    ],
)
def test_strict_input_before_storage(tmp_path, monkeypatch, capsys, raw):
    document = tmp_path / "doc.json"
    document.write_bytes(raw)
    before = files(tmp_path)
    monkeypatch.setattr(
        store, "open_local", lambda *a, **k: pytest.fail("must validate input first")
    )
    # CLI imports open_local directly.
    monkeypatch.setattr("ragwrench.bm25.cli.open_local", store.open_local)
    assert main(args(tmp_path / "absent", document)) == 2
    output = capsys.readouterr()
    assert output.out == "" and "error:" in output.err and "Traceback" not in output.err
    assert files(tmp_path) == before


@pytest.mark.parametrize(
    "damage",
    [
        "wrong-id",
        "duplicates",
        "missing-vector",
        "duplicate-index",
        "nan",
        "bad-index",
        "short-values",
    ],
)
def test_unestablishable_retrieval_refused(client, monkeypatch, damage):
    store.build_collection(client, "kb", [DOC])
    record = client.retrieve("kb", ids=[store.point_id(7)], with_vectors=True)[0]
    if damage == "wrong-id":
        record.id = store.point_id("7")
    elif damage == "missing-vector":
        record.vector = {}
    else:
        vector = record.vector["bm25"]
        if damage == "duplicate-index":
            vector.indices[1] = vector.indices[0]
        elif damage == "nan":
            vector.values[0] = float("nan")
        elif damage == "bad-index":
            vector.indices[0] = -1
        elif damage == "short-values":
            vector.values.pop()
    monkeypatch.setattr(
        client, "retrieve", lambda *a, **k: [record, record] if damage == "duplicates" else [record]
    )
    with pytest.raises(ValueError):
        store.verify_document(client, "kb", DOC)


def test_sparse_order_and_collision_privacy(client, monkeypatch, capsys):
    doc = {"id": 7, "text": "private alpha beta"}
    store.build_collection(client, "kb", [doc], tokenizer="whitespace")
    record = client.retrieve("kb", ids=[store.point_id(7)], with_vectors=True)[0]
    vector = record.vector["bm25"]
    vector.indices.reverse()
    vector.values.reverse()
    monkeypatch.setattr(client, "retrieve", lambda *a, **k: [record])
    assert store.verify_document(client, "kb", doc)["matches"] is True
    encode = Encoder.encode_documents

    def warning(self, docs):
        warnings.warn("private alpha beta", UserWarning)
        return encode(self, docs)

    monkeypatch.setattr(Encoder, "encode_documents", warning)
    with warnings.catch_warnings(record=True) as seen:
        assert store.verify_document(client, "kb", doc)["matches"] is True
    assert not seen and capsys.readouterr().err == ""
