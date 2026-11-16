"""Opt-in indexed-count agreement gates selected comparisons, not authenticity."""

from contextlib import closing
from copy import deepcopy
import json

import pytest

from ragwrench.bm25 import store
from ragwrench.bm25.cli import main
from ragwrench.bm25.encoder import Encoder
from .test_verify_documents import DOCS, batch_args, guard, no_access
from .test_verify_document import files


def test_complete_api_regression(client):
    store.build_collection(client, "kb", DOCS)
    subset = store.verify_documents(client, "kb", DOCS[:1])
    assert subset["matches"] is True and subset["totals"]["documents"] == 1
    with pytest.raises(ValueError, match="document count mismatch: expected=3 actual=1"):
        store.verify_documents(client, "kb", DOCS[:1], require_complete=True)


def test_complete_cli_regression(tmp_path, capsys):
    path, source = tmp_path / "qd", tmp_path / "sources.jsonl"
    source.write_text(json.dumps(DOCS[0]))
    with closing(store.open_local(path)) as client:
        store.build_collection(client, "kb", DOCS)
    assert main(batch_args(path, source)) == 0
    assert json.loads(capsys.readouterr().out)["matches"] is True
    assert main([*batch_args(path, source), "--require-complete"]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "document count mismatch: expected=3 actual=1" in output.err


@pytest.mark.parametrize("gate", [None, 0, 1, 0.0, 1.0, "true", [], {}])
def test_exact_boolean_before_client(gate):
    class NoClient:
        def __getattr__(self, name):
            no_access()

    with pytest.raises(ValueError, match="require_complete must be a boolean"):
        store.verify_documents(NoClient(), "kb", DOCS, require_complete=gate)


def test_gate_keyword_only(client):
    with pytest.raises(TypeError):
        store.verify_documents(client, "kb", DOCS, True)


@pytest.mark.parametrize("documents", [DOCS[:1], [*DOCS, {"id": "skipped", "text": ""}]])
def test_count_gate_before_selected_comparison(client, monkeypatch, documents):
    store.build_collection(client, "kb", DOCS)
    before = deepcopy(documents)
    guard(monkeypatch, client)
    monkeypatch.setattr(client, "retrieve", no_access)
    monkeypatch.setattr(Encoder, "encode_documents", no_access)
    with pytest.raises(
        ValueError, match=f"document count mismatch: expected=3 actual={len(documents)}"
    ):
        store.verify_documents(client, "kb", documents, require_complete=True)
    assert documents == before


@pytest.mark.parametrize("kind", ["agree", "payload", "missing"])
def test_equal_count_same_json_read_only(tmp_path, monkeypatch, capsys, kind):
    path, source = tmp_path / "qd", tmp_path / "sources.jsonl"
    documents = deepcopy([DOCS[1], DOCS[0], DOCS[2]])
    if kind == "payload":
        documents[0]["metadata"] = {"n": 1}
    elif kind == "missing":
        documents[-1] = {"id": "absent", "text": "private"}
    source.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in documents))
    with closing(store.open_local(path)) as client:
        store.build_collection(client, "kb", DOCS, stopwords=("private",), avgdl=19, k1=1.8, b=0.3)
    before = files(tmp_path)
    assert main(batch_args(path, source)) == int(kind != "agree")
    default = capsys.readouterr()
    calls = {"config": 0, "info": 0, "stats": 0, "retrieve": []}
    config, info = Encoder.from_config, store.QdrantClient.get_collection
    stats, retrieve = store._validate_ready_statistics, store.QdrantClient.retrieve
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
        assert kwargs == {
            "ids": [store.point_id(documents[position]["id"])],
            "with_payload": True,
            "with_vectors": ["bm25"],
        }
        calls["retrieve"].extend(kwargs["ids"])
        return retrieve(self, collection, **kwargs)

    def encoding(self, texts):
        assert (self.avgdl, self.k1, self.b, self.config["stopwords"]) == (
            19,
            1.8,
            0.3,
            ["private"],
        )
        texts = list(texts)
        assert texts == [documents[len(calls["retrieve"])]["text"]]
        return encode(self, texts)

    guard(monkeypatch, store.QdrantClient)
    monkeypatch.setattr(Encoder, "from_config", staticmethod(from_config))
    monkeypatch.setattr(store.QdrantClient, "get_collection", get_collection)
    monkeypatch.setattr(store, "_validate_ready_statistics", statistics)
    monkeypatch.setattr(store.QdrantClient, "retrieve", selected)
    monkeypatch.setattr(Encoder, "encode_documents", encoding)
    assert main([*batch_args(path, source), "--require-complete"]) == int(kind != "agree")
    output = capsys.readouterr()
    assert output.out == default.out and output.err == default.err == ""
    report = json.loads(output.out)
    assert report["totals"]["missing"] == int(kind == "missing")
    assert [type(r["id"]) for r in report["results"][:2]] == [str, int]
    assert calls == {
        "config": 1,
        "info": 1,
        "stats": 1,
        "retrieve": [store.point_id(d["id"]) for d in documents],
    }
    assert files(tmp_path) == before


def test_zero_indexed_nonempty_contract(tmp_path, monkeypatch, capsys):
    path, source = tmp_path / "qd", tmp_path / "sources.jsonl"
    with closing(store.open_local(path)) as client:
        store.build_collection(client, "kb", DOCS)
        store.update_collection(client, "kb", delete_ids=[d["id"] for d in DOCS])
        assert store.status_collection(client, "kb")["corpus"]["document_count"] == 0
        with pytest.raises(ValueError, match="corpus contains no documents"):
            store.verify_documents(client, "kb", [], require_complete=True)
    source.write_text(json.dumps(DOCS[0]))
    before = files(tmp_path)
    assert main(batch_args(path, source)) == 1
    assert json.loads(capsys.readouterr().out)["totals"]["missing"] == 1
    guard(monkeypatch, store.QdrantClient)
    monkeypatch.setattr(store.QdrantClient, "retrieve", no_access)
    monkeypatch.setattr(Encoder, "encode_documents", no_access)
    assert main([*batch_args(path, source), "--require-complete"]) == 2
    output = capsys.readouterr()
    assert output.out == "" and "document count mismatch: expected=0 actual=1" in output.err
    assert files(tmp_path) == before


def test_skipped_originals_and_legacy_manifest(client):
    documents = [*DOCS, {"id": "skipped", "text": "private"}]
    with pytest.warns(UserWarning):
        store.build_collection(client, "kb", documents, stopwords=("private",))
    assert store.verify_documents(client, "kb", documents)["totals"]["missing"] == 1
    with pytest.raises(ValueError, match="document count mismatch: expected=3 actual=4"):
        store.verify_documents(client, "kb", documents, require_complete=True)
    assert store.verify_documents(client, "kb", DOCS, require_complete=True)["matches"] is True
    manifest = deepcopy(client.get_collection("kb").config.metadata["qbm25"])
    del manifest["corpus"]  # Existing ready legacy support still uses indexed, not token totals.
    client.update_collection("kb", metadata={"qbm25": manifest})
    assert store.verify_documents(client, "kb", DOCS, require_complete=True)["matches"] is True
