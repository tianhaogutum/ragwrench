"""Selected stored sources export all-or-error, independently of comparison inputs."""

from contextlib import closing
from copy import deepcopy
import json
import os
import stat
from types import SimpleNamespace

import pytest

from ragwrench.bm25 import cli, store
from ragwrench.bm25.cli import main
from ragwrench.bm25.encoder import Encoder
from .test_verify_document import files
from .test_verify_documents import guard, no_access


DOCS = [
    {"id": 7, "text": "原文 ＡＴＬＡＳ 北京\n서울", "metadata": {"n": 1, "a": [True, None, 1.0]}},
    {"id": "7", "text": "東京 データ", "metadata": {"n": True, "s": "𠮷 é"}},
    {"id": "other", "text": "unselected"},
]


def args(path, selection, output):
    return [
        "export-documents",
        "--path",
        str(path),
        "--collection",
        "kb",
        "--ids",
        str(selection),
        "--output",
        str(output),
    ]


def test_export_typed_ids_cli(tmp_path, capsys):
    path, selection, output = tmp_path / "qd", tmp_path / "ids.json", tmp_path / "out.jsonl"
    with closing(store.open_local(path)) as client:
        store.build_collection(client, "kb", DOCS)
        assert (
            store.status_collection(client, "kb", require_ready=True)["corpus"]["document_count"]
            == 3
        )
    selection.write_text('["7", 7]', encoding="utf-8")
    assert main(args(path, selection, output)) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"schema": 1, "collection": "kb", "exported": 2}
    assert captured.err == ""
    expected = [DOCS[1], DOCS[0]]
    assert store.read_corpus(output) == expected
    assert [type(doc["id"]) for doc in store.read_corpus(output)] == [str, int]
    assert "東京" in output.read_text(encoding="utf-8")
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_only_selected_payloads_read(client, monkeypatch):
    store.build_collection(client, "kb", DOCS, avgdl=19, k1=1.8, b=0.3, stopwords=("private",))
    # Unselected malformed payload and unavailable vectors must be irrelevant.
    client.set_payload("kb", {"text": None}, points=[store.point_id("other")])
    client.delete_vectors("kb", vectors=["bm25"], points=[store.point_id(7)])
    original = client.retrieve
    calls = []

    def retrieve(collection, **kwargs):
        calls.append(kwargs)
        assert kwargs == {
            "ids": [store.point_id(["7", 7][len(calls) - 1])],
            "with_payload": True,
            "with_vectors": False,
        }
        return original(collection, **kwargs)

    info = client.get_collection
    configs = []

    def get_collection(collection):
        configs.append(collection)
        return info(collection)

    guard(monkeypatch, client)
    monkeypatch.setattr(client, "get_collection", get_collection)
    monkeypatch.setattr(client, "retrieve", retrieve)
    monkeypatch.setattr(Encoder, "encode_documents", no_access)
    monkeypatch.setattr(Encoder, "encode_query", no_access)
    ids = ["7", 7]
    exported = store.export_documents(client, "kb", ids)
    assert json.dumps(exported, sort_keys=True) == json.dumps([DOCS[1], DOCS[0]], sort_keys=True)
    assert ids == ["7", 7] and configs == ["kb"] and len(calls) == 2


@pytest.mark.parametrize(
    "ids",
    [
        [],
        {},
        None,
        "7",
        [7, 7],
        ["7", "7"],
        [True],
        [-1],
        [2**64],
        [1.0],
        [None],
        [""],
        ["x" * 257],
    ],
)
def test_api_bad_selection_before_client(monkeypatch, ids):
    monkeypatch.setattr(store, "_verification_encoder", no_access)
    with pytest.raises(ValueError):
        store.export_documents(None, "kb", ids)


@pytest.mark.parametrize(
    "raw",
    [
        b"[]",
        b"{}",
        b"[7,7]",
        b"[true]",
        b"[1.0]",
        b"[-1]",
        b"[null]",
        b'[""]',
        b"[NaN]",
        b"[7] trailing",
        b'["\xff"]',
        b'[{"a":1,"a":2}]',
    ],
)
def test_cli_bad_selection_before_storage(tmp_path, monkeypatch, capsys, raw):
    selection, output = tmp_path / "ids.json", tmp_path / "out"
    selection.write_bytes(raw)
    monkeypatch.setattr(cli, "open_local", no_access)
    assert main(args(tmp_path / "absent", selection, output)) == 2
    assert capsys.readouterr().out == ""
    assert not output.exists() and not list(tmp_path.glob(".qbm25-export-*"))


@pytest.mark.parametrize(
    "change",
    [
        {"source_id": True},
        {"source_id": "7"},
        {"source_id": None},
        {"text": None},
        {"text": 7},
        {"metadata": []},
        {"metadata": None},
    ],
)
def test_later_malformed_record_no_output(tmp_path, capsys, change):
    path, selection, output = tmp_path / "qd", tmp_path / "ids", tmp_path / "out"
    with closing(store.open_local(path)) as client:
        store.build_collection(client, "kb", DOCS)
        client.overwrite_payload(
            "kb",
            {"source_id": 7, "text": DOCS[0]["text"], "metadata": DOCS[0]["metadata"], **change},
            points=[store.point_id(7)],
        )
    selection.write_text('["7", 7]')
    before = files(tmp_path)
    assert main(args(path, selection, output)) == 2
    assert capsys.readouterr().out == ""
    assert files(tmp_path) == before


@pytest.mark.parametrize("missing", ["absent", "skipped"])
def test_later_missing_no_output(tmp_path, capsys, missing):
    path, selection, output = tmp_path / "qd", tmp_path / "ids", tmp_path / "out"
    with closing(store.open_local(path)) as client:
        with pytest.warns(UserWarning, match="empty"):
            store.build_collection(client, "kb", [*DOCS, {"id": "skipped", "text": ""}])
    selection.write_text(json.dumps(["7", missing]))
    before = files(tmp_path)
    assert main(args(path, selection, output)) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "does not exist" in captured.err
    assert files(tmp_path) == before


def test_metadata_normalized_and_internal_fields_omitted(client):
    store.build_collection(client, "kb", DOCS)
    client.delete_payload("kb", keys=["metadata"], points=[store.point_id(7)])
    client.set_payload("kb", {"internal": {"sparse": [999]}}, points=[store.point_id(7)])
    exported = store.export_documents(client, "kb", [7, "other"])
    assert exported == [{**DOCS[0], "metadata": {}}, {**DOCS[2], "metadata": {}}]
    exported[0]["text"] = "caller change"
    assert store.export_documents(client, "kb", [7])[0]["text"] == DOCS[0]["text"]


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate",
        "wrong-point",
        "no-payload",
        "missing-source",
        "nonfinite",
        "unicode",
        "long-text",
    ],
)
def test_unavailable_or_invalid_decoding(client, monkeypatch, kind):
    store.build_collection(client, "kb", DOCS)
    record = client.retrieve("kb", ids=[store.point_id(7)], with_payload=True, with_vectors=False)[
        0
    ]
    record = SimpleNamespace(id=record.id, payload=deepcopy(record.payload))
    records = [record]
    if kind == "duplicate":
        records *= 2
    elif kind == "wrong-point":
        record.id = store.point_id("7")
    elif kind == "no-payload":
        record.payload = None
    elif kind == "missing-source":
        del record.payload["source_id"]
    elif kind == "nonfinite":
        record.payload["metadata"] = {"n": float("nan")}
    elif kind == "unicode":
        record.payload["text"] = "\ud800"
    elif kind == "long-text":
        record.payload["text"] = "x" * (store.MAX_TEXT_CHARS + 1)
    monkeypatch.setattr(client, "retrieve", lambda *a, **kw: records)
    with pytest.raises((ValueError, UnicodeError)):
        store.export_documents(client, "kb", [7])


@pytest.mark.parametrize("boundary", ["ids-count", "ids-bytes", "text", "line", "total"])
def test_limits_before_publication(tmp_path, monkeypatch, capsys, boundary):
    path, selection, output = tmp_path / "qd", tmp_path / "ids", tmp_path / "out"
    with closing(store.open_local(path)) as client:
        store.build_collection(client, "kb", DOCS)
    selection.write_text('["7", 7]')
    if boundary == "ids-count":
        monkeypatch.setattr(store, "MAX_DOCUMENTS", 1)
        monkeypatch.setattr(cli, "open_local", no_access)
    elif boundary == "ids-bytes":
        selection.write_bytes(b" " * (store.MAX_LINE_BYTES + 1))
        monkeypatch.setattr(cli, "open_local", no_access)
    elif boundary == "text":
        monkeypatch.setattr(store, "MAX_TEXT_CHARS", 3)
    elif boundary == "line":
        # Selection fits; a source line does not.
        monkeypatch.setattr(store, "MAX_LINE_BYTES", 100)
    else:
        line = (json.dumps(DOCS[1], ensure_ascii=False) + "\n").encode("utf-8")
        monkeypatch.setattr(store, "MAX_CORPUS_BYTES", len(line))
    before = files(tmp_path)
    assert main(args(path, selection, output)) == 2
    assert capsys.readouterr().out == ""
    assert files(tmp_path) == before


@pytest.mark.parametrize(
    "state",
    [
        "building",
        "updating",
        "bad-statistics",
        "bad-config",
        "wrong-count",
        "foreign",
        "missing",
        "legacy",
    ],
)
def test_saved_readiness_before_retrieval(client, monkeypatch, state):
    store.build_collection(client, "kb", DOCS)
    info = client.get_collection("kb")
    manifest = deepcopy(info.config.metadata["qbm25"])
    if state in ("building", "updating"):
        manifest["state"] = state
    elif state == "bad-statistics":
        manifest["corpus"]["total_tokens"] = False
    elif state == "bad-config":
        manifest["encoder"]["schema"] = True
    elif state == "wrong-count":
        manifest["indexed"] += 1
    elif state == "legacy":
        del manifest["corpus"]
    client.update_collection(
        "kb", metadata={"qbm25": None} if state == "foreign" else {"qbm25": manifest}
    )
    if state == "legacy":
        assert store.export_documents(client, "kb", [7]) == [DOCS[0]]
    else:
        monkeypatch.setattr(client, "retrieve", no_access)
        with pytest.raises(ValueError):
            store.export_documents(client, "absent" if state == "missing" else "kb", [7])


@pytest.mark.parametrize(
    "destination",
    [
        "selection",
        "file",
        "directory",
        "hardlink",
        "symlink",
        "dangling",
        "storage",
        "storage-alias",
        "no-parent",
    ],
)
def test_output_rejected_before_storage(tmp_path, monkeypatch, capsys, destination):
    path, selection, output = tmp_path / "qd", tmp_path / "ids", tmp_path / "out"
    path.mkdir()
    selection.write_text("[7]")
    if destination == "selection":
        output = selection
    elif destination == "file":
        output.write_text("existing")
    elif destination == "directory":
        output.mkdir()
    elif destination == "hardlink":
        os.link(selection, output)
    elif destination in ("symlink", "dangling"):
        output.symlink_to(selection if destination == "symlink" else tmp_path / "missing")
    elif destination == "storage":
        output = path / "fresh"
    elif destination == "storage-alias":
        alias = tmp_path / "alias"
        alias.symlink_to(path, target_is_directory=True)
        output = alias / "fresh"
    elif destination == "no-parent":
        output = tmp_path / "missing" / "out"
    before = files(tmp_path) if destination != "dangling" else None
    monkeypatch.setattr(cli, "open_local", no_access)
    assert main(args(path, selection, output)) == 2
    assert capsys.readouterr().out == ""
    if before is not None:
        assert files(tmp_path) == before
    else:
        assert output.is_symlink()


@pytest.mark.parametrize("failure", ["race", "link", "write"])
def test_publication_failure_cleans_temporary(tmp_path, monkeypatch, capsys, failure):
    path, selection, output = tmp_path / "qd", tmp_path / "ids", tmp_path / "out"
    with closing(store.open_local(path)) as client:
        store.build_collection(client, "kb", DOCS)
    selection.write_text('["7", 7]')
    link = os.link

    def broken_link(source, target):
        if failure == "race":
            target.write_bytes(b"race winner")
            return link(source, target)
        raise OSError("injected link failure")

    if failure == "write":
        original = cli.tempfile.NamedTemporaryFile

        def broken_file(*a, **kw):
            stream = original(*a, **kw)
            write = stream.write
            calls = []

            def broken_write(data):
                calls.append(data)
                if len(calls) == 2:
                    raise OSError("injected write failure after first record")
                return write(data)

            stream.write = broken_write
            return stream

        monkeypatch.setattr(cli.tempfile, "NamedTemporaryFile", broken_file)
    else:
        monkeypatch.setattr(os, "link", broken_link)
    assert main(args(path, selection, output)) == 2
    assert capsys.readouterr().out == ""
    assert not list(tmp_path.glob(".qbm25-export-*"))
    if failure == "race":
        assert output.read_bytes() == b"race winner"
    else:
        assert not output.exists()


def test_read_only_files_search_and_instantiating_client(tmp_path, monkeypatch, capsys):
    path, selection, output = tmp_path / "qd", tmp_path / "ids", tmp_path / "out"
    with closing(store.open_local(path)) as client:
        store.build_collection(client, "kb", DOCS, avgdl=19, k1=1.8, b=0.3)
        hits = store.search_collection(client, "kb", "北京 東京")
    selection.write_text('["7", 7]')
    before = files(path)
    selection_before = files(tmp_path)["ids"]
    with monkeypatch.context() as m:
        guard(m, store.QdrantClient)
        m.setattr(Encoder, "encode_documents", no_access)
        assert main(args(path, selection, output)) == 0
    assert json.loads(capsys.readouterr().out)["exported"] == 2
    assert files(path) == before
    assert files(tmp_path)["ids"] == selection_before
    with closing(store.open_local(path, must_exist=True)) as client:
        assert store.search_collection(client, "kb", "北京 東京") == hits


def test_exact_json_types_and_id_boundaries(client):
    documents = [
        {
            "id": 0,
            "text": "𠮷 서울",
            "metadata": {
                "int": 2**60 + 3,
                "float": 1.0,
                "zero": -0.0,
                "nested": [None, False, True, {"n": 1, "s": "é"}],
            },
        },
        {"id": 2**64 - 1, "text": "北京"},
        {"id": "x" * 256, "text": "東京"},
    ]
    store.build_collection(client, "kb", documents)
    actual = store.export_documents(client, "kb", [d["id"] for d in documents])
    expected = [{**d, "metadata": d.get("metadata", {})} for d in documents]
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


def test_ready_zero_rejects_any_selected_id(client):
    store.build_collection(client, "kb", DOCS)
    store.update_collection(client, "kb", delete_ids=[d["id"] for d in DOCS])
    with pytest.raises(ValueError, match="does not exist"):
        store.export_documents(client, "kb", [7])
