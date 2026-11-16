import copy
from pathlib import Path
import uuid

import pytest

from ragwrench.sync.core import (SyncError, build_snapshot, chunk_text, make_plan, normalize,
                            snapshot_dir, snapshot_jsonl, validate_plan, validate_snapshot)

FIXTURES = Path(__file__).parent / "fixtures"


def row(text, doc="guide", **kw):
    return {"doc_id": doc, "text": text, **kw}


def test_normalization_ids_duplicates():
    a = build_snapshot([row(" cafe\u0301\r\nline  "), row("café\nline")])
    b = build_snapshot([row("café\nline"), row("café\nline")])
    assert a == b
    assert a["chunks"][0]["content_hash"] == a["chunks"][1]["content_hash"]
    assert a["chunks"][0]["id"] != a["chunks"][1]["id"]
    for i, chunk in enumerate(a["chunks"]):
        assert uuid.UUID(chunk["id"]).version == 5
        assert chunk["occurrence"] == i
    validate_snapshot(a)


def test_embedding_inputs_and_metadata_are_separate():
    a = build_snapshot([row("same", metadata={"access": "public"})])
    b = build_snapshot([row("same", metadata={"access": "private"})])
    c = build_snapshot([row("same", embed_metadata={"title": "new"})])
    assert a["chunks"][0]["content_hash"] == b["chunks"][0]["content_hash"]
    assert make_plan(a, b)["summary"]["payload_only"] == 1
    assert make_plan(a, c)["summary"]["upsert"] == 1


def test_classification_and_move_reuse():
    old = build_snapshot([row("stable", "a"), row("meta", "b", metadata={"x": 1}),
                          row("renamed", "c"), row("gone", "d"), row("one", "e"), row("two", "e")])
    new = build_snapshot([row("stable", "a"), row("meta", "b", metadata={"x": 2}),
                          row("renamed", "renamed"), row("added", "f"), row("two", "e"), row("one", "e")])
    plan = make_plan(old, new)
    assert plan["summary"] == {"upsert": 1, "delete": 1, "unchanged": 1, "payload_only": 1,
                              "moved": 3, "new_chunks": 6, "embeddings_avoided": 5, "embed_model_changed": False}
    assert validate_plan(plan) == plan


def test_model_change_requires_all_embeddings():
    old = build_snapshot([row("a"), row("b")])
    new = build_snapshot([row("a"), row("b")], embed_model="new-model")
    s = make_plan(old, new)["summary"]
    assert s["embed_model_changed"] and s["upsert"] == s["delete"] == 2
    assert s["embeddings_avoided"] == 0


@pytest.mark.parametrize("corrupt", ["version", "text", "hash", "id", "fingerprint", "index", "occurrence", "extra", "order", "bool_index"])
def test_reject_corrupt_manifest(corrupt):
    value = build_snapshot([row("first"), row("second")])
    if corrupt == "version":
        value["version"] = 0
    elif corrupt == "text":
        value["chunks"][0]["text"] = "altered"
    elif corrupt == "hash":
        value["chunks"][0]["content_hash"] = "bad"
    elif corrupt == "id":
        value["chunks"][0]["id"] = "not-a-uuid"
    elif corrupt == "fingerprint":
        value["fingerprint"] = "bad"
    elif corrupt == "extra":
        value["unknown"] = 1
    elif corrupt == "order":
        value["chunks"].reverse()
    elif corrupt == "bool_index":
        value["chunks"][0]["index"] = False
    else:
        value["chunks"][0][corrupt] = 99
    with pytest.raises(SyncError):
        validate_snapshot(value)


@pytest.mark.parametrize("bad", [None, [], {"version": 1}, {"kind": "chunksync.snapshot", "version": 1}])
def test_malformed_snapshot(bad):
    with pytest.raises(SyncError):
        validate_snapshot(bad)


def test_plan_tampering_and_namespace():
    old = build_snapshot([])
    new = build_snapshot([row("a")])
    plan = make_plan(old, new)
    plan["upsert"] = []
    with pytest.raises(SyncError, match="operations mismatch"):
        validate_plan(plan)
    other = build_snapshot([], namespace=str(uuid.uuid4()))
    with pytest.raises(SyncError, match="namespaces"):
        make_plan(other, new)


@pytest.mark.parametrize("rows", [[row("")], [row(" ")], [row(3)], [{"doc_id": "a"}], [row("a", metadata=[])], [row("a", unexpected=True)], [row("a", doc="")]])
def test_bad_chunk_rows(rows):
    with pytest.raises(SyncError):
        build_snapshot(rows)


@pytest.mark.parametrize("method", ["fixed", "cdc"])
def test_chunker_deterministic_and_bounds(method):
    text = "Words and symbols 你好。 " * 200
    chunks = chunk_text(text, method, 32, 64, 100)
    assert chunks == chunk_text(text, method, 32, 64, 100)
    assert all(0 < len(c) <= 100 for c in chunks)
    assert "".join("".join(chunks).split()) == "".join(text.split())
    assert chunk_text("", method) == []
    assert chunk_text("x" * 3000, method)
    assert max(map(len, chunk_text("x" * 3000, method))) <= 512


@pytest.mark.parametrize("sizes", [(0, 2, 4), (5, 4, 6), (2, 8, 4)])
def test_invalid_chunk_bounds(sizes):
    with pytest.raises(SyncError):
        chunk_text("hello", "cdc", *sizes)


def test_real_markdown_fixture_and_cdc_stability():
    old = snapshot_dir(FIXTURES / "docs_v1")
    new = snapshot_dir(FIXTURES / "docs_v2")
    assert new["chunks"][0]["text"].startswith("# A new season")
    plan = make_plan(old, new)
    assert plan["summary"]["upsert"] == 1
    assert plan["summary"]["embeddings_avoided"] == len(old["chunks"])
    fixed = make_plan(snapshot_dir(FIXTURES / "docs_v1", method="fixed"),
                      snapshot_dir(FIXTURES / "docs_v2", method="fixed"))
    assert fixed["summary"]["upsert"] > plan["summary"]["upsert"]


def test_jsonl_and_input_refusals(tmp_path):
    file = tmp_path / "chunks.jsonl"
    file.write_text('{"doc_id":"a","text":"one"}\n{"doc_id":"a","text":"one"}\n')
    assert len(snapshot_jsonl(file)["chunks"]) == 2
    file.write_text('{"doc_id": bad}\n')
    with pytest.raises(SyncError, match="line 1"):
        snapshot_jsonl(file)
    file.unlink()
    (tmp_path / "unsupported.pdf").write_bytes(b"%PDF")
    with pytest.raises(SyncError, match="unsupported"):
        snapshot_dir(tmp_path)
    (tmp_path / "unsupported.pdf").unlink()
    (tmp_path / "outside.md").symlink_to(FIXTURES / "docs_v1/guide.md")
    with pytest.raises(SyncError, match="symlink"):
        snapshot_dir(tmp_path)


def test_size_limits_fail_before_writing_or_allocating(tmp_path, monkeypatch):
    from ragwrench.sync import core
    monkeypatch.setattr(core, "MAX_JSON_BYTES", 16)
    destination = tmp_path / "out.json"
    with pytest.raises(SyncError, match="output exceeds"):
        core.write_json(destination, {"large": "x" * 100})
    assert not destination.exists()
    with pytest.raises(SyncError, match="chunk input exceeds"):
        build_snapshot([row("large text" * 10)])
    monkeypatch.setattr(core, "MAX_CHUNKS", 2)
    with pytest.raises(SyncError, match="chunk count"):
        chunk_text("x" * 200, "fixed", 1, 1, 2)


def test_empty_dir_still_checks_chunk_options(tmp_path):
    with pytest.raises(SyncError, match="min-size"):
        snapshot_dir(tmp_path, min_size=0)


def test_non_regular_input_refused_without_opening(tmp_path):
    import os
    from ragwrench.sync.core import load_json
    from ragwrench.sync.embedders import PrecomputedEmbedder
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    for operation in (load_json, snapshot_jsonl, PrecomputedEmbedder):
        with pytest.raises(SyncError, match="regular file"):
            operation(fifo)
