"""An intended corpus survives interrupted local point writes without input files."""

from contextlib import closing
from copy import deepcopy
import json
import os
import subprocess
import sys

import pytest

from ragwrench.bm25 import store
from ragwrench.bm25.cli import main


def manifest(client):
    return deepcopy(client.get_collection("kb").config.metadata["qbm25"])


def fail(*args, **kwargs):
    raise RuntimeError("injected crash")


def test_intent_is_persisted_before_first_point(tmp_path, monkeypatch):
    path = tmp_path / "qd"
    documents = [{"id": 7, "text": "数据库 备份", "metadata": {"typed": "int"}}]
    with closing(store.open_local(path)) as client:
        monkeypatch.setattr(client, "upsert", fail)
        with pytest.raises(RuntimeError, match="injected"):
            store.build_collection(client, "kb", documents)
        assert manifest(client)["recovery"]["documents"] == documents
    with closing(store.open_local(path)) as client:
        report = store.recover_collection(client, "kb")
        assert report["state"] == "ready" and report["recovered"] is True
        assert report["indexed"] == 1
        assert "recovery" not in manifest(client)
        assert store.search_collection(client, "kb", "备份")[0]["metadata"] == {"typed": "int"}
        assert store.recover_collection(client, "kb") == {**report, "recovered": False}


# Exit without finally/close: Qdrant must reopen persisted state in a new process.
CRASH_SCRIPT = r"""
import json, os, sys
from ragwrench.bm25 import store
path, operation, stage, source = sys.argv[1:]
client = store.open_local(path)
documents = json.load(open(source, encoding="utf-8"))
upsert, metadata, delete = client.upsert, client.update_collection, client.delete
batches = 0
def crash():
    os._exit(77)
def wrapped_upsert(*args, **kwargs):
    global batches
    batches += 1
    if stage == "before-points" and batches == 1:
        crash()
    if stage == "partial-batch" and batches == 1:
        upsert(*args, **{**kwargs, "points": kwargs["points"][:63]})
        crash()
    result = upsert(*args, **kwargs)
    if stage == f"batch-{batches}":
        crash()
    return result
def wrapped_metadata(*args, **kwargs):
    ready = kwargs.get("metadata", {}).get("qbm25", {}).get("state") == "ready"
    if ready and stage == "before-ready":
        crash()
    result = metadata(*args, **kwargs)
    if ready and stage == "after-ready":
        crash()
    return result
def wrapped_delete(*args, **kwargs):
    if stage == "before-delete":
        crash()
    result = delete(*args, **kwargs)
    if stage == "after-delete":
        crash()
    return result
client.upsert, client.update_collection, client.delete = wrapped_upsert, wrapped_metadata, wrapped_delete
if operation == "build":
    store.build_collection(client, "kb", documents, stopwords=("the",), k1=1.7, b=0.6)
elif operation == "update":
    store.update_collection(client, "kb", additions=documents, delete_ids=[7, "old"])
else:
    store.recover_collection(client, "kb")
raise RuntimeError("crash boundary was not reached")
"""


def crash_process(path, operation, stage, documents, tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(documents, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-c", CRASH_SCRIPT, str(path), operation, stage, str(source)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "20"},
    )
    source.unlink()
    assert result.returncode == 77, result.stderr


@pytest.mark.parametrize(
    ("operation", "stage"),
    [
        (operation, stage)
        for operation in ("build", "update")
        for stage in (
            "before-points",
            "partial-batch",
            "batch-1",
            "batch-2",
            "before-ready",
            "after-ready",
        )
    ]
    + [("update", "before-delete")],
)
def test_process_crash_recover_matches_full_rebuild(tmp_path, operation, stage):
    path = tmp_path / "qd"
    options = {"stopwords": ("the",), "k1": 1.7, "b": 0.6}
    documents = [
        {"id": i, "text": "数据库 备份 " + "alpha " * (i % 4), "metadata": {"row": i}}
        for i in range(130)
    ]
    retained = {"id": "7", "text": "서울 数据库 the", "metadata": {"typed": "string"}}
    if operation == "update":
        with closing(store.open_local(path)) as client:
            store.build_collection(
                client,
                "kb",
                [
                    {"id": 7, "text": "old"},
                    {"id": "old", "text": "obsolete"},
                    retained,
                ],
                **options,
            )
    crash_process(path, operation, stage, documents, tmp_path)
    expected = documents + ([retained] if operation == "update" else [])
    with closing(store.open_local(path)) as client:
        if stage != "after-ready":
            with pytest.raises(ValueError, match="incomplete"):
                store.search_collection(client, "kb", "数据库")
        report = store.recover_collection(client, "kb")
        assert report["recovered"] is (stage != "after-ready")
        rebuilt = store.build_collection(client, "rebuilt", expected, **options)
        assert manifest(client) == {k: v for k, v in rebuilt.items() if k != "created"}
        for query in ("数据库 备份", "alpha", "서울", "old obsolete", "missing"):
            actual = store.search_collection(client, "kb", query, top_k=200)
            reference = store.search_collection(client, "rebuilt", query, top_k=200)
            assert actual == reference
        # Success has no pending snapshot, and later updates work normally.
        assert store.recover_collection(client, "kb")["recovered"] is False
        store.update_collection(client, "kb", additions=[{"id": "later", "text": "快照"}])


def test_recovery_can_itself_crash_after_delete_and_batch(tmp_path):
    path = tmp_path / "qd"
    documents = [{"id": i, "text": "数据库 备份"} for i in range(130)]
    with closing(store.open_local(path)) as client:
        store.build_collection(
            client, "kb", [{"id": 7, "text": "old"}, {"id": "old", "text": "old"}]
        )
    crash_process(path, "update", "before-delete", documents, tmp_path)
    crash_process(path, "recover", "after-delete", [], tmp_path)
    crash_process(path, "recover", "batch-1", [], tmp_path)
    crash_process(path, "recover", "before-ready", [], tmp_path)
    with closing(store.open_local(path)) as client:
        assert store.recover_collection(client, "kb")["indexed"] == 130
        assert len(store.search_collection(client, "kb", "数据库", top_k=200)) == 130
        assert store.search_collection(client, "kb", "old") == []


@pytest.mark.parametrize("avgdl", [None, 9])
def test_delete_all_and_skipped_replacement_recovery(tmp_path, monkeypatch, avgdl):
    path = tmp_path / "qd"
    with closing(store.open_local(path)) as client:
        store.build_collection(client, "kb", [{"id": 7, "text": "alpha"}], avgdl=avgdl)
        original = client.update_collection

        def fail_ready(*args, **kwargs):
            if kwargs["metadata"]["qbm25"]["state"] == "ready":
                fail()
            return original(*args, **kwargs)

        monkeypatch.setattr(client, "update_collection", fail_ready)
        with pytest.warns(UserWarning, match="empty"), pytest.raises(RuntimeError):
            store.update_collection(client, "kb", delete_ids=[7], additions=[{"id": 7, "text": ""}])
    with closing(store.open_local(path)) as client:
        report = store.recover_collection(client, "kb")
        assert report["indexed"] == 0 and report["skipped"] == 1
        assert report["encoder"]["avgdl"] == (1 if avgdl is None else 9)
        assert store.search_collection(client, "kb", "alpha") == []


@pytest.mark.parametrize(
    "damage", ["missing", "checksum", "schema", "config", "stats", "duplicate", "collection"]
)
def test_bad_snapshot_fails_before_writes(client, monkeypatch, damage):
    with monkeypatch.context() as m:
        m.setattr(client, "upsert", fail)
        with pytest.raises(RuntimeError):
            store.build_collection(client, "kb", [{"id": 7, "text": "数据库"}])
    data = manifest(client)
    snapshot = data.get("recovery", {})
    if damage == "missing":
        data.pop("recovery", None)
    elif damage == "checksum":
        snapshot["documents"][0]["text"] = "changed"
    elif damage == "schema":
        snapshot["schema"] = 99
    elif damage == "config":
        snapshot["manifest"]["encoder"]["unicode_version"] = "incompatible"
    elif damage == "stats":
        snapshot["manifest"]["indexed"] = 99
    elif damage == "duplicate":
        snapshot["documents"] *= 2
    else:
        snapshot["collection"] = "another"
    if damage not in {"missing", "checksum"}:
        # A consistent checksum cannot make invalid content/schema/config usable.
        snapshot["sha256"] = store._snapshot_digest(snapshot)
        if damage in {"config", "stats"}:
            data.update({k: v for k, v in snapshot["manifest"].items() if k != "state"})
    client.update_collection("kb", metadata={"qbm25": data})

    def unexpected(*args, **kwargs):
        pytest.fail("invalid recovery must not write")

    for name in ("upsert", "delete", "update_collection", "create_collection"):
        monkeypatch.setattr(client, name, unexpected)
    with pytest.raises(ValueError):
        store.recover_collection(client, "kb")
    assert manifest(client) == data


def test_cli_recovery_and_negative_contract(tmp_path, monkeypatch, capsys):
    path = tmp_path / "qd"
    with closing(store.open_local(path)) as client:
        monkeypatch.setattr(client, "upsert", fail)
        with pytest.raises(RuntimeError):
            store.build_collection(client, "kb", [{"id": 7, "text": "数据库"}])
    args = ["recover", "--path", str(path), "--collection", "kb"]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["recovered"] is True
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["recovered"] is False
    for name in ("missing", "../escape"):
        assert main(["recover", "--path", str(path), "--collection", name]) == 2
        output = capsys.readouterr()
        assert output.out == "" and "error" in output.err
    missing = tmp_path / "never-created"
    assert main(["recover", "--path", str(missing), "--collection", "kb"]) == 2
    assert capsys.readouterr().out == "" and not missing.exists()


@pytest.mark.parametrize("boundary", ["snapshot-limit", "metadata-sync"])
def test_checkpoint_failure_does_not_write_points(tmp_path, monkeypatch, boundary):
    with closing(store.open_local(tmp_path / "qd")) as client:
        if boundary == "snapshot-limit":
            monkeypatch.setattr(store, "MAX_SNAPSHOT_BYTES", 100)
        else:
            monkeypatch.setattr(store, "_sync_metadata", fail)

        def unexpected(*args, **kwargs):
            pytest.fail("checkpoint failure must precede point writes")

        monkeypatch.setattr(client, "upsert", unexpected)
        monkeypatch.setattr(client, "delete", unexpected)
        with pytest.raises((ValueError, RuntimeError)):
            store.build_collection(client, "kb", [{"id": 7, "text": "数据库"}])
        if boundary == "snapshot-limit":
            assert not client.collection_exists("kb")
        else:
            assert manifest(client)["state"] == "building"
            assert manifest(client)["recovery"]["documents"]


def test_fixed_average_and_collision_recovery(client, monkeypatch):
    docs = [{"id": 7, "text": "collision402"}, {"id": "7", "text": "collision169677"}]
    with monkeypatch.context() as m:
        m.setattr(client, "upsert", fail)
        with pytest.warns(UserWarning, match="collision"), pytest.raises(RuntimeError):
            store.build_collection(client, "kb", docs, tokenizer="whitespace", avgdl=9)
    with pytest.warns(UserWarning, match="collision"):
        report = store.recover_collection(client, "kb")
    assert report["encoder"]["avgdl"] == 9
    assert report["corpus"]["avgdl_mode"] == "fixed"
    assert report["collision_stats"]["collision_count"] == 1
    assert {type(h["id"]) for h in store.search_collection(client, "kb", "collision402")} == {
        str,
        int,
    }


def test_ready_recovery_is_read_only_and_foreign_rejected(client, monkeypatch):
    from qdrant_client import models

    store.build_collection(client, "kb", [{"id": 7, "text": "数据库"}])
    client.create_collection("foreign", vectors_config={})
    client.create_collection(
        "unmanaged",
        vectors_config={},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )

    def unexpected(*args, **kwargs):
        pytest.fail("ready or unmanaged recovery must not write")

    for method in ("update_collection", "upsert", "delete", "create_collection"):
        monkeypatch.setattr(client, method, unexpected)
    assert store.recover_collection(client, "kb")["recovered"] is False
    for name in ("foreign", "unmanaged"):
        with pytest.raises(ValueError):
            store.recover_collection(client, name)
