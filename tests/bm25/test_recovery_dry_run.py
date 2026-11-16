"""Recovery inspection shares replay validation and never crosses the write boundary."""

from contextlib import closing
from copy import deepcopy
import hashlib
import json
import os

import pytest
from qdrant_client import models

from ragwrench.bm25 import store
from ragwrench.bm25.cli import main


def crash(*args, **kwargs):
    raise RuntimeError("injected interruption")


def files(path):
    return {
        str(p.relative_to(path)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in path.rglob("*")
        if p.is_file()
    }


def no_writes(client, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("inspection/invalid recovery must not write or fsync")

    for name in ("create_collection", "delete_collection", "delete", "upsert", "update_collection"):
        monkeypatch.setattr(client, name, unexpected)
    monkeypatch.setattr(os, "fsync", unexpected)


def pending_build(client, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(client, "upsert", crash)
        with pytest.raises(RuntimeError, match="interruption"):
            store.build_collection(client, "kb", [{"id": 7, "text": "private document 数据库"}])


@pytest.mark.parametrize(
    ("scenario", "current", "remove", "rewrite"),
    [
        ("build-partial", 63, 0, 130),
        ("replace-before-delete", 3, 1, 131),
        ("replace-partial", 63, 0, 131),
        ("delete-all-before-delete", 3, 3, 0),
        ("delete-all-after-delete", 0, 0, 0),
        ("ready", 3, 0, 0),
    ],
)
def test_counts_match_actual_replay(tmp_path, monkeypatch, scenario, current, remove, rewrite):
    path = tmp_path / "qd"
    docs = [{"id": i, "text": "数据库 备份 " + "alpha " * (i % 4)} for i in range(130)]
    with closing(store.open_local(path)) as client:
        if scenario != "build-partial":
            store.build_collection(
                client,
                "kb",
                [
                    {"id": 7, "text": "old replacement"},
                    {"id": "obsolete", "text": "old"},
                    {"id": "7", "text": "retained 数据库"},
                ],
            )
        with monkeypatch.context() as m:
            upsert = client.upsert

            def partial(*args, **kwargs):
                upsert(*args, **{**kwargs, "points": kwargs["points"][:63]})
                crash()

            m.setattr(client, "upsert", partial)
            if scenario in {"replace-before-delete", "delete-all-before-delete"}:
                m.setattr(client, "delete", crash)
            elif scenario == "delete-all-after-delete":
                update = client.update_collection

                def before_ready(*args, **kwargs):
                    if kwargs["metadata"]["qbm25"]["state"] == "ready":
                        crash()
                    return update(*args, **kwargs)

                m.setattr(client, "update_collection", before_ready)
            if scenario != "ready":
                with pytest.raises(RuntimeError, match="interruption"):
                    if scenario == "build-partial":
                        store.build_collection(client, "kb", docs)
                    elif scenario.startswith("replace"):
                        store.update_collection(
                            client, "kb", additions=docs, delete_ids=[7, "obsolete"]
                        )
                    else:
                        store.update_collection(client, "kb", delete_ids=[7, "obsolete", "7"])
    before = files(path)
    with closing(store.open_local(path)) as client:
        manifest = deepcopy(client.get_collection("kb").config.metadata["qbm25"])
        target = manifest.get("recovery", {}).get("manifest", manifest)
        with monkeypatch.context() as m:
            no_writes(client, m)
            report = store.recover_collection(client, "kb", dry_run=True)
            assert store.recover_collection(client, "kb", dry_run=True) == report
        assert report == {
            "schema": 1,
            "collection": "kb",
            "dry_run": True,
            "state": manifest["state"],
            "pending": scenario != "ready",
            "config": target["encoder"],
            "corpus": {
                "document_count": target["indexed"],
                "avgdl_mode": target["corpus"]["avgdl_mode"],
                "total_tokens": target["corpus"]["total_tokens"],
            },
            "points": {"current": current, "remove": remove, "rewrite": rewrite},
        }
        assert "private" not in json.dumps(report) and "documents" not in report
        assert client.get_collection("kb").config.metadata["qbm25"] == manifest
    assert files(path) == before
    # Count actual public point operations, including full replacement/retained rewrites.
    with closing(store.open_local(path)) as client:
        deleted, rewritten = [], []
        delete, upsert = client.delete, client.upsert

        def count_delete(*args, **kwargs):
            deleted.extend(kwargs["points_selector"].points)
            return delete(*args, **kwargs)

        def count_upsert(*args, **kwargs):
            rewritten.extend(p.id for p in kwargs["points"])
            return upsert(*args, **kwargs)

        monkeypatch.setattr(client, "delete", count_delete)
        monkeypatch.setattr(client, "upsert", count_upsert)
        actual = store.recover_collection(client, "kb")
        assert (len(deleted), len(rewritten)) == (remove, rewrite)
        assert actual["indexed"] == target["indexed"]
        assert store.recover_collection(client, "kb", dry_run=True)["points"] == {
            "current": target["indexed"],
            "remove": 0,
            "rewrite": 0,
        }
        hits = store.search_collection(client, "kb", "数据库", top_k=200)
        expected_hits = (
            0
            if scenario.startswith("delete-all")
            else (1 if scenario == "ready" else target["indexed"])
        )
        assert len(hits) == expected_hits


@pytest.mark.parametrize(
    "damage",
    ["missing", "checksum", "schema", "runtime", "stats", "collision", "duplicate", "collection"],
)
def test_invalid_snapshot_same_error_and_no_json(tmp_path, monkeypatch, capsys, damage):
    path = tmp_path / "qd"
    with closing(store.open_local(path)) as client:
        pending_build(client, monkeypatch)
        data = deepcopy(client.get_collection("kb").config.metadata["qbm25"])
        snapshot = data["recovery"]
        if damage == "missing":
            data.pop("recovery")
        elif damage == "checksum":
            snapshot["documents"][0]["text"] = "changed"
        elif damage == "schema":
            snapshot["schema"] = 99
        elif damage == "runtime":
            snapshot["manifest"]["encoder"]["unicode_version"] = "incompatible"
        elif damage == "stats":
            snapshot["manifest"]["corpus"]["total_tokens"] += 1
        elif damage == "collision":
            snapshot["manifest"]["collision_stats"]["collision_count"] = 99
        elif damage == "duplicate":
            snapshot["documents"] *= 2
        else:
            snapshot["collection"] = "another"
        if damage not in {"missing", "checksum"}:
            snapshot["sha256"] = store._snapshot_digest(snapshot)
            data.update({k: v for k, v in snapshot["manifest"].items() if k != "state"})
        client.update_collection("kb", metadata={"qbm25": data})
        with monkeypatch.context() as m:
            no_writes(client, m)
            with pytest.raises(ValueError) as actual:
                store.recover_collection(client, "kb")
            with pytest.raises(ValueError) as inspected:
                store.recover_collection(client, "kb", dry_run=True)
            assert str(actual.value) == str(inspected.value)
    before = files(path)
    args = ["recover", "--path", str(path), "--collection", "kb"]
    for flags in ([], ["--dry-run"]):
        assert main(args + flags) == 2
        output = capsys.readouterr()
        assert output.out == "" and "error" in output.err
    assert files(path) == before


def test_cli_pending_ready_and_missing(tmp_path, monkeypatch, capsys):
    path = tmp_path / "qd"
    with closing(store.open_local(path)) as client:
        pending_build(client, monkeypatch)
    args = ["recover", "--path", str(path), "--collection", "kb"]
    before = files(path)
    with monkeypatch.context() as m:

        def unexpected(*args, **kwargs):
            pytest.fail("CLI dry-run must not mutate")

        for name in (
            "create_collection",
            "delete_collection",
            "delete",
            "upsert",
            "update_collection",
        ):
            m.setattr(store.QdrantClient, name, unexpected)
        m.setattr(os, "fsync", unexpected)
        assert main(args + ["--dry-run"]) == 0
    pending = json.loads(capsys.readouterr().out)
    assert pending["state"] == "building" and pending["pending"]
    assert pending["points"] == {"current": 0, "remove": 0, "rewrite": 1}
    assert "private document" not in json.dumps(pending)
    assert files(path) == before
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["recovered"] is True
    before = files(path)
    assert main(args + ["--dry-run"]) == 0
    ready = json.loads(capsys.readouterr().out)
    assert ready["state"] == "ready" and ready["pending"] is False
    assert ready["config"] == pending["config"] and ready["corpus"] == pending["corpus"]
    assert ready["points"] == {"current": 1, "remove": 0, "rewrite": 0}
    assert files(path) == before
    for collection in ("missing", "../escape"):
        assert main(["recover", "--path", str(path), "--collection", collection, "--dry-run"]) == 2
        assert capsys.readouterr().out == ""
    missing = tmp_path / "never-created"
    assert main(["recover", "--path", str(missing), "--collection", "kb", "--dry-run"]) == 2
    assert capsys.readouterr().out == "" and not missing.exists()


def test_ready_normal_checks_and_legacy_stats(client, monkeypatch):
    store.build_collection(client, "kb", [{"id": 7, "text": "数据库"}])
    ready = deepcopy(client.get_collection("kb").config.metadata["qbm25"])
    ready.pop("corpus")  # Pre-update ready collections are still searchable/no-op recoverable.
    client.update_collection("kb", metadata={"qbm25": ready})
    client.create_collection("foreign", vectors_config={})
    client.create_collection(
        "unmanaged",
        vectors_config={},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    with monkeypatch.context() as m:
        no_writes(client, m)
        report = store.recover_collection(client, "kb", dry_run=True)
        assert report["corpus"] == {"document_count": 1, "avgdl_mode": None, "total_tokens": None}
        for collection in ("foreign", "unmanaged"):
            with pytest.raises(ValueError):
                store.recover_collection(client, collection, dry_run=True)
    client.delete("kb", points_selector=models.PointIdsList(points=[store.point_id(7)]))
    with monkeypatch.context() as m:
        no_writes(client, m)
        with pytest.raises(ValueError, match="point count"):
            store.recover_collection(client, "kb", dry_run=True)


@pytest.mark.parametrize(
    "damage",
    [
        "corpus-string",
        "corpus-null",
        "corpus-list",
        "corpus-empty-list",
        "corpus-empty-object",
        "missing-schema",
        "schema-bool",
        "schema-float",
        "schema-version",
        "missing-mode",
        "mode-unknown",
        "mode-list",
        "missing-total",
        "total-negative",
        "total-bool",
        "total-float",
        "total-string",
        "total-zero",
        "total-overflow",
        "auto-avgdl",
        "indexed-bool",
        "indexed-float",
    ],
)
def test_invalid_ready_statistics_fail_without_json_or_writes(
    tmp_path, monkeypatch, capsys, damage
):
    path = tmp_path / "qd"
    with closing(store.open_local(path)) as client:
        store.build_collection(client, "kb", [{"id": 7, "text": "private 数据库 备份"}])
        data = deepcopy(client.get_collection("kb").config.metadata["qbm25"])
        corpus = data["corpus"]
        if damage.startswith("corpus-"):
            data["corpus"] = {
                "corpus-string": "private malformed text",
                "corpus-null": None,
                "corpus-list": ["private malformed text"],
                "corpus-empty-list": [],
                "corpus-empty-object": {},
            }[damage]
        elif damage.startswith("missing-"):
            corpus.pop(
                {"schema": "schema", "mode": "avgdl_mode", "total": "total_tokens"}[
                    damage.removeprefix("missing-")
                ]
            )
        elif damage.startswith("schema-"):
            corpus["schema"] = {"schema-bool": True, "schema-float": 1.0, "schema-version": 99}[
                damage
            ]
        elif damage.startswith("mode-"):
            corpus["avgdl_mode"] = "unknown" if damage == "mode-unknown" else []
        elif damage.startswith("total-"):
            corpus["total_tokens"] = {
                "total-negative": -1,
                "total-bool": True,
                "total-float": float(corpus["total_tokens"]),
                "total-string": "private malformed text",
                "total-zero": 0,
                "total-overflow": 10**400,
            }[damage]
        elif damage == "auto-avgdl":
            corpus["total_tokens"] += 1
        else:
            data["indexed"] = True if damage == "indexed-bool" else 1.0
        client.update_collection("kb", metadata={"qbm25": data})
    before = files(path)
    with closing(store.open_local(path)) as client:
        with monkeypatch.context() as m:
            no_writes(client, m)
            for dry_run in (True, False):
                with pytest.raises(ValueError, match="ready.*statistics"):
                    store.recover_collection(client, "kb", dry_run=dry_run)
    for flags in (["--dry-run"], []):
        assert main(["recover", "--path", str(path), "--collection", "kb"] + flags) == 2
        output = capsys.readouterr()
        assert output.out == "" and "ready" in output.err and "statistics" in output.err
        assert "private" not in output.err
    assert files(path) == before


@pytest.mark.parametrize("scenario", ["auto", "fixed", "empty-auto", "empty-fixed", "legacy"])
def test_ready_statistics_valid_without_scanning(tmp_path, monkeypatch, capsys, scenario):
    path = tmp_path / "qd"
    with closing(store.open_local(path)) as client:
        options = {"avgdl": 19.0} if "fixed" in scenario else {}
        store.build_collection(client, "kb", [{"id": 7, "text": "数据库 备份"}], **options)
        if scenario.startswith("empty"):
            store.update_collection(client, "kb", delete_ids=[7])
        if scenario == "legacy":
            data = deepcopy(client.get_collection("kb").config.metadata["qbm25"])
            data.pop("corpus")
            client.update_collection("kb", metadata={"qbm25": data})
        hits = store.search_collection(client, "kb", "数据库")
    before = files(path)
    with closing(store.open_local(path)) as client:
        with monkeypatch.context() as m:
            no_writes(client, m)

            def unexpected(*args, **kwargs):
                pytest.fail("ready statistics validation must not scan or encode the corpus")

            m.setattr(client, "scroll", unexpected)
            m.setattr(store, "_encode_points", unexpected)
            m.setattr(store.Encoder, "encode_documents", unexpected)
            report = store.recover_collection(client, "kb", dry_run=True)
            assert store.recover_collection(client, "kb")["recovered"] is False
        assert store.search_collection(client, "kb", "数据库") == hits
    assert main(["recover", "--path", str(path), "--collection", "kb", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out) == report
    assert report["points"] == {
        "current": 0 if scenario.startswith("empty") else 1,
        "remove": 0,
        "rewrite": 0,
    }
    assert report["corpus"]["avgdl_mode"] == (
        None if scenario == "legacy" else ("fixed" if "fixed" in scenario else "auto")
    )
    assert files(path) == before


@pytest.mark.parametrize("scenario", ["nonempty-too-few-tokens", "empty-auto", "empty-fixed"])
def test_invalid_ready_count_token_relationship(client, monkeypatch, scenario):
    options = {"avgdl": 19.0} if scenario == "empty-fixed" else {}
    store.build_collection(
        client, "kb", [{"id": 7, "text": "数据库"}, {"id": "7", "text": "数据库"}], **options
    )
    if scenario.startswith("empty"):
        store.update_collection(client, "kb", delete_ids=[7, "7"])
    data = deepcopy(client.get_collection("kb").config.metadata["qbm25"])
    data["corpus"]["total_tokens"] = 1
    # For nonempty, the average still agrees: each indexed document must have >=1 token.
    if not scenario.startswith("empty"):
        data["encoder"]["avgdl"] = 0.5
    client.update_collection("kb", metadata={"qbm25": data})
    no_writes(client, monkeypatch)
    for dry_run in (True, False):
        with pytest.raises(ValueError, match="ready.*statistics"):
            store.recover_collection(client, "kb", dry_run=dry_run)
