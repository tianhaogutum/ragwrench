"""Real local Qdrant, with deterministic failures around its write calls."""
from collections import Counter
from copy import deepcopy

import pytest
from qdrant_client import models

from ragwrench.sync.core import SyncError, build_snapshot, make_plan
from ragwrench.sync.embedders import HashEmbedder
from ragwrench.sync.store import apply_plan, local_client, stored_payload, verify_snapshot
from .conftest import linux_durable

pytestmark = linux_durable


def snapshots():
    old = build_snapshot([{"doc_id": "a", "text": "old1"}, {"doc_id": "a", "text": "old2"}])
    new = build_snapshot([{"doc_id": "a", "text": "new1"}, {"doc_id": "a", "text": "new2"}])
    return old, new, make_plan(old, new)


@pytest.mark.parametrize("failure", ["upsert_before", "upsert_after", "delete_before", "delete_after",
                                     "checkpoint_before", "checkpoint_after", "commit_before"])
def test_resume_journal_never_repeats_persisted_mutations(tmp_path, monkeypatch, failure):
    old, new, plan = snapshots()
    db = str(tmp_path / "db")
    successful = Counter()

    def instrument(client, fail):
        for name in ("upsert", "delete"):
            original = getattr(client, name)

            def write(*args, _name=name, _original=original, **kwargs):
                ids = ([str(p.id) for p in kwargs["points"]] if _name == "upsert"
                       else [str(pid) for pid in kwargs["points_selector"].points])
                # Fail on the second upsert / first delete, before or after
                # the actual pinned Qdrant API persists its changes.
                target = plan["upsert"][1]["id"] if _name == "upsert" else plan["delete"][0]["id"]
                hit = fail and target in ids and failure.startswith(_name)
                if hit and failure.endswith("before"):
                    raise RuntimeError("injected vector-store failure")
                result = _original(*args, **kwargs)
                successful.update((_name, pid) for pid in ids)
                if hit:
                    raise RuntimeError("injected vector-store failure")
                return result

            monkeypatch.setattr(client, name, write)
        if fail and failure in ("checkpoint_before", "checkpoint_after", "commit_before"):
            original = client.update_collection

            def update(*args, **kwargs):
                state = kwargs.get("metadata", {}).get("chunksync", {})
                if state.get("fingerprint") == new["fingerprint"] and state.get("pending") is None:
                    if failure == "commit_before":
                        raise RuntimeError("injected vector-store failure")
                journal = state.get("pending")
                hit = (failure.startswith("checkpoint") and isinstance(journal, dict)
                       and journal["next_operation"] == 1 and journal["in_flight"] is None)
                if hit and failure.endswith("before"):
                    raise RuntimeError("injected vector-store failure")
                result = original(*args, **kwargs)
                if hit:
                    raise RuntimeError("injected vector-store failure")
                return result

            monkeypatch.setattr(client, "update_collection", update)

    with local_client(db) as client:
        apply_plan(client, "kb", make_plan(build_snapshot([]), old), HashEmbedder())
        instrument(client, True)
        with pytest.raises(RuntimeError, match="injected"):
            apply_plan(client, "kb", plan, HashEmbedder())
    with local_client(db) as client:
        instrument(client, False)
        apply_plan(client, "kb", plan, HashEmbedder())
        assert verify_snapshot(client, "kb", new, HashEmbedder())["points"] == 2
        assert apply_plan(client, "kb", plan, HashEmbedder())["status"] == "already_applied"
    expected = Counter({("upsert", c["id"]): 1 for c in plan["upsert"]})
    expected.update({("delete", c["id"]): 1 for c in plan["delete"]})
    assert successful == expected


@pytest.mark.parametrize("field", ["plan_fingerprint", "old_fingerprint", "new_fingerprint", "source_data_fingerprint"])
def test_journal_binding_refused_before_embedding_or_mutation(tmp_path, monkeypatch, field):
    old, new, plan = snapshots()
    db = str(tmp_path / "db")
    with local_client(db) as client:
        apply_plan(client, "kb", make_plan(build_snapshot([]), old), HashEmbedder())
        monkeypatch.setattr(client, "upsert", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fail")))
        with pytest.raises(RuntimeError):
            apply_plan(client, "kb", plan, HashEmbedder())
        state = deepcopy(client.get_collection("kb").config.metadata["chunksync"])
        assert isinstance(state["pending"], dict), "pending plan has no apply journal"
        state["pending"][field] = "0" * 64
        client.update_collection("kb", metadata={"chunksync": state})
    with local_client(db) as client:
        embedder = HashEmbedder()
        embedder.embed = lambda *a: pytest.fail("binding mismatch must precede embedding")
        for name in ("upsert", "delete", "overwrite_payload", "update_collection"):
            monkeypatch.setattr(client, name, lambda *a, **kw: pytest.fail("binding mismatch must precede mutation"))
        with pytest.raises(SyncError, match="journal.*binding"):
            apply_plan(client, "kb", plan, embedder)


def test_pending_journal_detects_vector_change(tmp_path, monkeypatch):
    old, new, plan = snapshots()
    db = str(tmp_path / "db")
    with local_client(db) as client:
        apply_plan(client, "kb", make_plan(build_snapshot([]), old), HashEmbedder())
        original = client.upsert

        def fail_after(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected")

        monkeypatch.setattr(client, "upsert", fail_after)
        with pytest.raises(RuntimeError):
            apply_plan(client, "kb", plan, HashEmbedder())
        pid = plan["upsert"][0]["id"]
        point = client.retrieve("kb", [pid])[0]
        original("kb", points=[models.PointStruct(id=pid, vector=[0.] * 32, payload=point.payload)])
    with local_client(db) as client:
        with pytest.raises(SyncError, match="journal.*fingerprint"):
            apply_plan(client, "kb", plan, HashEmbedder())


@pytest.mark.parametrize("kind", ["payload_only", "moved"])
def test_resume_payload_and_move_without_repeating_writes(tmp_path, monkeypatch, kind):
    old = build_snapshot([{"doc_id": "a", "text": "same", "metadata": {"v": 1}}])
    new = build_snapshot([{"doc_id": "a" if kind == "payload_only" else "b", "text": "same",
                           "metadata": {"v": 2}}])
    plan = make_plan(old, new)
    method = "overwrite_payload" if kind == "payload_only" else "upsert"
    db = str(tmp_path / "db")
    with local_client(db) as client:
        apply_plan(client, "kb", make_plan(build_snapshot([]), old), HashEmbedder())
        original = getattr(client, method)

        def fail_after(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected")

        monkeypatch.setattr(client, method, fail_after)
        with pytest.raises(RuntimeError):
            apply_plan(client, "kb", plan, HashEmbedder())
    with local_client(db) as client:
        monkeypatch.setattr(client, method, lambda *a, **kw: pytest.fail("repeated successful write"))
        embedder = HashEmbedder()
        assert apply_plan(client, "kb", plan, embedder)["embedded"] == 0
        assert embedder.calls == 0
        verify_snapshot(client, "kb", new, HashEmbedder())


def test_legacy_pending_marker_upgrades_without_repeating_writes(tmp_path, monkeypatch):
    old, new, plan = snapshots()
    db = str(tmp_path / "db")
    with local_client(db) as client:
        apply_plan(client, "kb", make_plan(build_snapshot([]), old), HashEmbedder())
        # Exact legacy state format, with all new writes and deletes already
        # persisted but its final collection-state commit still outstanding.
        state = deepcopy(client.get_collection("kb").config.metadata["chunksync"])
        state["pending"] = plan["fingerprint"]
        for c in new["chunks"]:
            client.upsert("kb", points=[models.PointStruct(id=c["id"], payload=stored_payload(c),
                                                           vector=HashEmbedder().embed(c, new["embed_model"]))])
        client.delete("kb", points_selector=models.PointIdsList(points=[c["id"] for c in old["chunks"]]))
        client.update_collection("kb", metadata={"chunksync": state})
    with local_client(db) as client:
        for name in ("upsert", "delete"):
            monkeypatch.setattr(client, name, lambda *a, **kw: pytest.fail("repeated legacy mutation"))
        embedder = HashEmbedder()
        assert apply_plan(client, "kb", plan, embedder)["embedded"] == 0
        verify_snapshot(client, "kb", new, HashEmbedder())


@pytest.mark.parametrize("target", ["source", "completed"])
def test_pending_journal_protects_points_outside_inflight_operation(tmp_path, monkeypatch, target):
    old, new, plan = snapshots()
    db = str(tmp_path / "db")
    with local_client(db) as client:
        apply_plan(client, "kb", make_plan(build_snapshot([]), old), HashEmbedder())
        original = client.upsert

        def fail_second(*args, **kwargs):
            if kwargs["points"][0].id == plan["upsert"][1]["id"]:
                raise RuntimeError("injected")
            return original(*args, **kwargs)

        monkeypatch.setattr(client, "upsert", fail_second)
        with pytest.raises(RuntimeError):
            apply_plan(client, "kb", plan, HashEmbedder())
        pid = (old if target == "source" else new)["chunks"][0]["id"]
        point = client.retrieve("kb", [pid])[0]
        original("kb", points=[models.PointStruct(id=pid, vector=[0.] * 32, payload=point.payload)])
    with local_client(db) as client:
        monkeypatch.setattr(client, "upsert", lambda *a, **kw: pytest.fail("mutation after fingerprint mismatch"))
        with pytest.raises(SyncError, match="journal.*fingerprint"):
            apply_plan(client, "kb", plan, HashEmbedder())


def test_retry_refuses_changed_embedding_for_unacknowledged_intent(tmp_path, monkeypatch):
    old, new, plan = snapshots()
    db = str(tmp_path / "db")
    with local_client(db) as client:
        apply_plan(client, "kb", make_plan(build_snapshot([]), old), HashEmbedder())

        def fail_before(*args, **kwargs):
            raise RuntimeError("injected")

        monkeypatch.setattr(client, "upsert", fail_before)
        with pytest.raises(RuntimeError):
            apply_plan(client, "kb", plan, HashEmbedder())
    with local_client(db) as client:
        embedder = HashEmbedder()
        # Same declared identity, different output (e.g. edited command code).
        embedder.embed = lambda *a: [0.] * 32
        monkeypatch.setattr(client, "upsert", lambda *a, **kw: pytest.fail("wrote changed embedding source"))
        with pytest.raises(SyncError, match="intended fingerprint mismatch"):
            apply_plan(client, "kb", plan, embedder)
