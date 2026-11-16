"""Recovery of real Qdrant metadata torn during journal persistence."""
from collections import Counter
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import stat

import pytest
from qdrant_client.local import qdrant_local

from ragwrench.sync.core import build_snapshot, make_plan
from ragwrench.sync.embedders import HashEmbedder
from ragwrench.sync.store import apply_plan, local_client, verify_snapshot
from ragwrench.sync.recovery import SIDECAR


def snapshots():
    old = build_snapshot([{"doc_id": "a", "text": t} for t in ("old1", "old2")])
    new = build_snapshot([{"doc_id": "a", "text": t} for t in ("new1", "new2")])
    return old, new, make_plan(old, new)


def record_mutations(client, monkeypatch, successful):
    for name in ("upsert", "delete"):
        original = getattr(client, name)

        def write(*args, _name=name, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            ids = ([str(p.id) for p in kwargs["points"]] if _name == "upsert"
                   else [str(pid) for pid in kwargs["points_selector"].points])
            successful.update((_name, pid) for pid in ids)
            return result

        monkeypatch.setattr(client, name, write)


@pytest.mark.parametrize("phase", ["upsert_checkpoint", "delete_checkpoint", "commit"])
def test_torn_meta_recovers_without_replaying_successful_point_writes(tmp_path, monkeypatch, phase):
    old, new, plan = snapshots()
    db = tmp_path / "db"
    successful = Counter()
    with local_client(str(db)) as client:
        apply_plan(client, "kb", make_plan(build_snapshot([]), old), HashEmbedder())
        # A second collection proves recovery retains the whole store catalog.
        apply_plan(client, "other", make_plan(build_snapshot([]), old), HashEmbedder())
        record_mutations(client, monkeypatch, successful)

        @contextmanager
        def torn_open(path, mode="r", *args, **kwargs):
            with open(path, mode, *args, **kwargs) as out:
                state = client.get_collection("kb").config.metadata["chunksync"]
                pending = state["pending"]
                hit = (pending is None and state["fingerprint"] == new["fingerprint"] if phase == "commit"
                       else isinstance(pending, dict) and pending["in_flight"] is None
                       and pending["next_operation"] == (2 if phase == "upsert_checkpoint" else 3))
                if Path(path) == db / "meta.json" and mode == "w" and hit:
                    out.write('{"collections":')
                    out.flush()
                    raise OSError("injected torn metadata checkpoint")
                yield out

        with monkeypatch.context() as patch:
            # Inject below the adapter, at the pinned backend's real file open.
            patch.setattr(qdrant_local, "open", torn_open, raising=False)
            with pytest.raises(OSError, match="torn metadata"):
                apply_plan(client, "kb", plan, HashEmbedder())
    assert (db / "meta.json").read_text() == '{"collections":'
    with local_client(str(db)) as client:
        record_mutations(client, monkeypatch, successful)
        embedder = HashEmbedder()
        resumed = apply_plan(client, "kb", plan, embedder)
        assert resumed["embedded"] == embedder.calls == 0
        assert resumed["points"] == 2
        assert verify_snapshot(client, "kb", new, HashEmbedder())["points"] == 2
        assert verify_snapshot(client, "other", old, HashEmbedder())["points"] == 2
        assert apply_plan(client, "kb", plan, HashEmbedder())["status"] == "already_applied"
    expected = Counter({("upsert", c["id"]): 1 for c in plan["upsert"]})
    expected.update({("delete", c["id"]): 1 for c in plan["delete"]})
    assert successful == expected


def bootstrap(db):
    old, new, plan = snapshots()
    with local_client(str(db)) as client:
        apply_plan(client, "kb", make_plan(build_snapshot([]), old), HashEmbedder())
    return old, new, plan


def test_sidecar_replacement_failure_preserves_previous_files_and_intent(tmp_path, monkeypatch):
    import os
    from ragwrench.sync.recovery import SIDECAR

    db = tmp_path / "db"
    old, new, plan = bootstrap(db)
    successful = Counter()
    with local_client(str(db)) as client:
        record_mutations(client, monkeypatch, successful)
        original_replace = os.replace
        previous = {}

        def fail_checkpoint(src, dst):
            state = client.get_collection("kb").config.metadata["chunksync"]
            pending = state["pending"]
            if (Path(dst).name == SIDECAR and isinstance(pending, dict)
                    and pending["next_operation"] == 1 and pending["in_flight"] is None):
                previous.update({name: (db / name).read_bytes() for name in ("meta.json", SIDECAR)})
                raise OSError("injected atomic replacement failure")
            return original_replace(src, dst)

        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", fail_checkpoint)
            with pytest.raises(OSError, match="atomic replacement"):
                apply_plan(client, "kb", plan, HashEmbedder())
        assert previous
        assert all((db / name).read_bytes() == content for name, content in previous.items())
        assert not list(db.glob(f".{SIDECAR}.*"))
    with local_client(str(db)) as client:
        record_mutations(client, monkeypatch, successful)
        assert apply_plan(client, "kb", plan, HashEmbedder())["embedded"] == 1
        verify_snapshot(client, "kb", new, HashEmbedder())
    assert successful == Counter({**{("upsert", c["id"]): 1 for c in plan["upsert"]},
                                  **{("delete", c["id"]): 1 for c in plan["delete"]}})


@pytest.mark.parametrize("damage", ["truncated", "missing", "invalid_utf8"])
def test_committed_catalog_recovers_then_replays_without_writes(tmp_path, monkeypatch, damage):
    from ragwrench.sync.recovery import SIDECAR

    db = tmp_path / "db"
    old, _, _ = bootstrap(db)
    sidecar = (db / SIDECAR).read_bytes()
    if damage == "missing":
        (db / "meta.json").unlink()
    else:
        (db / "meta.json").write_bytes(b"{" if damage == "truncated" else b"\xff")
    with local_client(str(db)) as client:
        for name in ("upsert", "delete", "update_collection"):
            monkeypatch.setattr(client, name, lambda *a, **kw: pytest.fail("replayed committed mutation"))
        assert apply_plan(client, "kb", make_plan(build_snapshot([]), old), HashEmbedder())["status"] == "already_applied"
        verify_snapshot(client, "kb", old, HashEmbedder())
    assert (db / SIDECAR).read_bytes() == sidecar


@pytest.mark.parametrize("damage", ["missing", "truncated", "fingerprint", "version", "schema"])
def test_bad_fallback_refused_without_replacing_meta(tmp_path, damage):
    from ragwrench.sync.core import SyncError, load_json, write_json
    from ragwrench.sync.recovery import SIDECAR

    db = tmp_path / "db"
    bootstrap(db)
    fallback = db / SIDECAR
    if damage == "missing":
        fallback.unlink()
    elif damage == "truncated":
        fallback.write_text("{")
    else:
        saved = load_json(fallback)
        if damage == "fingerprint":
            saved["metadata"]["collections"]["kb"]["metadata"]["chunksync"]["fingerprint"] = "0" * 64
        elif damage == "version":
            saved["version"] = 999
        else:
            saved["extra"] = True
        write_json(fallback, saved)
    (db / "meta.json").write_text("{")
    with pytest.raises(SyncError):
        with local_client(str(db)):
            pytest.fail("invalid recovery opened")
    assert (db / "meta.json").read_text() == "{"


@pytest.mark.parametrize("damage", ["traversal", "aliases", "sqlite"])
def test_fallback_cannot_bypass_preflight(tmp_path, damage):
    from ragwrench.sync.core import SyncError, digest, load_json, write_json
    from ragwrench.sync.recovery import SIDECAR

    db = tmp_path / "db"
    bootstrap(db)
    saved = load_json(db / SIDECAR)
    if damage == "traversal":
        saved["metadata"]["collections"]["../escape"] = saved["metadata"]["collections"].pop("kb")
    elif damage == "aliases":
        saved["metadata"]["aliases"] = {"alias": "kb"}
    else:
        (db / "collection" / "kb" / "storage.sqlite").write_bytes(b"invalid SQLite")
    saved["fingerprint"] = digest({key: value for key, value in saved.items() if key != "fingerprint"})
    write_json(db / SIDECAR, saved)
    (db / "meta.json").write_text("{")
    (db / ".lock").unlink()
    with pytest.raises(SyncError):
        with local_client(str(db)):
            pytest.fail("unsafe fallback opened")
    assert (db / "meta.json").read_text() == "{"
    assert not (db / ".lock").exists()


def test_valid_primary_metadata_is_authoritative(tmp_path):
    from ragwrench.sync.core import SyncError
    from ragwrench.sync.recovery import SIDECAR

    db = tmp_path / "db"
    old, _, _ = bootstrap(db)
    # A stale or broken fallback must not roll back valid primary metadata.
    (db / SIDECAR).write_text("{")
    with local_client(str(db)) as client:
        verify_snapshot(client, "kb", old, HashEmbedder())
    primary = '{"collections": {}, "aliases": {"unsafe": "kb"}}'
    (db / "meta.json").write_text(primary)
    with pytest.raises(SyncError, match="aliases"):
        with local_client(str(db)):
            pytest.fail("semantic primary error bypassed")
    assert (db / "meta.json").read_text() == primary


def test_legacy_store_without_sidecar_acquires_protection_on_next_write(tmp_path):
    from ragwrench.sync.recovery import SIDECAR

    db = tmp_path / "db"
    old, new, plan = bootstrap(db)
    (db / SIDECAR).unlink()
    with local_client(str(db)) as client:
        verify_snapshot(client, "kb", old, HashEmbedder())
        assert not (db / SIDECAR).exists()
        apply_plan(client, "kb", plan, HashEmbedder())
    assert (db / SIDECAR).is_file()
    with local_client(str(db)) as client:
        verify_snapshot(client, "kb", new, HashEmbedder())


@pytest.mark.parametrize("change", ["plan", "review", "vector"])
def test_recovered_journal_still_refuses_drift_before_point_writes(tmp_path, monkeypatch, change):
    from qdrant_client import models
    from ragwrench.sync.core import SyncError
    from ragwrench.sync.review import review_plan

    db = tmp_path / "db"
    old, new, plan = bootstrap(db)
    with local_client(str(db)) as client:
        original = client.upsert

        def persist_then_fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected after write")

        monkeypatch.setattr(client, "upsert", persist_then_fail)
        with pytest.raises(RuntimeError, match="after write"):
            apply_plan(client, "kb", plan, HashEmbedder())
        if change == "vector":
            pid = plan["upsert"][0]["id"]
            point = client.retrieve("kb", [pid])[0]
            original("kb", points=[models.PointStruct(id=pid, vector=[0.] * 32, payload=point.payload)])
    (db / "meta.json").write_text("{")
    candidate = make_plan(old, build_snapshot([{"doc_id": "a", "text": "other"}])) if change == "plan" else plan
    review = (review_plan(plan, max_deletion_fraction=0.5, overrides=["max_deletion_fraction"],
                          reviewer="test", reason="changed receipt") if change == "review" else None)
    with local_client(str(db)) as client:
        embedder = HashEmbedder()
        embedder.embed = lambda *a: pytest.fail("embedding after recovered integrity mismatch")
        for name in ("upsert", "delete", "overwrite_payload", "update_collection"):
            monkeypatch.setattr(client, name, lambda *a, **kw: pytest.fail("mutation after recovered integrity mismatch"))
        with pytest.raises(SyncError, match="journal.*(binding|fingerprint)"):
            apply_plan(client, "kb", candidate, embedder, review=review)


@pytest.mark.parametrize("operation", ["save", "repair"])
def test_recovery_fsync_order_and_flushed_contents(tmp_path, monkeypatch, operation):
    """Observe real calls, including bytes visible through a separate fd."""
    from ragwrench.sync.core import digest

    db = tmp_path / "db"
    bootstrap(db)
    target = db / (SIDECAR if operation == "save" else "meta.json")
    expected = json.loads((db / SIDECAR).read_text())
    if operation == "repair":
        (db / "meta.json").write_text("{")
    events, handles = [], []
    original_open, original_fdopen = os.open, os.fdopen
    original_fsync, original_replace, original_close = os.fsync, os.replace, os.close

    def directory_open(path, flags, *args, **kwargs):
        fd = original_open(path, flags, *args, **kwargs)
        if Path(path) == db:
            assert flags & os.O_DIRECTORY
            events.append("directory_open")
            handles.append(fd)
        return fd

    @contextmanager
    def fdopen(fd, *args, **kwargs):
        with original_fdopen(fd, *args, **kwargs) as out:
            handles.append(out)

            class ObservedFile:
                def __getattr__(self, name):
                    return getattr(out, name)

                def flush(self):
                    out.flush()
                    events.append("flush")

            yield ObservedFile()

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            assert json.loads(target.read_text()) == expected
            events.append("directory_fsync")
        else:
            temps = list(db.glob(f".{target.name}.*"))
            assert len(temps) == 1
            saved = json.loads(temps[0].read_text())  # cannot see unflushed bytes
            assert saved == expected
            if operation == "save":
                assert saved["fingerprint"] == digest({k: saved[k] for k in ("kind", "version", "metadata")})
            events.append("file_fsync")
        return original_fsync(fd)

    def replace(src, dst):
        assert Path(dst) == target
        assert handles[-1].closed, "temporary file must close before rename"
        events.append("rename")
        return original_replace(src, dst)

    def close(fd):
        if handles and fd == handles[0]:
            events.append("directory_close")
        return original_close(fd)

    def primary_open(path, mode="r", *args, **kwargs):
        if Path(path) == db / "meta.json" and mode == "w":
            events.append("primary_write")
        return open(path, mode, *args, **kwargs)

    if operation == "repair":
        expected = expected["metadata"]
    with local_client(str(db)) if operation == "save" else nullcontext() as client:
        with monkeypatch.context() as patch:
            for name, function in (("open", directory_open), ("fdopen", fdopen),
                                   ("fsync", fsync), ("replace", replace), ("close", close)):
                patch.setattr(os, name, function)
            patch.setattr(qdrant_local, "open", primary_open, raising=False)
            if operation == "save":
                client._client._save()
            else:
                with local_client(str(db)):
                    pass
    assert events == ["directory_open", "flush", "file_fsync", "rename",
                      "directory_fsync", "directory_close"] + (["primary_write"] if operation == "save" else [])
    with pytest.raises(OSError):
        os.fstat(handles[0])  # directory descriptor closed


@pytest.mark.parametrize("phase,completed", [("upsert_intent", 0), ("upsert_checkpoint", 1),
                                              ("delete_intent", 2), ("delete_checkpoint", 3), ("commit", 4)])
@pytest.mark.parametrize("boundary", ["file_before", "file_after", "rename_before", "rename_after",
                                      "directory_before", "directory_after"])
@pytest.mark.parametrize("primary", ["valid", "torn"])
def test_fsync_failure_stops_apply_and_reopens_without_repeated_writes(
        tmp_path, monkeypatch, phase, completed, boundary, primary):
    from ragwrench.sync.core import digest

    db = tmp_path / "db"
    _, new, plan = bootstrap(db)
    successful = Counter()
    previous = {}
    with local_client(str(db)) as client:
        record_mutations(client, monkeypatch, successful)
        original_fsync, original_replace = os.fsync, os.replace

        def hit():
            state = client.get_collection("kb").config.metadata["chunksync"]
            pending = state["pending"]
            if phase == "commit":
                return pending is None and state["fingerprint"] == new["fingerprint"]
            return (isinstance(pending, dict) and pending["next_operation"] == completed
                    and (pending["in_flight"] is not None) == phase.endswith("intent"))

        def inject(kind, call):
            if not boundary.startswith(kind) or not hit():
                return call()
            assert not previous, "failure should stop this apply immediately"
            previous.update({name: (db / name).read_bytes() for name in ("meta.json", SIDECAR)})
            if boundary.endswith("after"):
                call()
            raise OSError(f"injected {boundary}")

        def fsync(fd):
            kind = "directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
            return inject(kind, lambda: original_fsync(fd))

        def replace(src, dst):
            assert Path(dst).name == SIDECAR
            return inject("rename", lambda: original_replace(src, dst))

        with monkeypatch.context() as patch:
            patch.setattr(os, "fsync", fsync)
            patch.setattr(os, "replace", replace)
            with pytest.raises(OSError, match=f"injected {boundary}"):
                apply_plan(client, "kb", plan, HashEmbedder())
        assert sum(successful.values()) == completed, "no point write after failed persistence"
        assert (db / "meta.json").read_bytes() == previous["meta.json"]
        saved = json.loads((db / SIDECAR).read_text())
        assert saved["fingerprint"] == digest({k: saved[k] for k in ("kind", "version", "metadata")})
        published = boundary in ("rename_after", "directory_before", "directory_after")
        if published:
            assert saved["metadata"]["collections"]["kb"]["metadata"]["chunksync"] == client.get_collection("kb").config.metadata["chunksync"]
        else:
            assert (db / SIDECAR).read_bytes() == previous[SIDECAR]
        assert not list(db.glob(f".{SIDECAR}.*"))
    if primary == "torn":
        (db / "meta.json").write_text("{")
    with local_client(str(db)) as client:
        record_mutations(client, monkeypatch, successful)
        apply_plan(client, "kb", plan, HashEmbedder())
        verify_snapshot(client, "kb", new, HashEmbedder())
        assert apply_plan(client, "kb", plan, HashEmbedder())["status"] == "already_applied"
    expected = Counter({("upsert", c["id"]): 1 for c in plan["upsert"]})
    expected.update({("delete", c["id"]): 1 for c in plan["delete"]})
    assert successful == expected


@pytest.mark.parametrize("failure", ["directory_open", "unsupported_platform"])
def test_durability_unavailable_refuses_before_publication_or_point_writes(tmp_path, monkeypatch, failure):
    from ragwrench.sync import core

    db = tmp_path / "db"
    _, new, plan = bootstrap(db)
    successful = Counter()
    before = {name: (db / name).read_bytes() for name in ("meta.json", SIDECAR)}
    with local_client(str(db)) as client:
        record_mutations(client, monkeypatch, successful)
        with monkeypatch.context() as patch:
            if failure == "unsupported_platform":
                patch.setattr(core.sys, "platform", "darwin")
                error, message = core.SyncError, "require a local Linux filesystem"
            else:
                original_open = os.open

                def fail_open(path, flags, *args, **kwargs):
                    if Path(path) == db:
                        raise OSError("injected directory open failure")
                    return original_open(path, flags, *args, **kwargs)

                patch.setattr(os, "open", fail_open)
                error, message = OSError, "directory open failure"
            with pytest.raises(error, match=message):
                apply_plan(client, "kb", plan, HashEmbedder())
        assert not successful
        assert all((db / name).read_bytes() == value for name, value in before.items())
        assert not list(db.glob(f".{SIDECAR}.*"))
    with local_client(str(db)) as client:
        apply_plan(client, "kb", plan, HashEmbedder())
        verify_snapshot(client, "kb", new, HashEmbedder())


@pytest.mark.parametrize("phase,completed", [("admission", 0), ("intent", 0),
                                              ("checkpoint", 1), ("commit", 4)])
@pytest.mark.parametrize("boundary", ["rename_before", "directory_after", "primary_torn"])
def test_failed_metadata_save_fences_live_apply_until_reopen(
        tmp_path, monkeypatch, phase, completed, boundary):
    """A failed save leaves live metadata ahead of the authoritative catalog."""
    from ragwrench.sync.core import SyncError
    from ragwrench.sync.store import inspect_policy

    db = tmp_path / "db"
    old, new, plan = bootstrap(db)
    successful = Counter()
    error = OSError(f"injected {boundary} at {phase}")
    with local_client(str(db)) as client:
        record_mutations(client, monkeypatch, successful)

        def hit():
            state = client.get_collection("kb").config.metadata["chunksync"]
            pending = state["pending"]
            if phase == "commit":
                return pending is None and state["fingerprint"] == new["fingerprint"]
            return (isinstance(pending, dict) and pending["next_operation"] == completed
                    and (pending["in_flight"] is not None) == (phase == "intent"))

        original_replace, original_fsync = os.replace, os.fsync

        def replace(src, dst):
            if Path(dst).name == SIDECAR and hit():
                raise error
            return original_replace(src, dst)

        def fsync(fd):
            result = original_fsync(fd)
            if stat.S_ISDIR(os.fstat(fd).st_mode) and hit():
                raise error
            return result

        @contextmanager
        def torn_open(path, mode="r", *args, **kwargs):
            with open(path, mode, *args, **kwargs) as out:
                if Path(path) == db / "meta.json" and mode == "w" and hit():
                    out.write('{"collections":')
                    out.flush()
                    raise error
                yield out

        with monkeypatch.context() as patch:
            if boundary == "rename_before":
                patch.setattr(os, "replace", replace)
            elif boundary == "directory_after":
                patch.setattr(os, "fsync", fsync)
            else:
                patch.setattr(qdrant_local, "open", torn_open, raising=False)
            with pytest.raises(OSError) as first:
                apply_plan(client, "kb", plan, HashEmbedder())
        assert first.value is error, "the first save error must propagate unchanged"
        assert sum(successful.values()) == completed
        # These reads remain available, but describe live, possibly unsaved state.
        assert client.count("kb").count == 2 + min(completed, 2) - max(completed - 2, 0)
        assert inspect_policy(client, "kb")["collection"] == "kb"
        assert len(client.get_collections().collections) == 1
        before = {name: (db / name).read_bytes() for name in ("meta.json", SIDECAR)}
        retry_calls, unpersisted_intents = [], []
        with monkeypatch.context() as patch:
            for method in ("upsert", "delete", "overwrite_payload", "create_collection", "update_collection"):
                original = getattr(client, method)

                def observe(*args, _method=method, _original=original, **kwargs):
                    retry_calls.append(_method)
                    if _method == "upsert" and boundary == "rename_before" and phase == "intent":
                        pending = json.loads((db / "meta.json").read_text())["collections"]["kb"]["metadata"]["chunksync"]["pending"]
                        if pending["in_flight"] is None:
                            unpersisted_intents.append(str(kwargs["points"][0].id))
                    return _original(*args, **kwargs)

                patch.setattr(client, method, observe)
            retry_embedder = HashEmbedder()
            for target_collection, candidate in (("kb", plan), ("kb", plan),
                                                  ("fresh", make_plan(build_snapshot([]), old))):
                try:
                    apply_plan(client, target_collection, candidate, retry_embedder)
                except SyncError as exc:
                    assert str(exc) == "persistent metadata save failed; close and reopen the client before applying again"
                else:
                    pytest.fail(f"failed client was not fenced: embeddings={retry_embedder.calls}, "
                                f"mutation_calls={retry_calls}, writes_without_persisted_intent={unpersisted_intents}")
            assert retry_embedder.calls == 0
            assert retry_calls == []
        assert all((db / name).read_bytes() == content for name, content in before.items())
        # A separately opened store and in-memory client are unaffected.
        for other_db in (str(tmp_path / "independent"), ":memory:"):
            with local_client(other_db) as independent:
                assert apply_plan(independent, "kb", make_plan(build_snapshot([]), old), HashEmbedder())["points"] == 2
    with local_client(str(db)) as client:
        record_mutations(client, monkeypatch, successful)
        embedder = HashEmbedder()
        apply_plan(client, "kb", plan, embedder)
        assert embedder.calls == 2 - min(completed, 2)
        verify_snapshot(client, "kb", new, HashEmbedder())
        assert apply_plan(client, "kb", plan, HashEmbedder())["status"] == "already_applied"
    assert successful == Counter({**{("upsert", c["id"]): 1 for c in plan["upsert"]},
                                  **{("delete", c["id"]): 1 for c in plan["delete"]}})


@pytest.mark.parametrize("failure", ["embedding", "point_before", "point_after", "review", "memory_metadata"])
def test_nonpersistent_save_errors_do_not_fence_apply(tmp_path, monkeypatch, failure):
    from ragwrench.sync.core import SyncError
    from ragwrench.sync.review import review_plan

    old, new, plan = snapshots()
    db = ":memory:" if failure == "memory_metadata" else str(tmp_path / "db")
    with local_client(db) as client:
        apply_plan(client, "kb", make_plan(build_snapshot([]), old), HashEmbedder())
        embedder = HashEmbedder()
        original_upsert = client.upsert

        def fail_point(*args, **kwargs):
            if failure == "point_after":
                original_upsert(*args, **kwargs)
            raise RuntimeError("injected non-metadata failure")

        with monkeypatch.context() as patch:
            if failure == "embedding":
                patch.setattr(embedder, "embed", lambda *a: (_ for _ in ()).throw(RuntimeError("injected embedding")))
            elif failure == "memory_metadata":
                patch.setattr(client._client, "_save", lambda: (_ for _ in ()).throw(RuntimeError("injected memory save")))
            elif failure.startswith("point"):
                patch.setattr(client, "upsert", fail_point)
            with pytest.raises(SyncError if failure == "review" else RuntimeError):
                apply_plan(client, "kb", plan, embedder,
                           review=review_plan(plan, max_deletion_fraction=0) if failure == "review" else None)
        assert apply_plan(client, "kb", plan, HashEmbedder())["points"] == 2
        verify_snapshot(client, "kb", new, HashEmbedder())


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_collection_creation_save_error_fences_even_other_collections(tmp_path, monkeypatch, error_type):
    from ragwrench.sync import recovery
    from ragwrench.sync.core import SyncError

    plan = make_plan(build_snapshot([]), build_snapshot([{"doc_id": "a", "text": "new"}]))
    db = str(tmp_path / "fresh")
    error = error_type("injected creation save")
    with local_client(db) as client:
        with monkeypatch.context() as patch:
            patch.setattr(recovery, "write_json", lambda *a, **kw: (_ for _ in ()).throw(error))
            with pytest.raises(error_type, match="creation save") as first:
                apply_plan(client, "kb", plan, HashEmbedder())
            assert first.value is error
        with monkeypatch.context() as patch:
            patch.setattr(client, "create_collection", lambda *a, **kw: pytest.fail("created after save error"))
            with pytest.raises(SyncError, match="close and reopen the client"):
                apply_plan(client, "other", plan, HashEmbedder())
        assert client.get_collections().collections  # reads and close still work
    with local_client(db) as client:
        assert apply_plan(client, "kb", plan, HashEmbedder())["points"] == 1
