"""Metadata-only aggregate policy inventory contracts."""
import json

import pytest
from qdrant_client import models

from ragwrench.sync.cli import main
from ragwrench.sync.core import SyncError, build_snapshot, make_plan
from ragwrench.sync.embedders import HashEmbedder
from ragwrench.sync.review import review_plan
from ragwrench.sync.store import apply_plan, local_client
from .conftest import linux_durable


@pytest.fixture
def inventory_store(tmp_path):
    db = tmp_path / "db"
    plan = make_plan(build_snapshot([]), build_snapshot([{"doc_id": "guide", "text": "private text"}]))
    with local_client(str(db)) as client:
        for name, limit in (("zeta", 0.75), ("alpha", 0.25)):
            apply_plan(client, name, plan, HashEmbedder(),
                       review=review_plan(plan, max_deletion_fraction=limit))
        client.create_collection("middle", vectors_config=models.VectorParams(size=32, distance=models.Distance.DOT))
    return db


def inspection(name, limit, source="explicit", pending=None, override=None):
    return {"kind": "chunksync.policy_inspection", "version": 1, "collection": name,
            "max_deletion_fraction": limit, "policy_source": source,
            "pending_plan_fingerprint": pending, "policy_override": override}


@linux_durable
def test_lists_two_managed_and_one_unmanaged_without_collection_argument(inventory_store, capsys):
    assert main(["list-policies", "--qdrant-path", str(inventory_store)]) == 0
    out = capsys.readouterr()
    assert not out.err
    report = json.loads(out.out)
    assert set(report) == {"kind", "version", "catalog_source", "collections"}
    assert report["kind"] == "chunksync.policy_inventory" and report["version"] == 1
    assert report["catalog_source"] == "primary"
    a, m, z = report["collections"]
    assert a == {"collection": "alpha", "management_state": "managed", "status": "ok",
                 "policy": inspection("alpha", 0.25), "diagnostic": None}
    assert z == {"collection": "zeta", "management_state": "managed", "status": "ok",
                 "policy": inspection("zeta", 0.75), "diagnostic": None}
    assert m == {"collection": "middle", "management_state": "unmanaged", "status": "unmanaged",
                 "policy": None, "diagnostic": {"code": "unmanaged_collection",
                                                   "message": "collection has no management state"}}
    assert "private text" not in out.out


def files(db):
    return {p.relative_to(db): p.read_bytes() for p in db.rglob("*") if p.is_file() and not p.is_symlink()}


def edit_catalog(db, change):
    primary = db / "meta.json"
    metadata = json.loads(primary.read_text())
    change(metadata)
    primary.write_text(json.dumps(metadata))


def guarded_list(db, monkeypatch, capsys, *, no_repair=False, exit_code=0):
    import io
    import sqlite3
    import ragwrench.sync.cli as cli_module
    import ragwrench.sync.core as core
    import ragwrench.sync.recovery as recovery
    import ragwrench.sync.safe_store as safe_store
    import ragwrench.sync.store as store

    # Reading an inventory must not recreate the backend lock either.
    (db / ".lock").unlink(missing_ok=True)
    before = files(db)
    def forbidden(*a, **kw):
        pytest.fail("inventory must not open points/backend, embed, preflight points or write")
    real_open = io.open
    def catalog_only(path, mode="r", *a, **kw):
        from pathlib import Path
        assert Path(path).name in ("meta.json", "chunksync-recovery.json")
        assert mode in ("r", "rb")
        return real_open(path, mode, *a, **kw)
    with monkeypatch.context() as guard:
        guard.setattr(io, "open", catalog_only)
        guard.setattr(sqlite3, "connect", forbidden)
        guard.setattr(store, "QdrantClient", forbidden)
        guard.setattr(cli_module, "local_client", forbidden)
        guard.setattr(cli_module, "make_embedder", forbidden)
        guard.setattr(HashEmbedder, "embed", forbidden)
        guard.setattr(safe_store, "preflight", forbidden)
        guard.setattr(recovery, "preflight", forbidden)
        guard.setattr(recovery, "write_json", forbidden)
        guard.setattr(core, "write_json", forbidden)
        assert main(["list-policies", "--qdrant-path", str(db),
                     *(["--no-repair"] if no_repair else [])]) == exit_code
    out = capsys.readouterr()
    assert files(db) == before
    assert not (db / ".lock").exists()
    if exit_code == 0:
        assert not out.err
        return json.loads(out.out)
    assert not out.out and out.err.startswith("ragwrench sync: ") and "Traceback" not in out.err
    return out.err


@pytest.mark.parametrize("no_repair", [False, True])
@linux_durable
def test_reads_only_catalog_and_is_deterministic(inventory_store, monkeypatch, capsys, no_repair):
    first = guarded_list(inventory_store, monkeypatch, capsys, no_repair=no_repair)
    edit_catalog(inventory_store, lambda m: m.update(collections=dict(reversed(list(m["collections"].items())))))
    assert guarded_list(inventory_store, monkeypatch, capsys, no_repair=no_repair) == first


@pytest.mark.parametrize("source,limit", [("legacy-review", 0.25), ("default", 1.0), ("explicit", 0.25)])
@pytest.mark.parametrize("no_repair", [False, True])
@linux_durable
def test_legacy_policy_precedence_without_adoption(inventory_store, monkeypatch, capsys, source, limit, no_repair):
    def change(m):
        state = m["collections"]["alpha"]["metadata"]["chunksync"]
        if source == "explicit":
            state["last_review"] = []  # unused legacy state must not override explicit policy
        else:
            state.pop("review_policy")
            if source == "default":
                state.pop("last_review")
    edit_catalog(inventory_store, change)
    report = guarded_list(inventory_store, monkeypatch, capsys, no_repair=no_repair)
    assert report["collections"][0]["policy"] == inspection("alpha", limit, source)


@pytest.mark.parametrize("state", [None, [], {}, {"version": True}, {"version": 2}])
@linux_durable
def test_missing_and_unsupported_states_are_explicit(inventory_store, monkeypatch, capsys, state):
    edit_catalog(inventory_store, lambda m: m["collections"]["alpha"]["metadata"].update(chunksync=state))
    rows = guarded_list(inventory_store, monkeypatch, capsys)["collections"]
    row = rows[0]
    assert row["status"] == ("unmanaged" if state is None else "unsupported")
    assert row["management_state"] == row["status"] and row["policy"] is None
    assert row["diagnostic"]
    assert rows[2]["status"] == "ok"


@pytest.mark.parametrize("field,value", [
    ("review_policy", None), ("review_policy", {}), ("review_policy", {"max_deletion_fraction": True}),
    ("review_policy", {"max_deletion_fraction": 1.01}),
    ("review_policy", {"max_deletion_fraction": "0.25"}),
    ("pending", {}), ("pending", "A" * 64), ("last_policy_override", {}),
])
@pytest.mark.parametrize("no_repair", [False, True])
@linux_durable
def test_one_invalid_policy_does_not_hide_other_collections(inventory_store, monkeypatch, capsys, field, value, no_repair):
    edit_catalog(inventory_store, lambda m: m["collections"]["alpha"]["metadata"]["chunksync"].update({field: value}))
    a, m, z = guarded_list(inventory_store, monkeypatch, capsys, no_repair=no_repair)["collections"]
    assert a["management_state"] == "managed" and a["status"] == "invalid" and a["policy"] is None
    assert a["diagnostic"]["code"] == "invalid_policy_metadata"
    assert m["status"] == "unmanaged" and z["policy"] == inspection("zeta", 0.75)


@pytest.mark.parametrize("change", ["config", "distance", "name", "metadata"])
@linux_durable
def test_unavailable_config_is_a_row_diagnostic(inventory_store, monkeypatch, capsys, change):
    def mutate(m):
        config = m["collections"]["alpha"]
        if change == "config":
            m["collections"]["alpha"] = []
        elif change == "distance":
            config["vectors"]["distance"] = "Cosine"
        elif change == "name":
            m["collections"]["../bad"] = m["collections"].pop("alpha")
        else:
            config["metadata"] = []
    edit_catalog(inventory_store, mutate)
    rows = guarded_list(inventory_store, monkeypatch, capsys)["collections"]
    assert rows[0]["status"] == "unavailable" and rows[0]["management_state"] == "unknown"
    assert rows[0]["diagnostic"]["code"] == "unavailable_collection_metadata"
    assert rows[0]["policy"] is None and rows[-1]["status"] == "ok"


@pytest.mark.parametrize("pending", ["a" * 64, {"plan_fingerprint": "b" * 64}])
@linux_durable
def test_pending_and_historical_override_reuse_inspection_validation(inventory_store, monkeypatch, capsys, pending):
    from ragwrench.sync.policy import policy_override
    record = policy_override(make_plan(build_snapshot([]), build_snapshot([])), "alpha",
                             from_max_deletion_fraction=0.1, max_deletion_fraction=0.5,
                             reviewer="operator", reason="synthetic policy exception")
    def change(m):
        m["collections"]["alpha"]["metadata"]["chunksync"].update(pending=pending, last_policy_override=record)
    edit_catalog(inventory_store, change)
    row = guarded_list(inventory_store, monkeypatch, capsys)["collections"][0]
    assert row["policy"] == inspection("alpha", 0.25, pending=pending if isinstance(pending, str) else pending["plan_fingerprint"], override=record)
    record["provenance"]["reason"] = "tampered"
    edit_catalog(inventory_store, change)
    assert guarded_list(inventory_store, monkeypatch, capsys)["collections"][0]["status"] == "invalid"


@pytest.mark.parametrize("no_repair", [False, True])
def test_initialized_empty_catalog(tmp_path, monkeypatch, capsys, no_repair):
    db = tmp_path / "empty"
    db.mkdir()
    (db / "meta.json").write_text('{"collections": {}, "aliases": {}}')
    assert guarded_list(db, monkeypatch, capsys, no_repair=no_repair) == {
        "kind": "chunksync.policy_inventory", "version": 1, "catalog_source": "primary", "collections": []}


@pytest.mark.parametrize("damage", ["missing", "truncated", "invalid-utf8"])
@linux_durable
def test_default_recovers_catalog_in_memory_only_and_no_repair_refuses(inventory_store, monkeypatch, capsys, damage):
    before = guarded_list(inventory_store, monkeypatch, capsys)
    primary = inventory_store / "meta.json"
    if damage == "missing":
        primary.unlink()
    else:
        primary.write_bytes(b'{"collections":' if damage == "truncated" else b"\xff")
    assert "primary metadata repair required" in guarded_list(inventory_store, monkeypatch, capsys, no_repair=True, exit_code=2)
    recovered = guarded_list(inventory_store, monkeypatch, capsys)
    assert recovered == {**before, "catalog_source": "recovery"}


@pytest.mark.parametrize("case", ["bad-json", "fingerprint", "version", "metadata", "aliases", "missing"])
@linux_durable
def test_bad_recovery_refused_but_healthy_primary_authoritative(inventory_store, monkeypatch, capsys, case):
    from ragwrench.sync.core import digest
    sidecar = inventory_store / "chunksync-recovery.json"
    saved = json.loads(sidecar.read_text())
    if case == "bad-json":
        sidecar.write_text("bad")
    elif case == "missing":
        sidecar.unlink()
    else:
        if case == "fingerprint":
            saved["fingerprint"] = "0" * 64
        elif case == "version":
            saved["version"] = True
        else:
            saved["metadata"] = [] if case == "metadata" else {"collections": {}, "aliases": {"a": "alpha"}}
            saved["fingerprint"] = digest({k: v for k, v in saved.items() if k != "fingerprint"})
        sidecar.write_text(json.dumps(saved))
    assert guarded_list(inventory_store, monkeypatch, capsys)["catalog_source"] == "primary"
    (inventory_store / "meta.json").write_text("{")
    assert "primary metadata repair required" in guarded_list(inventory_store, monkeypatch, capsys, no_repair=True, exit_code=2)
    guarded_list(inventory_store, monkeypatch, capsys, exit_code=2)


@pytest.mark.parametrize("metadata", [[], {}, {"collections": [], "aliases": {}}, {"collections": {}, "aliases": {"x": "alpha"}}])
@linux_durable
def test_semantic_primary_error_never_uses_valid_sidecar(inventory_store, monkeypatch, capsys, metadata):
    (inventory_store / "meta.json").write_text(json.dumps(metadata))
    assert "unsupported Qdrant metadata or aliases" in guarded_list(inventory_store, monkeypatch, capsys, exit_code=2)


@linux_durable
def test_count_limit_before_rows_and_exact_boundary(inventory_store, monkeypatch, capsys):
    import ragwrench.sync.inventory as inventory
    config = json.loads((inventory_store / "meta.json").read_text())["collections"]["alpha"]
    def populate(n):
        edit_catalog(inventory_store, lambda m: m.update(collections={f"c{i:04}": config for i in range(n)}))
    populate(1000)
    assert len(guarded_list(inventory_store, monkeypatch, capsys)["collections"]) == 1000
    populate(1001)
    def forbidden(*a):
        pytest.fail("must check count before decoding rows")
    monkeypatch.setattr(inventory, "_row", forbidden)
    assert "1000 collection limit" in guarded_list(inventory_store, monkeypatch, capsys, exit_code=2)


@linux_durable
def test_output_byte_bound_includes_newline(inventory_store, monkeypatch, capsys):
    import ragwrench.sync.inventory as inventory
    report = guarded_list(inventory_store, monkeypatch, capsys)
    length = len(inventory.inventory_json(report).encode("utf-8"))
    monkeypatch.setattr(inventory, "MAX_JSON_BYTES", length)
    assert guarded_list(inventory_store, monkeypatch, capsys) == report
    monkeypatch.setattr(inventory, "MAX_JSON_BYTES", length - 1)
    assert "output exceeds" in guarded_list(inventory_store, monkeypatch, capsys, exit_code=2)


@pytest.mark.parametrize("no_repair", [False, True])
@linux_durable
def test_oversized_primary_refuses_without_fallback(inventory_store, monkeypatch, capsys, no_repair):
    from ragwrench.sync.core import MAX_JSON_BYTES
    primary = inventory_store / "meta.json"
    with primary.open("r+b") as stream:
        stream.truncate(MAX_JSON_BYTES + 1)
    # Avoid reading the sparse file into the file-invariance helper.
    with pytest.raises(SyncError, match="64 MiB"):
        from ragwrench.sync.inventory import list_policies
        list_policies(inventory_store, repair_metadata=not no_repair)
    assert primary.stat().st_size == MAX_JSON_BYTES + 1


@pytest.mark.parametrize("case", ["missing", "empty-directory", "file", "memory", "symlink", "catalog-symlink"])
def test_invalid_paths_are_never_created(tmp_path, capsys, case):
    db = tmp_path / "db"
    if case == "empty-directory":
        db.mkdir()
    elif case == "file":
        db.write_text("not a store")
    elif case in ("symlink", "catalog-symlink"):
        target = tmp_path / "target"
        target.mkdir()
        (target / "meta.json").write_text('{"collections": {}, "aliases": {}}')
        if case == "symlink":
            db.symlink_to(target, target_is_directory=True)
        else:
            db.mkdir()
            (db / "meta.json").symlink_to(target / "meta.json")
    assert main(["list-policies", "--qdrant-path", ":memory:" if case == "memory" else str(db)]) == 2
    out = capsys.readouterr()
    assert not out.out and "ragwrench sync:" in out.err
    if case in ("missing", "memory"):
        assert not db.exists()
    elif case == "empty-directory":
        assert list(db.iterdir()) == []


@pytest.mark.parametrize("no_repair", [False, True])
@linux_durable
def test_catalog_io_error_does_not_read_fallback(inventory_store, monkeypatch, capsys, no_repair):
    import ragwrench.sync.inventory as inventory
    def unreadable(path):
        assert path.name == "meta.json"
        raise PermissionError("synthetic catalog read refusal")
    monkeypatch.setattr(inventory, "load_json", unreadable)
    assert "synthetic catalog read refusal" in guarded_list(inventory_store, monkeypatch, capsys, no_repair=no_repair, exit_code=2)


@pytest.mark.parametrize("damaged", [False, True])
@linux_durable
def test_no_repair_never_reads_sidecar(inventory_store, monkeypatch, capsys, damaged):
    import ragwrench.sync.inventory as inventory
    if damaged:
        (inventory_store / "meta.json").write_bytes(b"\xff")
    original = inventory.load_json
    def primary_only(path):
        assert path.name == "meta.json"
        return original(path)
    monkeypatch.setattr(inventory, "load_json", primary_only)
    guarded_list(inventory_store, monkeypatch, capsys, no_repair=True, exit_code=2 if damaged else 0)


@linux_durable
def test_recovery_reports_invalid_one_collection_metadata(inventory_store, monkeypatch, capsys):
    from ragwrench.sync.core import digest
    sidecar = inventory_store / "chunksync-recovery.json"
    saved = json.loads(sidecar.read_text())
    saved["metadata"]["collections"]["alpha"]["metadata"]["chunksync"]["review_policy"] = {}
    saved["fingerprint"] = digest({k: v for k, v in saved.items() if k != "fingerprint"})
    sidecar.write_text(json.dumps(saved))
    (inventory_store / "meta.json").unlink()
    report = guarded_list(inventory_store, monkeypatch, capsys)
    assert report["catalog_source"] == "recovery"
    assert [r["status"] for r in report["collections"]] == ["invalid", "unmanaged", "ok"]
