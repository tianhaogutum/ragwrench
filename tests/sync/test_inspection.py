"""Collection policy discovery is read-only apart from local-client recovery."""
from copy import deepcopy
import json
import subprocess
import sys

import pytest

from ragwrench.sync.core import build_snapshot, make_plan
from ragwrench.sync.embedders import HashEmbedder
from ragwrench.sync.review import review_plan
from ragwrench.sync.store import apply_plan, local_client


def snapshot(*texts):
    return build_snapshot([{"doc_id": "guide", "text": text} for text in texts])


def bootstrap(db):
    plan = make_plan(snapshot(), snapshot("a", "b", "c", "d"))
    with local_client(str(db)) as client:
        apply_plan(client, "kb", plan, HashEmbedder(),
                   review=review_plan(plan, max_deletion_fraction=0.5))


def cli(db, collection="kb", *options):
    return subprocess.run([sys.executable, "-m", "ragwrench.sync.cli", "inspect-policy",
                           "--qdrant-path", str(db), "--collection", collection, *options],
                          capture_output=True, text=True)


def expected(*, limit=0.5, source="explicit", pending=None, override=None):
    return {"kind": "chunksync.policy_inspection", "version": 1, "collection": "kb",
            "max_deletion_fraction": limit, "policy_source": source,
            "pending_plan_fingerprint": pending, "policy_override": override}


def test_cli_inspects_strict_explicit_policy_without_changing_store(tmp_path):
    db = tmp_path / "db"
    bootstrap(db)
    before = {p.relative_to(db): p.read_bytes() for p in db.rglob("*") if p.is_file()}
    result = cli(db)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected()
    assert not result.stderr
    assert {p.relative_to(db): p.read_bytes() for p in db.rglob("*") if p.is_file()} == before


def change_state(db, change):
    with local_client(str(db)) as client:
        state = deepcopy(client.get_collection("kb").config.metadata["chunksync"])
        change(state)
        client.update_collection("kb", metadata={"chunksync": state})


def guarded_inspection(db, monkeypatch, capsys, *, collection="kb", exit_code=0, no_repair=False):
    """Exercise the command with actual reopen and forbid work after opening."""
    from contextlib import contextmanager
    import ragwrench.sync.cli as cli_module

    before = {p.relative_to(db): p.read_bytes() for p in db.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("inspection must not embed, scan points, resume, or mutate")

    @contextmanager
    def guarded_client(path, **options):
        with local_client(path, **options) as client:
            with monkeypatch.context() as guard:
                for name in ("upsert", "delete", "overwrite_payload", "set_payload", "delete_payload",
                             "clear_payload", "update_collection", "create_collection", "delete_collection",
                             "scroll", "retrieve", "count"):
                    guard.setattr(client, name, forbidden)
                yield client

    with monkeypatch.context() as guard:
        guard.setattr(cli_module, "local_client", guarded_client)
        guard.setattr(cli_module, "make_embedder", forbidden)
        guard.setattr(cli_module, "apply_plan", forbidden)
        guard.setattr(HashEmbedder, "embed", forbidden)
        assert cli_module.main(["inspect-policy", "--qdrant-path", str(db),
                                "--collection", collection, *(["--no-repair"] if no_repair else [])]) == exit_code
    output = capsys.readouterr()
    assert {p.relative_to(db): p.read_bytes() for p in db.rglob("*") if p.is_file()} == before
    if exit_code == 0:
        assert not output.err
        return json.loads(output.out)
    assert not output.out and output.err.startswith("ragwrench sync: ") and "Traceback" not in output.err
    return output.err


@pytest.mark.parametrize("case,source,limit", [
    ("explicit", "explicit", 0.5), ("last_review", "legacy-review", 0.5),
    ("no_review", "default", 1.0), ("null_review", "default", 1.0),
])
@pytest.mark.parametrize("no_repair", [False, True])
def test_policy_precedence_and_legacy_fallbacks_do_not_adopt_policy(tmp_path, monkeypatch, capsys, case, source, limit, no_repair):
    db = tmp_path / "db"
    bootstrap(db)
    def change(state):
        if case == "explicit":
            # Explicit policy takes precedence, even if the old receipt is unusable.
            state["last_review"] = {"policy": {"max_deletion_fraction": 0.9}}
        else:
            state.pop("review_policy")
            if case == "no_review":
                state.pop("last_review")
            elif case == "null_review":
                state["last_review"] = None
    change_state(db, change)
    assert guarded_inspection(db, monkeypatch, capsys, no_repair=no_repair) == expected(limit=limit, source=source)


@pytest.mark.parametrize("no_repair", [False, True])
def test_pending_transaction_reports_admitted_override_without_resuming(tmp_path, monkeypatch, capsys, no_repair):
    from ragwrench.sync.policy import policy_override
    db = tmp_path / "db"
    bootstrap(db)
    plan = make_plan(snapshot("a", "b", "c", "d"), snapshot("a", "b", "c", "d", "e"))
    override = policy_override(plan, "kb", from_max_deletion_fraction=0.5,
                               max_deletion_fraction=0.75, reviewer="ops", reason="DOC-42")
    with local_client(str(db)) as client:
        embedder = HashEmbedder()
        with monkeypatch.context() as failure:
            def interrupt(*args):
                raise RuntimeError("interrupted embedding")
            failure.setattr(embedder, "embed", interrupt)
            with pytest.raises(RuntimeError, match="interrupted embedding"):
                apply_plan(client, "kb", plan, embedder,
                           review=review_plan(plan, max_deletion_fraction=0.75), policy_override=override)
    assert guarded_inspection(db, monkeypatch, capsys, no_repair=no_repair) == expected(
        limit=0.75, pending=plan["fingerprint"], override=override)


@pytest.mark.parametrize("journal", [False, True])
@pytest.mark.parametrize("no_repair", [False, True])
def test_pending_binding_is_reported_without_requiring_full_journal(tmp_path, monkeypatch, capsys, journal, no_repair):
    db = tmp_path / "db"
    bootstrap(db)
    fingerprint = "a" * 64
    change_state(db, lambda s: s.update(pending={"plan_fingerprint": fingerprint} if journal else fingerprint))
    assert guarded_inspection(db, monkeypatch, capsys, no_repair=no_repair) == expected(pending=fingerprint)


@pytest.mark.parametrize("no_repair", [False, True])
def test_latest_exception_retained_when_later_policy_is_stricter(tmp_path, monkeypatch, capsys, no_repair):
    from ragwrench.sync.policy import policy_override
    db = tmp_path / "db"
    bootstrap(db)
    old, middle, new = snapshot("a", "b", "c", "d"), snapshot("a", "b"), snapshot("a", "b", "e")
    plan = make_plan(old, middle)
    override = policy_override(plan, "kb", from_max_deletion_fraction=0.5,
                               max_deletion_fraction=0.75, reviewer="policy-owner", reason="DOC-42")
    with local_client(str(db)) as client:
        apply_plan(client, "kb", plan, HashEmbedder(),
                   review=review_plan(plan, max_deletion_fraction=0.75), policy_override=override)
        tightened = make_plan(middle, new)
        apply_plan(client, "kb", tightened, HashEmbedder(),
                   review=review_plan(tightened, max_deletion_fraction=0.25))
    assert guarded_inspection(db, monkeypatch, capsys, no_repair=no_repair) == expected(limit=0.25, override=override)


@pytest.mark.parametrize("policy", [
    None, [], {}, {"max_deletion_fraction": True}, {"max_deletion_fraction": "0.5"},
    {"max_deletion_fraction": -0.1}, {"max_deletion_fraction": 1.1},
    {"max_deletion_fraction": 10**400}, {"max_deletion_fraction": 0.5, "extra": 0},
])
@pytest.mark.parametrize("source", ["explicit", "legacy-review"])
@pytest.mark.parametrize("no_repair", [False, True])
def test_malformed_policy_fails_closed_without_fallback(tmp_path, monkeypatch, capsys, policy, source, no_repair):
    db = tmp_path / "db"
    bootstrap(db)
    def change(state):
        if source == "explicit":
            state["review_policy"] = policy
        else:
            state.pop("review_policy")
            state["last_review"] = {"policy": policy}
    change_state(db, change)
    assert "policy" in guarded_inspection(db, monkeypatch, capsys, no_repair=no_repair, exit_code=2)


@pytest.mark.parametrize("pending", [{}, [], 12, "", "A" * 64, {"plan_fingerprint": None}])
@pytest.mark.parametrize("no_repair", [False, True])
def test_invalid_pending_binding_is_an_error(tmp_path, monkeypatch, capsys, pending, no_repair):
    db = tmp_path / "db"
    bootstrap(db)
    change_state(db, lambda s: s.update(pending=pending))
    assert "pending plan" in guarded_inspection(db, monkeypatch, capsys, no_repair=no_repair, exit_code=2)


@pytest.mark.parametrize("state", [None, [], {}, {"version": True}, {"version": 2}])
@pytest.mark.parametrize("no_repair", [False, True])
def test_unmanaged_or_unsupported_state_is_an_error(tmp_path, monkeypatch, capsys, state, no_repair):
    db = tmp_path / "db"
    bootstrap(db)
    with local_client(str(db)) as client:
        client.update_collection("kb", metadata={"chunksync": state})
    assert "management state" in guarded_inspection(db, monkeypatch, capsys, no_repair=no_repair, exit_code=2)


@pytest.mark.parametrize("record", [[], {}, {"fingerprint": "0" * 64}])
@pytest.mark.parametrize("no_repair", [False, True])
def test_malformed_override_is_an_error(tmp_path, monkeypatch, capsys, record, no_repair):
    db = tmp_path / "db"
    bootstrap(db)
    change_state(db, lambda s: s.update(last_policy_override=record))
    assert "policy override" in guarded_inspection(db, monkeypatch, capsys, no_repair=no_repair, exit_code=2)


@pytest.mark.parametrize("change", ["tampered", "wrong-collection", "bad-plan"])
@pytest.mark.parametrize("no_repair", [False, True])
def test_invalid_historical_override_is_an_error(tmp_path, monkeypatch, capsys, change, no_repair):
    from ragwrench.sync.policy import policy_override
    db = tmp_path / "db"
    bootstrap(db)
    record = policy_override(make_plan(snapshot("a"), snapshot()),
                             "other" if change == "wrong-collection" else "kb",
                             from_max_deletion_fraction=0.5, max_deletion_fraction=0.75,
                             reviewer="ops", reason="DOC-42")
    if change == "tampered":
        record["provenance"]["reason"] = "edited"
    elif change == "bad-plan":
        record["plan_fingerprint"] = "bogus"
    change_state(db, lambda s: s.update(last_policy_override=record))
    assert "policy override" in guarded_inspection(db, monkeypatch, capsys, no_repair=no_repair, exit_code=2)


@pytest.mark.parametrize("no_repair", [False, True])
def test_missing_collection_is_not_created(tmp_path, monkeypatch, capsys, no_repair):
    db = tmp_path / "db"
    bootstrap(db)
    assert "does not exist" in guarded_inspection(db, monkeypatch, capsys, no_repair=no_repair, collection="missing", exit_code=2)


@pytest.mark.parametrize("case", ["missing-path", "empty-directory", "file", "memory", "invalid-name"])
@pytest.mark.parametrize("no_repair", [False, True])
def test_invalid_inputs_refuse_before_opening(tmp_path, monkeypatch, capsys, case, no_repair):
    import ragwrench.sync.cli as cli_module
    db = tmp_path / "absent"
    if case == "empty-directory":
        db.mkdir()
    elif case == "file":
        db.write_text("not a store")
    def forbidden(*a):
        pytest.fail("must refuse before local_client is opened")
    monkeypatch.setattr(cli_module, "local_client", forbidden)
    assert cli_module.main(["inspect-policy", "--qdrant-path", ":memory:" if case == "memory" else str(db),
                            "--collection", "../bad" if case == "invalid-name" else "kb",
                            *(["--no-repair"] if no_repair else [])]) == 2
    output = capsys.readouterr()
    assert not output.out and "ragwrench sync: " in output.err
    if case in ("missing-path", "memory", "invalid-name"):
        assert not db.exists()
    elif case == "empty-directory":
        assert list(db.iterdir()) == []
    else:
        assert db.read_text() == "not a store"


def test_inspection_retains_documented_metadata_repair_on_open(tmp_path):
    db = tmp_path / "db"
    bootstrap(db)
    sidecar = (db / "chunksync-recovery.json").read_bytes()
    (db / "meta.json").write_text('{"collections":')
    result = cli(db)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected()
    assert json.loads((db / "meta.json").read_text())["collections"]["kb"]["metadata"]["chunksync"]["review_policy"] == {"max_deletion_fraction": 0.5}
    assert (db / "chunksync-recovery.json").read_bytes() == sidecar


@pytest.mark.parametrize("no_repair", [False, True])
def test_unmanaged_collection_with_no_state_is_an_error(tmp_path, monkeypatch, capsys, no_repair):
    from qdrant_client import models
    db = tmp_path / "db"
    with local_client(str(db)) as client:
        client.create_collection("kb", vectors_config=models.VectorParams(size=32, distance=models.Distance.DOT))
    assert "management state" in guarded_inspection(db, monkeypatch, capsys, no_repair=no_repair, exit_code=2)


@pytest.mark.parametrize("review", [[], {}, {"policy": "invalid"}])
@pytest.mark.parametrize("no_repair", [False, True])
def test_malformed_legacy_review_is_an_error(tmp_path, monkeypatch, capsys, review, no_repair):
    db = tmp_path / "db"
    bootstrap(db)
    def change(state):
        state.pop("review_policy")
        state["last_review"] = review
    change_state(db, change)
    assert "policy" in guarded_inspection(db, monkeypatch, capsys, no_repair=no_repair, exit_code=2)


@pytest.mark.parametrize("limit", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_policy_is_refused_by_reused_validator(tmp_path, monkeypatch, capsys, limit):
    # Recovery serialization already refuses these, so inject at the read boundary.
    import ragwrench.sync.store as store_module
    db = tmp_path / "db"
    bootstrap(db)
    monkeypatch.setattr(store_module, "_state", lambda *a: {
        "version": 1, "review_policy": {"max_deletion_fraction": limit}})
    assert "policy" in guarded_inspection(db, monkeypatch, capsys, exit_code=2)


@pytest.mark.parametrize("limit", [0.0, 1.0])
def test_valid_explicit_limit_boundaries(tmp_path, monkeypatch, capsys, limit):
    db = tmp_path / "db"
    bootstrap(db)
    change_state(db, lambda s: s.update(review_policy={"max_deletion_fraction": limit}, last_review=[]))
    assert guarded_inspection(db, monkeypatch, capsys) == expected(limit=limit)


REPAIR_REQUIRED = ("ragwrench sync: primary metadata repair required; reopen without --no-repair "
                   "to attempt validated recovery\n")


@pytest.mark.parametrize("damage", ["missing", "truncated", "invalid-utf8"])
@pytest.mark.parametrize("sidecar", ["valid", "bad", "missing"])
def test_no_repair_refuses_damaged_primary_without_touching_files(tmp_path, damage, sidecar):
    db = tmp_path / "db"
    bootstrap(db)
    primary, recovery = db / "meta.json", db / "chunksync-recovery.json"
    if damage == "missing":
        primary.unlink()
    else:
        primary.write_bytes(b'{"collections":' if damage == "truncated" else b"\xff")
    if sidecar == "bad":
        recovery.write_text("broken sidecar")
    elif sidecar == "missing":
        recovery.unlink()
    # An absent lock must remain absent: damaged refusal happens before backend open.
    (db / ".lock").unlink(missing_ok=True)
    before = {p.relative_to(db): p.read_bytes() for p in db.rglob("*") if p.is_file()}
    result = cli(db, "kb", "--no-repair")
    assert result.returncode == 2
    assert not result.stdout
    assert result.stderr == ("ragwrench sync: inspect-policy requires an existing persistent Qdrant store\n"
                             if damage == sidecar == "missing" else REPAIR_REQUIRED)
    assert {p.relative_to(db): p.read_bytes() for p in db.rglob("*") if p.is_file()} == before
    # The opt-in refusal must leave default recovery available and unchanged.
    result = cli(db)
    if sidecar == "valid":
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == expected()
        assert json.loads(primary.read_text()) == json.loads(recovery.read_text())["metadata"]
        assert recovery.read_bytes() == before[recovery.relative_to(db)]
    else:
        assert result.returncode == 2 and not result.stdout
        assert "Traceback" not in result.stderr
        assert {p.relative_to(db): p.read_bytes() for p in db.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("sidecar", ["bad", "missing"])
def test_no_repair_healthy_primary_ignores_unusable_sidecar_and_keeps_lock_handling(tmp_path, sidecar):
    db = tmp_path / "db"
    bootstrap(db)
    recovery = db / "chunksync-recovery.json"
    if sidecar == "bad":
        recovery.write_bytes(b"\xff")
    else:
        recovery.unlink()
    (db / ".lock").unlink()
    before = {p.relative_to(db): p.read_bytes() for p in db.rglob("*") if p.is_file()}
    result = cli(db, "kb", "--no-repair")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected()
    assert not result.stderr
    assert (db / ".lock").is_file()
    after = {p.relative_to(db): p.read_bytes() for p in db.rglob("*") if p.is_file() and p.name != ".lock"}
    assert after == before


@pytest.mark.parametrize("case,diagnostic", [
    ("aliases", "aliases"), ("name", "collection must"), ("config", "configuration"),
    ("pickle", "pickle"), ("symlink", "symlink"), ("oversize", "64 MiB"),
    ("nonregular", "regular file"),
])
def test_no_repair_preserves_safe_store_preflight(tmp_path, case, diagnostic):
    import sqlite3
    db = tmp_path / "db"
    bootstrap(db)
    primary = db / "meta.json"
    metadata = json.loads(primary.read_text())
    if case == "aliases":
        metadata["aliases"] = {"alias": "kb"}
    elif case == "name":
        metadata["collections"]["../bad"] = metadata["collections"].pop("kb")
    elif case == "config":
        metadata["collections"]["kb"]["vectors"]["distance"] = "Cosine"
    elif case == "pickle":
        with sqlite3.connect(db / "collection" / "kb" / "storage.sqlite") as con:
            # Executable REDUCE opcode must be rejected before Qdrant unpickles.
            con.execute("UPDATE points SET point = ?", (b"cbuiltins\neval\n(V1+1\ntR.",))
    elif case == "symlink":
        (db / "link").symlink_to(primary)
    primary.write_text(json.dumps(metadata))
    if case == "oversize":
        from ragwrench.sync.core import MAX_JSON_BYTES
        with primary.open("r+b") as stream:
            stream.truncate(MAX_JSON_BYTES + 1)
    elif case == "nonregular":
        primary.unlink()
        primary.mkdir()
    (db / ".lock").unlink()
    # Hash sparse/oversized files without retaining their contents in the test.
    import hashlib
    def files():
        result = {}
        for p in db.rglob("*"):
            if p.is_file():
                with p.open("rb") as stream:
                    result[p.relative_to(db)] = hashlib.file_digest(stream, "sha256").hexdigest()
        return result
    before = files()
    result = cli(db, "kb", "--no-repair")
    assert result.returncode == 2 and not result.stdout
    assert diagnostic in result.stderr and "Traceback" not in result.stderr
    assert "repair required" not in result.stderr
    assert files() == before
    assert not (db / ".lock").exists()


def test_no_repair_propagates_metadata_io_error_without_open_or_fallback(tmp_path, monkeypatch, capsys):
    import ragwrench.sync.recovery as recovery
    import ragwrench.sync.store as store
    import ragwrench.sync.cli as cli_module
    db = tmp_path / "db"
    bootstrap(db)
    before = {p.relative_to(db): p.read_bytes() for p in db.rglob("*") if p.is_file()}
    def unreadable(path):
        assert path == db / "meta.json"  # no fallback read
        raise PermissionError("synthetic primary read refusal")
    def forbidden(*a, **kw):
        pytest.fail("must refuse before backend open or metadata write")
    monkeypatch.setattr(recovery, "load_json", unreadable)
    monkeypatch.setattr(recovery, "write_json", forbidden)
    monkeypatch.setattr(store, "QdrantClient", forbidden)
    assert cli_module.main(["inspect-policy", "--qdrant-path", str(db), "--no-repair"]) == 2
    output = capsys.readouterr()
    assert not output.out and output.err == "ragwrench sync: synthetic primary read refusal\n"
    assert {p.relative_to(db): p.read_bytes() for p in db.rglob("*") if p.is_file()} == before
