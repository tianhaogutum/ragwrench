"""Offline archived policy comparison: validate completely before reporting drift."""
import json
from copy import deepcopy

import pytest

from ragwrench.sync.cli import main
from .test_inventory import inventory_store


def report(rows=(), source="primary"):
    return {"kind": "chunksync.policy_inventory", "version": 1,
            "catalog_source": source, "collections": list(rows)}


def invoke(tmp_path, capsys, before, after):
    paths = [tmp_path / name for name in ("before.json", "after.json")]
    for path, value in zip(paths, (before, after)):
        path.write_text(json.dumps(value), encoding="utf-8")
    code = main(["compare-policies", *map(str, paths)])
    out = capsys.readouterr()
    return code, out


def test_equal_archives_without_opening_a_store(tmp_path, capsys):
    code, out = invoke(tmp_path, capsys, report(), report())
    assert code == 0
    assert not out.err
    assert json.loads(out.out) == {
        "kind": "chunksync.policy_comparison", "version": 1, "identical": True,
        "catalog_source": {"before": "primary", "after": "primary"},
        "added": [], "removed": [], "changed": []}


def ok(name="kb", limit=0.5, source="explicit", pending=None, override=None):
    return {"collection": name, "management_state": "managed", "status": "ok",
            "policy": {"kind": "chunksync.policy_inspection", "version": 1, "collection": name,
                       "max_deletion_fraction": limit, "policy_source": source,
                       "pending_plan_fingerprint": pending, "policy_override": override},
            "diagnostic": None}


def bad(name="kb", status="invalid", message=None):
    management, code, default = {
        "unmanaged": ("unmanaged", "unmanaged_collection", "collection has no management state"),
        "unsupported": ("unsupported", "unsupported_management_state", "unsupported collection management state"),
        "invalid": ("managed", "invalid_policy_metadata", "invalid collection review policy"),
        "unavailable": ("unknown", "unavailable_collection_metadata", "unsupported collection configuration"),
    }[status]
    return {"collection": name, "management_state": management, "status": status,
            "policy": None, "diagnostic": {"code": code, "message": message or default}}


def override(name="kb", reviewer="Ada", reason="approved exception"):
    from ragwrench.sync.core import digest
    value = {"kind": "chunksync.policy_override", "version": 1, "collection": name,
             "plan_fingerprint": "a" * 64, "from_policy": {"max_deletion_fraction": 0.2},
             "to_policy": {"max_deletion_fraction": 0.8},
             "provenance": {"reviewer": reviewer, "reason": reason}}
    value["fingerprint"] = digest(value)
    return value


def test_reordered_and_source_only_are_equal(tmp_path, capsys):
    rows = [ok("zeta", 1), bad("alpha", "unmanaged"), ok("middle", 1, "default")]
    before, after = report(rows), report(list(reversed(deepcopy(rows))), "recovery")
    after["collections"][-1]["policy"]["max_deletion_fraction"] = 1.0
    code, out = invoke(tmp_path, capsys, before, after)
    assert code == 0 and not out.err
    result = json.loads(out.out)
    assert result["identical"] and not any(result[k] for k in ("added", "removed", "changed"))
    assert result["catalog_source"] == {"before": "primary", "after": "recovery"}


@pytest.mark.parametrize("variant", ["limit", "source", "pending", "override", "reviewer", "reason", "message"])
def test_every_reported_policy_and_diagnostic_field_counts(tmp_path, capsys, variant):
    before, after = ok(), ok()
    if variant == "limit":
        after["policy"]["max_deletion_fraction"] = 0.25
    elif variant == "source":
        after["policy"]["policy_source"] = "legacy-review"
    elif variant == "pending":
        after["policy"]["pending_plan_fingerprint"] = "b" * 64
    elif variant == "override":
        after["policy"]["policy_override"] = override()
    elif variant in ("reviewer", "reason"):
        before["policy"]["policy_override"] = override()
        after["policy"]["policy_override"] = override(**{variant: "different"})
    else:
        before, after = bad(), bad(message="a different diagnostic")
    code, out = invoke(tmp_path, capsys, report([before]), report([after]))
    assert code == 10 and not out.err
    result = json.loads(out.out)
    assert not result["identical"] and result["added"] == result["removed"] == []
    assert result["changed"] == [{"collection": "kb", "before": before, "after": after}]


@pytest.mark.parametrize("old_status", ["ok", "unmanaged", "unsupported", "invalid", "unavailable"])
@pytest.mark.parametrize("new_status", ["ok", "unmanaged", "unsupported", "invalid", "unavailable"])
def test_all_diagnostic_transitions(tmp_path, capsys, old_status, new_status):
    old = ok() if old_status == "ok" else bad(status=old_status)
    new = ok() if new_status == "ok" else bad(status=new_status)
    code, out = invoke(tmp_path, capsys, report([old]), report([new], "recovery"))
    assert code == (0 if old_status == new_status else 10) and not out.err
    result = json.loads(out.out)
    assert result["changed"] == ([] if old_status == new_status else [{"collection": "kb", "before": old, "after": new}])


def test_mixed_sorted_full_rows_and_deterministic_bytes(tmp_path, capsys):
    from ragwrench.sync.policy_comparison import compare_policies, comparison_json
    old = [bad("z", "unavailable"), ok("keep"), ok("d"), bad("a", "unmanaged"), ok("q")]
    new = [ok("z"), ok("keep"), bad("d"), ok("c"), bad("b", "unsupported")]
    before, after = report(old), report(new, "recovery")
    code, out = invoke(tmp_path, capsys, before, after)
    assert code == 10 and not out.err
    result = json.loads(out.out)
    assert result["added"] == [new[4], new[3]]
    assert result["removed"] == [old[3], old[4]]
    assert result["changed"] == [{"collection": "d", "before": old[2], "after": new[2]},
                                  {"collection": "z", "before": old[0], "after": new[0]}]
    # Object order, collection order and number spelling cannot change bytes.
    def reorder(v):
        if isinstance(v, dict): return {k: reorder(v[k]) for k in reversed(v)}
        if isinstance(v, list): return [reorder(r) for r in reversed(v)]
        return v
    assert comparison_json(compare_policies(reorder(before), reorder(after))) == out.out
    assert out.out.endswith("\n") and not out.out.endswith("\n\n")


@pytest.mark.parametrize("name", ["", "../bad", "é", "with space", "nul\0name"])
def test_unavailable_identities_emitted_by_inventory_are_supported(tmp_path, capsys, name):
    row = bad(name, "unavailable", "invalid collection name")
    code, out = invoke(tmp_path, capsys, report(), report([row]))
    assert code == 10 and json.loads(out.out)["added"] == [row] and not out.err


# One mutation per invalid schema boundary; valid rows may precede the malformed row.
INVALID = [
    ((), None), (("kind",), "other"), (("version",), True), (("version",), 2),
    (("catalog_source",), "other"), (("collections",), {}),
    (("collections", 0), []), (("collections", 0, "collection"), []),
    (("collections", 0, "collection"), "bad/name"), (("collections", 0, "status"), []),
    (("collections", 0, "status"), "other"), (("collections", 0, "management_state"), "unmanaged"),
    (("collections", 0, "diagnostic"), {"code": "x", "message": "y"}),
    (("collections", 0, "policy"), None),
    *(( ("collections", 0, "policy", key), value) for key, value in [
        ("kind", "other"), ("version", True), ("version", 2), ("collection", "other"),
        ("max_deletion_fraction", True), ("max_deletion_fraction", -0.1),
        ("max_deletion_fraction", 1.1), ("max_deletion_fraction", "0.5"),
        ("max_deletion_fraction", None), ("policy_source", "other"), ("policy_source", "default"),
        ("pending_plan_fingerprint", "A" * 64), ("pending_plan_fingerprint", "x"),
        ("pending_plan_fingerprint", {"plan_fingerprint": "a" * 64}),
        ("policy_override", {}), ("policy_override", [])]),
]


def mutate(value, path, replacement):
    if not path: return replacement
    target = value
    for key in path[:-1]: target = target[key]
    target[path[-1]] = replacement
    return value


@pytest.mark.parametrize("path,replacement", INVALID)
@pytest.mark.parametrize("side", ["before", "after"])
def test_rejects_invalid_archive_completely(tmp_path, capsys, path, replacement, side):
    invalid = mutate(report([ok()]), path, replacement)
    before, after = (invalid, report()) if side == "before" else (report(), invalid)
    code, out = invoke(tmp_path, capsys, before, after)
    assert code == 2 and not out.out and out.err.startswith("ragwrench sync: ")
    assert "Traceback" not in out.err


@pytest.mark.parametrize("level", ["report", "row", "policy", "diagnostic", "override"])
@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_exact_keys(tmp_path, capsys, level, mutation):
    row = bad() if level == "diagnostic" else ok(override=override())
    value = report([row])
    obj = {"report": value, "row": row, "policy": row["policy"],
           "diagnostic": row["diagnostic"], "override": row["policy"]["policy_override"] if row["policy"] else None}[level]
    if mutation == "extra": obj["extra"] = None
    else: obj.pop(next(iter(obj)))
    code, out = invoke(tmp_path, capsys, report(), value)
    assert code == 2 and not out.out and out.err


@pytest.mark.parametrize("status", ["unmanaged", "unsupported", "invalid", "unavailable"])
@pytest.mark.parametrize("mutation", ["policy", "management", "code", "message-type", "message-empty", "diagnostic-null"])
def test_diagnostic_consistency(tmp_path, capsys, status, mutation):
    row = bad(status=status)
    if mutation == "policy": row["policy"] = ok()["policy"]
    elif mutation == "management": row["management_state"] = "wrong"
    elif mutation == "code": row["diagnostic"]["code"] = "wrong"
    elif mutation == "message-type": row["diagnostic"]["message"] = []
    elif mutation == "message-empty": row["diagnostic"]["message"] = "  "
    else: row["diagnostic"] = None
    code, out = invoke(tmp_path, capsys, report(), report([ok("first"), row]))
    assert code == 2 and not out.out and out.err


@pytest.mark.parametrize("field,value", [("collection", "other"), ("fingerprint", "0" * 64),
    ("plan_fingerprint", "X" * 64), ("provenance", {"reviewer": "", "reason": "x"}),
    ("from_policy", {"max_deletion_fraction": True}), ("to_policy", {"max_deletion_fraction": 0.1}),
    ("version", True)])
def test_override_tampering_refused(tmp_path, capsys, field, value):
    record = override(); record[field] = value
    code, out = invoke(tmp_path, capsys, report(), report([ok(override=record)]))
    assert code == 2 and not out.out and out.err


@pytest.mark.parametrize("row", [ok(), bad(), bad("bad/name", "unavailable")])
def test_duplicate_collection_identities_refused(tmp_path, capsys, row):
    code, out = invoke(tmp_path, capsys, report(), report([row, deepcopy(row)]))
    assert code == 2 and not out.out and "duplicate collection" in out.err


@pytest.mark.parametrize("raw", [b'{', b'\xff', b'null', b'[]', b'{"version":1,"version":1}',
    b'{"x":{"x":1,"x":2}}', b'NaN', b'Infinity', b'-Infinity', b'[' * 2000 + b']' * 2000,
    b'{"x":' + b'9' * 5000 + b'}',
    json.dumps(report([ok(limit=float('inf'))])).encode(),
    json.dumps(report([ok(limit=float('nan'))])).encode(),
    json.dumps(report([bad(message='bad\ud800')])).encode(),
    json.dumps(report([ok()])).replace('0.5', '1e999').encode()])
@pytest.mark.parametrize("side", [0, 1])
def test_malformed_json_no_stdout(tmp_path, capsys, raw, side):
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path in paths: path.write_text(json.dumps(report()))
    paths[side].write_bytes(raw)
    code = main(["compare-policies", *map(str, paths)])
    out = capsys.readouterr()
    assert code == 2 and not out.out and out.err and "Traceback" not in out.err


@pytest.mark.parametrize("case", ["missing", "directory", "oversize", "count", "unreadable"])
@pytest.mark.parametrize("side", [0, 1])
def test_file_and_count_bounds(tmp_path, capsys, monkeypatch, case, side):
    from pathlib import Path
    from ragwrench.sync.core import MAX_JSON_BYTES
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path in paths: path.write_text(json.dumps(report()))
    path = paths[side]
    if case == "missing": path.unlink()
    elif case == "directory": path.unlink(); path.mkdir()
    elif case == "oversize":
        with path.open("wb") as f: f.truncate(MAX_JSON_BYTES + 1)
    elif case == "count": path.write_text(json.dumps(report([ok(f"c{i}") for i in range(1001)])))
    else:
        real = Path.open
        def denied(self, *a, **kw):
            if self == path: raise PermissionError("synthetic unreadable archive")
            return real(self, *a, **kw)
        monkeypatch.setattr(Path, "open", denied)
    code = main(["compare-policies", *map(str, paths)])
    out = capsys.readouterr()
    assert code == 2 and not out.out and out.err


def test_exact_count_boundary_and_python_validation():
    from ragwrench.sync.policy_comparison import compare_policies
    from ragwrench.sync.core import SyncError
    value = report([ok(f"c{i}") for i in range(1000)])
    assert compare_policies(value, value)["identical"]
    with pytest.raises(SyncError): compare_policies(value, report([ok(), ok()]))


def test_output_limit_no_partial_stdout(tmp_path, capsys, monkeypatch):
    import ragwrench.sync.policy_comparison as comparison
    # Each input fits, but complete before/after output must not be truncated.
    before = report([bad(message="a" * 800)])
    after = report([bad(message="b" * 800)])
    limit = max(len(comparison.comparison_json(v).encode()) for v in (before, after))
    monkeypatch.setattr(comparison, "MAX_JSON_BYTES", limit)
    code, out = invoke(tmp_path, capsys, before, after)
    assert code == 2 and not out.out and out.err


def test_reads_only_two_files_without_backend_or_mutation(tmp_path, capsys, monkeypatch):
    import io
    import sqlite3
    import ragwrench.sync.cli as cli
    import ragwrench.sync.inventory as inventory
    import ragwrench.sync.recovery as recovery
    import ragwrench.sync.store as store
    from ragwrench.sync.embedders import HashEmbedder
    before, after = tmp_path / "a.json", tmp_path / "b.json"
    before.write_text(json.dumps(report([ok()])))
    after.write_text(json.dumps(report([ok(limit=0.25)], "recovery")))
    files = {p: p.read_bytes() for p in tmp_path.iterdir()}
    real_open = io.open
    reads = []
    def only_archives(path, mode="r", *a, **kw):
        assert path in (before, after) and mode == "rb"
        reads.append(path)
        return real_open(path, mode, *a, **kw)
    def forbidden(*a, **kw): pytest.fail("comparison must only read the two archives")
    with monkeypatch.context() as guard:
        guard.setattr(io, "open", only_archives)
        for obj, name in [(cli, "local_client"), (cli, "make_embedder"), (cli, "list_policies"),
                          (store, "QdrantClient"), (sqlite3, "connect"), (HashEmbedder, "embed"),
                          (inventory, "load_json"), (recovery, "recovery_catalog")]:
            guard.setattr(obj, name, forbidden)
        assert main(["compare-policies", str(before), str(after)]) == 10
    out = capsys.readouterr()
    assert not out.err and json.loads(out.out)["changed"]
    assert reads == [before, after]
    assert files == {p: p.read_bytes() for p in tmp_path.iterdir()}


def test_real_archives_include_all_statuses_and_in_memory_recovery(inventory_store, tmp_path, capsys):
    from ragwrench.sync.core import digest
    from .test_inventory import files
    primary = inventory_store / "meta.json"
    catalog = json.loads(primary.read_text())
    healthy = catalog["collections"]["alpha"]
    invalid, unsupported = deepcopy(healthy), deepcopy(healthy)
    invalid["metadata"]["chunksync"]["review_policy"] = {}
    unsupported["metadata"]["chunksync"]["version"] = 2
    catalog["collections"].update(invalid=invalid, unsupported=unsupported, unavailable=[], **{"bad/name": []})
    primary.write_text(json.dumps(catalog))
    paths = [tmp_path / "primary.json", tmp_path / "recovered.json"]
    assert main(["list-policies", "--qdrant-path", str(inventory_store), "--output", str(paths[0]), "--fail-on-diagnostics"]) == 3
    capsys.readouterr()
    saved = json.loads((inventory_store / "chunksync-recovery.json").read_text())
    saved["metadata"] = catalog
    saved["fingerprint"] = digest({k:v for k,v in saved.items() if k != "fingerprint"})
    (inventory_store / "chunksync-recovery.json").write_text(json.dumps(saved))
    primary.write_bytes(b'{')
    assert main(["list-policies", "--qdrant-path", str(inventory_store), "--output", str(paths[1])]) == 0
    capsys.readouterr()
    store_before = files(inventory_store)
    # Removing the entire source store is unnecessary: assert its bytes unchanged.
    assert main(["compare-policies", *map(str, paths)]) == 0
    out = capsys.readouterr()
    assert not out.err and json.loads(out.out)["identical"]
    assert json.loads(out.out)["catalog_source"] == {"before": "primary", "after": "recovery"}
    assert files(inventory_store) == store_before


@pytest.mark.parametrize("status", ["unmanaged", "unsupported"])
def test_fixed_diagnostic_message_consistency(tmp_path, capsys, status):
    code, out = invoke(tmp_path, capsys, report(), report([bad(status=status, message="forged message")]))
    assert code == 2 and not out.out and out.err


def test_input_byte_boundary_and_python_api_bound(tmp_path, monkeypatch):
    import ragwrench.sync.policy_comparison as comparison
    from ragwrench.sync.core import SyncError
    value = report()
    raw = comparison.comparison_json(value).encode()
    path = tmp_path / "archive.json"
    path.write_bytes(raw)
    monkeypatch.setattr(comparison, "MAX_JSON_BYTES", len(raw))
    assert comparison.load_inventory(path) == value
    assert comparison.validate_inventory(value) == {}
    path.write_bytes(raw + b' ')
    with pytest.raises(SyncError, match="64 MiB"):
        comparison.load_inventory(path)
    with pytest.raises(SyncError, match="64 MiB"):
        comparison.validate_inventory(report([ok()]))


def test_default_and_retained_override_with_later_pending_plan(tmp_path, capsys):
    row = ok(limit=1, source="default", override=override(), pending="b" * 64)
    code, out = invoke(tmp_path, capsys, report([row]), report([deepcopy(row)]))
    assert code == 0 and not out.err
    # Legacy and explicit retained overrides can report a stricter effective limit.
    row["policy"].update(policy_source="legacy-review", max_deletion_fraction=0.1)
    code, out = invoke(tmp_path, capsys, report([row]), report())
    assert code == 10 and json.loads(out.out)["removed"] == [row]
