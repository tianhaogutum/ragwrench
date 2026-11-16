"""Optional diagnostic exit changes status, never the complete inventory."""
import json
from copy import deepcopy

import pytest

from ragwrench.sync.cli import main
from ragwrench.sync.core import digest
from .test_inventory import files, inventory_store
from .conftest import linux_durable

pytestmark = linux_durable


def test_fail_on_diagnostics_keeps_complete_report(inventory_store, capsys):
    argv = ["list-policies", "--qdrant-path", str(inventory_store)]
    assert main(argv) == 0
    default = capsys.readouterr()
    assert main([*argv, "--fail-on-diagnostics"]) == 3
    strict = capsys.readouterr()
    assert strict == default
    assert not strict.err
    assert [r["status"] for r in json.loads(strict.out)["collections"]] == ["ok", "unmanaged", "ok"]


@pytest.mark.parametrize("case,statuses", [
    ("empty", []), ("healthy", ["ok", "ok"]),
    ("unmanaged", ["unmanaged"]), ("unsupported", ["unsupported"]),
    ("invalid", ["invalid"]), ("unavailable", ["unavailable"]),
    ("mixed", ["ok", "unmanaged", "unsupported", "invalid", "unavailable", "ok"]),
    ("last-bad", ["ok", "unmanaged"]),
])
@pytest.mark.parametrize("recovered,no_repair", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("archive_mode", ["none", "new", "existing"])
def test_diagnostic_exit_and_exact_output(inventory_store, tmp_path, monkeypatch, capsys,
                                          case, statuses, recovered, no_repair, archive_mode):
    import sqlite3
    import ragwrench.sync.cli as cli
    import ragwrench.sync.inventory as inventory
    import ragwrench.sync.store as store

    primary = inventory_store / "meta.json"
    catalog = json.loads(primary.read_text())
    healthy = catalog["collections"]["alpha"]
    variants = {"ok": healthy, "unmanaged": catalog["collections"]["middle"],
                "unsupported": deepcopy(healthy), "invalid": deepcopy(healthy), "unavailable": []}
    variants["unsupported"]["metadata"]["chunksync"] = {"version": 2}
    variants["invalid"]["metadata"]["chunksync"]["review_policy"] = {}
    # Insert in reverse order to prove sorted, complete rows (including bad last).
    catalog["collections"] = {f"c{i}": variants[status] for i, status in reversed(list(enumerate(statuses)))}
    if recovered:
        sidecar = inventory_store / "chunksync-recovery.json"
        saved = json.loads(sidecar.read_text())
        saved["metadata"] = catalog
        saved["fingerprint"] = digest({k: v for k, v in saved.items() if k != "fingerprint"})
        sidecar.write_text(json.dumps(saved))
        primary.write_bytes(b"\xff")
    else:
        primary.write_text(json.dumps(catalog))
    (inventory_store / ".lock").unlink()
    before = files(inventory_store)
    archive = tmp_path / "report.json"
    previous = b"previous archive\n" if archive_mode == "existing" else None
    if previous is not None:
        archive.write_bytes(previous)
    original = inventory.load_json
    reads = []
    def catalog_only(path):
        assert path.parent == inventory_store
        assert path.name in ("meta.json", "chunksync-recovery.json")
        reads.append(path.name)
        return original(path)
    def forbidden(*a, **kw):
        pytest.fail("diagnostic exit must not open backend/SQLite or embed")
    monkeypatch.setattr(inventory, "load_json", catalog_only)
    monkeypatch.setattr(cli, "local_client", forbidden)
    monkeypatch.setattr(cli, "make_embedder", forbidden)
    monkeypatch.setattr(store, "QdrantClient", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    argv = ["list-policies", "--qdrant-path", str(inventory_store)]
    if no_repair:
        argv.append("--no-repair")
    if archive_mode != "none":
        argv.extend(["--output", str(archive)])
    refused = recovered and no_repair
    assert main(argv) == (2 if refused else 0)
    default = capsys.readouterr()
    assert main([*argv, "--fail-on-diagnostics"]) == (2 if refused else 3 if any(s != "ok" for s in statuses) else 0)
    strict = capsys.readouterr()
    assert strict == default
    if refused:
        assert not strict.out and "primary metadata repair required" in strict.err
        assert reads == ["meta.json", "meta.json"]
        assert (archive.read_bytes() if archive.exists() else None) == previous
    else:
        assert not strict.err
        report = json.loads(strict.out)
        assert set(report) == {"kind", "version", "catalog_source", "collections"}
        assert report["catalog_source"] == ("recovery" if recovered else "primary")
        assert [r["status"] for r in report["collections"]] == statuses
        assert [r["collection"] for r in report["collections"]] == [f"c{i}" for i in range(len(statuses))]
        assert all((r["policy"] is None and r["diagnostic"]) if r["status"] != "ok"
                   else (r["policy"] is not None and r["diagnostic"] is None) for r in report["collections"])
        if archive_mode != "none":
            assert archive.read_bytes() == strict.out.encode("utf-8")
        assert strict.out.endswith("\n") and not strict.out.endswith("\n\n")
    assert files(inventory_store) == before
    assert not (inventory_store / ".lock").exists()
    assert not list(tmp_path.glob(".chunksync-inventory-*"))
