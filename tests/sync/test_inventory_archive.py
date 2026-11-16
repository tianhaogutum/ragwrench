"""Atomic external archive contracts for metadata-only policy inventories."""
import json
import os
from pathlib import Path

import pytest

from ragwrench.sync.cli import main
from .test_inventory import edit_catalog, files, inventory_store


def test_archive_exact_stdout_two_managed_one_unmanaged(inventory_store, tmp_path, capsys):
    archive = tmp_path / "report.json"
    before = files(inventory_store)
    assert main(["list-policies", "--qdrant-path", str(inventory_store), "--output", str(archive)]) == 0
    out = capsys.readouterr()
    assert not out.err
    assert archive.read_bytes() == out.out.encode("utf-8")
    report = json.loads(out.out)
    assert [r["management_state"] for r in report["collections"]] == ["managed", "unmanaged", "managed"]
    assert files(inventory_store) == before


def invoke(db, archive, capsys, *, no_repair=False, fail_on_diagnostics=False, expected=0):
    before = files(db)
    assert main(["list-policies", "--qdrant-path", str(db), "--output", str(archive),
                 *(["--fail-on-diagnostics"] if fail_on_diagnostics else []),
                 *(["--no-repair"] if no_repair else [])]) == expected
    out = capsys.readouterr()
    assert files(db) == before
    if expected in (0, 3):
        assert not out.err
        assert archive.read_bytes() == out.out.encode("utf-8")
        assert out.out.endswith("\n") and not out.out.endswith("\n\n")
        return json.loads(out.out)
    assert not out.out and out.err.startswith("ragwrench sync: ") and "Traceback" not in out.err
    return out.err


@pytest.mark.parametrize("case", ["explicit", "legacy-review", "default", "partial", "recovery", "unicode"])
@pytest.mark.parametrize("no_repair", [False, True])
def test_archive_existing_report_and_policy_variants(inventory_store, tmp_path, monkeypatch, capsys, case, no_repair):
    import ragwrench.sync.cli as cli
    import ragwrench.sync.inventory as inventory
    import sqlite3
    if case in ("legacy-review", "default", "partial", "unicode"):
        def change(m):
            state = m["collections"]["alpha"]["metadata"]["chunksync"]
            if case == "partial":
                state["review_policy"] = {}
            elif case == "unicode":
                from ragwrench.sync.core import build_snapshot, make_plan
                from ragwrench.sync.policy import policy_override
                state["last_policy_override"] = policy_override(
                    make_plan(build_snapshot([]), build_snapshot([])), "alpha",
                    from_max_deletion_fraction=0.1, max_deletion_fraction=0.5,
                    reviewer="Prüfer", reason="Überprüfung 合成")
            else:
                state.pop("review_policy")
                if case == "default":
                    state.pop("last_review")
        edit_catalog(inventory_store, change)
    if case == "recovery":
        (inventory_store / "meta.json").write_text("{")
    (inventory_store / ".lock").unlink()
    archive = tmp_path / "report.json"
    archive.write_bytes(b"previous report\n")
    original = inventory.load_json
    def catalog_only(path):
        assert path.parent == inventory_store
        assert path.name in ("meta.json", "chunksync-recovery.json")
        return original(path)
    def forbidden(*a, **kw):
        pytest.fail("archive must not open backend/SQLite or embed")
    monkeypatch.setattr(inventory, "load_json", catalog_only)
    monkeypatch.setattr(cli, "local_client", forbidden)
    monkeypatch.setattr(cli, "make_embedder", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    if case == "recovery" and no_repair:
        assert "primary metadata repair required" in invoke(inventory_store, archive, capsys, no_repair=True, expected=2)
        assert archive.read_bytes() == b"previous report\n"
    else:
        report = invoke(inventory_store, archive, capsys, no_repair=no_repair)
        assert report["catalog_source"] == ("recovery" if case == "recovery" else "primary")
        row = report["collections"][0]
        if case in ("legacy-review", "default", "explicit"):
            assert row["policy"]["policy_source"] == case
        if case == "partial":
            assert [r["status"] for r in report["collections"]] == ["invalid", "unmanaged", "ok"]
        if case == "unicode":
            assert "Überprüfung 合成" in archive.read_text()
    assert not (inventory_store / ".lock").exists()
    assert not list(tmp_path.glob(".chunksync-inventory-*"))


@pytest.mark.parametrize("relative", [".", "meta.json", "chunksync-recovery.json", ".lock",
                                      "collection/alpha/storage.sqlite", "new-report.json", "sub/../new-report.json"])
def test_refuses_all_store_destinations_before_inventory(inventory_store, monkeypatch, capsys, relative):
    import ragwrench.sync.cli as cli
    before = files(inventory_store)
    def forbidden(*a, **kw):
        pytest.fail("unsafe destination must refuse before inventory")
    monkeypatch.setattr(cli, "list_policies", forbidden)
    target = inventory_store / relative
    assert "outside the Qdrant store" in invoke(inventory_store, target, capsys, expected=2)
    assert files(inventory_store) == before


@pytest.mark.parametrize("case", ["file-into-store", "directory-into-store", "external-file", "external-directory",
                                  "dangling", "ancestor-dotdot", "store-alias"])
def test_refuses_symlink_destinations_and_ancestors(inventory_store, tmp_path, monkeypatch, capsys, case):
    import ragwrench.sync.cli as cli
    old = tmp_path / "old-report.json"
    old.write_bytes(b"prior archive")
    link = tmp_path / "link"
    if case == "file-into-store":
        link.symlink_to(inventory_store / "meta.json")
        target = link
    elif case == "external-file":
        link.symlink_to(old)
        target = link
    elif case == "dangling":
        link.symlink_to(tmp_path / "absent")
        target = link
    else:
        link.symlink_to(inventory_store if case in ("directory-into-store", "store-alias") else tmp_path,
                        target_is_directory=True)
        target = link / ("meta.json" if case == "store-alias" else "report.json")
        if case == "ancestor-dotdot":
            target = link / ".." / "old-report.json"
    def forbidden(*a, **kw):
        pytest.fail("symlink destination must refuse before inventory")
    monkeypatch.setattr(cli, "list_policies", forbidden)
    assert "may not traverse symlinks" in invoke(inventory_store, target, capsys, expected=2)
    assert old.read_bytes() == b"prior archive"
    assert link.is_symlink()


@pytest.mark.parametrize("case", ["missing-parent", "file-parent", "directory", "fifo", "null-byte", "empty"])
def test_invalid_output_paths_do_not_create_directories(inventory_store, tmp_path, capsys, case):
    target = tmp_path / "report.json"
    if case == "missing-parent":
        target = tmp_path / "absent" / "report.json"
    elif case == "file-parent":
        target.write_text("parent is a file")
        target = target / "report.json"
    elif case == "directory":
        target.mkdir()
    elif case == "fifo":
        os.mkfifo(target)
    elif case == "null-byte":
        target = str(target) + "\0"
    else:
        target = ""
    before = set(tmp_path.iterdir())
    assert main(["list-policies", "--qdrant-path", str(inventory_store), "--output", str(target)]) == 2
    out = capsys.readouterr()
    assert not out.out and out.err.startswith("ragwrench sync:")
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("case", ["semantic", "bad-recovery", "count", "input-size", "output-size", "missing-store"])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("fail_on_diagnostics", [False, True])
def test_inventory_refusals_preserve_archive(inventory_store, tmp_path, monkeypatch, capsys, case, existing, fail_on_diagnostics):
    import ragwrench.sync.inventory as inventory
    if case == "semantic":
        (inventory_store / "meta.json").write_text("[]")
    elif case == "bad-recovery":
        (inventory_store / "meta.json").write_text("{")
        (inventory_store / "chunksync-recovery.json").write_text("{}")
    elif case == "count":
        monkeypatch.setattr(inventory, "MAX_COLLECTIONS", 2)
    elif case == "input-size":
        import ragwrench.sync.core as core
        monkeypatch.setattr(core, "MAX_JSON_BYTES", 1)
    elif case == "output-size":
        monkeypatch.setattr(inventory, "MAX_JSON_BYTES", 1)
    else:
        inventory_store = tmp_path / "missing-store"
    archive = tmp_path / "report.json"
    if existing:
        archive.write_bytes(b"old archive")
    invoke(inventory_store, archive, capsys, fail_on_diagnostics=fail_on_diagnostics, expected=2)
    assert archive.read_bytes() == b"old archive" if existing else not archive.exists()
    assert not list(tmp_path.glob(".chunksync-inventory-*"))
    if case == "missing-store":
        assert not inventory_store.exists()


@pytest.mark.parametrize("phase", ["create", "write", "flush", "fsync", "close", "replace"])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("fail_on_diagnostics", [False, True])
def test_publication_failures_preserve_old_report(inventory_store, tmp_path, monkeypatch, capsys, phase, existing, fail_on_diagnostics):
    import ragwrench.sync.inventory as inventory
    archive = tmp_path / "report.json"
    if existing:
        archive.write_bytes(b"old archive")
    real_temp = inventory.tempfile.NamedTemporaryFile
    real_sync = os.fsync
    def fail():
        raise OSError("synthetic archive " + phase + " refusal")
    class FaultyFile:
        def __init__(self, stream):
            self.stream = stream
            self.name = stream.name
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.__exit__(*args)
            if phase == "close":
                fail()
        def write(self, data):
            if phase == "write":
                self.stream.write(data[:7])
                fail()
            return self.stream.write(data)
        def flush(self):
            self.stream.flush()
            if phase == "flush":
                fail()
        def fileno(self):
            return self.stream.fileno()
    def create(*a, **kw):
        if phase == "create":
            fail()
        return FaultyFile(real_temp(*a, **kw))
    def sync(fd):
        real_sync(fd)
        if phase == "fsync":
            fail()
    def replace(src, dst):
        assert Path(src).parent == archive.parent
        assert Path(src).read_bytes().endswith(b"\n")
        assert archive.read_bytes() == b"old archive" if existing else not archive.exists()
        fail()
    monkeypatch.setattr(inventory.tempfile, "NamedTemporaryFile", create)
    monkeypatch.setattr(inventory.os, "fsync", sync)
    if phase == "replace":
        monkeypatch.setattr(inventory.os, "replace", replace)
    assert "synthetic archive " + phase in invoke(inventory_store, archive, capsys, fail_on_diagnostics=fail_on_diagnostics, expected=2)
    assert archive.read_bytes() == b"old archive" if existing else not archive.exists()
    assert not list(tmp_path.glob(".chunksync-inventory-*"))


def test_atomic_replace_follows_complete_closed_synced_temp(inventory_store, tmp_path, monkeypatch, capsys):
    import ragwrench.sync.inventory as inventory
    archive = tmp_path / "report.json"
    archive.write_bytes(b"old report")
    real_temp, real_sync, real_replace = inventory.tempfile.NamedTemporaryFile, os.fsync, os.replace
    streams, events = [], []
    def create(*a, **kw):
        stream = real_temp(*a, **kw)
        streams.append(stream)
        return stream
    def sync(fd):
        with open(streams[0].name, "rb") as reader:
            assert reader.read(1) == b"{"  # flushed before sync
        real_sync(fd)
        events.append("sync")
    def replace(src, dst):
        assert events == ["sync"]
        assert streams[0].closed
        assert Path(src).parent == archive.parent
        assert archive.read_bytes() == b"old report"
        payload = Path(src).read_bytes()
        assert json.loads(payload)["kind"] == "chunksync.policy_inventory"
        real_replace(src, dst)
        assert archive.read_bytes() == payload
        events.append("replace")
    monkeypatch.setattr(inventory.tempfile, "NamedTemporaryFile", create)
    monkeypatch.setattr(inventory.os, "fsync", sync)
    monkeypatch.setattr(inventory.os, "replace", replace)
    invoke(inventory_store, archive, capsys)
    assert events == ["sync", "replace"]
    assert not list(tmp_path.glob(".chunksync-inventory-*"))
