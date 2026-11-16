"""Document intent filtering changes detail arrays only, after full validation."""
from copy import deepcopy
import hashlib
import json

import pytest

from ragwrench.port.bundle import fingerprint, load_bundle, write_bundle
from ragwrench.port.cli import main
from ragwrench.port.common import PortError, content_hash, locked
from ragwrench.port.status import import_status
from .test_import_status_bundle import files_snapshot, forbid_io, interrupted

STAGES = ("document_creation", "bootstrap_cleanup", "chunk_creation", "enabled_flag")


@pytest.fixture
def mixed(interrupted, bundle, tmp_path):
    manifest, docs, chunks = load_bundle(bundle)
    documents, records, saved = [], [], {}
    phases = ["creating", "creating", "clearing", "clearing", "create-chunk", "create-chunk",
              "flag", "flag", "new", "chunks", "complete", "skipped", "unvisited", "empty"]
    for i, phase in reversed(list(enumerate(phases))):
        source = f"d{i:02}"
        documents.append({**deepcopy(docs[0]), "id": source})
        if phase != "empty":
            for j in range(2):
                content = "x" * 100 + str(j) if phase == "skipped" else f"chunk {j}"
                records.append({**deepcopy(chunks[j]), "id": f"c{j}", "doc_id": source,
                                "content": content, "content_sha256": content_hash(content)})
        if phase == "unvisited":
            continue
        saved[source] = {
            "id": None if phase in ("creating", "new", "empty") else f"t-{source}",
            "phase": "chunks" if phase in ("create-chunk", "flag") else
                     "complete" if phase == "skipped" else "new" if phase == "empty" else phase,
            "chunks": {"c0": "t-c0", "c1": "t-c1"} if phase == "complete" else
                      {"c0": "t-c0"} if phase in ("create-chunk", "flag", "chunks") else {},
            "pending": {"source": "c1", "target": "t-c1" if phase == "flag" else None}
                       if phase in ("create-chunk", "flag") else None,
        }
    manifest = write_bundle(bundle, manifest["source"], manifest["settings"], documents, records)
    interrupted["identity"].update(bundle=fingerprint(manifest), max_chunk_bytes=20, on_error="skip")
    interrupted["documents"] = saved
    path = tmp_path / "state.json"
    path.write_text(json.dumps(interrupted))
    return path, bundle, interrupted


def assert_filter(path, bundle, capsys, stage, full, ids):
    expected = deepcopy(full)
    expected["documents"] = [row for row in full["documents"] if row["source_id"] in ids]
    if bundle is not None:
        expected["bundle_progress"]["documents"] = [row for row in full["bundle_progress"]["documents"]
                                                      if row["source_id"] in ids]
    args = ["import-status", "--state", str(path), "--pending-stage", stage]
    if bundle is not None:
        args += ["--bundle", str(bundle)]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert not captured.err and json.loads(captured.out) == expected
    assert import_status(path, bundle, pending_stage=stage) == expected
    assert [r["source_id"] for r in expected["documents"]] == ids


@pytest.mark.parametrize("has_bundle", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_pending_stage_mixed(mixed, capsys, request, monkeypatch, has_bundle, legacy):
    path, bundle, state = mixed
    if legacy:
        del state["identity"]["api_profile"]
        path.write_text(json.dumps(state))
    # A concurrently held writer lock does not block an offline snapshot.
    with locked(path):
        request.getfixturevalue("forbid_io")
        if not has_bundle:
            bundle = None
            monkeypatch.setattr("ragwrench.port.status.load_bundle", lambda *a: pytest.fail("state-only loaded bundle"))
        before = files_snapshot(path.parent), set(path.parent.rglob("*"))
        full = import_status(path, bundle)
        assert import_status(path, bundle, pending_stage=None) == full
        assert full["mapped_documents"] == 9 and full["completed_chunks"] == 7
        counted = import_status(path, bundle, summary_only=True, pending_counts=True)["pending_counts"]
        for i, stage in enumerate(STAGES):
            ids = [f"d{2*i:02}", f"d{2*i+1:02}"]
            assert counted[stage] == len(ids) == 2
            assert [r["source_id"] for r in full["documents"] if r["pending_stage"] == stage] == ids
            assert_filter(path, bundle, capsys, stage, full, ids)
        assert (files_snapshot(path.parent), set(path.parent.rglob("*"))) == before


@pytest.mark.parametrize("boundary", ["no-match", "initial", "dataset-intent", "empty-bundle"])
@pytest.mark.parametrize("has_bundle", [False, True])
def test_pending_stage_no_matches(mixed, capsys, request, boundary, has_bundle):
    path, bundle, state = mixed
    state["documents"] = {k: v for k, v in state["documents"].items() if k >= "d08"}
    if boundary != "no-match":
        state.update(documents={}, dataset=None, dataset_pending=boundary == "dataset-intent")
    if boundary == "empty-bundle":
        manifest, _, _ = load_bundle(bundle)
        manifest = write_bundle(bundle, manifest["source"], manifest["settings"], [], [])
        state["identity"]["bundle"] = fingerprint(manifest)
    path.write_text(json.dumps(state))
    request.getfixturevalue("forbid_io")
    full = import_status(path, bundle if has_bundle else None)
    before = files_snapshot(path.parent)
    for stage in STAGES:
        assert_filter(path, bundle if has_bundle else None, capsys, stage, full, [])
    assert files_snapshot(path.parent) == before


@pytest.mark.parametrize("has_bundle", [False, True])
@pytest.mark.parametrize("extra,options", [
    (["--summary-only"], {"summary_only": True}),
    (["--pending-counts"], {"pending_counts": True}),
    (["--summary-only", "--pending-counts"], {"summary_only": True, "pending_counts": True}),
    (["--document", "d"], {"document": "d"}),
    (["--document", ""], {"document": ""}),
    (["--chunk", "c"], {"chunk": "c"}),
    (["--chunk", ""], {"chunk": ""}),
    (["--list-chunks"], {"list_chunks": True}),
    (["--chunk-status", "pending"], {"chunk_status": "pending"}),
    (["--page-size", "1"], {"page_size": 1}),
    (["--cursor", ""], {"cursor": ""}),
    (["--document", "d", "--list-chunks", "--page-size", "1", "--cursor", "token"],
     {"document": "d", "list_chunks": True, "page_size": 1, "cursor": "token"}),
])
def test_pending_stage_conflicts(tmp_path, capsys, monkeypatch, forbid_io, has_bundle, extra, options):
    def forbidden(*a, **k):
        pytest.fail("argument error reached input reads")
    for name in ("read_json", "read_json_snapshot", "load_bundle"):
        monkeypatch.setattr(f"ragwrench.port.status.{name}", forbidden)
    path = tmp_path / "absent" / "state.json"
    args = ["import-status", "--state", str(path), "--pending-stage", "enabled_flag"]
    if has_bundle:
        args += ["--bundle", str(tmp_path / "bundle")]
    with pytest.raises(SystemExit) as exc:
        main(args + extra)
    captured = capsys.readouterr()
    assert exc.value.code == 2 and not captured.out and "--pending-stage cannot be combined" in captured.err
    with pytest.raises(PortError, match="Pending stage cannot be combined"):
        import_status(path, tmp_path if has_bundle else None, pending_stage="enabled_flag", **options)
    assert not path.parent.exists()


@pytest.mark.parametrize("stage", ["dataset_creation", "", "pending", "Enabled_flag", " enabled_flag", 0, True, [], {}])
def test_pending_stage_invalid(tmp_path, capsys, monkeypatch, forbid_io, stage):
    def forbidden(*a, **k):
        pytest.fail("invalid stage reached input reads")
    for name in ("read_json", "read_json_snapshot", "load_bundle"):
        monkeypatch.setattr(f"ragwrench.port.status.{name}", forbidden)
    path = tmp_path / "absent" / "state.json"
    if isinstance(stage, str):
        with pytest.raises(SystemExit) as exc:
            main(["import-status", "--state", str(path), "--pending-stage", stage])
        captured = capsys.readouterr()
        assert exc.value.code == 2 and not captured.out and "invalid choice" in captured.err
    with pytest.raises(PortError, match="Pending stage must be"):
        import_status(path, pending_stage=stage)
    assert not path.parent.exists()


@pytest.mark.parametrize("paging", [False, True, "cursor"])
@pytest.mark.parametrize("case", [
    "state-schema", "state-consistency", "wrong-document-chunk", "wrong-document-pending",
    "unknown-document", "skipped-mapped", "skipped-pending", "incomplete-complete",
    "abort-preflight", "document-preflight", "hash", "fingerprint", "bundle-schema",
    "relationship", "content-hash", "counts",
])
def test_pending_stage_validates_nonmatching_details(mixed, capsys, request, monkeypatch, case, paging):
    path, bundle, state = mixed
    manifest, docs, chunks = load_bundle(bundle)
    # The invalid detail belongs to a nonmatching document; d00/d01 still match.
    doc = state["documents"]["d09"]
    if case == "state-schema":
        doc["chunks"] = []
    elif case == "state-consistency":
        doc["pending"] = {"source": "c0", "target": None}  # already mapped
    elif case == "wrong-document-chunk":
        doc["chunks"]["foreign"] = "t-foreign"
    elif case == "wrong-document-pending":
        doc["pending"] = {"source": "foreign", "target": None}
    elif case == "unknown-document":
        state["documents"]["unknown"] = state["documents"].pop("d09")
    elif case in ("skipped-mapped", "skipped-pending"):
        state["documents"]["d11"].update(phase="chunks",
            chunks={"c0": "t-c0"} if case == "skipped-mapped" else {},
            pending={"source": "c0", "target": None} if case == "skipped-pending" else None)
    elif case == "incomplete-complete":
        doc["phase"] = "complete"
    elif case == "abort-preflight":
        state["identity"]["on_error"] = "abort"
    elif case == "document-preflight":
        docs[-1]["metadata"] = {"private": "not restorable"}
        manifest = write_bundle(bundle, manifest["source"], manifest["settings"], docs, chunks)
        state["identity"]["bundle"] = fingerprint(manifest)
    elif case == "hash":
        with (bundle / "chunks.jsonl").open("a") as stream:
            stream.write("\n")
    elif case == "fingerprint":
        write_bundle(bundle, manifest["source"], {"different": True}, docs, chunks)
    else:
        if case == "bundle-schema":
            chunks[-1]["content"] = 42
        elif case == "relationship":
            chunks[-1]["doc_id"] = "unknown"
        elif case == "content-hash":
            chunks[-1]["content"] = "private changed content"
        else:
            manifest["counts"]["chunks"] += 1
        blob = "".join(json.dumps(c) + "\n" for c in chunks).encode()
        (bundle / "chunks.jsonl").write_bytes(blob)
        manifest["sha256"]["chunks.jsonl"] = hashlib.sha256(blob).hexdigest()
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        state["identity"]["bundle"] = fingerprint(manifest)
    path.write_text(json.dumps(state))
    request.getfixturevalue("forbid_io")
    if case in ("hash", "fingerprint", "bundle-schema", "relationship", "content-hash", "counts"):
        monkeypatch.setattr("ragwrench.port.status.make_plan", lambda *a, **k: pytest.fail("invalid input reached preflight"))
    before = files_snapshot(path.parent), set(path.parent.rglob("*"))
    # Also cover state-only schema/consistency validation of excluded rows.
    for scope in ([None, bundle] if case.startswith("state-") else [bundle]):
        args = ["import-status", "--state", str(path), "--pending-stage", "document_creation"]
        if scope is not None:
            args += ["--bundle", str(scope)]
        options = {}
        if paging:
            args += ["--document-page-size", "1"]
            options["document_page_size"] = 1
        if paging == "cursor":
            args += ["--document-cursor", "private-invalid-cursor"]
            options["document_cursor"] = "private-invalid-cursor"
        assert main(args) == 1
        captured = capsys.readouterr()
        assert not captured.out and captured.err and "private" not in captured.err and "Traceback" not in captured.err
        assert "continuation cursor" not in captured.err  # Validate before cursor lookup.
        with pytest.raises(PortError):
            import_status(path, scope, pending_stage="document_creation", **options)
    assert (files_snapshot(path.parent), set(path.parent.rglob("*"))) == before


@pytest.mark.parametrize("paging", [False, True])
@pytest.mark.parametrize("kind", ["missing", "malformed", "symlink", "directory", "oversized"])
def test_pending_stage_file_errors(tmp_path, capsys, forbid_io, kind, paging):
    path = tmp_path / "absent" / "state.json"
    if kind != "missing":
        path.parent.mkdir()
    if kind == "malformed":
        path.write_text('{"private":')
    elif kind == "symlink":
        target = tmp_path / "target.json"
        target.write_text("{}")
        path.symlink_to(target)
    elif kind == "directory":
        path.mkdir()
    elif kind == "oversized":
        with path.open("wb") as stream:
            stream.truncate(256 * 1024 * 1024 + 1)
    before = set(tmp_path.rglob("*"))
    args = ["import-status", "--state", str(path), "--pending-stage", "enabled_flag"]
    if paging:
        args += ["--document-page-size", "1"]
    assert main(args) == 1
    captured = capsys.readouterr()
    assert not captured.out and captured.err and "Traceback" not in captured.err and "private" not in captured.err
    with pytest.raises((PortError, OSError)):
        import_status(path, pending_stage="enabled_flag", **({"document_page_size": 1} if paging else {}))
    assert set(tmp_path.rglob("*")) == before
