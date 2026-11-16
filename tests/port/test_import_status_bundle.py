"""Bundle-aware status counts are local progress through the saved preflight plan."""
import base64
from copy import deepcopy
import hashlib
import json

import pytest

from ragwrench.port.bundle import fingerprint, load_bundle, write_bundle
from ragwrench.port.cli import main
from ragwrench.port.common import PortError, atomic_json, read_json
from ragwrench.port.fakes import FakeState, running_fake
from ragwrench.port.operations import import_bundle


@pytest.fixture
def interrupted(example):
    manifest, documents, chunks = load_bundle(example)
    return {
        "identity": {
            "version": 1, "bundle": fingerprint(manifest), "system": "ragflow",
            "api_profile": "ragflow-0.27.3", "base_url": "http://127.0.0.1:1/api/v1",
            "dataset_name": "handbook", "reuse_dataset": None, "allow_lossy": False,
            "max_chunk_bytes": 16000, "on_error": "abort", "indexing": "high_quality",
        },
        "dataset": "target-dataset", "dataset_pending": False,
        "documents": {documents[0]["id"]: {
            "id": "target-doc", "phase": "chunks",
            "chunks": {chunks[0]["id"]: "t-one", chunks[1]["id"]: "t-two"},
            "pending": {"source": chunks[2]["id"], "target": "t-three"},
        }},
    }


def invoke(path, bundle, capsys, document=None, chunk=None, *, list_chunks=False, page_size=None, cursor=None, chunk_status=None, summary_only=False, pending_counts=False):
    args = ["import-status", "--state", str(path), "--bundle", str(bundle)]
    if document is not None:
        args += ["--document", document]
    if chunk is not None:
        args += ["--chunk", chunk]
    if list_chunks:
        args += ["--list-chunks"]
    if page_size is not None:
        args += ["--page-size", str(page_size)]
    if cursor is not None:
        args += ["--cursor", cursor]
    if chunk_status is not None:
        args += ["--chunk-status", chunk_status]
    if summary_only:
        args += ["--summary-only"]
    if pending_counts:
        args += ["--pending-counts"]
    code = main(args)
    out = capsys.readouterr()
    return code, out.out, out.err


def assert_chunk_list(path, bundle, capsys, selected):
    """Listing changes only one field and every row matches an independent selector."""
    source = selected["bundle_progress"]["documents"][0]["source_id"]
    code, out, err = invoke(path, bundle, capsys, source, list_chunks=True)
    assert code == 0 and not err
    listed = json.loads(out)
    rows = listed["bundle_progress"].pop("chunks")
    assert listed == selected
    _, _, chunks = load_bundle(bundle)
    assert [r["source_id"] for r in rows] == sorted(c["id"] for c in chunks if c["doc_id"] == source)
    for row in rows:
        code, out, err = invoke(path, bundle, capsys, source, row["source_id"])
        assert code == 0 and not err
        single = json.loads(out)
        assert row == single["bundle_progress"].pop("chunk")
        assert single == selected
    assert_chunk_pages(path, bundle, capsys, selected, rows)
    assert_chunk_filters(path, bundle, capsys, selected, rows)
    return rows


def assert_chunk_pages(path, bundle, capsys, selected, rows, page_size=2, chunk_status=None):
    """Every page preserves global fields and equals the independent full-list slice."""
    source = selected["bundle_progress"]["documents"][0]["source_id"]
    cursor, offset, seen = None, 0, set()
    while True:
        code, out, err = invoke(path, bundle, capsys, source, list_chunks=True,
                                page_size=page_size, cursor=cursor, chunk_status=chunk_status)
        assert code == 0 and not err
        page = json.loads(out)
        actual = page["bundle_progress"].pop("chunks")
        next_cursor = page["bundle_progress"].pop("next_cursor")
        assert actual == rows[offset:offset + page_size]
        assert page == selected
        offset += len(actual)
        if offset == len(rows):
            assert next_cursor is None
            break
        assert isinstance(next_cursor, str) and next_cursor and next_cursor not in seen
        seen.add(next_cursor)
        cursor = next_cursor


def assert_chunk_filters(path, bundle, capsys, selected, rows):
    """Filter precedes slicing and affects only chunks; single-row equality is checked above."""
    source = selected["bundle_progress"]["documents"][0]["source_id"]
    for status in ("completed", "pending", "unvisited", "skipped"):
        expected = [row for row in rows if row["status"] == status]
        code, out, err = invoke(path, bundle, capsys, source, list_chunks=True, chunk_status=status)
        assert code == 0 and not err
        filtered = json.loads(out)
        assert filtered["bundle_progress"].pop("chunks") == expected
        assert filtered == selected
        assert_chunk_pages(path, bundle, capsys, selected, expected, page_size=1, chunk_status=status)


def files_snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def assert_breakdown_totals(result):
    p = result["bundle_progress"]
    rows = p["documents"]
    assert [r["source_id"] for r in rows] == sorted(r["source_id"] for r in rows)
    assert len(rows) == p["planned_documents"]
    assert sum(r["complete"] for r in rows) == p["completed_documents"]
    assert sum(not r["complete"] for r in rows) == p["remaining_documents"]
    assert sum(not r["visited"] for r in rows) == p["unvisited_documents"]
    assert sum(r["completed_chunks"] for r in rows) == result["completed_chunks"]
    assert sum(len(r["skipped_chunks"]) for r in rows) == p["skipped_chunks"]
    for field in ("planned_chunks", "remaining_chunks", "pending_chunks", "unvisited_chunks"):
        assert sum(r[field] for r in rows) == p[field]
    for row in rows:
        assert row["remaining_chunks"] == row["pending_chunks"] + row["unvisited_chunks"]
        assert row["planned_chunks"] == row["completed_chunks"] + row["remaining_chunks"]
        assert not row["complete"] or row["visited"]
        assert row["skipped_chunks"] == sorted(row["skipped_chunks"], key=lambda r: r["source_id"])


@pytest.fixture
def forbid_io(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("bundle status attempted client/network/credentials/write/lock access")
    monkeypatch.delenv("DIFY_KB_API_KEY", raising=False)
    monkeypatch.delenv("RAGFLOW_API_KEY", raising=False)
    monkeypatch.setattr("ragwrench.port.cli.API", forbidden)
    monkeypatch.setattr("httpx.Client", forbidden)
    monkeypatch.setattr("socket.socket", forbidden)
    monkeypatch.setattr("os.getenv", forbidden)
    monkeypatch.setattr("ragwrench.port.common.atomic_bytes", forbidden)
    monkeypatch.setattr("ragwrench.port.common.locked", forbidden)
    monkeypatch.setattr("ragwrench.port.operations.locked", forbidden)


def test_bundle_status_interrupted_contract(interrupted, bundle, tmp_path, capsys, forbid_io):
    path = tmp_path / "mapping.json"
    # Write setup independently of the functions forbidden for the command itself.
    path.write_text(json.dumps(interrupted))
    before = files_snapshot(tmp_path), set(tmp_path.rglob("*"))
    assert main(["import-status", "--state", str(path)]) == 0
    state_only = json.loads(capsys.readouterr().out)
    code, out, err = invoke(path, bundle, capsys)
    assert code == 0 and err == ""
    result = json.loads(out)
    assert_breakdown_totals(result)
    progress = result.pop("bundle_progress")
    assert result == state_only
    assert progress.pop("documents") == [
        {"source_id": "00000000-0000-0000-0000-000000000064", "visited": True, "complete": False,
         "planned_chunks": 5, "completed_chunks": 2, "remaining_chunks": 3,
         "pending_chunks": 1, "unvisited_chunks": 2, "skipped_chunks": []},
        {"source_id": "00000000-0000-0000-0000-000000000065", "visited": False, "complete": False,
         "planned_chunks": 5, "completed_chunks": 0, "remaining_chunks": 5,
         "pending_chunks": 0, "unvisited_chunks": 5, "skipped_chunks": []},
        {"source_id": "00000000-0000-0000-0000-000000000066", "visited": False, "complete": False,
         "planned_chunks": 5, "completed_chunks": 0, "remaining_chunks": 5,
         "pending_chunks": 0, "unvisited_chunks": 5, "skipped_chunks": []},
    ]
    assert progress == {
        "bundle": str(bundle), "manifest_fingerprint": interrupted["identity"]["bundle"],
        "planned_documents": 3, "planned_chunks": 15, "skipped_chunks": 0,
        "completed_documents": 0, "remaining_documents": 3, "remaining_chunks": 13,
        "pending_chunks": 1, "unvisited_documents": 2, "unvisited_chunks": 12, "warnings": [],
    }
    assert result["mapped_documents"] == 1 and result["completed_chunks"] == 2
    assert result["documents"][0]["pending_stage"] == "enabled_flag"
    assert (files_snapshot(tmp_path), set(tmp_path.rglob("*"))) == before


@pytest.mark.parametrize("case", ["unknown", "empty", "unsafe", "wrong-document", "unknown-document"])
def test_bundle_status_unknown_chunk(interrupted, bundle, tmp_path, capsys, forbid_io, case):
    _, docs, chunks = load_bundle(bundle)
    source = "private-unknown-document" if case == "unknown-document" else docs[0]["id"]
    selected = {"unknown": "private-unknown-chunk", "empty": "", "unsafe": "../private/path\nprivate-content",
                "wrong-document": chunks[5]["id"], "unknown-document": chunks[0]["id"]}[case]
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(interrupted))
    before = files_snapshot(tmp_path), set(tmp_path.rglob("*"))
    code, out, err = invoke(path, bundle, capsys, source, selected)
    assert code == 1 and out == ""
    assert err == ("ragwrench port: Selected source document is not in the bundle\n" if case == "unknown-document" else
                   "ragwrench port: Selected source chunk is not in the selected bundle document\n")
    assert (files_snapshot(tmp_path), set(tmp_path.rglob("*"))) == before


@pytest.mark.parametrize("has_bundle,has_document", [(False, False), (True, False), (False, True)])
def test_import_status_chunk_requires_scope(tmp_path, capsys, monkeypatch, forbid_io, has_bundle, has_document):
    from ragwrench.port.status import import_status

    def forbidden(*args, **kwargs):
        pytest.fail("chunk selector without scope attempted to read state")
    monkeypatch.setattr("ragwrench.port.status.read_json", forbidden)
    path = tmp_path / "absent" / "mapping.json"
    args = ["import-status", "--state", str(path), "--chunk", "private-chunk"]
    if has_bundle:
        args += ["--bundle", str(tmp_path / "absent-bundle")]
    if has_document:
        args += ["--document", "private-doc"]
    with pytest.raises(SystemExit) as exc:
        main(args)
    captured = capsys.readouterr()
    assert exc.value.code == 2 and captured.out == ""
    assert "--chunk requires --bundle and --document" in captured.err and "private" not in captured.err
    with pytest.raises(PortError, match="Chunk selection requires a bundle and source document"):
        import_status(path, tmp_path if has_bundle else None,
                      document="private-doc" if has_document else None, chunk="private-chunk")
    assert not path.parent.exists()


@pytest.mark.parametrize("selected", [False, True, "chunk", "list", "page", "cursor", "filter", "filter-page", "filter-cursor", "summary", "pending-counts", "skip-list", "skip-page", "skip-cursor", "bundle-list", "bundle-filter", "bundle-page", "bundle-cursor"])
@pytest.mark.parametrize("case", [
    "hash", "fingerprint", "document", "chunk", "wrong-document-chunk", "pending",
    "wrong-document-pending", "incomplete-complete", "skipped-mapped", "skipped-pending",
    "abort-preflight", "document-preflight", "missing-bundle", "symlink-bundle",
    "state-schema", "state-consistency",
])
def test_bundle_status_failures(interrupted, bundle, tmp_path, capsys, case, selected, monkeypatch):
    manifest, docs, chunks = load_bundle(bundle)
    state = deepcopy(interrupted)
    doc = next(iter(state["documents"].values()))
    if case == "state-schema":
        doc["chunks"] = []
    elif case == "state-consistency":
        state["dataset"] = None
    elif case == "hash":
        with (bundle / "chunks.jsonl").open("a") as stream:
            stream.write("\n")
    elif case == "fingerprint":
        # Internally valid bundle, but different manifest including descriptive settings.
        write_bundle(bundle, manifest["source"], {"different": True}, docs, chunks)
    elif case == "document":
        state["documents"]["private-unknown-doc"] = state["documents"].pop(docs[0]["id"])
    elif case in ("chunk", "wrong-document-chunk"):
        doc["chunks"].pop(chunks[0]["id"])
        doc["chunks"]["private-unknown-chunk" if case == "chunk" else chunks[5]["id"]] = "t-one"
    elif case in ("pending", "wrong-document-pending"):
        doc["pending"]["source"] = "private-unknown-chunk" if case == "pending" else chunks[5]["id"]
    elif case == "incomplete-complete":
        doc.update(phase="complete", pending=None)
    elif case in ("skipped-mapped", "skipped-pending", "abort-preflight"):
        state["identity"].update(max_chunk_bytes=1, on_error="abort" if case == "abort-preflight" else "skip")
        if case == "skipped-pending":
            doc["chunks"] = {}
    elif case == "document-preflight":
        docs[0]["metadata"] = {"private": "not restorable"}
        manifest = write_bundle(bundle, manifest["source"], manifest["settings"], docs, chunks)
        state["identity"].update(bundle=fingerprint(manifest), on_error="skip")
    elif case == "missing-bundle":
        bundle = tmp_path / "absent" / "bundle"
    elif case == "symlink-bundle":
        link = tmp_path / "link"
        link.symlink_to(bundle, target_is_directory=True)
        bundle = link
    path = tmp_path / "mapping.json"
    atomic_json(path, state)
    before = files_snapshot(tmp_path), set(tmp_path.rglob("*"))
    def forbidden(*args, **kwargs):
        pytest.fail("rejected status attempted client/write/lock access")
    monkeypatch.setattr("ragwrench.port.cli.API", forbidden)
    monkeypatch.setattr("socket.socket", forbidden)
    monkeypatch.setattr("ragwrench.port.common.atomic_bytes", forbidden)
    monkeypatch.setattr("ragwrench.port.common.locked", forbidden)
    monkeypatch.setattr("ragwrench.port.operations.locked", forbidden)
    if case in ("fingerprint", "hash"):
        monkeypatch.setattr("ragwrench.port.status.make_plan", forbidden)
    # Selecting the untouched second document must not hide invalid state in the first.
    if selected in ("bundle-list", "bundle-filter", "bundle-page", "bundle-cursor"):
        from ragwrench.port.status import import_status
        options = {"list_bundle_chunks": True}
        args = ["import-status", "--state", str(path), "--bundle", str(bundle), "--list-bundle-chunks"]
        if selected != "bundle-list":
            options["bundle_chunk_status"] = "skipped"; args += ["--bundle-chunk-status", "skipped"]
        if selected in ("bundle-page", "bundle-cursor"):
            options["bundle_chunk_page_size"] = 1; args += ["--bundle-chunk-page-size", "1"]
        if selected == "bundle-cursor":
            options["bundle_chunk_cursor"] = "private-invalid-cursor"; args += ["--bundle-chunk-cursor", "private-invalid-cursor"]
        from ragwrench.port.common import PortError
        # An absent bundle surfaces as OSError from load_bundle, as in every other bundle mode; CLI maps it to exit 1.
        with pytest.raises((PortError, OSError)): import_status(path, bundle, **options)
        code = main(args)
        captured = capsys.readouterr()
        out, err = captured.out, captured.err
    elif selected in ("skip-list", "skip-page", "skip-cursor"):
        args = ["import-status", "--state", str(path), "--bundle", str(bundle), "--list-skipped-chunks"]
        if selected != "skip-list": args += ["--skip-page-size", "1"]
        if selected == "skip-cursor": args += ["--skip-cursor", "private-invalid-cursor"]
        code = main(args)
        captured = capsys.readouterr()
        out, err = captured.out, captured.err
    else:
        code, out, err = invoke(path, bundle, capsys, docs[1]["id"] if selected and selected not in ("summary", "pending-counts") else None,
                                chunks[5]["id"] if selected == "chunk" else None,
                                list_chunks=selected in ("list", "page", "cursor", "filter", "filter-page", "filter-cursor"),
                                page_size=1 if selected in ("page", "cursor", "filter-page", "filter-cursor") else None,
                                cursor="private-invalid-cursor" if selected in ("cursor", "filter-cursor") else None,
                                chunk_status="completed" if selected in ("filter", "filter-page", "filter-cursor") else None,
                                summary_only=selected in ("summary", "pending-counts"), pending_counts=selected == "pending-counts")
    assert code == 1 and out == "" and err and "private" not in err and "Traceback" not in err
    assert "continuation cursor" not in err  # Full validation precedes cursor checks, too.
    assert (files_snapshot(tmp_path), set(tmp_path.rglob("*"))) == before


@pytest.mark.parametrize("system", ["dify", "ragflow"])
@pytest.mark.parametrize("legacy", [False, True])
def test_bundle_status_saved_preflight_and_skips(interrupted, bundle, tmp_path, capsys, system, legacy, request):
    manifest, docs, chunks = load_bundle(bundle)
    # Saved lossy projection must run before the saved byte guard: RAGFlow's
    # answer flattening pushes chunk 1 over the limit. Dify drops questions.
    chunks[0]["answer" if system == "ragflow" else "questions"] = "a" if system == "ragflow" else ["q"]
    chunks[1]["metadata"] = {"layout": "descriptive"}
    manifest = write_bundle(bundle, manifest["source"], manifest["settings"], docs, chunks)
    state = deepcopy(interrupted)
    state["identity"].update(system=system, bundle=fingerprint(manifest), allow_lossy=True,
                             max_chunk_bytes=len(chunks[0]["content"].encode("utf-8")), on_error="skip")
    if legacy:
        del state["identity"]["api_profile"]
    else:
        state["identity"]["api_profile"] = "dify-f8fc371a" if system == "dify" else "ragflow-0.27.3"
    state["documents"] = {docs[0]["id"]: {
        "id": "target-doc", "phase": "chunks", "chunks": {chunks[1]["id"]: "t-two"},
        "pending": {"source": chunks[2]["id"], "target": None},
    }}
    path = tmp_path / "mapping.json"
    atomic_json(path, state)
    request.getfixturevalue("forbid_io")
    before = files_snapshot(tmp_path), set(tmp_path.rglob("*"))
    code, out, err = invoke(path, bundle, capsys)
    assert code == 0 and not err
    result = json.loads(out)
    assert_breakdown_totals(result)
    assert_summary(path, bundle, capsys, result)
    code, out, err = invoke(path, bundle, capsys, docs[0]["id"])
    assert code == 0 and not err
    selected = json.loads(out)
    expected = deepcopy(result)
    expected["bundle_progress"]["documents"] = [result["bundle_progress"]["documents"][0]]
    assert selected == expected
    assert_chunk_list(path, bundle, capsys, selected)
    p = result["bundle_progress"]
    skipped = 1 if system == "ragflow" else 0
    assert p["skipped_chunks"] == skipped and p["planned_chunks"] == 15 - skipped
    assert p["remaining_chunks"] == 14 - skipped and p["unvisited_chunks"] == 13 - skipped
    assert p["pending_chunks"] == 1 and result["completed_chunks"] == 1
    assert p["warnings"]
    from .test_document_pages import page_result, assert_slice
    assert assert_slice(page_result(path, bundle, capsys, "chunk_creation", 1), result, [docs[0]["id"]]) is None
    assert assert_slice(page_result(path, bundle, capsys, "enabled_flag", 1), result, []) is None
    first = p["documents"][0]
    assert first["completed_chunks"] == first["pending_chunks"] == 1
    assert first["planned_chunks"] == 5 - skipped
    assert first["skipped_chunks"] == ([{"source_id": chunks[0]["id"],
        "reason": "Chunk exceeds configured target UTF-8 byte limit"}] if skipped else [])
    from .test_skip_pages import invoke as skip_invoke, assert_page, expected_rows
    skip_rows = expected_rows(result)
    assert_page(skip_invoke(path, bundle, capsys), result, skip_rows, False)
    assert assert_page(skip_invoke(path, bundle, capsys, 1), result, skip_rows) is None
    from .test_bundle_chunk_pages import invoke as bundle_invoke, assert_page as bundle_assert_page, expected_rows as bundle_rows
    all_rows = bundle_rows(path, bundle)
    for status in (None, "completed", "pending", "unvisited", "skipped"):
        rows = [r for r in all_rows if status is None or r["status"] == status]
        bundle_assert_page(bundle_invoke(path, bundle, capsys, status=status), result, rows, False)
        assert bundle_assert_page(bundle_invoke(path, bundle, capsys, 10**30, status=status), result, rows) is None
    assert result["api_profile"] == ("ragflow-1.0" if legacy and system == "ragflow" else state["identity"].get("api_profile", "dify-f8fc371a"))
    code, out, err = invoke(path, bundle, capsys, docs[0]["id"], chunks[0]["id"])
    assert code == 0 and not err
    expected = deepcopy(selected)
    expected["bundle_progress"]["chunk"] = {
        "document_source_id": docs[0]["id"], "source_id": chunks[0]["id"],
        "status": "skipped" if skipped else "unvisited", "target_id": None, "pending_stage": None,
        "reason": "Chunk exceeds configured target UTF-8 byte limit" if skipped else None}
    assert json.loads(out) == expected
    assert (files_snapshot(tmp_path), set(tmp_path.rglob("*"))) == before


@pytest.mark.parametrize("stage", ["initial", "creating", "clearing", "all-skipped", "complete"])
def test_bundle_status_scope_boundaries(interrupted, bundle, tmp_path, capsys, stage, request):
    manifest, docs, chunks = load_bundle(bundle)
    state = deepcopy(interrupted)
    doc = next(iter(state["documents"].values()))
    if stage == "initial":
        state.update(dataset=None, dataset_pending=True, documents={})
    elif stage in ("creating", "clearing"):
        doc.update(id=None if stage == "creating" else "t-doc", phase=stage, chunks={}, pending=None)
    else:
        if stage == "all-skipped":
            state["identity"].update(max_chunk_bytes=1, on_error="skip")
        state["documents"] = {d["id"]: {
            "id": f't-doc-{i}', "phase": "complete", "pending": None,
            "chunks": {} if stage == "all-skipped" else {c["id"]: f't-{j}' for j, c in enumerate(chunks) if c["doc_id"] == d["id"]},
        } for i, d in enumerate(docs)}
    path = tmp_path / "mapping.json"
    atomic_json(path, state)
    request.getfixturevalue("forbid_io")
    before = files_snapshot(tmp_path), set(tmp_path.rglob("*"))
    code, out, err = invoke(path, bundle, capsys)
    assert code == 0 and not err
    result = json.loads(out)
    p = result["bundle_progress"]
    assert_breakdown_totals(result)
    assert_summary(path, bundle, capsys, result)
    finished = stage in ("all-skipped", "complete")
    assert p["planned_documents"] == 3
    assert p["remaining_documents"] == (0 if finished else 3)
    assert p["completed_documents"] == (3 if finished else 0)
    assert p["unvisited_documents"] == (0 if finished else 3 if stage == "initial" else 2)
    assert p["remaining_chunks"] == p["unvisited_chunks"] == (0 if finished else 15)
    assert p["skipped_chunks"] == (15 if stage == "all-skipped" else 0)
    assert result["completed_chunks"] == (15 if stage == "complete" else 0)
    assert "verified" not in result and "verified" not in p
    code, out, err = invoke(path, bundle, capsys, docs[0]["id"], chunks[0]["id"])
    assert code == 0 and not err
    expected = deepcopy(result)
    expected["bundle_progress"].update(documents=[p["documents"][0]], chunk={
        "document_source_id": docs[0]["id"], "source_id": chunks[0]["id"],
        "status": "completed" if stage == "complete" else "skipped" if stage == "all-skipped" else "unvisited",
        "target_id": "t-0" if stage == "complete" else None, "pending_stage": None,
        "reason": "Chunk exceeds configured target UTF-8 byte limit" if stage == "all-skipped" else None})
    assert json.loads(out) == expected
    del expected["bundle_progress"]["chunk"]
    listed = assert_chunk_list(path, bundle, capsys, expected)
    assert {r["status"] for r in listed} == {"completed" if stage == "complete" else "skipped" if stage == "all-skipped" else "unvisited"}
    assert (files_snapshot(tmp_path), set(tmp_path.rglob("*"))) == before


def test_bundle_status_actual_skip_checkpoint(bundle, tmp_path, capsys, api_factory):
    manifest, docs, chunks = load_bundle(bundle)
    chunks[0]["metadata"] = {"unsupported": True}
    write_bundle(bundle, manifest["source"], manifest["settings"], docs, chunks)
    path = tmp_path / "mapping.json"
    target = FakeState("ragflow")
    with running_fake(target) as url:
        api = api_factory("ragflow", url)
        original = api.set_enabled
        calls = 0
        def interrupt(*args):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise PortError("simulated interruption")
            return original(*args)
        api.set_enabled = interrupt
        with pytest.raises(PortError, match="simulated interruption"):
            import_bundle(api, bundle, path, dataset_name="skip-checkpoint", on_error="skip")
        before = len(target.requests), files_snapshot(tmp_path), set(tmp_path.rglob("*"))
        code, out, err = invoke(path, bundle, capsys)
        assert code == 0 and not err
        result = json.loads(out)
        p = result["bundle_progress"]
        assert_breakdown_totals(result)
        assert_summary(path, bundle, capsys, result)
        assert result["mapped_documents"] == 1 and result["completed_chunks"] == 2
        assert (p["planned_chunks"], p["skipped_chunks"], p["remaining_chunks"], p["unvisited_chunks"]) == (14, 1, 12, 11)
        assert p["documents"][0] == {
            "source_id": docs[0]["id"], "visited": True, "complete": False,
            "planned_chunks": 4, "completed_chunks": 2, "remaining_chunks": 2,
            "pending_chunks": 1, "unvisited_chunks": 1,
            "skipped_chunks": [{"source_id": chunks[0]["id"], "reason":
                "Unsupported fields: chunk metadata (including image/layout/tags); use --allow-lossy to acknowledge"}],
        }
        code, out, err = invoke(path, bundle, capsys, docs[0]["id"])
        assert code == 0 and not err
        expected = deepcopy(result)
        expected["bundle_progress"]["documents"] = [p["documents"][0]]
        assert json.loads(out) == expected
        assert (len(target.requests), files_snapshot(tmp_path), set(tmp_path.rglob("*"))) == before
        api.set_enabled = original
        assert import_bundle(api, bundle, path, dataset_name="skip-checkpoint", on_error="skip")["skipped"] == 1
        code, out, err = invoke(path, bundle, capsys)
        assert code == 0 and not err
        result = json.loads(out)
        assert_breakdown_totals(result)
        assert_summary(path, bundle, capsys, result)
        assert result["completed_chunks"] == 14
        assert result["bundle_progress"]["skipped_chunks"] == 1
        assert result["bundle_progress"]["remaining_chunks"] == 0
        assert result["bundle_progress"]["completed_documents"] == 3


@pytest.mark.parametrize("phase", [None, "new", "complete"])
def test_bundle_status_empty_and_excluded_documents(interrupted, bundle, tmp_path, capsys, phase, monkeypatch):
    manifest, docs, chunks = load_bundle(bundle)
    # One empty document, one entirely skipped, and one untouched importable document.
    chunks = [c for c in chunks if c["doc_id"] != docs[0]["id"]]
    for chunk in chunks:
        if chunk["doc_id"] == docs[1]["id"]:
            chunk["metadata"] = {"private-content": "must not be echoed"}
    manifest = write_bundle(bundle, manifest["source"], manifest["settings"], list(reversed(docs)), list(reversed(chunks)))
    state = deepcopy(interrupted)
    state["identity"].update(bundle=fingerprint(manifest), on_error="skip")
    state["documents"] = {} if phase is None else {
        doc["id"]: {"id": None if phase == "new" else f"t-{i}", "phase": phase, "chunks": {}, "pending": None}
        for i, doc in enumerate(docs[:2])}
    path = tmp_path / "mapping.json"
    atomic_json(path, state)
    # Exercise the command with the same forbidden I/O as the interrupted contract.
    def forbidden(*args, **kwargs):
        pytest.fail("status attempted network/client/write/lock access")
    monkeypatch.setattr("ragwrench.port.cli.API", forbidden)
    monkeypatch.setattr("socket.socket", forbidden)
    monkeypatch.setattr("ragwrench.port.common.atomic_bytes", forbidden)
    monkeypatch.setattr("ragwrench.port.operations.locked", forbidden)
    before = files_snapshot(tmp_path), set(tmp_path.rglob("*"))
    code, out, err = invoke(path, bundle, capsys)
    assert code == 0 and not err and "private-content" not in out and "must not be echoed" not in out
    assert (files_snapshot(tmp_path), set(tmp_path.rglob("*"))) == before
    result = json.loads(out)
    assert_breakdown_totals(result)
    assert_summary(path, bundle, capsys, result)
    empty, excluded, untouched = result["bundle_progress"]["documents"]
    assert empty == {
        "source_id": docs[0]["id"], "visited": phase is not None, "complete": phase == "complete",
        "planned_chunks": 0, "completed_chunks": 0, "remaining_chunks": 0,
        "pending_chunks": 0, "unvisited_chunks": 0, "skipped_chunks": []}
    assert excluded == {**empty, "source_id": docs[1]["id"], "skipped_chunks": [
        {"source_id": c["id"], "reason":
            "Unsupported fields: chunk metadata (including image/layout/tags); use --allow-lossy to acknowledge"}
        for c in chunks if c["doc_id"] == docs[1]["id"]]}
    assert not untouched["visited"] and not untouched["complete"] and untouched["unvisited_chunks"] == 5
    for row in (empty, excluded, untouched):
        code, out, err = invoke(path, bundle, capsys, row["source_id"])
        assert code == 0 and not err
        expected = deepcopy(result)
        expected["bundle_progress"]["documents"] = [row]
        assert json.loads(out) == expected
        assert_chunk_list(path, bundle, capsys, expected)
    code, out, err = invoke(path, bundle, capsys, empty["source_id"], chunks[0]["id"])
    assert code == 1 and out == "" and err == "ragwrench port: Selected source chunk is not in the selected bundle document\n"
    code, out, err = invoke(path, bundle, capsys, excluded["source_id"], excluded["skipped_chunks"][0]["source_id"])
    assert code == 0 and not err
    expected = deepcopy(result)
    expected["bundle_progress"].update(documents=[excluded], chunk={
        "document_source_id": excluded["source_id"], "source_id": excluded["skipped_chunks"][0]["source_id"],
        "status": "skipped", "target_id": None, "pending_stage": None,
        "reason": excluded["skipped_chunks"][0]["reason"]})
    assert json.loads(out) == expected
    assert (files_snapshot(tmp_path), set(tmp_path.rglob("*"))) == before


def test_bundle_status_duplicate_chunk_ids_are_document_scoped(interrupted, bundle, tmp_path, capsys, request):
    manifest, docs, chunks = load_bundle(bundle)
    chunks[5]["id"] = chunks[0]["id"]
    chunks[0]["metadata"] = {"unsupported": True}
    # Same source chunk ID is skipped in doc 1 and mapped in doc 2.
    manifest = write_bundle(bundle, manifest["source"], manifest["settings"], docs, chunks)
    state = deepcopy(interrupted)
    state["identity"].update(bundle=fingerprint(manifest), on_error="skip")
    state["documents"] = {docs[1]["id"]: {
        "id": "t-doc", "phase": "chunks", "chunks": {chunks[5]["id"]: "t-one"}, "pending": None}}
    path = tmp_path / "mapping.json"
    atomic_json(path, state)
    request.getfixturevalue("forbid_io")
    before = files_snapshot(tmp_path), set(tmp_path.rglob("*"))
    code, out, err = invoke(path, bundle, capsys)
    assert code == 0 and not err
    result = json.loads(out)
    assert_breakdown_totals(result)
    first, second, _ = result["bundle_progress"]["documents"]
    assert first["planned_chunks"] == 4 and first["completed_chunks"] == 0
    assert first["skipped_chunks"][0]["source_id"] == chunks[0]["id"]
    assert second["planned_chunks"] == 5 and second["completed_chunks"] == 1
    assert second["skipped_chunks"] == []
    for row in (first, second):
        code, out, err = invoke(path, bundle, capsys, row["source_id"])
        assert code == 0 and not err
        expected = deepcopy(result)
        expected["bundle_progress"]["documents"] = [row]
        assert json.loads(out) == expected
        assert_chunk_list(path, bundle, capsys, expected)
        code, out, err = invoke(path, bundle, capsys, row["source_id"], chunks[0]["id"])
        assert code == 0 and not err
        expected["bundle_progress"]["chunk"] = {
            "document_source_id": row["source_id"], "source_id": chunks[0]["id"],
            "status": "skipped" if row == first else "completed", "pending_stage": None,
            "target_id": None if row == first else "t-one",
            "reason": first["skipped_chunks"][0]["reason"] if row == first else None}
        assert json.loads(out) == expected
    assert (files_snapshot(tmp_path), set(tmp_path.rglob("*"))) == before


@pytest.mark.parametrize("source,visited,completed,pending,remaining,unvisited", [
    ("00000000-0000-0000-0000-000000000064", True, 2, 1, 3, 2),
    ("00000000-0000-0000-0000-000000000065", False, 0, 0, 5, 5),
])
def test_bundle_status_document_selection_contract(
        interrupted, bundle, tmp_path, capsys, forbid_io,
        source, visited, completed, pending, remaining, unvisited):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(interrupted))
    before = files_snapshot(tmp_path), set(tmp_path.rglob("*"))
    code, out, err = invoke(path, bundle, capsys)
    assert code == 0 and not err
    full = json.loads(out)
    assert main(["import-status", "--state", str(path), "--bundle", str(bundle),
                 "--document", source]) == 0
    captured = capsys.readouterr()
    assert not captured.err
    selected = json.loads(captured.out)
    row = {
        "source_id": source, "visited": visited, "complete": False,
        "planned_chunks": 5, "completed_chunks": completed, "remaining_chunks": remaining,
        "pending_chunks": pending, "unvisited_chunks": unvisited, "skipped_chunks": [],
    }
    assert selected["bundle_progress"]["documents"] == [row]
    # Only detail rows are filtered: aggregate progress and saved snapshot stay global.
    full["bundle_progress"]["documents"] = [row]
    assert selected == full
    assert (selected["bundle_progress"]["planned_chunks"], selected["completed_chunks"],
            selected["bundle_progress"]["remaining_chunks"]) == (15, 2, 13)
    assert (files_snapshot(tmp_path), set(tmp_path.rglob("*"))) == before


@pytest.mark.parametrize("source", ["private-unknown-document", "", "../private/path\nprivate-content"])
@pytest.mark.parametrize("list_chunks", [False, True])
def test_bundle_status_unknown_document(interrupted, bundle, tmp_path, capsys, forbid_io, source, list_chunks):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(interrupted))
    before = files_snapshot(tmp_path), set(tmp_path.rglob("*"))
    code, out, err = invoke(path, bundle, capsys, source, list_chunks=list_chunks)
    assert code == 1 and out == ""
    assert err == "ragwrench port: Selected source document is not in the bundle\n"
    assert (files_snapshot(tmp_path), set(tmp_path.rglob("*"))) == before


def test_import_status_document_requires_bundle(tmp_path, capsys, monkeypatch, forbid_io):
    from ragwrench.port.status import import_status

    def forbidden(*args, **kwargs):
        pytest.fail("selector without bundle attempted to read state")
    monkeypatch.setattr("ragwrench.port.status.read_json", forbidden)
    path = tmp_path / "absent" / "mapping.json"
    with pytest.raises(SystemExit) as exc:
        main(["import-status", "--state", str(path), "--document", "source-doc"])
    captured = capsys.readouterr()
    assert exc.value.code == 2 and captured.out == ""
    assert "--document requires --bundle" in captured.err
    with pytest.raises(PortError, match="Document selection requires a bundle"):
        import_status(path, document="source-doc")
    assert not path.parent.exists()


@pytest.mark.parametrize("case", ["completed", "pending-flag", "pending-create", "unvisited", "untouched", "skipped"])
def test_bundle_status_chunk_contract(interrupted, bundle, tmp_path, capsys, request, case):
    manifest, docs, chunks = load_bundle(bundle)
    state = deepcopy(interrupted)
    source = docs[0]["id"]
    selected = chunks[0]["id"]
    detail = {"document_source_id": source, "source_id": selected,
              "status": "completed", "target_id": "t-one", "pending_stage": None, "reason": None}
    if case.startswith("pending"):
        selected = chunks[2]["id"]
        target = "t-three" if case == "pending-flag" else None
        state["documents"][source]["pending"]["target"] = target
        detail.update(source_id=selected, status="pending", target_id=target,
                      pending_stage="enabled_flag" if target else "chunk_creation")
    elif case in ("unvisited", "untouched", "skipped"):
        selected = chunks[5 if case == "untouched" else 4]["id"]
        if case == "untouched":
            source = docs[1]["id"]
        detail.update(document_source_id=source, source_id=selected, status="unvisited", target_id=None)
        if case == "skipped":
            chunks[4]["metadata"] = {"private-content": "never echo"}
            manifest = write_bundle(bundle, manifest["source"], manifest["settings"], docs, chunks)
            state["identity"].update(bundle=fingerprint(manifest), on_error="skip")
            detail.update(status="skipped", reason=
                          "Unsupported fields: chunk metadata (including image/layout/tags); use --allow-lossy to acknowledge")
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(state))
    request.getfixturevalue("forbid_io")
    before = files_snapshot(tmp_path), set(tmp_path.rglob("*"))
    code, out, err = invoke(path, bundle, capsys, source)
    assert code == 0 and not err
    expected = json.loads(out)
    expected["bundle_progress"]["chunk"] = detail
    code, out, err = invoke(path, bundle, capsys, source, selected)
    assert code == 0 and not err and "private-content" not in out and "never echo" not in out
    assert json.loads(out) == expected
    progress = expected["bundle_progress"]
    skipped = int(case == "skipped")
    assert (progress["planned_chunks"], expected["completed_chunks"], progress["remaining_chunks"],
            progress["unvisited_chunks"], progress["pending_chunks"], progress["skipped_chunks"]) == (
                15 - skipped, 2, 13 - skipped, 12 - skipped, 1, skipped)
    assert (files_snapshot(tmp_path), set(tmp_path.rglob("*"))) == before


@pytest.mark.parametrize("skipped", [False, True])
@pytest.mark.parametrize("pending_target", [None, "t-three"])
def test_bundle_status_chunk_list_contract(interrupted, bundle, tmp_path, capsys, request, skipped, pending_target):
    manifest, docs, chunks = load_bundle(bundle)
    source = docs[0]["id"]
    # Source-ID ordering differs from position and from the bundle's canonical order.
    chunks[3]["id"], chunks[4]["id"] = "z-last", "a-first"
    if skipped:
        chunks[4]["metadata"] = {"private-content": "never echo"}
    manifest = write_bundle(bundle, manifest["source"], manifest["settings"], docs, chunks)
    interrupted["identity"].update(bundle=fingerprint(manifest), on_error="skip" if skipped else "abort")
    interrupted["documents"][source]["pending"]["target"] = pending_target
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(interrupted))
    request.getfixturevalue("forbid_io")
    before = files_snapshot(tmp_path), set(tmp_path.rglob("*"))
    code, out, err = invoke(path, bundle, capsys, source)
    assert code == 0 and not err
    selected = json.loads(out)
    detail = lambda chunk, status, target=None, stage=None, reason=None: {
        "document_source_id": source, "source_id": chunk, "status": status,
        "target_id": target, "pending_stage": stage, "reason": reason}
    expected = sorted([
        detail(chunks[0]["id"], "completed", "t-one"),
        detail(chunks[1]["id"], "completed", "t-two"),
        detail(chunks[2]["id"], "pending", pending_target,
               "enabled_flag" if pending_target else "chunk_creation"),
        detail("z-last", "unvisited"),
        detail("a-first", "skipped" if skipped else "unvisited", reason=(
            "Unsupported fields: chunk metadata (including image/layout/tags); use --allow-lossy to acknowledge"
            if skipped else None)),
    ], key=lambda row: row["source_id"])
    assert assert_chunk_list(path, bundle, capsys, selected) == expected
    p = selected["bundle_progress"]
    assert (p["planned_chunks"], selected["completed_chunks"], p["remaining_chunks"], p["skipped_chunks"]) == (
        15 - skipped, 2, 13 - skipped, int(skipped))
    assert "private-content" not in json.dumps(expected) and "never echo" not in json.dumps(expected)
    assert (files_snapshot(tmp_path), set(tmp_path.rglob("*"))) == before


@pytest.mark.parametrize("has_bundle,has_document,has_chunk", [
    (False, False, False), (True, False, False), (False, True, False),
    (True, True, True), (False, False, True),
])
def test_import_status_chunk_list_argument_errors(tmp_path, capsys, monkeypatch, forbid_io,
                                                 has_bundle, has_document, has_chunk):
    from ragwrench.port.status import import_status

    def forbidden(*args, **kwargs):
        pytest.fail("invalid list arguments attempted to read state")
    monkeypatch.setattr("ragwrench.port.status.read_json", forbidden)
    path = tmp_path / "absent" / "mapping.json"
    args = ["import-status", "--state", str(path), "--list-chunks"]
    if has_bundle:
        args += ["--bundle", str(tmp_path / "absent-bundle")]
    if has_document:
        args += ["--document", "private-document"]
    if has_chunk:
        args += ["--chunk", "private-chunk"]
    with pytest.raises(SystemExit) as exc:
        main(args)
    captured = capsys.readouterr()
    assert exc.value.code == 2 and not captured.out and "private" not in captured.err
    assert ("--list-chunks cannot be combined with --chunk" if has_chunk else
            "--list-chunks requires --bundle and --document") in captured.err
    with pytest.raises(PortError, match=("Chunk listing cannot be combined with chunk selection" if has_chunk else
                                        "Chunk listing requires a bundle and source document")):
        import_status(path, tmp_path if has_bundle else None,
                      document="private-document" if has_document else None,
                      chunk="private-chunk" if has_chunk else None, list_chunks=True)
    assert not path.parent.exists()


@pytest.mark.parametrize("page_size", [1, 2, 5, 6, 10**30])
def test_chunk_page_boundaries(interrupted, bundle, tmp_path, capsys, forbid_io, page_size):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(interrupted))
    source = next(iter(interrupted["documents"]))
    code, out, err = invoke(path, bundle, capsys, source, list_chunks=True)
    assert code == 0 and not err
    selected = json.loads(out)
    rows = selected["bundle_progress"].pop("chunks")
    assert_chunk_pages(path, bundle, capsys, selected, rows, page_size)


@pytest.fixture
def first_page(interrupted, bundle, tmp_path, capsys):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(interrupted))
    source = next(iter(interrupted["documents"]))
    code, out, err = invoke(path, bundle, capsys, source, list_chunks=True, page_size=2)
    assert code == 0 and not err
    return path, source, json.loads(out)


def decode_cursor(token):
    return json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))


def encode_cursor(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def test_chunk_cursor_determinism_size_change_and_copy(first_page, bundle, tmp_path, capsys, request):
    from ragwrench.port.status import import_status

    path, source, first = first_page
    cursor = first["bundle_progress"]["next_cursor"]
    assert decode_cursor(cursor) == {
        "version": 1, "document": source, "bundle": first["bundle_progress"]["manifest_fingerprint"],
        "state": hashlib.sha256(path.read_bytes()).hexdigest(),
        "after": first["bundle_progress"]["chunks"][-1]["source_id"]}
    assert import_status(path, bundle, document=source, list_chunks=True, page_size=2) == first
    # Page boundaries are after IDs: a caller can change size and move identical snapshots.
    copy = tmp_path / "copy.json"
    copy.write_bytes(path.read_bytes())
    m, docs, chunks = load_bundle(bundle)
    bundle_copy = tmp_path / "copy-bundle"
    write_bundle(bundle_copy, m["source"], m["settings"], docs, chunks)
    request.getfixturevalue("forbid_io")
    before = files_snapshot(tmp_path), set(tmp_path.rglob("*"))
    code, out, err = invoke(copy, bundle_copy, capsys, source, list_chunks=True, page_size=1, cursor=cursor)
    assert code == 0 and not err
    second = json.loads(out)
    assert [row["status"] for row in second["bundle_progress"]["chunks"]] == ["pending"]
    code, out, err = invoke(path, bundle, capsys, source, list_chunks=True, page_size=9,
                            cursor=second["bundle_progress"]["next_cursor"])
    assert code == 0 and not err
    final = json.loads(out)
    assert [row["status"] for row in final["bundle_progress"]["chunks"]] == ["unvisited", "unvisited"]
    assert final["bundle_progress"]["next_cursor"] is None
    assert (files_snapshot(tmp_path), set(tmp_path.rglob("*"))) == before


@pytest.mark.parametrize("case", [
    "empty", "unsafe", "unicode", "oversized", "truncated", "padding", "not-json", "array", "noncanonical",
    "version", "bool-version", "missing", "extra", "document", "bundle", "state", "after", "end", "after-type",
    "changed-document", "changed-bundle", "changed-state", "whitespace", "extension", "empty-document",
])
@pytest.mark.parametrize("chunk_status", [None, "completed"])
def test_chunk_cursor_rejections(first_page, interrupted, bundle, tmp_path, capsys, case, request, chunk_status):
    path, source, first = first_page
    if chunk_status is not None:
        code, out, err = invoke(path, bundle, capsys, source, list_chunks=True,
                                page_size=1, chunk_status=chunk_status)
        assert code == 0 and not err
        first = json.loads(out)
    cursor = first["bundle_progress"]["next_cursor"]
    value = decode_cursor(cursor)
    _, docs, chunks = load_bundle(bundle)
    if case == "empty":
        cursor = ""
    elif case == "unsafe":
        cursor = "../private-content"
    elif case == "unicode":
        cursor = "界"
    elif case == "oversized":
        cursor = "x" * 2049
    elif case == "truncated":
        cursor = cursor[:1]
    elif case == "padding":
        cursor += "="
    elif case == "not-json":
        cursor = base64.urlsafe_b64encode(b"private-content").decode().rstrip("=")
    elif case == "array":
        cursor = encode_cursor([])
    elif case == "noncanonical":
        cursor = base64.urlsafe_b64encode(json.dumps(value, indent=2).encode()).decode().rstrip("=")
    elif case in ("version", "bool-version", "document", "bundle", "state", "after", "end", "after-type"):
        key = "version" if case == "bool-version" else "after" if case in ("end", "after-type") else case
        value[key] = (True if case == "bool-version" else 99 if case == "version" else
                      max(c["id"] for c in chunks if c["doc_id"] == source) if case == "end" else
                      [] if case == "after-type" else "private-mismatch")
        cursor = encode_cursor(value)
    elif case == "missing":
        del value["state"]
        cursor = encode_cursor(value)
    elif case == "extra":
        cursor = encode_cursor({**value, "private": "private-content"})
    elif case == "changed-document":
        source = docs[1]["id"]
    elif case == "changed-bundle":
        m, docs, chunks = load_bundle(bundle)
        m = write_bundle(bundle, m["source"], {"different": True}, docs, chunks)
        interrupted["identity"]["bundle"] = fingerprint(m)
        path.write_text(json.dumps(interrupted))
    elif case == "changed-state":
        interrupted["documents"][source]["pending"]["target"] = None
        path.write_text(json.dumps(interrupted))
    elif case == "whitespace":
        path.write_bytes(path.read_bytes() + b" ")
    elif case == "extension":
        interrupted["private-extension"] = "private-content"
        path.write_text(json.dumps(interrupted))
    elif case == "empty-document":
        m, docs, chunks = load_bundle(bundle)
        m = write_bundle(bundle, m["source"], m["settings"], docs,
                         [c for c in chunks if c["doc_id"] != source])
        interrupted["documents"] = {}
        interrupted["identity"]["bundle"] = fingerprint(m)
        path.write_text(json.dumps(interrupted))
        # Even a correctly bound cursor cannot continue an empty document.
        cursor = encode_cursor({**value, "bundle": fingerprint(m),
                                "state": hashlib.sha256(path.read_bytes()).hexdigest()})
    before = files_snapshot(tmp_path), set(tmp_path.rglob("*"))
    request.getfixturevalue("forbid_io")
    code, out, err = invoke(path, bundle, capsys, source, list_chunks=True, page_size=2, cursor=cursor,
                            chunk_status=chunk_status)
    assert code == 1 and out == ""
    assert err == "ragwrench port: Invalid chunk continuation cursor or snapshot mismatch; restart without --cursor\n"
    assert (files_snapshot(tmp_path), set(tmp_path.rglob("*"))) == before


@pytest.mark.parametrize("extra", [
    ["--page-size", "0"], ["--page-size", "-1"], ["--page-size", "1.5"], ["--page-size", "private-value"],
    ["--page-size", "2"], ["--cursor", "private-cursor"],
    ["--list-chunks", "--cursor", "private-cursor"],
    ["--list-chunks", "--chunk", "private-chunk", "--page-size", "2"],
])
def test_chunk_page_cli_argument_errors(tmp_path, capsys, monkeypatch, forbid_io, extra):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid pagination arguments read saved state")
    monkeypatch.setattr("ragwrench.port.status.read_json", forbidden)
    monkeypatch.setattr("ragwrench.port.status.read_json_snapshot", forbidden)
    path = tmp_path / "absent" / "mapping.json"
    with pytest.raises(SystemExit) as exc:
        main(["import-status", "--state", str(path), "--bundle", str(tmp_path / "bundle"),
              "--document", "private-document"] + extra)
    captured = capsys.readouterr()
    assert exc.value.code == 2 and not captured.out and "private" not in captured.err
    assert not path.parent.exists()


@pytest.mark.parametrize("options", [
    {"page_size": 0}, {"page_size": -1}, {"page_size": 1.5}, {"page_size": True}, {"page_size": "2"},
    {"page_size": 2, "list_chunks": False}, {"cursor": "private-cursor"},
    {"page_size": 2, "chunk": "private-chunk"},
])
def test_chunk_page_api_argument_errors(tmp_path, monkeypatch, forbid_io, options):
    from ragwrench.port.status import import_status

    def forbidden(*args, **kwargs):
        pytest.fail("invalid pagination arguments read saved state")
    monkeypatch.setattr("ragwrench.port.status.read_json", forbidden)
    monkeypatch.setattr("ragwrench.port.status.read_json_snapshot", forbidden)
    with pytest.raises(PortError):
        import_status(tmp_path / "absent.json", tmp_path / "absent-bundle", document="private-document",
                      **{"list_chunks": True, **options})


def test_chunk_page_reads_one_state_snapshot(first_page, bundle, monkeypatch, forbid_io):
    from pathlib import Path
    from ragwrench.port.status import import_status

    path, source, first = first_page
    original = Path.read_bytes
    reads = []
    before = original(path)
    def replace_after_read(selected):
        data = original(selected)
        if selected == path:
            reads.append(selected)
            path.write_bytes(data + b" ")  # Simulate a concurrent atomic writer after our snapshot.
        return data
    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    assert import_status(path, bundle, document=source, list_chunks=True, page_size=2) == first
    assert reads == [path]
    assert decode_cursor(first["bundle_progress"]["next_cursor"])["state"] == hashlib.sha256(before).hexdigest()


@pytest.mark.parametrize("original", [None, "completed", "unvisited"])
@pytest.mark.parametrize("requested", [None, "completed", "pending", "unvisited", "skipped"])
def test_chunk_filter_cursor_binding(first_page, bundle, tmp_path, capsys, original, requested, request):
    path, source, _ = first_page
    code, out, err = invoke(path, bundle, capsys, source, list_chunks=True,
                            page_size=1, chunk_status=original)
    assert code == 0 and not err
    first = json.loads(out)
    cursor = first["bundle_progress"]["next_cursor"]
    assert decode_cursor(cursor) == {
        "version": 1 if original is None else 2,
        "document": source, "bundle": first["bundle_progress"]["manifest_fingerprint"],
        "state": hashlib.sha256(path.read_bytes()).hexdigest(),
        "after": first["bundle_progress"]["chunks"][-1]["source_id"],
        **({"chunk_status": original} if original is not None else {}),
    }
    copy = tmp_path / "copy.json"
    copy.write_bytes(path.read_bytes())
    request.getfixturevalue("forbid_io")
    before = files_snapshot(tmp_path), set(tmp_path.rglob("*"))
    # Exact bytes at a different path and page-size change are accepted.
    code, out, err = invoke(copy, bundle, capsys, source, list_chunks=True,
                            page_size=9, cursor=cursor, chunk_status=requested)
    if original != requested:
        assert code == 1 and out == ""
        assert err == "ragwrench port: Invalid chunk continuation cursor or snapshot mismatch; restart without --cursor\n"
    else:
        assert code == 0 and not err
        page = json.loads(out)["bundle_progress"]
        assert page["next_cursor"] is None
        if requested is not None:
            assert all(r["status"] == requested for r in page["chunks"])
        assert all(r["source_id"] > decode_cursor(cursor)["after"] for r in page["chunks"])
    assert (files_snapshot(tmp_path), set(tmp_path.rglob("*"))) == before


@pytest.mark.parametrize("extra", [
    ["--chunk-status", "all", "--list-chunks", "--bundle", "missing", "--document", "d"],
    ["--chunk-status", "Completed"], ["--chunk-status", ""],
    ["--chunk-status", "completed"],
    ["--chunk-status", "pending", "--bundle", "missing", "--document", "d"],
    ["--chunk-status", "pending", "--list-chunks"],
    ["--chunk-status", "pending", "--list-chunks", "--bundle", "missing"],
    ["--chunk-status", "pending", "--list-chunks", "--document", "d"],
    ["--chunk-status", "pending", "--chunk", "c", "--bundle", "missing", "--document", "d"],
    ["--chunk-status", "pending", "--list-chunks", "--chunk", "c", "--bundle", "missing", "--document", "d"],
])
def test_chunk_filter_cli_argument_errors(tmp_path, capsys, monkeypatch, forbid_io, extra):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid chunk filter attempted to read state")
    monkeypatch.setattr("ragwrench.port.status.read_json", forbidden)
    monkeypatch.setattr("ragwrench.port.status.read_json_snapshot", forbidden)
    with pytest.raises(SystemExit) as exc:
        main(["import-status", "--state", str(tmp_path / "missing" / "state.json")] + extra)
    captured = capsys.readouterr()
    assert exc.value.code == 2 and captured.out == ""
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("options", [
    {"chunk_status": "all"}, {"chunk_status": "Completed"}, {"chunk_status": ""},
    {"chunk_status": []}, {"chunk_status": True},
    {"chunk_status": "pending", "list_chunks": False},
    {"chunk_status": "pending", "chunk": "c"},
    {"chunk_status": "pending", "document": None},
    {"chunk_status": "pending", "bundle": None},
])
def test_chunk_filter_api_argument_errors(tmp_path, monkeypatch, forbid_io, options):
    from ragwrench.port.status import import_status
    def forbidden(*args, **kwargs):
        pytest.fail("invalid chunk filter attempted to read state")
    monkeypatch.setattr("ragwrench.port.status.read_json", forbidden)
    monkeypatch.setattr("ragwrench.port.status.read_json_snapshot", forbidden)
    with pytest.raises(PortError):
        import_status(tmp_path / "missing" / "state.json",
                      **{"bundle": tmp_path, "document": "d", "list_chunks": True, **options})
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("boundary", ["last-match", "outside-filter", "missing-filter", "null-filter"])
def test_chunk_filtered_cursor_boundary(first_page, bundle, capsys, forbid_io, boundary):
    path, source, first = first_page
    value = decode_cursor(first["bundle_progress"]["next_cursor"])
    # Last matching ID and a real full-list ID excluded by the filter are both invalid.
    code, out, err = invoke(path, bundle, capsys, source, list_chunks=True)
    assert code == 0 and not err
    rows = json.loads(out)["bundle_progress"]["chunks"]
    completed = [row for row in rows if row["status"] == "completed"]
    value.update(version=2, chunk_status="completed", after=completed[-1]["source_id"])
    if boundary == "outside-filter":
        value["after"] = next(row["source_id"] for row in rows if row["status"] != "completed")
    elif boundary == "missing-filter":
        del value["chunk_status"]
    elif boundary == "null-filter":
        value["chunk_status"] = None
    code, out, err = invoke(path, bundle, capsys, source, list_chunks=True, chunk_status="completed",
                            page_size=1, cursor=encode_cursor(value))
    assert code == 1 and not out
    assert err == "ragwrench port: Invalid chunk continuation cursor or snapshot mismatch; restart without --cursor\n"


def assert_summary(path, bundle, capsys, full):
    """Summary omits exactly the two detail arrays; explicit False preserves full JSON."""
    from ragwrench.port.status import import_status

    expected = {key: value for key, value in full.items() if key != "documents"}
    expected["bundle_progress"] = {key: value for key, value in full["bundle_progress"].items()
                                   if key != "documents"}
    code, out, err = invoke(path, bundle, capsys, summary_only=True)
    assert code == 0 and not err
    assert json.loads(out) == expected
    assert import_status(path, bundle, summary_only=True) == expected
    assert import_status(path, bundle, summary_only=False) == full
    counts = {"dataset_creation": int(full["dataset_pending"])}
    for stage in ("document_creation", "bootstrap_cleanup", "chunk_creation", "enabled_flag"):
        counts[stage] = sum(row["pending_stage"] == stage for row in full["documents"])
        # The same saved lossy/byte/skip plan and legacy defaults apply to filtering.
        matching = [row for row in full["documents"] if row["pending_stage"] == stage]
        sources = {row["source_id"] for row in matching}
        filtered = {**full, "documents": matching, "bundle_progress": {
            **full["bundle_progress"], "documents": [row for row in full["bundle_progress"]["documents"]
                                                      if row["source_id"] in sources]}}
        assert import_status(path, bundle, pending_stage=stage) == filtered
    counted_expected = {**expected, "pending_counts": counts}
    code, out, err = invoke(path, bundle, capsys, summary_only=True, pending_counts=True)
    assert code == 0 and not err and json.loads(out) == counted_expected
    assert all(type(value) is int for value in json.loads(out)["pending_counts"].values())
    assert import_status(path, bundle, summary_only=True, pending_counts=True) == counted_expected
    assert import_status(path, bundle, summary_only=True, pending_counts=False) == expected
    assert import_status(path, bundle, pending_counts=False) == full


@pytest.mark.parametrize("pending_target", [None, "t-three"])
@pytest.mark.parametrize("skipped", [False, True])
def test_bundle_status_summary_contract(interrupted, bundle, tmp_path, capsys, request, skipped, pending_target):
    manifest, docs, chunks = load_bundle(bundle)
    if skipped:
        chunks[4]["metadata"] = {"private": "not restorable"}
        manifest = write_bundle(bundle, manifest["source"], manifest["settings"], docs, chunks)
        interrupted["identity"].update(bundle=fingerprint(manifest), on_error="skip")
    next(iter(interrupted["documents"].values()))["pending"]["target"] = pending_target
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(interrupted))
    request.getfixturevalue("forbid_io")
    before = files_snapshot(tmp_path), set(tmp_path.rglob("*"))
    code, out, err = invoke(path, bundle, capsys)
    assert code == 0 and not err
    full = json.loads(out)
    p = full["bundle_progress"]
    assert {k: v for k, v in p.items() if k not in ("bundle", "manifest_fingerprint", "documents")} == {
        "planned_documents": 3, "planned_chunks": 15 - skipped, "skipped_chunks": int(skipped),
        "warnings": [], "completed_documents": 0, "remaining_documents": 3,
        "remaining_chunks": 13 - skipped, "pending_chunks": 1, "unvisited_documents": 2,
        "unvisited_chunks": 12 - skipped,
    }
    assert full["completed_chunks"] == 2 and p["manifest_fingerprint"] == fingerprint(manifest)
    assert_summary(path, bundle, capsys, full)
    assert (files_snapshot(tmp_path), set(tmp_path.rglob("*"))) == before


@pytest.mark.parametrize("extra,options", [
    (["--document", "d"], {"document": "d"}),
    (["--document", ""], {"document": ""}),
    (["--chunk", "c"], {"chunk": "c"}),
    (["--chunk", ""], {"chunk": ""}),
    (["--list-chunks"], {"list_chunks": True}),
    (["--chunk-status", "completed"], {"chunk_status": "completed"}),
    (["--page-size", "1"], {"page_size": 1}),
    (["--cursor", "private-token"], {"cursor": "private-token"}),
    (["--cursor", ""], {"cursor": ""}),
    (["--document", "d", "--chunk", "c"], {"document": "d", "chunk": "c"}),
    (["--document", "d", "--list-chunks"], {"document": "d", "list_chunks": True}),
    (["--document", "d", "--list-chunks", "--chunk-status", "pending"],
     {"document": "d", "list_chunks": True, "chunk_status": "pending"}),
    (["--document", "d", "--list-chunks", "--page-size", "1", "--cursor", "private-token"],
     {"document": "d", "list_chunks": True, "page_size": 1, "cursor": "private-token"}),
])
@pytest.mark.parametrize("pending_counts", [False, True])
@pytest.mark.parametrize("has_bundle", [False, True])
def test_summary_argument_conflicts(tmp_path, capsys, monkeypatch, forbid_io, extra, options, has_bundle, pending_counts):
    from ragwrench.port.status import import_status

    def forbidden(*args, **kwargs):
        pytest.fail("invalid summary combination attempted state/bundle reads")
    for name in ("ragwrench.port.status.read_json", "ragwrench.port.status.read_json_snapshot", "ragwrench.port.status.load_bundle"):
        monkeypatch.setattr(name, forbidden)
    path = tmp_path / "absent" / "state.json"
    args = ["import-status", "--state", str(path), "--summary-only"]
    if pending_counts:
        args += ["--pending-counts"]
    if has_bundle:
        args += ["--bundle", str(tmp_path / "bundle")]
    with pytest.raises(SystemExit) as exc:
        main(args + extra)
    captured = capsys.readouterr()
    assert exc.value.code == 2 and not captured.out
    assert "--summary-only cannot be combined" in captured.err
    with pytest.raises(PortError, match="Summary-only cannot be combined"):
        import_status(path, tmp_path if has_bundle else None, summary_only=True, pending_counts=pending_counts, **options)
    assert not path.parent.exists()


def test_summary_empty_bundle(interrupted, bundle, tmp_path, capsys, request):
    manifest, _, _ = load_bundle(bundle)
    manifest = write_bundle(bundle, manifest["source"], manifest["settings"], [], [])
    interrupted["identity"]["bundle"] = fingerprint(manifest)
    interrupted.update(dataset=None, documents={})
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(interrupted))
    request.getfixturevalue("forbid_io")
    before = files_snapshot(tmp_path), set(tmp_path.rglob("*"))
    code, out, err = invoke(path, bundle, capsys)
    assert code == 0 and not err
    full = json.loads(out)
    p = full["bundle_progress"]
    assert all(p[field] == 0 for field in ("planned_documents", "planned_chunks", "skipped_chunks",
        "completed_documents", "remaining_documents", "remaining_chunks", "pending_chunks",
        "unvisited_documents", "unvisited_chunks"))
    assert p["documents"] == full["documents"] == p["warnings"] == []
    assert_summary(path, bundle, capsys, full)
    assert (files_snapshot(tmp_path), set(tmp_path.rglob("*"))) == before


@pytest.mark.parametrize("case,error", [
    ("schema", "Bundle schema violation"),
    ("relationship", "Chunk references an unknown document"),
    ("content-hash", "Chunk content_sha256 mismatch"),
    ("counts", "Manifest counts do not match JSONL records"),
])
@pytest.mark.parametrize("pending_counts", [False, True])
def test_summary_rehashed_invalid_bundle(interrupted, bundle, tmp_path, capsys, request, monkeypatch, case, error, pending_counts):
    from ragwrench.port.status import import_status

    manifest, _, chunks = load_bundle(bundle)
    # Rehash invalid detail outside the visited document, with matching saved
    # fingerprint, so neither file hashing nor fingerprint alone can catch it.
    if case == "schema":
        chunks[-1]["content"] = 42
    elif case == "relationship":
        chunks[-1]["doc_id"] = "unknown-document"
    elif case == "content-hash":
        chunks[-1]["content"] = "private changed content"
    else:
        manifest["counts"]["chunks"] += 1
    blob = "".join(json.dumps(chunk) + "\n" for chunk in chunks).encode()
    (bundle / "chunks.jsonl").write_bytes(blob)
    manifest["sha256"]["chunks.jsonl"] = hashlib.sha256(blob).hexdigest()
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    interrupted["identity"]["bundle"] = fingerprint(manifest)
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(interrupted))
    request.getfixturevalue("forbid_io")
    def forbidden(*args, **kwargs):
        pytest.fail("invalid bundle reached saved preflight")
    monkeypatch.setattr("ragwrench.port.status.make_plan", forbidden)
    before = files_snapshot(tmp_path), set(tmp_path.rglob("*"))
    code, out, err = invoke(path, bundle, capsys, summary_only=True, pending_counts=pending_counts)
    assert code == 1 and not out and error in err and "private" not in err
    with pytest.raises(PortError, match=error):
        import_status(path, bundle, summary_only=True, pending_counts=pending_counts)
    assert (files_snapshot(tmp_path), set(tmp_path.rglob("*"))) == before


@pytest.mark.parametrize("has_bundle", [False, True])
@pytest.mark.parametrize("extra,options", [
    ([], {}),
    (["--document", ""], {"document": ""}),
    (["--chunk", "c"], {"chunk": "c"}),
    (["--list-chunks"], {"list_chunks": True}),
    (["--chunk-status", "pending"], {"chunk_status": "pending"}),
    (["--page-size", "1"], {"page_size": 1}),
    (["--cursor", ""], {"cursor": ""}),
    (["--document", "d", "--list-chunks", "--chunk-status", "pending", "--page-size", "1"],
     {"document": "d", "list_chunks": True, "chunk_status": "pending", "page_size": 1}),
])
def test_pending_counts_requires_summary(tmp_path, capsys, monkeypatch, forbid_io, has_bundle, extra, options):
    from ragwrench.port.status import import_status

    def forbidden(*args, **kwargs):
        pytest.fail("pending-counts without summary attempted state/bundle reads")
    for name in ("ragwrench.port.status.read_json", "ragwrench.port.status.read_json_snapshot", "ragwrench.port.status.load_bundle"):
        monkeypatch.setattr(name, forbidden)
    path = tmp_path / "absent" / "state.json"
    args = ["import-status", "--state", str(path), "--pending-counts"]
    if has_bundle:
        args += ["--bundle", str(tmp_path / "bundle")]
    with pytest.raises(SystemExit) as exc:
        main(args + extra)
    captured = capsys.readouterr()
    assert exc.value.code == 2 and not captured.out
    assert "--pending-counts requires --summary-only" in captured.err
    with pytest.raises(PortError, match="Pending counts require summary-only"):
        import_status(path, tmp_path if has_bundle else None, pending_counts=True, **options)
    assert not path.parent.exists()


@pytest.mark.parametrize("stage", ["document_creation", "bootstrap_cleanup", "chunk_creation", "enabled_flag"])
def test_pending_counts_all_bundle_documents(interrupted, bundle, tmp_path, capsys, request, stage):
    from ragwrench.port.status import import_status

    _, docs, chunks = load_bundle(bundle)
    interrupted["documents"] = {}
    for i, doc in enumerate(docs):
        first = next(c["id"] for c in chunks if c["doc_id"] == doc["id"])
        interrupted["documents"][doc["id"]] = {
            "id": None if stage == "document_creation" else f"target-{i}",
            "phase": {"document_creation": "creating", "bootstrap_cleanup": "clearing"}.get(stage, "chunks"),
            "chunks": {},
            "pending": {"source": first, "target": "t-pending" if stage == "enabled_flag" else None}
                       if stage in ("chunk_creation", "enabled_flag") else None,
        }
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(interrupted))
    request.getfixturevalue("forbid_io")
    before = files_snapshot(tmp_path), set(tmp_path.rglob("*"))
    full = import_status(path, bundle)
    assert all(row["pending_stage"] == stage for row in full["documents"])
    counts = dict(dataset_creation=0, document_creation=0, bootstrap_cleanup=0, chunk_creation=0, enabled_flag=0)
    counts[stage] = 3
    assert import_status(path, bundle, summary_only=True, pending_counts=True)["pending_counts"] == counts
    assert_summary(path, bundle, capsys, full)
    assert (files_snapshot(tmp_path), set(tmp_path.rglob("*"))) == before
