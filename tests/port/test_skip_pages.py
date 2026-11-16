"""Bundle-wide saved-preflight exclusions and compound-boundary continuations."""
import base64
from copy import deepcopy
import hashlib
import json
import shutil

import pytest

from ragwrench.port.bundle import fingerprint, load_bundle, write_bundle
from ragwrench.port.cli import main
from ragwrench.port.common import PortError, content_hash, locked
from ragwrench.port.status import import_status
from .test_import_status_bundle import interrupted, forbid_io, files_snapshot, encode_cursor, decode_cursor

META = "Unsupported fields: chunk metadata (including image/layout/tags); use --allow-lossy to acknowledge"
BYTE = "Chunk exceeds configured target UTF-8 byte limit"


@pytest.fixture
def skip_mix(interrupted, bundle, tmp_path):
    m, docs, chunks = load_bundle(bundle)
    specs = {"a": [("c3", "byte"), ("c1", "good"), ("c2", "meta"), ("c0", "good")],
             "b": [("c3", "byte"), ("c2", "good"), ("c1", "meta")],
             "c": [("c1", "byte"), ("c0", "meta")], "d": [], "e": [("c0", "good")]}
    documents, records = [], []
    for source, entries in reversed(list(specs.items())):
        documents.append({**deepcopy(docs[0]), "id": source})
        for position, (cid, mode) in enumerate(entries):
            text = "界" * 30 if mode == "byte" else f"text {cid}"
            records.append({**deepcopy(chunks[0]), "id": cid, "doc_id": source, "position": position,
                            "content": text, "content_sha256": content_hash(text),
                            "metadata": {"layout": True} if mode == "meta" else {}})
    m = write_bundle(bundle, m["source"], m["settings"], documents, records)
    interrupted["identity"].update(bundle=fingerprint(m), max_chunk_bytes=20, on_error="skip")
    interrupted["documents"] = {
        "a": {"id": "ta", "phase": "chunks", "chunks": {"c0": "tc0"},
              "pending": {"source": "c1", "target": "tc1"}},
        "c": {"id": "tc", "phase": "complete", "chunks": {}, "pending": None},
        "d": {"id": None, "phase": "new", "chunks": {}, "pending": None},
        "e": {"id": "te", "phase": "complete", "chunks": {"c0": "tc0"}, "pending": None}}
    path = tmp_path / "state.json"
    path.write_text(json.dumps(interrupted))
    return path, bundle, interrupted


def expected_rows(full):
    return [{"document_source_id": d["source_id"], **c}
            for d in full["bundle_progress"]["documents"] for c in d["skipped_chunks"]]


def invoke(path, bundle, capsys, size=None, cursor=None):
    args = ["import-status", "--state", str(path), "--bundle", str(bundle), "--list-skipped-chunks"]
    if size is not None: args += ["--skip-page-size", str(size)]
    if cursor is not None: args += ["--skip-cursor", cursor]
    assert main(args) == 0
    out = capsys.readouterr()
    assert not out.err
    result = json.loads(out.out)
    assert result == import_status(path, bundle, list_skipped_chunks=True, skip_page_size=size, skip_cursor=cursor)
    return result


def assert_page(result, full, rows, paged=True):
    p = result["bundle_progress"]
    assert p.pop("skipped_chunk_details") == rows
    token = p.pop("next_skip_cursor") if paged else None
    assert result == full
    return token


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("size", [None, 1, 2, 3, 6, 7, 10**30])
def test_bundle_skip_contract(skip_mix, capsys, request, legacy, size):
    path, bundle, state = skip_mix
    if legacy:
        del state["identity"]["api_profile"]
        path.write_text(json.dumps(state))
    with locked(path):
        request.getfixturevalue("forbid_io")
        full = import_status(path, bundle)
        assert import_status(path, bundle, list_skipped_chunks=False, skip_page_size=None, skip_cursor=None) == full
        rows = expected_rows(full)
        assert [(r["document_source_id"], r["source_id"], r["reason"]) for r in rows] == [
            ("a", "c2", META), ("a", "c3", BYTE), ("b", "c1", META),
            ("b", "c3", BYTE), ("c", "c0", META), ("c", "c1", BYTE)]
        before = files_snapshot(path.parent), set(path.parent.rglob("*"))
        offset, token, seen = 0, None, []
        while True:
            page = invoke(path, bundle, capsys, size, token)
            actual = page["bundle_progress"]["skipped_chunk_details"]
            assert actual == rows[offset:offset+size] if size is not None else actual == rows
            seen.extend(actual)
            token = assert_page(page, full, actual, size is not None)
            offset += len(actual)
            if offset == len(rows):
                assert token is None
                break
            assert token and len(token) <= 2048
        assert seen == rows
        assert (files_snapshot(path.parent), set(path.parent.rglob("*"))) == before


@pytest.mark.parametrize("boundary", ["initial", "dataset-pending", "no-skips", "empty", "all-skipped"])
def test_bundle_skip_boundaries(skip_mix, capsys, request, boundary):
    path, bundle, state = skip_mix
    m, docs, chunks = load_bundle(bundle)
    if boundary in ("initial", "dataset-pending"):
        state.update(dataset=None, dataset_pending=boundary == "dataset-pending", documents={})
    else:
        state["documents"] = {}
        if boundary == "empty": docs, chunks = [], []
        elif boundary == "no-skips": chunks = [c for c in chunks if not c["metadata"] and len(c["content"].encode()) <= 20]
        else: chunks = [c for c in chunks if c["metadata"] or len(c["content"].encode()) > 20]
        state["identity"]["bundle"] = fingerprint(write_bundle(bundle, m["source"], m["settings"], docs, chunks))
    path.write_text(json.dumps(state))
    request.getfixturevalue("forbid_io")
    full = import_status(path, bundle)
    rows = expected_rows(full)
    assert_page(invoke(path, bundle, capsys), full, rows, False)
    token = assert_page(invoke(path, bundle, capsys, 10**30), full, rows)
    assert token is None


CONFLICTS = [
    ({"summary_only": True}, ["--summary-only"]), ({"pending_counts": True}, ["--pending-counts"]),
    ({"pending_stage": "enabled_flag"}, ["--pending-stage", "enabled_flag"]),
    ({"document": ""}, ["--document", ""]), ({"chunk": ""}, ["--chunk", ""]),
    ({"list_chunks": True}, ["--list-chunks"]), ({"chunk_status": "skipped"}, ["--chunk-status", "skipped"]),
    ({"page_size": 1}, ["--page-size", "1"]), ({"cursor": ""}, ["--cursor", ""]),
    ({"document_page_size": 1}, ["--document-page-size", "1"]), ({"document_cursor": ""}, ["--document-cursor", ""])]


@pytest.mark.parametrize("options,extra", CONFLICTS + [
    ({"skip_page_size": 0}, ["--skip-page-size", "0"]),
    ({"skip_page_size": -1}, ["--skip-page-size", "-1"]),
    ({"skip_page_size": 1.5}, ["--skip-page-size", "1.5"]),
    ({"skip_page_size": "private"}, ["--skip-page-size", "private"]),
    ({"skip_cursor": ""}, ["--skip-cursor", ""])])
def test_skip_arguments_before_reads(tmp_path, monkeypatch, capsys, forbid_io, options, extra):
    def forbidden(*a, **k): pytest.fail("argument error read inputs")
    for name in ("read_json", "read_json_snapshot", "load_bundle"):
        monkeypatch.setattr(f"ragwrench.port.status.{name}", forbidden)
    with pytest.raises(PortError):
        import_status(tmp_path / "absent", tmp_path, list_skipped_chunks=True, **options)
    with pytest.raises(SystemExit) as exc:
        main(["import-status", "--state", str(tmp_path / "absent"), "--bundle", str(tmp_path), "--list-skipped-chunks", *extra])
    assert exc.value.code == 2 and not capsys.readouterr().out


@pytest.mark.parametrize("options,extra,has_bundle", [
    ({"list_skipped_chunks": True}, ["--list-skipped-chunks"], False),
    ({"skip_page_size": 1}, ["--skip-page-size", "1"], True),
    ({"skip_page_size": 1}, ["--skip-page-size", "1"], False),
    ({"skip_cursor": ""}, ["--skip-cursor", ""], True),
    ({"skip_page_size": 1, "skip_cursor": "x"}, ["--skip-page-size", "1", "--skip-cursor", "x"], True)])
def test_skip_requires_scope(tmp_path, monkeypatch, capsys, forbid_io, options, extra, has_bundle):
    def forbidden(*a, **k): pytest.fail("argument error read inputs")
    for name in ("read_json", "read_json_snapshot", "load_bundle"):
        monkeypatch.setattr(f"ragwrench.port.status.{name}", forbidden)
    with pytest.raises(PortError): import_status(tmp_path / "absent", tmp_path if has_bundle else None, **options)
    args = ["import-status", "--state", str(tmp_path / "absent"), *extra]
    if has_bundle: args += ["--bundle", str(tmp_path)]
    with pytest.raises(SystemExit) as exc: main(args)
    assert exc.value.code == 2 and not capsys.readouterr().out


@pytest.mark.parametrize("value", [True, False, 1.0, "2"])
def test_skip_api_integer_only(skip_mix, forbid_io, value):
    with pytest.raises(PortError): import_status(*skip_mix[:2], list_skipped_chunks=True, skip_page_size=value)


@pytest.mark.parametrize("case", ["empty", "alphabet", "unicode", "oversized", "truncated", "padded", "non-json", "array",
    "noncanonical", "version", "bool-version", "extra", "kind", "state", "bundle", "after-type", "after-length",
    "after-member", "after-unknown", "after-nonskip", "after-cross-document", "after-final", "state-bytes",
    "state-order", "state-unknown", "state-value", "chunk-v1", "chunk-v2", "document"])
def test_skip_cursor_rejection(skip_mix, capsys, request, case):
    path, bundle, state = skip_mix
    token = invoke(path, bundle, capsys, 1)["bundle_progress"]["next_skip_cursor"]
    value = decode_cursor(token)
    if case == "empty": token = ""
    elif case == "alphabet": token = "../private"
    elif case == "unicode": token = "界"
    elif case == "oversized": token = "x" * 2049
    elif case == "truncated": token = token[:1]
    elif case == "padded": token += "="
    elif case == "non-json": token = base64.urlsafe_b64encode(b"private").decode().rstrip("=")
    elif case == "array": token = encode_cursor([])
    elif case == "noncanonical": token = base64.urlsafe_b64encode(json.dumps(value, indent=1).encode()).decode().rstrip("=")
    elif case == "state-bytes": path.write_bytes(path.read_bytes() + b" ")
    elif case == "state-order": path.write_text(json.dumps(state, sort_keys=True))
    elif case == "state-unknown": path.write_text(json.dumps({**state, "extra": True}))
    elif case == "state-value":
        state["identity"]["base_url"] = "http://different.invalid"
        path.write_text(json.dumps(state))
    elif case.startswith("chunk-"):
        token = import_status(path, bundle, document="a", list_chunks=True, page_size=1,
                              chunk_status="skipped" if case == "chunk-v2" else None)["bundle_progress"]["next_cursor"]
    elif case == "document":
        token = encode_cursor({"version": 1, "kind": "pending-stage-documents", "scope": "bundle",
            "state": value["state"], "bundle": value["bundle"], "pending_stage": "enabled_flag", "after": "a"})
    else:
        key, changed = {"version": ("version", 99), "bool-version": ("version", True), "extra": ("extra", 1),
            "kind": ("kind", "chunks"), "state": ("state", "f"*64), "bundle": ("bundle", "f"*64),
            "after-type": ("after", "a/c2"), "after-length": ("after", ["a"]), "after-member": ("after", ["a", 2]),
            "after-unknown": ("after", ["unknown", "c2"]), "after-nonskip": ("after", ["a", "c0"]),
            "after-cross-document": ("after", ["b", "c2"]), "after-final": ("after", ["c", "c1"])}[case]
        value[key] = changed; token = encode_cursor(value)
    request.getfixturevalue("forbid_io")
    before = files_snapshot(path.parent)
    with pytest.raises(PortError, match="Invalid skip continuation"):
        import_status(path, bundle, list_skipped_chunks=True, skip_page_size=1, skip_cursor=token)
    assert main(["import-status", "--state", str(path), "--bundle", str(bundle), "--list-skipped-chunks",
                 "--skip-page-size", "1", "--skip-cursor", token]) == 1
    out = capsys.readouterr()
    assert not out.out and "Invalid skip continuation" in out.err and "private" not in out.err
    assert files_snapshot(path.parent) == before


def test_skip_cursor_copies_sizes_snapshot_and_kinds(skip_mix, tmp_path, monkeypatch, capsys, forbid_io):
    from ragwrench.port import status
    path, bundle, _ = skip_mix
    full = import_status(path, bundle)
    rows = expected_rows(full)
    original = status.read_json_snapshot
    data, calls = path.read_bytes(), []
    def changed_after_read(p):
        calls.append(p)
        result = original(p)
        path.write_bytes(data + b" ")
        return result
    monkeypatch.setattr(status, "read_json_snapshot", changed_after_read)
    page = import_status(path, bundle, list_skipped_chunks=True, skip_page_size=1)
    token = page["bundle_progress"]["next_skip_cursor"]
    assert calls == [path] and decode_cursor(token)["state"] == hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(status, "read_json_snapshot", original)
    path.write_bytes(data)
    copy = tmp_path / "copy.json"; copy.write_bytes(data)
    other = tmp_path / "other"; shutil.copytree(bundle, other)
    copied_full = import_status(copy, other)
    assert import_status(copy, other, list_skipped_chunks=True, skip_page_size=1)["bundle_progress"]["next_skip_cursor"] == token
    page = invoke(copy, other, capsys, 3, token)
    next_token = assert_page(page, copied_full, rows[1:4])
    assert assert_page(invoke(copy, other, capsys, 2, next_token), copied_full, rows[4:]) is None
    for chunk_status in (None, "skipped"):
        with pytest.raises(PortError, match="Invalid chunk continuation"):
            import_status(path, bundle, document="a", list_chunks=True, page_size=1, cursor=token, chunk_status=chunk_status)
    with pytest.raises(PortError, match="Invalid document continuation"):
        import_status(path, bundle, pending_stage="enabled_flag", document_page_size=1, document_cursor=token)
    m, docs, chunks = load_bundle(bundle)
    state = json.loads(data)
    m["settings"] = {"different": True}
    (bundle / "manifest.json").write_text(json.dumps(m))
    state["identity"]["bundle"] = fingerprint(m)
    path.write_text(json.dumps(state))
    value = decode_cursor(token); value["state"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(PortError, match="Invalid skip continuation"):
        import_status(path, bundle, list_skipped_chunks=True, skip_page_size=1, skip_cursor=encode_cursor(value))


@pytest.mark.parametrize("mode", ["list", "page", "cursor"])
@pytest.mark.parametrize("case,error", [("schema", "Bundle schema violation"),
    ("relationship", "Chunk references an unknown document"), ("content-hash", "Chunk content_sha256 mismatch"),
    ("counts", "Manifest counts do not match JSONL records")])
def test_skip_validates_non_skipped_off_page_bundle(skip_mix, capsys, request, monkeypatch, mode, case, error):
    path, bundle, state = skip_mix
    m, _, chunks = load_bundle(bundle)
    # Last document e has no skips, and cannot appear in this output.
    if case == "schema": chunks[-1]["content"] = 42
    elif case == "relationship": chunks[-1]["doc_id"] = "private-unknown"
    elif case == "content-hash": chunks[-1]["content"] = "private changed"
    else: m["counts"]["chunks"] += 1
    blob = "".join(json.dumps(c) + "\n" for c in chunks).encode()
    (bundle / "chunks.jsonl").write_bytes(blob)
    m["sha256"]["chunks.jsonl"] = hashlib.sha256(blob).hexdigest()
    (bundle / "manifest.json").write_text(json.dumps(m))
    state["identity"]["bundle"] = fingerprint(m)
    path.write_text(json.dumps(state))
    request.getfixturevalue("forbid_io")
    monkeypatch.setattr("ragwrench.port.status.make_plan", lambda *a, **k: pytest.fail("invalid bundle reached preflight"))
    options = {"list_skipped_chunks": True}
    args = ["import-status", "--state", str(path), "--bundle", str(bundle), "--list-skipped-chunks"]
    if mode != "list":
        options["skip_page_size"] = 1; args += ["--skip-page-size", "1"]
    if mode == "cursor":
        options["skip_cursor"] = "private-invalid"; args += ["--skip-cursor", "private-invalid"]
    before = files_snapshot(path.parent)
    with pytest.raises(PortError, match=error): import_status(path, bundle, **options)
    assert main(args) == 1
    out = capsys.readouterr()
    assert not out.out and error in out.err and "private" not in out.err and "cursor" not in out.err
    assert files_snapshot(path.parent) == before


@pytest.mark.parametrize("mode", ["list", "page", "cursor"])
@pytest.mark.parametrize("case", ["off-page-skipped", "off-page-nonskipped", "incomplete", "pending-skip"])
def test_skip_invalid_off_page_mapping(skip_mix, capsys, request, mode, case):
    path, bundle, state = skip_mix
    if case == "off-page-skipped": state["documents"]["c"]["chunks"]["c1"] = "tc1"
    elif case == "off-page-nonskipped": state["documents"]["e"]["chunks"]["foreign"] = "tf"
    elif case == "incomplete": state["documents"]["e"]["chunks"] = {}
    else: state["documents"]["a"]["pending"]["source"] = "c3"
    path.write_text(json.dumps(state))
    request.getfixturevalue("forbid_io")
    options = dict(list_skipped_chunks=True, skip_page_size=None if mode == "list" else 1,
                   skip_cursor="private-invalid" if mode == "cursor" else None)
    with pytest.raises(PortError, match="Invalid import state"): import_status(path, bundle, **options)
    args = ["import-status", "--state", str(path), "--bundle", str(bundle), "--list-skipped-chunks"]
    if mode != "list": args += ["--skip-page-size", "1"]
    if mode == "cursor": args += ["--skip-cursor", "private-invalid"]
    assert main(args) == 1
    out = capsys.readouterr()
    assert not out.out and "Invalid import state" in out.err and "private" not in out.err


@pytest.mark.parametrize("mode", ["list", "page"])
@pytest.mark.parametrize("kind", ["missing", "malformed", "symlink", "directory", "oversized"])
def test_skip_state_file_errors(tmp_path, bundle, capsys, forbid_io, kind, mode):
    path = tmp_path / "missing" / "state.json"
    if kind != "missing": path.parent.mkdir()
    if kind == "malformed": path.write_text('{"secret":"private"')
    elif kind == "symlink": path.symlink_to(bundle / "manifest.json")
    elif kind == "directory": path.mkdir()
    elif kind == "oversized":
        with path.open("wb") as stream: stream.truncate(256 * 1024 * 1024 + 1)
    before = set(tmp_path.rglob("*"))
    args = ["import-status", "--state", str(path), "--bundle", str(bundle), "--list-skipped-chunks"]
    if mode == "page": args += ["--skip-page-size", "1"]
    assert main(args) == 1
    out = capsys.readouterr()
    assert not out.out and out.err and "private" not in out.err and "Traceback" not in out.err
    assert set(tmp_path.rglob("*")) == before
