"""Pending-stage pages preserve a fully validated, global offline snapshot."""
from copy import deepcopy
import base64
import hashlib
import json
import shutil

import pytest

from ragwrench.port.bundle import fingerprint, load_bundle, write_bundle
from ragwrench.port.cli import main
from ragwrench.port.common import PortError, locked
from ragwrench.port.status import import_status
from .test_import_status_bundle import files_snapshot, forbid_io, interrupted, encode_cursor, decode_cursor
from .test_pending_stage import mixed, STAGES


def page_result(path, bundle, capsys, stage, size, cursor=None):
    args = ["import-status", "--state", str(path), "--pending-stage", stage,
            "--document-page-size", str(size)]
    if bundle is not None:
        args += ["--bundle", str(bundle)]
    if cursor is not None:
        args += ["--document-cursor", cursor]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert not captured.err
    result = json.loads(captured.out)
    assert import_status(path, bundle, pending_stage=stage, document_page_size=size,
                         document_cursor=cursor) == result
    return result


def assert_slice(result, full, ids):
    expected = deepcopy(full)
    expected["documents"] = [row for row in full["documents"] if row["source_id"] in ids]
    if "bundle_progress" in full:
        expected["bundle_progress"]["documents"] = [row for row in full["bundle_progress"]["documents"]
                                                     if row["source_id"] in ids]
    token = result.pop("next_document_cursor")
    assert result == expected
    assert [r["source_id"] for r in result["documents"]] == ids
    if "bundle_progress" in result:
        assert [r["source_id"] for r in result["bundle_progress"]["documents"]] == ids
    return token


@pytest.mark.parametrize("has_bundle", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("size", [1, 2, 3, 10**30])
def test_document_page_contract(mixed, capsys, monkeypatch, request, has_bundle, legacy, size):
    path, bundle, state = mixed
    if legacy:
        del state["identity"]["api_profile"]
        path.write_text(json.dumps(state))
    with locked(path):
        request.getfixturevalue("forbid_io")
        if not has_bundle:
            bundle = None
            monkeypatch.setattr("ragwrench.port.status.load_bundle", lambda *a: pytest.fail("state-only bundle read"))
        before = files_snapshot(path.parent), set(path.parent.rglob("*"))
        full = import_status(path, bundle)
        assert full["mapped_documents"] == 9 and full["completed_chunks"] == 7
        assert import_status(path, bundle, document_page_size=None, document_cursor=None) == full
        for i, stage in enumerate(STAGES):
            ids = [f"d{2*i:02}", f"d{2*i+1:02}"]
            offset, token, seen = 0, None, []
            while True:
                result = page_result(path, bundle, capsys, stage, size, token)
                token = assert_slice(result, full, ids[offset:offset+size])
                seen.extend(row["source_id"] for row in result["documents"])
                offset += size
                if offset >= len(ids):
                    assert token is None
                    break
                assert isinstance(token, str) and token
            assert seen == ids
            if size >= 2:
                assert import_status(path, bundle, pending_stage=stage) == result
        assert (files_snapshot(path.parent), set(path.parent.rglob("*"))) == before


@pytest.mark.parametrize("has_bundle", [False, True])
@pytest.mark.parametrize("boundary", ["no-match", "initial", "dataset-intent", "empty-bundle"])
def test_document_pages_empty(mixed, capsys, request, boundary, has_bundle):
    path, bundle, state = mixed
    state["documents"] = {k: v for k, v in state["documents"].items() if k >= "d08"}
    if boundary != "no-match":
        state.update(documents={}, dataset=None, dataset_pending=boundary == "dataset-intent")
    if boundary == "empty-bundle":
        m, _, _ = load_bundle(bundle)
        state["identity"]["bundle"] = fingerprint(write_bundle(bundle, m["source"], m["settings"], [], []))
    path.write_text(json.dumps(state))
    request.getfixturevalue("forbid_io")
    bundle = bundle if has_bundle else None
    full = import_status(path, bundle)
    for stage in STAGES:
        assert assert_slice(page_result(path, bundle, capsys, stage, 1), full, []) is None


@pytest.mark.parametrize("options", [
    {"document_page_size": 0}, {"document_page_size": -1}, {"document_page_size": True},
    {"document_page_size": 1.5}, {"document_page_size": "2"},
    {"document_cursor": ""}, {"document_cursor": "private"},
    *[{"document_page_size": 1, **conflict} for conflict in (
        {"summary_only": True}, {"pending_counts": True}, {"document": ""}, {"chunk": ""},
        {"list_chunks": True}, {"chunk_status": "pending"}, {"page_size": 1}, {"cursor": ""})],
])
def test_document_page_api_arguments(tmp_path, monkeypatch, forbid_io, options):
    def forbidden(*a, **k):
        pytest.fail("argument error read input")
    for name in ("read_json", "read_json_snapshot", "load_bundle"):
        monkeypatch.setattr(f"ragwrench.port.status.{name}", forbidden)
    with pytest.raises(PortError):
        import_status(tmp_path / "missing", pending_stage="enabled_flag", **options)


@pytest.mark.parametrize("extra", [
    ["--document-page-size", "0"], ["--document-page-size", "-1"],
    ["--document-page-size", "1.5"], ["--document-page-size", "private"],
    ["--document-cursor", ""], ["--document-cursor", "private"],
    *[["--document-page-size", "1"] + conflict for conflict in (
        ["--summary-only"], ["--pending-counts"], ["--document", ""], ["--chunk", ""],
        ["--list-chunks"], ["--chunk-status", "pending"], ["--page-size", "1"], ["--cursor", ""])],
])
def test_document_page_cli_arguments(tmp_path, capsys, monkeypatch, forbid_io, extra):
    for name in ("read_json", "read_json_snapshot", "load_bundle"):
        monkeypatch.setattr(f"ragwrench.port.status.{name}", lambda *a: pytest.fail("argument error read input"))
    with pytest.raises(SystemExit) as exc:
        main(["import-status", "--state", str(tmp_path / "missing"), "--pending-stage", "enabled_flag"] + extra)
    out = capsys.readouterr()
    assert exc.value.code == 2 and not out.out


@pytest.mark.parametrize("options,extra", [
    ({"document_page_size": 1}, ["--document-page-size", "1"]),
    ({"document_cursor": ""}, ["--document-cursor", ""]),
    ({"document_page_size": 1, "document_cursor": "token"},
     ["--document-page-size", "1", "--document-cursor", "token"]),
])
def test_document_paging_requires_stage(tmp_path, capsys, monkeypatch, forbid_io, options, extra):
    for name in ("read_json", "read_json_snapshot", "load_bundle"):
        monkeypatch.setattr(f"ragwrench.port.status.{name}", lambda *a: pytest.fail("missing stage read input"))
    with pytest.raises(PortError):
        import_status(tmp_path / "missing", **options)
    with pytest.raises(SystemExit) as exc:
        main(["import-status", "--state", str(tmp_path / "missing")] + extra)
    assert exc.value.code == 2 and not capsys.readouterr().out


@pytest.mark.parametrize("has_bundle", [False, True])
def test_document_cursor_binding_and_copy(mixed, tmp_path, capsys, request, has_bundle):
    path, bundle, state = mixed
    bundle = bundle if has_bundle else None
    first = page_result(path, bundle, capsys, "enabled_flag", 1)
    token = first["next_document_cursor"]
    assert decode_cursor(token) == {"version": 1, "kind": "pending-stage-documents",
        "scope": "bundle" if has_bundle else "state", "bundle": state["identity"]["bundle"] if has_bundle else None,
        "state": hashlib.sha256(path.read_bytes()).hexdigest(), "pending_stage": "enabled_flag", "after": "d06"}
    copy = tmp_path / "copy.json"
    copy.write_bytes(path.read_bytes())
    if bundle is not None:
        copied_bundle = tmp_path / "copy-bundle"
        shutil.copytree(bundle, copied_bundle)
        bundle = copied_bundle
    request.getfixturevalue("forbid_io")
    before = files_snapshot(tmp_path)
    full = import_status(copy, bundle)
    assert assert_slice(page_result(copy, bundle, capsys, "enabled_flag", 9, token), full, ["d07"]) is None
    assert files_snapshot(tmp_path) == before


@pytest.mark.parametrize("has_bundle", [False, True])
@pytest.mark.parametrize("case", ["empty", "alphabet", "unicode", "oversized", "truncated", "padded", "not-json",
    "array", "noncanonical", "version", "boolean-version", "extra", "kind", "stage", "scope", "state",
    "state-bytes", "state-key-order", "state-unknown", "state-value", "bundle-binding", "nonmatching-after", "unknown-after", "last-after", "type-after", "chunk-token"])
def test_document_cursor_rejections(mixed, capsys, request, has_bundle, case):
    path, bundle, state = mixed
    scope = bundle if has_bundle else None
    token = page_result(path, scope, capsys, "enabled_flag", 1)["next_document_cursor"]
    value = decode_cursor(token)
    if case == "empty": token = ""
    elif case == "alphabet": token = "../private"
    elif case == "unicode": token = "界"
    elif case == "oversized": token = "x" * 2049
    elif case == "truncated": token = token[:1]
    elif case == "padded": token += "="
    elif case == "not-json": token = base64.urlsafe_b64encode(b"private").decode().rstrip("=")
    elif case == "array": token = encode_cursor([])
    elif case == "noncanonical": token = base64.urlsafe_b64encode(json.dumps(value, indent=2).encode()).decode().rstrip("=")
    elif case == "state-bytes": path.write_bytes(path.read_bytes() + b" ")
    elif case == "state-key-order": path.write_text(json.dumps(state, sort_keys=True))
    elif case == "state-unknown": path.write_text(json.dumps({**state, "unknown": True}))
    elif case == "state-value":
        state["identity"]["base_url"] = "http://different.invalid"
        path.write_text(json.dumps(state))
    elif case == "scope": scope = None if has_bundle else bundle
    elif case == "stage": value["pending_stage"] = "chunk_creation"; token = encode_cursor(value)
    elif case == "chunk-token":
        token = import_status(path, bundle, document="d10", list_chunks=True, page_size=1)["bundle_progress"]["next_cursor"]
    else:
        key, changed = {"version": ("version", 99), "boolean-version": ("version", True),
            "extra": ("private", "private"), "kind": ("kind", "chunks"), "state": ("state", "f"*64),
            "bundle-binding": ("bundle", "f"*64), "nonmatching-after": ("after", "d05"),
            "unknown-after": ("after", "unknown"), "last-after": ("after", "d07"),
            "type-after": ("after", 7)}[case]
        value[key] = changed
        token = encode_cursor(value)
    request.getfixturevalue("forbid_io")
    before = files_snapshot(path.parent)
    args = ["import-status", "--state", str(path), "--pending-stage", "enabled_flag",
            "--document-page-size", "1", "--document-cursor", token]
    if scope is not None: args += ["--bundle", str(scope)]
    assert main(args) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert captured.err == "ragwrench port: Invalid document continuation cursor or snapshot mismatch; restart without --document-cursor\n"
    with pytest.raises(PortError, match="Invalid document continuation"):
        import_status(path, scope, pending_stage="enabled_flag", document_page_size=1, document_cursor=token)
    assert files_snapshot(path.parent) == before


def test_document_cursor_cannot_be_chunk_cursor(mixed, forbid_io):
    path, bundle, _ = mixed
    token = import_status(path, bundle, pending_stage="enabled_flag", document_page_size=1)["next_document_cursor"]
    for status in (None, "completed"):
        with pytest.raises(PortError, match="Invalid chunk continuation"):
            import_status(path, bundle, document="d10", list_chunks=True, page_size=1, cursor=token, chunk_status=status)


def test_document_page_reads_one_snapshot(mixed, monkeypatch, forbid_io):
    from ragwrench.port import status
    path, bundle, _ = mixed
    before = path.read_bytes()
    original, calls = status.read_json_snapshot, []
    def changed_after_read(p):
        calls.append(p)
        snapshot = original(p)
        path.write_bytes(before + b" ")
        return snapshot
    monkeypatch.setattr(status, "read_json_snapshot", changed_after_read)
    result = import_status(path, bundle, pending_stage="enabled_flag", document_page_size=1)
    assert calls == [path]
    assert decode_cursor(result["next_document_cursor"])["state"] == hashlib.sha256(before).hexdigest()


@pytest.mark.parametrize("has_bundle", [False, True])
def test_document_pages_interleaved_size_changes(mixed, capsys, request, has_bundle):
    path, bundle, state = mixed
    ids = ["d00", "d02", "d04", "d06", "d07", "d08", "d09"]
    for source in ids:
        state["documents"][source] = {"id": f"t-{source}", "phase": "chunks", "chunks": {"c0": "t-c0"},
                                      "pending": {"source": "c1", "target": "t-c1"}}
    path.write_text(json.dumps(state))
    request.getfixturevalue("forbid_io")
    bundle = bundle if has_bundle else None
    full = import_status(path, bundle)
    offset, token, seen = 0, None, []
    for size in (1, 2, 3, 9):
        page = page_result(path, bundle, capsys, "enabled_flag", size, token)
        token = assert_slice(page, full, ids[offset:offset+size])
        seen.extend(row["source_id"] for row in page["documents"])
        offset += size
    assert seen == ids and token is None


@pytest.mark.parametrize("has_bundle,case", [(False, "schema"), (False, "consistency"),
    (True, "schema"), (True, "consistency"), (True, "foreign-chunk")])
def test_document_page_validates_off_page_match(mixed, capsys, request, has_bundle, case):
    path, bundle, state = mixed
    # d06 would be the first flag page; d07 remains an off-page flag intent.
    doc = state["documents"]["d07"]
    if case == "schema": doc["chunks"] = []
    elif case == "consistency": doc["chunks"]["c1"] = "duplicate-pending"
    else: doc["chunks"]["foreign"] = "t-foreign"
    path.write_text(json.dumps(state))
    request.getfixturevalue("forbid_io")
    scope = bundle if has_bundle else None
    with pytest.raises(PortError, match="Invalid import state"):
        import_status(path, scope, pending_stage="enabled_flag", document_page_size=1)
    args = ["import-status", "--state", str(path), "--pending-stage", "enabled_flag", "--document-page-size", "1"]
    if scope is not None: args += ["--bundle", str(scope)]
    assert main(args) == 1 and not capsys.readouterr().out


@pytest.mark.parametrize("has_bundle", [False, True])
def test_document_cursor_changed_stage_and_bundle(mixed, capsys, request, has_bundle):
    path, bundle, state = mixed
    scope = bundle if has_bundle else None
    token = page_result(path, scope, capsys, "enabled_flag", 1)["next_document_cursor"]
    with pytest.raises(PortError, match="Invalid document continuation"):
        import_status(path, scope, pending_stage="chunk_creation", document_page_size=1, document_cursor=token)
    m, docs, chunks = load_bundle(bundle)
    m = write_bundle(bundle, m["source"], {"changed": True}, docs, chunks)
    state["identity"]["bundle"] = fingerprint(m)
    path.write_text(json.dumps(state))
    value = decode_cursor(token)
    # Bind the valid new state bytes to isolate bundle fingerprint rejection.
    value["state"] = hashlib.sha256(path.read_bytes()).hexdigest()
    request.getfixturevalue("forbid_io")
    if has_bundle:
        with pytest.raises(PortError, match="Invalid document continuation"):
            import_status(path, bundle, pending_stage="enabled_flag", document_page_size=1,
                          document_cursor=encode_cursor(value))
    # Original state-byte binding rejects either state-only or bundle change.
    with pytest.raises(PortError, match="Invalid document continuation"):
        import_status(path, scope, pending_stage="enabled_flag", document_page_size=1, document_cursor=token)
