"""Offline checkpoint snapshots: counts are durable mappings, never bundle totals."""
from copy import deepcopy
import json

import pytest

from ragwrench.port.cli import main
from ragwrench.port.common import PortError, atomic_json, locked, read_json
from ragwrench.port.fakes import FakeState, running_fake
from ragwrench.port.operations import import_bundle


@pytest.fixture
def checkpoint():
    return {
        "identity": {
            "version": 1, "bundle": "a" * 64, "system": "ragflow",
            "api_profile": "ragflow-0.27.3", "base_url": "http://127.0.0.1:1/api/v1",
            "dataset_name": "handbook", "reuse_dataset": None, "allow_lossy": False,
            "max_chunk_bytes": 16000, "on_error": "abort", "indexing": "high_quality",
        },
        "dataset": "target-dataset", "dataset_pending": False,
        "documents": {"source-doc": {
            "id": "target-doc", "phase": "chunks", "chunks": {"one": "t-one", "two": "t-two"},
            "pending": {"source": "three", "target": "t-three"},
        }},
    }


def invoke(path, capsys, summary_only=False, pending_counts=False):
    code = main(["import-status", "--state", str(path)] + (["--summary-only"] if summary_only else []) +
                (["--pending-counts"] if pending_counts else []))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_import_status_offline_snapshot(checkpoint, tmp_path, capsys, monkeypatch):
    path = tmp_path / "mapping.json"
    atomic_json(path, checkpoint)
    before = path.read_bytes(), path.stat().st_mtime_ns, set(tmp_path.iterdir())
    monkeypatch.delenv("DIFY_KB_API_KEY", raising=False)
    monkeypatch.delenv("RAGFLOW_API_KEY", raising=False)
    # Offline status must not even construct a client, load a bundle or acquire a writer lock.
    def forbidden(*args, **kwargs):
        pytest.fail("offline status attempted API/bundle/lock access")
    monkeypatch.setattr("ragwrench.port.cli.API", forbidden)
    monkeypatch.setattr("ragwrench.port.cli.load_bundle", forbidden)
    monkeypatch.setattr("ragwrench.port.common.locked", forbidden)
    monkeypatch.setattr("ragwrench.port.operations.locked", forbidden)
    code, out, err = invoke(path, capsys)
    assert code == 0 and err == ""
    assert json.loads(out) == {
        "state": str(path), "target_system": "ragflow", "api_profile": "ragflow-0.27.3",
        "dataset_id": "target-dataset", "dataset_pending": False,
        "mapped_documents": 1, "completed_chunks": 2,
        "documents": [{"source_id": "source-doc", "target_id": "target-doc", "phase": "chunks",
                       "completed_chunks": 2, "pending_stage": "enabled_flag",
                       "pending_chunk": {"source_id": "three", "target_id": "t-three"}}],
    }
    assert (path.read_bytes(), path.stat().st_mtime_ns, set(tmp_path.iterdir())) == before


@pytest.mark.parametrize("stage", ["dataset_creation", "document_creation", "bootstrap_cleanup", "chunk_creation", "enabled_flag", None])
def test_import_status_pending_stages(checkpoint, tmp_path, capsys, stage):
    doc = checkpoint["documents"]["source-doc"]
    if stage == "dataset_creation":
        checkpoint.update(dataset=None, dataset_pending=True, documents={})
    elif stage in ("new", "document_creation"):
        doc.update(id=None, phase="new" if stage == "new" else "creating", chunks={}, pending=None)
    elif stage == "bootstrap_cleanup":
        doc.update(phase="clearing", chunks={}, pending=None)
    elif stage == "chunk_creation":
        doc["pending"]["target"] = None
    elif stage is None:
        doc.update(phase="complete", pending=None)
    path = tmp_path / "mapping.json"
    atomic_json(path, checkpoint)
    code, out, err = invoke(path, capsys)
    assert code == 0 and not err
    result = json.loads(out)
    assert result["dataset_pending"] == (stage == "dataset_creation")
    assert result["mapped_documents"] == (0 if stage in ("dataset_creation", "document_creation") else 1)
    assert result["completed_chunks"] == (0 if stage in ("dataset_creation", "document_creation", "bootstrap_cleanup") else 2)
    if result["documents"]:
        assert result["documents"][0]["pending_stage"] == stage
    assert "verified" not in result and "complete" not in result


@pytest.mark.parametrize("summary_only,pending_counts", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("system,default", [("dify", "dify-f8fc371a"), ("ragflow", "ragflow-1.0")])
def test_import_status_older_state_and_active_writer(checkpoint, system, default, tmp_path, capsys, summary_only, pending_counts):
    checkpoint["identity"]["system"] = system
    del checkpoint["identity"]["api_profile"]
    # Atomic replacement by a writer makes a lock-free snapshot safe to read, though it can go stale.
    path = tmp_path / "mapping.json"
    atomic_json(path, checkpoint)
    with locked(str(path) + ".lock"):
        code, out, err = invoke(path, capsys, summary_only, pending_counts)
    assert code == 0 and not err and json.loads(out)["api_profile"] == default


@pytest.mark.parametrize("summary_only,pending_counts", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("case", [
    "array", "export", "version", "bool-version", "system", "profile", "auto-profile", "missing-identity",
    "documents-list", "bad-id", "newline-id", "bad-source-id", "bad-phase", "chunks-list", "bad-chunk-id",
    "duplicate-document", "duplicate-chunk", "pending-completed", "pending-target-used", "bad-pending",
    "creating-mapped", "clearing-chunks", "complete-pending", "no-dataset", "dataset-and-pending", "bad-bool",
])
def test_import_status_rejects_invalid_state(checkpoint, case, tmp_path, capsys, summary_only, pending_counts):
    state = deepcopy(checkpoint)
    doc = state["documents"]["source-doc"]
    if case == "array":
        state = []
    elif case == "export":
        state = {"identity": {"version": 1, "system": "ragflow"}, "pages": {}}
    elif case == "version":
        state["identity"]["version"] = 2
    elif case == "bool-version":
        state["identity"]["version"] = True
    elif case == "system":
        state["identity"]["system"] = "unknown-private-value"
    elif case == "profile":
        state["identity"]["api_profile"] = "dify-f8fc371a"
    elif case == "auto-profile":
        state["identity"]["api_profile"] = "auto"
    elif case == "missing-identity":
        del state["identity"]
    elif case == "documents-list":
        state["documents"] = []
    elif case == "bad-id":
        doc["id"] = "private/id"
    elif case == "newline-id":
        doc["id"] = "target-doc\n"
    elif case == "bad-source-id":
        state["documents"]["private/id"] = state["documents"].pop("source-doc")
    elif case == "bad-phase":
        doc["phase"] = "unknown-private-value"
    elif case == "chunks-list":
        doc["chunks"] = []
    elif case == "bad-chunk-id":
        doc["chunks"]["one"] = "private/id"
    elif case == "duplicate-document":
        state["documents"]["other"] = deepcopy(doc)
    elif case == "duplicate-chunk":
        doc["chunks"]["two"] = "t-one"
    elif case == "pending-completed":
        doc["pending"]["source"] = "one"
    elif case == "pending-target-used":
        doc["pending"]["target"] = "t-one"
    elif case == "bad-pending":
        doc["pending"] = {}
    elif case == "creating-mapped":
        doc.update(phase="creating", chunks={}, pending=None)
    elif case == "clearing-chunks":
        doc.update(phase="clearing", pending=None)
    elif case == "complete-pending":
        doc["phase"] = "complete"
    elif case == "no-dataset":
        state["dataset"] = None
    elif case == "dataset-and-pending":
        state["dataset_pending"] = True
    elif case == "bad-bool":
        state["dataset_pending"] = 0
    path = tmp_path / "mapping.json"
    atomic_json(path, state)
    code, out, err = invoke(path, capsys, summary_only, pending_counts)
    assert code == 1 and out == "" and "Invalid import state" in err
    assert "Traceback" not in err and "private" not in err


def test_import_status_multiple_documents_sorted_and_initial_state(checkpoint, tmp_path, capsys):
    checkpoint["documents"]["another-doc"] = {
        "id": "another-target", "phase": "complete", "chunks": {"four": "t-four"}, "pending": None,
    }
    path = tmp_path / "mapping.json"
    atomic_json(path, checkpoint)
    code, out, err = invoke(path, capsys)
    assert code == 0 and not err
    result = json.loads(out)
    assert result["mapped_documents"] == 2 and result["completed_chunks"] == 3
    assert [d["source_id"] for d in result["documents"]] == ["another-doc", "source-doc"]
    # The initial pre-create checkpoint and a reused dataset with no recorded documents are valid.
    checkpoint.update(dataset=None, documents={})
    for reuse in (None, "reused-target"):
        checkpoint["identity"].update(dataset_name=None if reuse else "handbook", reuse_dataset=reuse)
        checkpoint["dataset"] = reuse
        atomic_json(path, checkpoint)
        code, out, err = invoke(path, capsys)
        assert code == 0 and not err
        result = json.loads(out)
        assert result["mapped_documents"] == result["completed_chunks"] == 0
        assert result["documents"] == [] and not result["dataset_pending"]


@pytest.mark.parametrize("summary_only,pending_counts", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("kind", ["missing", "malformed", "symlink", "directory", "oversized"])
def test_import_status_file_errors(tmp_path, capsys, kind, summary_only, pending_counts):
    path = tmp_path / "nested" / "mapping.json"
    if kind != "missing":
        path.parent.mkdir()
    if kind == "malformed":
        path.write_text('{"secret": "private-value"')
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
    code, out, err = invoke(path, capsys, summary_only, pending_counts)
    assert code == 1 and not out and "Traceback" not in err and "private-value" not in err
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("system", ["dify", "ragflow"])
@pytest.mark.parametrize("stage", ["dataset_creation", "document_creation", "bootstrap_cleanup", "chunk_creation", "enabled_flag"])
def test_import_status_real_interrupted_states(system, stage, example, tmp_path, api_factory, capsys):
    target = FakeState(system)
    path = tmp_path / "mapping.json"
    method = {"dataset_creation": "create_dataset", "document_creation": "create_document",
              "bootstrap_cleanup": "ready_and_clear", "chunk_creation": "add_chunk", "enabled_flag": "set_enabled"}[stage]
    with running_fake(target) as url:
        api = api_factory(system, url)
        original = getattr(api, method)
        def interrupt(*args, **kwargs):
            raise PortError("simulated interruption")
        setattr(api, method, interrupt)
        with pytest.raises(PortError, match="simulated interruption"):
            import_bundle(api, example, path, dataset_name="interrupted")
        before = len(target.requests), path.read_bytes()
        code, out, err = invoke(path, capsys)
        assert code == 0 and not err
        result = json.loads(out)
        assert result["dataset_pending"] == (stage == "dataset_creation")
        if result["documents"]:
            assert result["documents"][0]["pending_stage"] == stage
        assert (len(target.requests), path.read_bytes()) == before
        setattr(api, method, original)
        import_bundle(api, example, path, dataset_name="interrupted", retry_pending=True)
        code, out, err = invoke(path, capsys)
        assert code == 0 and not err
        result = json.loads(out)
        assert result["mapped_documents"] == 3 and result["completed_chunks"] == 15
        assert all(d["phase"] == "complete" and d["pending_stage"] is None for d in result["documents"])
        assert not read_json(path)["dataset_pending"]


@pytest.mark.parametrize("stage", ["initial", "new", "dataset_creation", "document_creation", "bootstrap_cleanup",
                                   "chunk_creation", "enabled_flag", "complete", "legacy", "reuse"])
def test_import_status_summary_contract(checkpoint, tmp_path, capsys, monkeypatch, stage):
    from ragwrench.port.status import import_status

    doc = checkpoint["documents"]["source-doc"]
    if stage in ("initial", "dataset_creation"):
        checkpoint.update(dataset=None, dataset_pending=stage == "dataset_creation", documents={})
    elif stage in ("new", "document_creation"):
        doc.update(id=None, phase="new" if stage == "new" else "creating", chunks={}, pending=None)
    elif stage == "bootstrap_cleanup":
        doc.update(phase="clearing", chunks={}, pending=None)
    elif stage == "chunk_creation":
        doc["pending"]["target"] = None
    elif stage == "complete":
        doc.update(phase="complete", pending=None)
    elif stage == "legacy":
        del checkpoint["identity"]["api_profile"]
    elif stage == "reuse":
        checkpoint["identity"].update(dataset_name=None, reuse_dataset="target-dataset")
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(checkpoint))
    before = path.read_bytes(), path.stat().st_mtime_ns, set(tmp_path.iterdir())

    def forbidden(*args, **kwargs):
        pytest.fail("summary attempted credentials/client/network/bundle/write/lock access")
    for name in ("ragwrench.port.cli.API", "httpx.Client", "socket.socket", "os.getenv",
                 "ragwrench.port.status.load_bundle", "ragwrench.port.common.atomic_bytes", "ragwrench.port.common.locked",
                 "ragwrench.port.operations.locked"):
        monkeypatch.setattr(name, forbidden)
    code, out, err = invoke(path, capsys)
    assert code == 0 and not err
    full = json.loads(out)
    expected = {key: value for key, value in full.items() if key != "documents"}
    assert main(["import-status", "--state", str(path), "--summary-only"]) == 0
    summary = capsys.readouterr()
    assert not summary.err and json.loads(summary.out) == expected
    assert import_status(path, summary_only=True) == expected
    assert import_status(path, summary_only=False) == full
    counts = dict(dataset_creation=int(stage == "dataset_creation"), document_creation=0,
                  bootstrap_cleanup=0, chunk_creation=0, enabled_flag=0)
    if stage in ("document_creation", "bootstrap_cleanup", "chunk_creation", "enabled_flag"):
        counts[stage] = 1
    elif stage in ("legacy", "reuse"):
        counts["enabled_flag"] = 1
    assert main(["import-status", "--state", str(path), "--summary-only", "--pending-counts"]) == 0
    counted = capsys.readouterr()
    assert not counted.err and json.loads(counted.out) == {**expected, "pending_counts": counts}
    assert all(type(value) is int for value in json.loads(counted.out)["pending_counts"].values())
    assert import_status(path, summary_only=True, pending_counts=True) == {**expected, "pending_counts": counts}
    assert import_status(path, summary_only=True, pending_counts=False) == expected
    assert import_status(path, pending_counts=False) == full
    if stage == "enabled_flag":
        assert expected == {
            "state": str(path), "target_system": "ragflow", "api_profile": "ragflow-0.27.3",
            "dataset_id": "target-dataset", "dataset_pending": False,
            "mapped_documents": 1, "completed_chunks": 2,
        }
    assert (path.read_bytes(), path.stat().st_mtime_ns, set(tmp_path.iterdir())) == before


def test_pending_counts_multiple_documents(checkpoint, tmp_path, capsys, monkeypatch):
    from ragwrench.port.status import import_status

    checkpoint["documents"] = {}
    # Multiple intents per stage, plus phases with no saved intent. Completed chunks
    # alongside pending chunks must not inflate counts; source IDs are document scoped.
    for phase, count in (("creating", 2), ("clearing", 3), ("create-chunk", 2), ("flag", 3),
                         ("new", 1), ("chunks", 1), ("complete", 1)):
        for i in range(count):
            source = f"private-{phase}-{i}"
            has_chunks = phase in ("create-chunk", "flag", "chunks", "complete")
            checkpoint["documents"][source] = {
                "id": None if phase in ("new", "creating") else f"target-{source}",
                "phase": "chunks" if phase in ("create-chunk", "flag") else phase,
                "chunks": {"done": "t-done"} if has_chunks else {},
                "pending": {"source": "pending", "target": "t-pending" if phase == "flag" else None}
                           if phase in ("create-chunk", "flag") else None,
            }
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(checkpoint))
    before = path.read_bytes(), path.stat().st_mtime_ns, set(tmp_path.iterdir())
    def forbidden(*args, **kwargs):
        pytest.fail("pending aggregate attempted client/credentials/HTTP/lock/write access")
    for name in ("ragwrench.port.cli.API", "httpx.Client", "socket.socket", "os.getenv",
                 "ragwrench.port.status.load_bundle", "ragwrench.port.common.atomic_bytes", "ragwrench.port.common.locked",
                 "ragwrench.port.operations.locked"):
        monkeypatch.setattr(name, forbidden)
    full = import_status(path)
    expected_counts = dict(dataset_creation=0, document_creation=2, bootstrap_cleanup=3,
                           chunk_creation=2, enabled_flag=3)
    assert sum(d["completed_chunks"] for d in full["documents"]) == 7
    for stage in expected_counts.keys() - {"dataset_creation"}:
        assert expected_counts[stage] == sum(d["pending_stage"] == stage for d in full["documents"])
    code, out, err = invoke(path, capsys, summary_only=True, pending_counts=True)
    assert code == 0 and not err
    assert json.loads(out) == {**{k: v for k, v in full.items() if k != "documents"},
                               "pending_counts": expected_counts}
    # No per-document source IDs leak into the summary ("state" is the checkpoint path,
    # which on macOS lives under /private/var, so check the IDs, not a substring).
    assert not any(source in out for source in checkpoint["documents"])
    assert "documents" not in json.loads(out)
    assert (path.read_bytes(), path.stat().st_mtime_ns, set(tmp_path.iterdir())) == before
