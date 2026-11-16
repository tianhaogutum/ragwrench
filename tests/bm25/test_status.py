"""Routine status validates saved aggregates, without auditing a recovery snapshot."""

from contextlib import closing
from copy import deepcopy
import hashlib
import json
import os

import pytest
from qdrant_client import models

from ragwrench.bm25 import store
from ragwrench.bm25.cli import main
from ragwrench.bm25.tokenizers import Tokenizer


def files(path):
    return {
        str(p.relative_to(path)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in path.rglob("*")
        if p.is_file()
    }


def interrupt(*args, **kwargs):
    raise RuntimeError("injected interruption")


def guard_status(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("status must not write, fsync, inspect points, encode or validate snapshots")

    for name in (
        "create_collection",
        "delete_collection",
        "delete",
        "upsert",
        "update_collection",
        "scroll",
        "retrieve",
        "query_points",
    ):
        monkeypatch.setattr(store.QdrantClient, name, unexpected)
    monkeypatch.setattr(os, "fsync", unexpected)
    monkeypatch.setattr(store, "_encode_points", unexpected)
    monkeypatch.setattr(store, "_snapshot_digest", unexpected)
    monkeypatch.setattr(store.Encoder, "encode_documents", unexpected)
    monkeypatch.setattr(Tokenizer, "__call__", unexpected)


def cli_args(path, collection="kb"):
    return ["status", "--path", str(path), "--collection", collection]


def test_cli_ready_status(tmp_path, monkeypatch, capsys):
    path = tmp_path / "qd"
    with closing(store.open_local(path)) as client:
        built = store.build_collection(
            client,
            "kb",
            [{"id": 7, "text": "private 数据库 备份", "metadata": {"private": True}}],
            stopwords=("private",),
        )
        hits = store.search_collection(client, "kb", "数据库")
    before = files(path)
    with monkeypatch.context() as m:
        guard_status(m)
        assert main(cli_args(path)) == 0
        report = json.loads(capsys.readouterr().out)
        assert main(cli_args(path)) == 0
        assert json.loads(capsys.readouterr().out) == report
    assert report == {
        "schema": 1,
        "collection": "kb",
        "state": "ready",
        "config": built["encoder"],
        "corpus": {"document_count": 1, "avgdl_mode": "auto", "total_tokens": 3},
        "points": {"current": 1},
        "recovery_snapshot_present": False,
    }
    # Literal stopwords remain config data; document text/metadata/collision vocabulary do not.
    assert "private 数据库" not in json.dumps(report, ensure_ascii=False)
    assert not {"documents", "metadata", "collision_stats", "skipped"} & report.keys()
    with closing(store.open_local(path)) as client:
        assert store.search_collection(client, "kb", "数据库") == hits
    assert files(path) == before


@pytest.mark.parametrize("scenario", ["fixed", "empty-auto", "empty-fixed", "legacy"])
def test_ready_policies(client, monkeypatch, scenario):
    options = {"avgdl": 19.0} if "fixed" in scenario else {}
    built = store.build_collection(client, "kb", [{"id": "7", "text": "数据库 备份"}], **options)
    if scenario.startswith("empty"):
        store.update_collection(client, "kb", delete_ids=["7"])
    if scenario == "legacy":
        data = deepcopy(client.get_collection("kb").config.metadata["qbm25"])
        data.pop("corpus")
        client.update_collection("kb", metadata={"qbm25": data})
    guard_status(monkeypatch)
    report = store.status_collection(client, "kb")
    assert store.status_collection(client, "kb", require_ready=True) == report
    count = 0 if scenario.startswith("empty") else 1
    assert store.status_collection(client, "kb", expect_documents=count) == report
    assert (
        store.status_collection(client, "kb", require_ready=True, expect_documents=count) == report
    )
    with pytest.raises(ValueError) as error:
        store.status_collection(client, "kb", expect_documents=count + 1)
    assert str(error.value) == f"document count mismatch: expected={count + 1} actual={count}"
    assert report["points"] == {"current": count}
    assert report["corpus"] == {
        "document_count": count,
        "avgdl_mode": None
        if scenario == "legacy"
        else ("fixed" if "fixed" in scenario else "auto"),
        "total_tokens": None if scenario == "legacy" else (0 if count == 0 else 3),
    }
    assert report["config"]["avgdl"] == (
        19.0 if "fixed" in scenario else (1.0 if count == 0 else 3.0)
    )
    assert report["config"]["tokenizer"] == built["encoder"]["tokenizer"]


@pytest.mark.parametrize("state", ["ready", "building", "updating"])
@pytest.mark.parametrize("snapshot", ["absent", "null", "corrupt"])
def test_require_ready(tmp_path, monkeypatch, capsys, state, snapshot):
    path = tmp_path / "qd"
    with closing(store.open_local(path)) as client:
        store.build_collection(client, "kb", [{"id": 7, "text": "private 数据库 备份"}])
        hits = store.search_collection(client, "kb", "数据库")
        original = deepcopy(client.get_collection("kb").config.metadata["qbm25"])
        data = deepcopy(original)
        data["state"] = state
        if snapshot != "absent":
            data["recovery"] = None if snapshot == "null" else {"documents": "private"}
        client.update_collection("kb", metadata={"qbm25": data})
    before = files(path)
    with monkeypatch.context() as m:
        guard_status(m)
        assert main(cli_args(path)) == 0  # Default pending status remains successful.
        output = capsys.readouterr()
        report = json.loads(output.out)
        assert output.err == ""
        assert report["state"] == state
        assert report["recovery_snapshot_present"] is (snapshot != "absent")
        with closing(store.open_local(path)) as client:
            assert store.status_collection(client, "kb") == report
            if state == "ready":
                assert store.status_collection(client, "kb", require_ready=True) == report
            else:
                with pytest.raises(ValueError) as error:
                    store.status_collection(client, "kb", require_ready=True)
                assert str(error.value) == f"collection is not ready: state={state}"
        assert main(cli_args(path) + ["--require-ready"]) == (0 if state == "ready" else 2)
        output = capsys.readouterr()
        if state == "ready":
            assert json.loads(output.out) == report and output.err == ""
        else:
            assert output.out == ""
            assert output.err == f"ragwrench bm25: error: collection is not ready: state={state}\n"
    assert files(path) == before
    # Restore only the explicit fixture marker, then verify points and search parity.
    with closing(store.open_local(path)) as client:
        client.update_collection("kb", metadata={"qbm25": original})
        assert store.search_collection(client, "kb", "数据库") == hits


@pytest.mark.parametrize(
    ("scenario", "state", "current", "target", "tokens"),
    [
        ("build-before", "building", 0, 130, 390),
        ("build-partial", "building", 63, 130, 390),
        ("update-before-delete", "updating", 2, 2, 7),
        ("update-after-delete", "updating", 1, 2, 7),
        ("delete-all", "updating", 0, 0, 0),
    ],
)
def test_pending_status_without_replay(
    tmp_path, monkeypatch, capsys, scenario, state, current, target, tokens
):
    path = tmp_path / "qd"
    with closing(store.open_local(path)) as client:
        if scenario.startswith("build"):
            upsert = client.upsert

            def partial(*args, **kwargs):
                if scenario == "build-partial":
                    upsert(*args, **{**kwargs, "points": kwargs["points"][:63]})
                interrupt()

            with monkeypatch.context() as m:
                m.setattr(client, "upsert", partial)
                with pytest.raises(RuntimeError, match="interruption"):
                    store.build_collection(
                        client, "kb", [{"id": i, "text": "数据库 备份"} for i in range(130)]
                    )
        else:
            store.build_collection(
                client, "kb", [{"id": i, "text": "数据库 备份"} for i in range(2)]
            )
            with monkeypatch.context() as m:
                if scenario == "delete-all":
                    m.setattr(store, "_finalize", interrupt)
                else:
                    m.setattr(
                        client,
                        "delete" if scenario == "update-before-delete" else "upsert",
                        interrupt,
                    )
                with pytest.raises(RuntimeError, match="interruption"):
                    store.update_collection(
                        client,
                        "kb",
                        delete_ids=[0, 1] if scenario == "delete-all" else [0],
                        additions=[]
                        if scenario == "delete-all"
                        else [{"id": 0, "text": "数据库 快照 恢复"}],
                    )
        manifest = deepcopy(client.get_collection("kb").config.metadata["qbm25"])
    before = files(path)
    with monkeypatch.context() as m:
        guard_status(m)
        with closing(store.open_local(path)) as client:
            report = store.status_collection(client, "kb")
            assert client.get_collection("kb").config.metadata["qbm25"] == manifest
        assert main(cli_args(path)) == 0
        assert json.loads(capsys.readouterr().out) == report
        with closing(store.open_local(path)) as client:
            with pytest.raises(ValueError) as error:
                store.status_collection(client, "kb", require_ready=True)
            assert str(error.value) == f"collection is not ready: state={state}"
        assert main(cli_args(path) + ["--require-ready"]) == 2
        output = capsys.readouterr()
        assert output.out == ""
        assert output.err == f"ragwrench bm25: error: collection is not ready: state={state}\n"
        # Count is the intended target, even before/partway through point writes.
        with closing(store.open_local(path)) as client:
            assert store.status_collection(client, "kb", expect_documents=target) == report
            with pytest.raises(ValueError) as error:
                store.status_collection(client, "kb", expect_documents=target + 1)
            assert str(error.value) == (
                f"document count mismatch: expected={target + 1} actual={target}"
            )
        assert main(cli_args(path) + ["--expect-documents", str(target)]) == 0
        assert json.loads(capsys.readouterr().out) == report
        wrong = current if current != target else target + 1
        assert main(cli_args(path) + ["--expect-documents", str(wrong)]) == 2
        output = capsys.readouterr()
        assert output.out == ""
        assert output.err == (
            f"ragwrench bm25: error: document count mismatch: expected={wrong} actual={target}\n"
        )
        for expected in (target, target + 1):
            with closing(store.open_local(path)) as client:
                with pytest.raises(ValueError) as error:
                    store.status_collection(
                        client, "kb", require_ready=True, expect_documents=expected
                    )
                assert str(error.value) == f"collection is not ready: state={state}"
            assert (
                main(cli_args(path) + ["--require-ready", "--expect-documents", str(expected)]) == 2
            )
            output = capsys.readouterr()
            assert output.out == ""
            assert output.err == f"ragwrench bm25: error: collection is not ready: state={state}\n"
    assert report["state"] == state
    assert report["points"] == {"current": current}
    assert report["corpus"] == {
        "document_count": target,
        "avgdl_mode": "auto",
        "total_tokens": tokens,
    }
    assert report["recovery_snapshot_present"] is True
    assert report["config"] == manifest["encoder"]
    assert "documents" not in json.dumps(report)
    assert files(path) == before


@pytest.mark.parametrize("state", ["ready", "building", "updating"])
@pytest.mark.parametrize("snapshot", ["absent", "null", "corrupt"])
def test_snapshot_presence_is_not_recovery_validation(client, monkeypatch, state, snapshot):
    store.build_collection(client, "kb", [{"id": 7, "text": "private 数据库"}])
    data = deepcopy(client.get_collection("kb").config.metadata["qbm25"])
    data["state"] = state
    if snapshot != "absent":
        data["recovery"] = (
            None if snapshot == "null" else {"sha256": "wrong", "documents": "private"}
        )
    client.update_collection("kb", metadata={"qbm25": data})
    guard_status(monkeypatch)
    report = store.status_collection(client, "kb")
    assert report["state"] == state
    assert report["recovery_snapshot_present"] is (snapshot != "absent")
    assert store.status_collection(client, "kb", expect_documents=1) == report
    with pytest.raises(ValueError, match="document count mismatch: expected=0 actual=1"):
        store.status_collection(client, "kb", expect_documents=0)
    if state != "ready":
        with pytest.raises(ValueError, match=f"collection is not ready: state={state}"):
            store.status_collection(client, "kb", require_ready=True, expect_documents=0)
    assert "private" not in json.dumps(report)


@pytest.mark.parametrize("state", ["ready", "building"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("encoder", None),
        ("encoder", {}),
        ("indexed", True),
        ("indexed", 1.0),
        ("indexed", -1),
        ("indexed", 10001),
        ("corpus", None),
        ("corpus", "private"),
        ("corpus.schema", True),
        ("corpus.schema", 1.0),
        ("corpus.schema", 99),
        ("corpus.avgdl_mode", []),
        ("corpus.total_tokens", -1),
        ("corpus.total_tokens", True),
        ("corpus.total_tokens", 4.0),
        ("corpus.total_tokens", 0),
        ("corpus.total_tokens", 10**400),
        ("corpus.total_tokens", 5),
        ("encoder.unicode_version", "incompatible"),
        ("encoder.k1", 10**400),
    ],
)
def test_invalid_manifest_no_json_or_writes(tmp_path, monkeypatch, capsys, state, field, value):
    path = tmp_path / "qd"
    with closing(store.open_local(path)) as client:
        store.build_collection(client, "kb", [{"id": 7, "text": "数据库 备份"}])
        data = deepcopy(client.get_collection("kb").config.metadata["qbm25"])
        data["state"] = state
        if "." in field:
            parent, key = field.split(".")
            data[parent][key] = value
        else:
            data[field] = value
        client.update_collection("kb", metadata={"qbm25": data})
    before = files(path)
    with monkeypatch.context() as m:
        guard_status(m)
        with closing(store.open_local(path)) as client:
            with pytest.raises(ValueError):
                store.status_collection(client, "kb")
        assert main(cli_args(path)) == 2
        output = capsys.readouterr()
        assert output.out == "" and "error:" in output.err
        assert "private" not in output.err and "Traceback" not in output.err
        # Validation takes precedence over the pending-state gate.
        with closing(store.open_local(path)) as client:
            with pytest.raises(ValueError):
                store.status_collection(client, "kb", require_ready=True)
        assert main(cli_args(path) + ["--require-ready"]) == 2
        assert capsys.readouterr() == output
        for flags in (["--expect-documents", "2"], ["--require-ready", "--expect-documents", "2"]):
            with closing(store.open_local(path)) as client:
                with pytest.raises(ValueError) as error:
                    store.status_collection(
                        client, "kb", expect_documents=2, require_ready="--require-ready" in flags
                    )
                assert output.err == f"ragwrench bm25: error: {error.value}\n"
            assert main(cli_args(path) + flags) == 2
            assert capsys.readouterr() == output
    assert files(path) == before


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "dense",
        "no-idf",
        "no-metadata",
        "manifest-list",
        "state-list",
        "state-unknown",
        "count",
    ],
)
def test_missing_foreign_and_invalid_state(tmp_path, monkeypatch, capsys, damage):
    path = tmp_path / "qd"
    with closing(store.open_local(path)) as client:
        if damage in ("dense", "no-idf", "no-metadata"):
            client.create_collection(
                "kb",
                vectors_config={},
                sparse_vectors_config={}
                if damage == "dense"
                else {
                    "bm25": models.SparseVectorParams(
                        modifier=None if damage == "no-idf" else models.Modifier.IDF
                    )
                },
            )
        elif damage != "missing":
            store.build_collection(client, "kb", [{"id": 7, "text": "数据库"}])
            data = deepcopy(client.get_collection("kb").config.metadata["qbm25"])
            if damage == "count":
                client.delete("kb", points_selector=models.PointIdsList(points=[store.point_id(7)]))
            else:
                if damage == "manifest-list":
                    data = []
                else:
                    data["state"] = [] if damage == "state-list" else "unknown"
                client.update_collection("kb", metadata={"qbm25": data})
    before = files(path)
    guard_status(monkeypatch)
    assert main(cli_args(path)) == 2
    output = capsys.readouterr()
    assert output.out == "" and "error:" in output.err and "Traceback" not in output.err
    assert main(cli_args(path) + ["--require-ready"]) == 2
    assert capsys.readouterr() == output
    for flags in (["--expect-documents", "2"], ["--require-ready", "--expect-documents", "2"]):
        assert main(cli_args(path) + flags) == 2
        assert capsys.readouterr() == output
    assert files(path) == before


def test_missing_path_and_bad_names(tmp_path, capsys):
    path = tmp_path / "never-created"
    for collection in ("kb", "../escape"):
        assert main(cli_args(path, collection)) == 2
        output = capsys.readouterr()
        assert output.out == ""
        assert main(cli_args(path, collection) + ["--require-ready"]) == 2
        assert capsys.readouterr() == output
        assert main(cli_args(path, collection) + ["--expect-documents", "2"]) == 2
        assert capsys.readouterr() == output
        assert (
            main(cli_args(path, collection) + ["--require-ready", "--expect-documents", "2"]) == 2
        )
        assert capsys.readouterr() == output
    assert not path.exists()


@pytest.mark.parametrize("state", ["ready", "building", "updating"])
@pytest.mark.parametrize("field", ["schema", "tokenizer_version"])
@pytest.mark.parametrize("value", [True, 1.0])
def test_status_rejects_noninteger_encoder_versions(
    tmp_path, monkeypatch, capsys, state, field, value
):
    path = tmp_path / "qd"
    with closing(store.open_local(path)) as client:
        store.build_collection(client, "kb", [{"id": 7, "text": "数据库 备份"}])
        hits = store.search_collection(client, "kb", "数据库")
        original = deepcopy(client.get_collection("kb").config.metadata["qbm25"])
        data = deepcopy(original)
        data["state"] = state
        # A snapshot is deliberately absent: status must only check the summary.
        client.update_collection("kb", metadata={"qbm25": data})
    before = files(path)
    with monkeypatch.context() as m:
        guard_status(m)
        assert main(cli_args(path)) == 0  # Positive control on the prior implementation.
        report = json.loads(capsys.readouterr().out)
        assert report["state"] == state and report["recovery_snapshot_present"] is False
        assert type(report["config"][field]) is int
    assert files(path) == before

    data["encoder"][field] = value
    with closing(store.open_local(path)) as client:
        client.update_collection("kb", metadata={"qbm25": data})
    before = files(path)
    with monkeypatch.context() as m:
        guard_status(m)
        with closing(store.open_local(path)) as client:
            with pytest.raises(ValueError, match=field):
                store.status_collection(client, "kb")
        assert main(cli_args(path)) == 2
        output = capsys.readouterr()
        assert output.out == "" and field in output.err and "error:" in output.err
        assert "Traceback" not in output.err
    assert files(path) == before
    # Restoring only the fixture manifest proves inspection did not alter points/results.
    with closing(store.open_local(path)) as client:
        client.update_collection("kb", metadata={"qbm25": original})
        assert store.search_collection(client, "kb", "数据库") == hits


@pytest.mark.parametrize(
    "flag", ["--k1", "--b", "--avgdl", "--tokenizer", "--stopwords", "--dry-run"]
)
def test_no_overrides(tmp_path, capsys, flag):
    path = tmp_path / "never-created"
    with pytest.raises(SystemExit) as error:
        main(cli_args(path) + [flag, "1"])
    assert error.value.code == 2 and capsys.readouterr().out == ""
    assert not path.exists()


def test_expect_documents_ready(tmp_path, monkeypatch, capsys):
    path = tmp_path / "qd"
    with closing(store.open_local(path)) as client:
        store.build_collection(client, "kb", [{"id": 7, "text": "数据库 备份"}])
        hits = store.search_collection(client, "kb", "数据库")
    before = files(path)
    with monkeypatch.context() as m:
        guard_status(m)
        assert main(cli_args(path)) == 0  # Positive control on the prior implementation.
        report = json.loads(capsys.readouterr().out)
        with closing(store.open_local(path)) as client:
            assert store.status_collection(client, "kb", expect_documents=1) == report
            assert store.status_collection(client, "kb", expect_documents=None) == report
            for expected in (0, 2, 10001):
                message = f"document count mismatch: expected={expected} actual=1"
                with pytest.raises(ValueError) as error:
                    store.status_collection(client, "kb", expect_documents=expected)
                assert str(error.value) == message
        for ready_flags in ([], ["--require-ready"]):
            assert main(cli_args(path) + ready_flags + ["--expect-documents", "1"]) == 0
            assert json.loads(capsys.readouterr().out) == report
            assert main(cli_args(path) + ready_flags + ["--expect-documents", "2"]) == 2
            output = capsys.readouterr()
            assert output.out == ""
            assert output.err == "ragwrench bm25: error: document count mismatch: expected=2 actual=1\n"
    assert files(path) == before
    with closing(store.open_local(path)) as client:
        assert store.search_collection(client, "kb", "数据库") == hits
    assert files(path) == before


@pytest.mark.parametrize("expected", [-1, True, False, 0.0, 1.0, "1", [], {}, float("nan")])
def test_invalid_expected_count_before_client_access(expected):
    class NoAccess:
        def __getattr__(self, name):
            pytest.fail(f"invalid expected count must fail before client access: {name}")

    with pytest.raises(ValueError) as error:
        store.status_collection(NoAccess(), "kb", expect_documents=expected)
    assert str(error.value) == "expect_documents must be a nonnegative integer or None"


@pytest.mark.parametrize("value", ["-1", "1.0", "true", "nan", "1e2", ""])
def test_invalid_expected_count_before_open(tmp_path, monkeypatch, capsys, value):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid expected count must fail before opening storage")

    monkeypatch.setattr("ragwrench.bm25.cli.open_local", unexpected)
    path = tmp_path / "never-created"
    with pytest.raises(SystemExit) as error:
        main(cli_args(path) + ["--expect-documents", value])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == "" and "--expect-documents" in output.err
    assert "expected a nonnegative integer" in output.err
    assert not path.exists()
