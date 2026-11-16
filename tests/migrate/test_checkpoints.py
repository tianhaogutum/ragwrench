"""Checkpoint recovery against real pinned stores, including uncertain writes."""
from __future__ import annotations

import json
import os
import pwd
import tempfile
from pathlib import Path, PurePath, PureWindowsPath
from contextlib import closing

import pytest
from qdrant_client import QdrantClient, models

from ragwrench.migrate import checkpoints, engine
from ragwrench.migrate.cli import main
from ragwrench.migrate.model import ParityError, StoreURI, physical_id
from ragwrench.migrate.stores import Store
from .test_integration import fingerprint, uristr
from .test_paths import tree_state


@pytest.fixture(params=["chroma", "qdrant"])
def source(request, chroma_store, qdrant_store):
    if request.param == "chroma":
        return chroma_store()
    return qdrant_store(distance="Euclid", points=[
        {"id": qid, "vector": vector, "payload": {"__id__": rid, "document": rid,
            "nested": {"flag": True, "null": None}}}
        for qid, rid, vector in [
            (77, "custom-id", [1., 0., 0., 0.]),
            (0, "zero", [0., 2., 0., 0.]),
            (2**64 - 1, "last", [0., 0., 3., 0.]),
            ("550e8400-e29b-41d4-a716-446655440000", "uuid-alias", [0., 0., 0., 4.]),
            (12, "other", [-5., 0., 0., 0.]),
        ]])


def target_for(source, tmp_path):
    return StoreURI("qdrant" if source.kind == "chroma" else "chroma", tmp_path / "target", "docs")


CHECKPOINT_ENCODING_ERROR = "Checkpoint path must be a local path encodable by the filesystem"
CHECKPOINT_NUL_ERROR = "Checkpoint path must be a local path without NUL bytes"
CHECKPOINT_HOME_ERROR = "Checkpoint path must be a local path with a resolvable home directory"
CHECKPOINT_TYPE_ERROR = "Checkpoint path must be a local str or pathlib.Path object"


class ConcreteCheckpointPath(type(Path())):
    pass


class CheckpointText(str):
    pass


class ConversionTrap:
    def __str__(self):
        pytest.fail("Unsupported checkpoint input was stringified")

    def __fspath__(self):
        pytest.fail("Unsupported checkpoint input invoked __fspath__")


class TextPathLike:
    def __fspath__(self):
        return "progress.json"


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("value", [b"progress.json", 42, False, [], {}, PurePath("progress.json"),
                                  PureWindowsPath("C:/progress.json"), object(), ConversionTrap(), TextPathLike()],
                         ids=["bytes", "int", "bool", "list", "dict", "pure-path", "windows-pure-path",
                              "object", "conversion-trap", "text-pathlike"])
def test_checkpoint_unsupported_input_types(source, tmp_path, monkeypatch, resume, value):
    target = target_for(source, tmp_path)
    before = tree_state(tmp_path)
    with monkeypatch.context() as guard:
        forbid_checkpoint_io(guard)
        # Lexical checkpoint validation must not run for unsupported types.
        guard.setattr(checkpoints, "_validate_path_text", lambda *a: pytest.fail("Lexical conversion before type check"))
        with pytest.raises(ParityError) as caught:
            engine.copy(source, target, checkpoint=value, resume=resume)
    assert str(caught.value) == CHECKPOINT_TYPE_ERROR
    assert caught.value.__cause__ is None
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize("factory", [ConcreteCheckpointPath, CheckpointText])
def test_checkpoint_supported_subclasses_create_resume(source, tmp_path, monkeypatch, factory):
    target = target_for(source, tmp_path)
    monkeypatch.chdir(tmp_path)
    value = factory("progress.json")
    before = fingerprint(source.path)
    report = engine.copy(source, target, checkpoint=value, top_k=2)
    assert report["copied"] == 5 and report["parity"]["ok"]
    assert report["checkpoint"] == str(tmp_path / "progress.json")
    resumed = engine.copy(source, target, checkpoint=value, resume=True, top_k=2)
    assert resumed["copied"] == 0 and resumed["skipped"] == 5 and resumed["parity"]["ok"]
    assert json.loads((tmp_path / "progress.json").read_text())["parity"]["ok"]
    assert fingerprint(source.path) == before


def test_checkpoint_none_sentinel(source, tmp_path, monkeypatch):
    target = target_for(source, tmp_path)
    before = tree_state(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("None sentinel entered checkpoint code")

    with monkeypatch.context() as guard:
        for name in ("checkpoint_path", "load", "save"):
            guard.setattr(checkpoints, name, forbidden)
        with pytest.raises(ParityError, match="--resume requires --checkpoint"):
            engine.copy(source, target, checkpoint=None, resume=True)
        assert tree_state(tmp_path) == before
        report = engine.copy(source, target, checkpoint=None, dry_run=True)
        assert report["would_copy"] == 5 and "checkpoint" not in report
        assert tree_state(tmp_path) == before
        report = engine.copy(source, target, checkpoint=None)
        assert report["copied"] == 5 and "checkpoint" not in report
    assert engine.verify(source, target, top_k=2)["ok"]


@pytest.fixture
def unresolved_checkpoint_home():
    username = "vecparity_no_such_user_r11_9f72a1"
    with pytest.raises(KeyError):
        pwd.getpwnam(username)
    return Path(f"~{username}/progress.json")


def forbid_checkpoint_io(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Checkpoint filesystem/snapshot/backend access before lexical validation")

    monkeypatch.setattr(checkpoints, "checked_store_path", forbidden)
    monkeypatch.setattr(checkpoints, "load", forbidden)
    monkeypatch.setattr(checkpoints, "save", forbidden)
    monkeypatch.setattr(engine, "read_store", forbidden)
    monkeypatch.setattr(engine, "Store", forbidden)


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("as_string", [False, True])
def test_unresolved_checkpoint_home_preflight(
    source, tmp_path, monkeypatch, unresolved_checkpoint_home, resume, as_string,
):
    target = target_for(source, tmp_path)
    before = tree_state(tmp_path)
    value = str(unresolved_checkpoint_home) if as_string else unresolved_checkpoint_home
    with monkeypatch.context() as guard:
        forbid_checkpoint_io(guard)
        with pytest.raises(ParityError) as caught:
            engine.copy(source, target, checkpoint=value, resume=resume)
    assert str(caught.value) == CHECKPOINT_HOME_ERROR
    assert type(caught.value.__cause__) is RuntimeError
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize("resume", [False, True])
def test_unresolved_checkpoint_home_python_cli(
    source, tmp_path, monkeypatch, capsys, unresolved_checkpoint_home, resume,
):
    target = target_for(source, tmp_path)
    before = tree_state(tmp_path)
    argv = ["copy", uristr(source), uristr(target), "--checkpoint", str(unresolved_checkpoint_home), "--json"]
    with monkeypatch.context() as guard:
        forbid_checkpoint_io(guard)
        assert main(argv + (["--resume"] if resume else [])) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"error": CHECKPOINT_HOME_ERROR}
    assert tree_state(tmp_path) == before


def nul_checkpoint(tmp_path, monkeypatch, location):
    if location == "parent":
        return tmp_path / "invalid\x00parent" / "progress.json"
    if location == "dot-dot":
        return tmp_path / "invalid\x00parent" / ".." / "progress.json"
    if location == "relative":
        monkeypatch.chdir(tmp_path)
        return Path("invalid\x00progress.json")
    if location == "tilde-user":
        return Path("~invalid\x00user/progress.json")
    if location == "expanded-home":
        # NUL cannot be assigned to the real OS environment either.
        monkeypatch.setattr(os, "environ", dict(os.environ, HOME=str(tmp_path / "invalid\x00home")))
        return Path("~/progress.json")
    return tmp_path / "invalid\x00progress.json"


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("as_string", [False, True])
@pytest.mark.parametrize("location", ["basename", "parent", "dot-dot", "relative", "tilde-user", "expanded-home"])
def test_nul_checkpoint_preflight(source, tmp_path, monkeypatch, resume, as_string, location):
    target = target_for(source, tmp_path)
    checkpoint = nul_checkpoint(tmp_path, monkeypatch, location)
    before = tree_state(tmp_path)
    with monkeypatch.context() as guard:
        forbid_checkpoint_io(guard)
        with pytest.raises(ParityError) as caught:
            engine.copy(source, target, checkpoint=str(checkpoint) if as_string else checkpoint, resume=resume)
    assert str(caught.value) == CHECKPOINT_NUL_ERROR
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("location", ["basename", "parent", "dot-dot", "relative", "tilde-user", "expanded-home"])
def test_nul_checkpoint_python_cli(source, tmp_path, monkeypatch, capsys, resume, location):
    target = target_for(source, tmp_path)
    checkpoint = nul_checkpoint(tmp_path, monkeypatch, location)
    before = tree_state(tmp_path)
    args = ["copy", uristr(source), uristr(target), "--checkpoint", str(checkpoint), "--json"]
    with monkeypatch.context() as guard:
        forbid_checkpoint_io(guard)
        assert main(args + (["--resume"] if resume else [])) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"error": CHECKPOINT_NUL_ERROR}
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("as_string", [False, True])
@pytest.mark.parametrize("location", ["basename", "parent", "expanded-home"])
def test_unencodable_checkpoint_preflight(source, tmp_path, monkeypatch, resume, as_string, location):
    target = target_for(source, tmp_path)
    checkpoint = tmp_path / "invalid\ud800progress.json"
    if location == "parent":
        checkpoint = tmp_path / "invalid\ud800parent" / "progress.json"
    elif location == "expanded-home":
        # OS environment variables cannot carry high surrogates; inject the mapping.
        monkeypatch.setattr(os, "environ", dict(os.environ, HOME=str(tmp_path / "invalid\ud800home")))
        checkpoint = Path("~/progress.json")
    before = tree_state(tmp_path)
    with monkeypatch.context() as guard:
        forbid_checkpoint_io(guard)
        with pytest.raises(ParityError, match=CHECKPOINT_ENCODING_ERROR):
            engine.copy(source, target, checkpoint=str(checkpoint) if as_string else checkpoint, resume=resume)
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize("resume", [False, True])
def test_unencodable_checkpoint_python_cli(source, tmp_path, monkeypatch, capsys, resume):
    target = target_for(source, tmp_path)
    before = tree_state(tmp_path)
    args = ["copy", uristr(source), uristr(target), "--checkpoint",
            str(tmp_path / "invalid\ud800progress.json"), "--json"]
    with monkeypatch.context() as guard:
        forbid_checkpoint_io(guard)
        assert main(args + (["--resume"] if resume else [])) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"error": CHECKPOINT_ENCODING_ERROR}
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize("as_string", [False, True])
@pytest.mark.parametrize("filename", ["progress.json", "進捗-é.json", os.fsdecode(b"progress-\xff.json")],
                         ids=["ascii", "unicode", "surrogateescape"])
def test_encodable_checkpoint_create_and_resume(source, tmp_path, as_string, filename):
    target = target_for(source, tmp_path)
    checkpoint = tmp_path / filename
    value = str(checkpoint) if as_string else checkpoint
    before = fingerprint(source.path)
    result = engine.copy(source, target, checkpoint=value, batch_size=2, top_k=2)
    assert result["copied"] == 5 and result["parity"]["ok"]
    assert json.loads(checkpoint.read_text())["parity"]["ok"]
    again = engine.copy(source, target, checkpoint=value, resume=True, top_k=2)
    assert again["copied"] == 0 and again["skipped"] == 5 and again["parity"]["ok"]
    assert fingerprint(source.path) == before
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("stage", ["encoder", "expanduser", "absolute", "later-path-check"])
@pytest.mark.parametrize("error_kind", ["os", "unicode", "value", "runtime", "type"])
def test_checkpoint_lexical_validation_preserves_unrelated_errors(tmp_path, monkeypatch, stage, error_kind):
    source = StoreURI("chroma", tmp_path / "source", "docs")
    target = target_for(source, tmp_path)
    error = {"os": OSError("unrelated filesystem failure"),
             "unicode": UnicodeEncodeError("utf-8", "\ud800", 0, 1, "filesystem failure"),
             "value": ValueError("unrelated value failure"),
             "runtime": RuntimeError("unrelated runtime failure"),
             "type": TypeError("unrelated downstream type failure")}[error_kind]

    def fail(*args, **kwargs):
        raise error

    if stage == "encoder":
        monkeypatch.setattr(checkpoints.os, "fsencode", fail)
    elif stage in {"expanduser", "absolute"}:
        monkeypatch.setattr(Path, stage, fail)
    else:
        monkeypatch.setattr(checkpoints, "checked_store_path", fail)
    if stage == "encoder" and error_kind == "unicode":
        with pytest.raises(ParityError, match=CHECKPOINT_ENCODING_ERROR) as caught:
            checkpoints.checkpoint_path(tmp_path / "progress.json", source, target, resume=False)
        assert caught.value.__cause__ is error
        return
    if stage == "expanduser" and error_kind == "runtime":
        with pytest.raises(ParityError, match=CHECKPOINT_HOME_ERROR) as caught:
            checkpoints.checkpoint_path(tmp_path / "progress.json", source, target, resume=False)
        assert caught.value.__cause__ is error
        return
    with pytest.raises(type(error)) as caught:
        checkpoints.checkpoint_path(tmp_path / "progress.json", source, target, resume=False)
    assert caught.value is error


@pytest.mark.parametrize("spelling", ["tilde", "known-user", "relative"])
@pytest.mark.parametrize("as_string", [False, True])
def test_checkpoint_home_and_relative_create_resume(source, tmp_path, monkeypatch, spelling, as_string):
    target = target_for(source, tmp_path)
    user = pwd.getpwuid(os.getuid())
    # Use the real named-user home; HOME only controls the bare '~' case.
    parent = user.pw_dir if spelling == "known-user" else tmp_path
    with tempfile.TemporaryDirectory(prefix="vp-checkpoint-home-", dir=parent) as temp:
        checkpoint = Path(temp) / "progress.json"
        if spelling == "known-user":
            value = Path(f"~{user.pw_name}") / checkpoint.relative_to(user.pw_dir)
        elif spelling == "tilde":
            monkeypatch.setenv("HOME", temp)
            value = Path("~/progress.json")
        else:
            monkeypatch.chdir(temp)
            value = Path("progress.json")
        if as_string:
            value = str(value)
        before = fingerprint(source.path)
        report = engine.copy(source, target, checkpoint=value, top_k=2)
        assert report["copied"] == 5 and report["parity"]["ok"]
        assert report["checkpoint"] == str(checkpoint)
        assert json.loads(checkpoint.read_text())["parity"]["ok"]
        resumed = engine.copy(source, target, checkpoint=value, resume=True, top_k=2)
        assert resumed["copied"] == 0 and resumed["skipped"] == 5 and resumed["parity"]["ok"]
        assert fingerprint(source.path) == before


@pytest.mark.parametrize("failure", ["before-write", "after-write", "partial-write"])
def test_resume_after_one_injected_write_failure(source, tmp_path, monkeypatch, failure):
    target = target_for(source, tmp_path)
    checkpoint = tmp_path / "copy.json"
    before = fingerprint(source.path)
    original = Store.write
    writes = []

    def fail_once(store, batch):
        writes.append([r.id for r in batch])
        if len(writes) == 2:
            if failure == "after-write":
                original(store, batch)
            elif failure == "partial-write":
                if store.uri.kind == "chroma":
                    store.collection.delete(ids=[r.id for r in batch])
                else:
                    original(store, batch[:1])
            raise OSError("injected write failure")
        original(store, batch)

    monkeypatch.setattr(Store, "write", fail_once)
    with pytest.raises(OSError, match="injected write failure"):
        engine.copy(source, target, batch_size=2, checkpoint=checkpoint, top_k=2)
    saved = json.loads(checkpoint.read_text())
    assert set(saved["completed_ids"]) == set(writes[0])
    assert saved["parity"] is None
    assert fingerprint(source.path) == before
    mapping = saved["manifest"]["records"]
    if source.kind == "chroma":
        assert mapping["guide.md::12"]["target_id"] == physical_id("guide.md::12")
    else:
        assert mapping["custom-id"]["target_id"] == "custom-id"

    resumed_writes = []

    def recording_write(store, batch):
        resumed_writes.extend(r.id for r in batch)
        original(store, batch)

    monkeypatch.setattr(Store, "write", recording_write)
    original_batches = Store.batches

    def reordered_batches(store, size):
        rows = [r for batch in original_batches(store, size) for r in batch][::-1]
        for start in range(0, len(rows), size):
            yield rows[start:start + size]

    monkeypatch.setattr(Store, "batches", reordered_batches)
    # Different batch sizes must not reinterpret an offset or lose records.
    result = engine.copy(source, target, batch_size=1, checkpoint=checkpoint, resume=True, top_k=2)
    assert result["copied"] == 3 and result["skipped"] == 2 and result["resumed"]
    assert set(resumed_writes).isdisjoint(saved["completed_ids"])
    assert result["parity"]["ok"]
    assert result["parity"]["mismatch_counts"] == dict.fromkeys(["vectors", "metadata", "documents", "id_mapping"], 0)
    assert result["parity"]["top_k"]["overlap_mean"] == 1.0
    final = json.loads(checkpoint.read_text())
    assert len(final["completed_ids"]) == 5
    assert final["manifest"] == saved["manifest"]
    assert final["parity"] == result["parity"]
    assert fingerprint(source.path) == before
    again = engine.copy(source, target, checkpoint=checkpoint, resume=True, top_k=2)
    assert again["copied"] == 0 and again["skipped"] == 5 and again["parity"]["ok"]


def test_checkpoint_refuses_changed_source_or_completed_target(chroma_store, tmp_path):
    source = chroma_store()
    target = target_for(source, tmp_path)
    checkpoint = tmp_path / "copy.json"
    engine.copy(source, target, checkpoint=checkpoint)
    saved = checkpoint.read_bytes()
    with closing(QdrantClient(path=str(target.path))) as client:
        client.set_payload("docs", payload={"document": "changed"}, points=[physical_id("guide.md::12")])
    target_before = fingerprint(target.path)
    with pytest.raises(ParityError, match="completed target"):
        engine.copy(source, target, checkpoint=checkpoint, resume=True)
    assert fingerprint(target.path) == target_before
    assert checkpoint.read_bytes() == saved
    # Restore target, then alter a source document without changing IDs or count.
    engine.copy(source, target)
    with Store(source) as store:
        vector = store.collection.get(ids=["guide.md::12"], include=["embeddings"])["embeddings"][0].tolist()
        store.collection.update(ids=["guide.md::12"], embeddings=[vector], documents=["new source"])
    target_before = fingerprint(target.path)
    source_before = fingerprint(source.path)
    with pytest.raises(ParityError, match="Checkpoint does not match"):
        engine.copy(source, target, checkpoint=checkpoint, resume=True)
    assert fingerprint(target.path) == target_before
    assert fingerprint(source.path) == source_before
    assert checkpoint.read_bytes() == saved


def test_checkpoint_cli_report_and_parity_exit(chroma_store, qdrant_store, tmp_path, capsys):
    source = chroma_store()
    target = qdrant_store(points=[{"id": 999, "vector": [1., 0., 0., 0.], "payload": {}}])
    checkpoint = tmp_path / "copy.json"
    args = ["copy", uristr(source), uristr(target), "--checkpoint", str(checkpoint), "--top-k", "2"]
    assert main(args + ["--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["parity"]["extra_ids"] == ["999"]
    assert not json.loads(checkpoint.read_text())["parity"]["ok"]
    assert main(args + ["--resume"]) == 1
    text = capsys.readouterr().out
    assert "skipped 5" in text and "PARITY MISMATCH" in text


@pytest.mark.parametrize("layout", ["source-inside", "target-inside", "symlink", "ancestor-link", "existing", "missing-parent", "directory", "fifo", "missing-resume", "no-checkpoint", "dry-run"])
def test_checkpoint_invalid_paths_and_modes_are_read_only(source, tmp_path, layout):
    target = target_for(source, tmp_path)
    checkpoint = tmp_path / "checkpoint.json"
    kwargs = {"checkpoint": checkpoint}
    if layout == "source-inside":
        kwargs["checkpoint"] = source.path / "copy.json"
    elif layout == "target-inside":
        kwargs["checkpoint"] = target.path / "copy.json"
    elif layout == "symlink":
        checkpoint.symlink_to(tmp_path / "absent.json")
    elif layout == "ancestor-link":
        (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
        kwargs["checkpoint"] = tmp_path / "link" / "checkpoint.json"
    elif layout == "existing":
        checkpoint.write_text("unrelated file")
    elif layout == "missing-parent":
        kwargs["checkpoint"] = tmp_path / "missing-parent" / "checkpoint.json"
    elif layout == "directory":
        checkpoint.mkdir()
    elif layout == "fifo":
        os.mkfifo(checkpoint)
    elif layout == "missing-resume":
        kwargs["resume"] = True
    elif layout == "no-checkpoint":
        kwargs = {"resume": True}
    elif layout == "dry-run":
        kwargs["dry_run"] = True
    before = tree_state(tmp_path)
    with pytest.raises(ParityError):
        engine.copy(source, target, **kwargs)
    assert tree_state(tmp_path) == before
    assert not target.path.exists()
    if layout == "existing":
        assert checkpoint.read_text() == "unrelated file"


@pytest.mark.parametrize("corrupt", ["json", "version", "unknown-id", "duplicate-id", "manifest", "wrong-target"])
def test_invalid_checkpoint_refused_before_target_writes(chroma_store, tmp_path, corrupt):
    source = chroma_store()
    target = target_for(source, tmp_path)
    checkpoint = tmp_path / "copy.json"
    engine.copy(source, target, checkpoint=checkpoint)
    data = json.loads(checkpoint.read_text())
    if corrupt == "json":
        checkpoint.write_text("{")
    else:
        if corrupt == "version":
            data["version"] = 100
        elif corrupt == "unknown-id":
            data["completed_ids"].append("not-in-source")
        elif corrupt == "duplicate-id":
            data["completed_ids"].append(data["completed_ids"][0])
        elif corrupt == "manifest":
            data["manifest"]["records"]["guide.md::12"]["target_id"] = 123
        elif corrupt == "wrong-target":
            target = StoreURI("qdrant", tmp_path / "wrong-target", "docs")
        checkpoint.write_text(json.dumps(data))
    before = fingerprint(tmp_path / "target")
    saved = checkpoint.read_bytes()
    with pytest.raises(ParityError, match="[Cc]heckpoint"):
        engine.copy(source, target, checkpoint=checkpoint, resume=True)
    assert fingerprint(tmp_path / "target") == before
    assert checkpoint.read_bytes() == saved
    if corrupt == "wrong-target":
        assert not target.path.exists()


def test_checkpoint_replace_failure_replays_unacknowledged_write(chroma_store, tmp_path, monkeypatch):
    from ragwrench.migrate import checkpoints

    source = chroma_store()
    target = target_for(source, tmp_path)
    checkpoint = tmp_path / "copy.json"
    before = fingerprint(source.path)
    original = checkpoints.os.replace
    replacements = 0

    def fail_once(src, dst):
        nonlocal replacements
        if Path(dst) == checkpoint:
            replacements += 1
        if Path(dst) == checkpoint and replacements == 2:
            raise OSError("injected checkpoint replace failure")
        return original(src, dst)

    monkeypatch.setattr(checkpoints.os, "replace", fail_once)
    with pytest.raises(OSError, match="injected checkpoint"):
        engine.copy(source, target, batch_size=2, checkpoint=checkpoint)
    saved = json.loads(checkpoint.read_text())
    assert saved["completed_ids"] == []  # Target accepted batch one; checkpoint did not.
    assert engine.inspect(target)["count"] == 2
    assert not list(tmp_path.glob(".copy.json.*.tmp"))
    result = engine.copy(source, target, checkpoint=checkpoint, resume=True, top_k=2)
    assert result["copied"] == 5 and result["skipped"] == 0 and result["parity"]["ok"]
    assert fingerprint(source.path) == before


def test_resume_after_parity_backend_failure(chroma_store, tmp_path, monkeypatch):
    source = chroma_store()
    target = target_for(source, tmp_path)
    checkpoint = tmp_path / "copy.json"
    original = engine.verify

    def fail(*args, **kwargs):
        raise OSError("injected parity failure")

    monkeypatch.setattr(engine, "verify", fail)
    with pytest.raises(OSError, match="injected parity"):
        engine.copy(source, target, checkpoint=checkpoint)
    saved = json.loads(checkpoint.read_text())
    assert len(saved["completed_ids"]) == 5 and saved["parity"] is None
    monkeypatch.setattr(engine, "verify", original)
    result = engine.copy(source, target, checkpoint=checkpoint, resume=True, top_k=2)
    assert result["copied"] == 0 and result["skipped"] == 5 and result["parity"]["ok"]


def test_missing_acknowledged_target_refused(chroma_store, tmp_path):
    source = chroma_store()
    target = target_for(source, tmp_path)
    checkpoint = tmp_path / "copy.json"
    engine.copy(source, target, checkpoint=checkpoint)
    with closing(QdrantClient(path=str(target.path))) as client:
        client.delete("docs", points_selector=models.PointIdsList(points=[physical_id("guide.md::12")]))
    before = fingerprint(target.path)
    saved = checkpoint.read_bytes()
    with pytest.raises(ParityError, match="completed target record is missing"):
        engine.copy(source, target, checkpoint=checkpoint, resume=True)
    assert fingerprint(target.path) == before
    assert checkpoint.read_bytes() == saved


def test_named_vector_checkpoint(qdrant_store, tmp_path):
    source = qdrant_store(vector_names=["dense"], points=[
        {"id": 77, "vector": {"dense": [1., 2., 3., 4.]}, "payload": {"__id__": "alias"}}])
    target = target_for(source, tmp_path)
    checkpoint = tmp_path / "copy.json"
    before = fingerprint(source.path)
    result = engine.copy(source, target, checkpoint=checkpoint, vector_name="dense", top_k=1)
    assert result["parity"]["ok"]
    assert json.loads(checkpoint.read_text())["manifest"]["vector_name"] == "dense"
    assert engine.copy(source, target, checkpoint=checkpoint, resume=True, vector_name="dense")["skipped"] == 1
    assert fingerprint(source.path) == before


NONFINITE_JSON_TOKENS = ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"]


NONOBJECT_IDENTITY_JSON = ["null", "[]", '[{"kind": "chroma"}]', '"identity"', '""',
                           "0", "-17", "1.25", "true", "false", "missing"]


def checkpoint_identity(data, field, value):
    if value == "missing":
        del data["manifest"][field]
    else:
        data["manifest"][field] = json.loads(value)
    return json.dumps(data)


@pytest.mark.parametrize("field", ["source", "target"])
@pytest.mark.parametrize("value", NONOBJECT_IDENTITY_JSON)
def test_checkpoint_load_requires_identity_object(tmp_path, field, value):
    checkpoint = tmp_path / "progress.json"
    data = {"version": 1, "manifest": {"source": {}, "target": {}, "schema": {}, "records": {}},
            "completed_ids": [], "parity": None}
    checkpoint.write_text(checkpoint_identity(data, field, value))
    before = tree_state(tmp_path)
    with pytest.raises(ParityError, match="^Invalid or unsupported checkpoint$"):
        checkpoints.load(checkpoint)
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize("field", ["source", "target"])
def test_checkpoint_nonobject_identity_resume(source, tmp_path, monkeypatch, capsys, field):
    target = target_for(source, tmp_path)
    checkpoint = tmp_path / "progress.json"
    assert engine.copy(source, target, checkpoint=checkpoint, top_k=2)["parity"]["ok"]
    valid = checkpoint.read_text()
    for uri, name in ((source, "source"), (target, "target")):
        assert json.loads(valid)["manifest"][name] == checkpoints.uri_identity(uri)

    def forbidden(*args, **kwargs):
        pytest.fail("Snapshot/backend access or checkpoint save after invalid identity read")

    for value in NONOBJECT_IDENTITY_JSON:
        checkpoint.write_text(checkpoint_identity(json.loads(valid), field, value))
        before = tree_state(tmp_path)
        with monkeypatch.context() as guard:
            guard.setattr(engine, "read_store", forbidden)
            guard.setattr(engine, "Store", forbidden)
            guard.setattr(Store, "__enter__", forbidden)
            guard.setattr(checkpoints, "save", forbidden)
            with pytest.raises(ParityError, match="^Invalid or unsupported checkpoint$"):
                engine.copy(source, target, checkpoint=checkpoint, resume=True)
            assert main(["copy", uristr(source), uristr(target), "--checkpoint", str(checkpoint),
                         "--resume", "--json"]) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == '{"error": "Invalid or unsupported checkpoint"}\n'
        assert tree_state(tmp_path) == before


@pytest.mark.parametrize("field", ["source", "target"])
@pytest.mark.parametrize("identity", [{}, {"kind": "chroma", "path": "/local", "collection": "docs"},
    {"kind": None, "path": ["NaN"], "collection": False, "extra": {"finite": 1.25}}])
def test_checkpoint_identity_object_load_is_shape_only(tmp_path, field, identity):
    checkpoint = tmp_path / "progress.json"
    data = {"version": 1, "manifest": {"source": {}, "target": {}, "schema": {}, "records": {}},
            "completed_ids": [], "parity": None}
    data["manifest"][field] = identity
    checkpoint.write_text(json.dumps(data))
    assert checkpoints.load(checkpoint) == data


def test_checkpoint_identity_objects_still_require_manifest_match(source, tmp_path, monkeypatch):
    target = target_for(source, tmp_path)
    checkpoint = tmp_path / "progress.json"
    assert engine.copy(source, target, checkpoint=checkpoint, top_k=2)["parity"]["ok"]
    valid = checkpoint.read_text()

    def forbidden(*args, **kwargs):
        pytest.fail("Write/save after identity manifest mismatch")

    for field in ("source", "target"):
        for identity in ({}, {"kind": None, "path": [], "extra": False}):
            data = json.loads(valid)
            data["manifest"][field] = identity
            checkpoint.write_text(json.dumps(data))
            assert checkpoints.load(checkpoint) == data
            before = tree_state(tmp_path)
            with monkeypatch.context() as guard:
                guard.setattr(Store, "write", forbidden)
                guard.setattr(checkpoints, "save", forbidden)
                with pytest.raises(ParityError, match="^Checkpoint does not match source contents, ID mapping, target or vector selection$"):
                    engine.copy(source, target, checkpoint=checkpoint, resume=True)
            assert tree_state(tmp_path) == before


NONOBJECT_SCHEMA_JSON = ["null", "[]", '[{"dimension": 4}]', '"schema"', '""',
                         "0", "-17", "1.25", "true", "false", "missing"]


def checkpoint_schema(data, value):
    if value == "missing":
        del data["manifest"]["schema"]
    else:
        data["manifest"]["schema"] = json.loads(value)
    return json.dumps(data)


@pytest.mark.parametrize("value", NONOBJECT_SCHEMA_JSON)
def test_checkpoint_load_requires_schema_object(tmp_path, value):
    checkpoint = tmp_path / "progress.json"
    data = {"version": 1, "manifest": {"source": {}, "target": {}, "schema": {}, "records": {}}, "completed_ids": [], "parity": None}
    checkpoint.write_text(checkpoint_schema(data, value))
    before = tree_state(tmp_path)
    with pytest.raises(ParityError, match="^Invalid or unsupported checkpoint$"):
        checkpoints.load(checkpoint)
    assert tree_state(tmp_path) == before


def test_checkpoint_nonobject_schema_resume(source, tmp_path, monkeypatch, capsys):
    target = target_for(source, tmp_path)
    checkpoint = tmp_path / "progress.json"
    assert engine.copy(source, target, checkpoint=checkpoint, top_k=2)["parity"]["ok"]
    valid = checkpoint.read_text()

    def forbidden(*args, **kwargs):
        pytest.fail("Snapshot/backend access or checkpoint save after invalid schema read")

    for value in NONOBJECT_SCHEMA_JSON:
        checkpoint.write_text(checkpoint_schema(json.loads(valid), value))
        before = tree_state(tmp_path)
        with monkeypatch.context() as guard:
            guard.setattr(engine, "read_store", forbidden)
            guard.setattr(engine, "Store", forbidden)
            guard.setattr(Store, "__enter__", forbidden)
            guard.setattr(checkpoints, "save", forbidden)
            with pytest.raises(ParityError, match="^Invalid or unsupported checkpoint$"):
                engine.copy(source, target, checkpoint=checkpoint, resume=True)
            assert main(["copy", uristr(source), uristr(target), "--checkpoint", str(checkpoint),
                         "--resume", "--json"]) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == '{"error": "Invalid or unsupported checkpoint"}\n'
        assert tree_state(tmp_path) == before


@pytest.mark.parametrize("schema", [{}, {"dimension": 4},
    {"dimension": 4, "distance": "l2", "count": 5},
    {"dimension": None, "distance": ["NaN"], "count": False, "extra": {"finite": 1.25}}])
def test_checkpoint_schema_object_load_is_shape_only(tmp_path, schema):
    checkpoint = tmp_path / "progress.json"
    data = {"version": 1, "manifest": {"source": {}, "target": {}, "schema": schema, "records": {}}, "completed_ids": [], "parity": None}
    checkpoint.write_text(json.dumps(data))
    assert checkpoints.load(checkpoint) == data


NONOBJECT_RECORDS_JSON = ["null", "[]", '[{"target_id": "alpha"}]', '"records"', '""',
                          "0", "-17", "1.25", "true", "false", "missing"]


def checkpoint_records(data, value):
    if value == "missing":
        del data["manifest"]["records"]
    else:
        data["manifest"]["records"] = json.loads(value)
    return json.dumps(data)


@pytest.mark.parametrize("value", NONOBJECT_RECORDS_JSON)
def test_checkpoint_load_requires_records_object(tmp_path, value):
    checkpoint = tmp_path / "progress.json"
    data = {"version": 1, "manifest": {"source": {}, "target": {}, "schema": {}, "records": {}}, "completed_ids": [], "parity": None}
    checkpoint.write_text(checkpoint_records(data, value))
    before = tree_state(tmp_path)
    with pytest.raises(ParityError, match="^Invalid or unsupported checkpoint$"):
        checkpoints.load(checkpoint)
    assert tree_state(tmp_path) == before


def test_checkpoint_nonobject_records_resume(source, tmp_path, monkeypatch, capsys):
    target = target_for(source, tmp_path)
    checkpoint = tmp_path / "progress.json"
    assert engine.copy(source, target, checkpoint=checkpoint, top_k=2)["parity"]["ok"]
    valid = checkpoint.read_text()
    mapping = json.loads(valid)["manifest"]["records"]
    assert type(mapping) is dict and len(mapping) == 5

    def forbidden(*args, **kwargs):
        pytest.fail("Snapshot/backend access or checkpoint save after invalid records read")

    for value in NONOBJECT_RECORDS_JSON:
        checkpoint.write_text(checkpoint_records(json.loads(valid), value))
        before = tree_state(tmp_path)
        with monkeypatch.context() as guard:
            guard.setattr(engine, "read_store", forbidden)
            guard.setattr(engine, "Store", forbidden)
            guard.setattr(Store, "__enter__", forbidden)
            guard.setattr(checkpoints, "save", forbidden)
            with pytest.raises(ParityError, match="^Invalid or unsupported checkpoint$"):
                engine.copy(source, target, checkpoint=checkpoint, resume=True)
            assert main(["copy", uristr(source), uristr(target), "--checkpoint", str(checkpoint),
                         "--resume", "--json"]) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == '{"error": "Invalid or unsupported checkpoint"}\n'
        assert tree_state(tmp_path) == before


@pytest.mark.parametrize("records", [{}, {"alpha": {"target_id": 77, "sha256": "a" * 64}},
    {"alpha": None}, {"alpha": {"target_id": [], "sha256": False}, "other": [1, "NaN"]}])
def test_checkpoint_records_object_load_is_shape_only(tmp_path, records):
    checkpoint = tmp_path / "progress.json"
    data = {"version": 1, "manifest": {"source": {}, "target": {}, "schema": {}, "records": records}, "completed_ids": [], "parity": None}
    checkpoint.write_text(json.dumps(data))
    assert checkpoints.load(checkpoint) == data


def checkpoint_number(data, location, token):
    """Put a raw JSON number in an object or a nested array, never a string."""
    if location == "dimension":
        data["manifest"]["schema"]["dimension"] = "RAW_NUMBER"
    elif location in ("source", "target"):
        data["manifest"][location]["extra"] = {"numbers": ["RAW_NUMBER"]}
    elif location == "manifest-list":
        data["manifest"]["extra"] = [{"numbers": [0, "RAW_NUMBER"]}]
    else:
        data["parity"] = {"extra": [None, {"numbers": ["RAW_NUMBER"]}]}
    return json.dumps(data).replace('"RAW_NUMBER"', token)


@pytest.mark.parametrize("token", NONFINITE_JSON_TOKENS)
@pytest.mark.parametrize("location", ["dimension", "manifest-list", "parity-list", "source", "target"])
def test_checkpoint_load_nonfinite(tmp_path, token, location):
    checkpoint = tmp_path / "progress.json"
    data = {"version": 1, "manifest": {"source": {}, "target": {}, "schema": {"dimension": 4}, "records": {}}, "completed_ids": [], "parity": None}
    checkpoint.write_text(checkpoint_number(data, location, token))
    before = tree_state(tmp_path)
    with pytest.raises(ParityError, match="^Invalid or unsupported checkpoint$"):
        checkpoints.load(checkpoint)
    assert tree_state(tmp_path) == before


def test_checkpoint_nonfinite_resume(source, tmp_path, monkeypatch, capsys):
    target = target_for(source, tmp_path)
    checkpoint = tmp_path / "progress.json"
    assert engine.copy(source, target, checkpoint=str(checkpoint), top_k=2)["parity"]["ok"]
    valid = checkpoint.read_text()

    def forbidden(*args, **kwargs):
        pytest.fail("Snapshot/backend access or checkpoint save after invalid checkpoint read")

    for location in ("dimension", "manifest-list", "parity-list", "source", "target"):
        for token in NONFINITE_JSON_TOKENS:
            checkpoint.write_text(checkpoint_number(json.loads(valid), location, token))
            before = tree_state(tmp_path)
            with monkeypatch.context() as guard:
                guard.setattr(engine, "read_store", forbidden)
                guard.setattr(engine, "Store", forbidden)
                guard.setattr(checkpoints, "save", forbidden)
                with pytest.raises(ParityError, match="^Invalid or unsupported checkpoint$"):
                    engine.copy(source, target, checkpoint=str(checkpoint), resume=True)
                assert main(["copy", uristr(source), uristr(target), "--checkpoint", str(checkpoint),
                             "--resume", "--json"]) == 2
            captured = capsys.readouterr()
            assert captured.out == ""
            assert captured.err == '{"error": "Invalid or unsupported checkpoint"}\n'
            assert tree_state(tmp_path) == before


def test_checkpoint_load_finite_numbers_and_token_strings(tmp_path):
    checkpoint = tmp_path / "progress.json"
    data = {"version": 1, "manifest": {"source": {}, "target": {}, "schema": {"dimension": 4}, "records": {},
            "nested": [0, -17, 2**100, 1.25, 1.7976931348623157e308, -1.7976931348623157e308,
                       1e-300, True, False, None, {t: t for t in NONFINITE_JSON_TOKENS}]},
            "completed_ids": ["NaN", "Infinity", "-Infinity", "1e999"],
            "parity": {"text": "NaN Infinity -Infinity 1e999", "values": [0.0, -0.0]}}
    checkpoint.write_text(json.dumps(data, allow_nan=False))
    assert checkpoints.load(checkpoint) == data
    # Underflow is finite; strict input does not impose a precision policy.
    checkpoint.write_text(checkpoint.read_text().replace("1e-300", "1e-999"))
    data["manifest"]["nested"][6] = 0.0
    assert checkpoints.load(checkpoint) == data


def test_checkpoint_token_strings_resume(source, tmp_path):
    target = target_for(source, tmp_path)
    checkpoint = tmp_path / "progress.json"
    before = fingerprint(source.path)
    assert engine.copy(source, target, checkpoint=str(checkpoint), top_k=2)["parity"]["ok"]
    data = json.loads(checkpoint.read_text())
    data["parity"]["notes"] = ["NaN Infinity -Infinity 1e999", {t: t for t in NONFINITE_JSON_TOKENS}]
    checkpoint.write_text(json.dumps(data, allow_nan=False))
    resumed = engine.copy(source, target, checkpoint=str(checkpoint), resume=True, top_k=2)
    assert resumed["copied"] == 0 and resumed["skipped"] == 5 and resumed["parity"]["ok"]
    assert fingerprint(source.path) == before


@pytest.mark.parametrize("invalid", [
    'null', '[]', '{}',
    '{"version":true,"manifest":{"source":{},"target":{},"schema":{},"records":{}},"completed_ids":[],"parity":null}',
    '{"version":2,"manifest":{"source":{},"target":{},"schema":{},"records":{}},"completed_ids":[],"parity":null}',
    '{"version":1,"manifest":[],"completed_ids":[],"parity":null}',
    '{"version":1,"manifest":{"source":{},"target":{},"schema":{},"records":{}},"completed_ids":[1],"parity":null}',
    '{"version":1,"manifest":{"source":{},"target":{},"schema":{},"records":{}},"completed_ids":["a","a"],"parity":null}',
    '{"version":1,"manifest":{"source":{},"target":{},"schema":{},"records":{}},"completed_ids":[],"parity":[]}',
    '{"version":1,"version":1,"manifest":{"source":{},"target":{},"schema":{},"records":{}},"completed_ids":[],"parity":null}',
    '{"version":1,"manifest":{"source":{},"target":{},"schema":{},"records":{},"nested":{"a":1,"a":2}},"completed_ids":[],"parity":null}',
    '{"version":1,"manifest":{"source":{},"target":{},"schema":{},"records":{}},"completed_ids":[],"parity":{"a":1,"a":2}}',
    '{"version":1,"manifest":{"source":{},"target":{},"schema":{},"records":{"a":1,"a":2}},"completed_ids":[],"parity":null}',
    '{"version":1,"manifest":{"source":{"kind":"chroma","kind":"qdrant"},"target":{},"schema":{},"records":{}},"completed_ids":[],"parity":null}',
    '{"version":1,"manifest":{"source":{},"target":{"path":"a","path":"b"},"schema":{},"records":{}},"completed_ids":[],"parity":null}',
])
def test_checkpoint_load_retains_shape_version_duplicate_checks(tmp_path, invalid):
    checkpoint = tmp_path / "progress.json"
    checkpoint.write_text(invalid)
    with pytest.raises(ParityError, match="^Invalid or unsupported checkpoint$"):
        checkpoints.load(checkpoint)


@pytest.mark.parametrize("stage", ["read", "parse"])
@pytest.mark.parametrize("error_kind", ["os", "type"])
def test_checkpoint_load_preserves_unrelated_errors(tmp_path, monkeypatch, stage, error_kind):
    checkpoint = tmp_path / "progress.json"
    checkpoint.write_text('{"version":1,"manifest":{"source":{},"target":{},"schema":{},"records":{}},"completed_ids":[],"parity":null}')
    error = OSError("unrelated I/O failure") if error_kind == "os" else TypeError("unrelated type failure")

    def fail(*args, **kwargs):
        raise error

    if stage == "read":
        monkeypatch.setattr(Path, "read_text", fail)
    else:
        monkeypatch.setattr(checkpoints.json, "loads", fail)
    with pytest.raises(type(error)) as caught:
        checkpoints.load(checkpoint)
    assert caught.value is error
