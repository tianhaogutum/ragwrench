"""Filesystem boundaries using real stores from the pinned embedded clients."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from ragwrench.migrate import engine, stores
from ragwrench.migrate.cli import main
from ragwrench.migrate.model import ParityError, Record, Schema, StoreURI, checked_store_path
from ragwrench.migrate.stores import Store, check_directory, read_store


@pytest.fixture(params=["chroma", "qdrant"])
def migration_source(request, chroma_store, qdrant_store):
    if request.param == "chroma":
        return chroma_store()
    return qdrant_store(points=[{"id": 7, "vector": [1., 2., 3., 4.], "payload": {}}])


def uri_text(uri):
    return f"{uri.kind}:{uri.path}#{uri.collection}"


def tree_state(root):
    """Include empty directories and dangling links as well as file contents."""
    state = {}
    for path in root.rglob("*"):
        key = str(path.relative_to(root))
        if path.is_symlink():
            state[key] = ("symlink", str(path.readlink()))
        elif path.is_dir():
            state[key] = ("directory",)
        elif path.is_file():
            state[key] = ("file", hashlib.sha256(path.read_bytes()).hexdigest())
        else:
            state[key] = ("special", stat.S_IFMT(path.lstat().st_mode))
    return state


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("layout", ["root", "existing-store", "dangling", "loop", "ancestor", "cancelled-ancestor"])
def test_cli_rejects_target_symlinks_before_mutation(
    migration_source, tmp_path, capsys, layout, dry_run,
):
    src = migration_source
    target_kind = "qdrant" if src.kind == "chroma" else "chroma"
    real = tmp_path / "real-target"
    link = tmp_path / "target-link"
    if layout != "dangling":
        real.mkdir()
    if layout == "existing-store":
        engine.copy(src, StoreURI(target_kind, real, "docs"))
    link.symlink_to(link if layout == "loop" else real, target_is_directory=True)
    path = link
    if layout == "ancestor":
        path = link / "missing-parent" / "new-store"
    elif layout == "cancelled-ancestor":
        path = link / ".." / "new-store"
    dst = StoreURI(target_kind, path, "docs")
    before = tree_state(tmp_path)

    result = main(["copy", uri_text(src), uri_text(dst), "--json"] + (["--dry-run"] if dry_run else []))

    assert result == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "symlink" in json.loads(output.err)["error"]
    assert tree_state(tmp_path) == before


def test_relative_separate_paths_and_missing_parents_still_work(migration_source, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "container").mkdir()
    src = replace(migration_source, path=Path(migration_source.path.name))
    target_path = Path("container") / ".." / "new-parent" / "new-store"
    dst = StoreURI("qdrant" if src.kind == "chroma" else "chroma", target_path, "docs")
    before = tree_state(src.path)

    preview = engine.copy(src, dst, dry_run=True)
    assert preview["would_copy"] == engine.inspect(src)["count"]
    assert not (tmp_path / "new-parent").exists()
    assert engine.copy(src, dst)["copied"] == preview["would_copy"]
    assert engine.verify(src, dst, top_k=1)["ok"]
    assert tree_state(src.path) == before


@pytest.mark.parametrize("command", ["inspect", "copy", "verify"])
@pytest.mark.parametrize("layout", ["root", "ancestor"])
def test_cli_rejects_source_symlinks(migration_source, tmp_path, capsys, command, layout):
    src = migration_source
    link = tmp_path / "source-link"
    link.symlink_to(src.path if layout == "root" else src.path.parent, target_is_directory=True)
    src = replace(src, path=link if layout == "root" else link / src.path.name)
    dst = StoreURI("qdrant" if src.kind == "chroma" else "chroma", tmp_path / "absent", "docs")
    before = tree_state(tmp_path)

    args = [command, uri_text(src)] + ([] if command == "inspect" else [uri_text(dst)])
    assert main(args + ["--json"]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "symlink" in json.loads(output.err)["error"]
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize("entry", ["check-directory", "inspect", "copy-target", "store"])
@pytest.mark.parametrize("layout", ["root", "ancestor", "dangling"])
def test_direct_python_callers_cannot_bypass_path_checks(migration_source, tmp_path, entry, layout):
    src = migration_source
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "missing" if layout == "dangling" else src.path, target_is_directory=True)
    unsafe = replace(src, path=link / "new-store" if layout == "ancestor" else link)
    before = tree_state(tmp_path)

    with pytest.raises(ParityError, match="symlink"):
        if entry == "check-directory":
            check_directory(unsafe, required=False)
        elif entry == "inspect":
            engine.inspect(unsafe)
        elif entry == "copy-target":
            target_kind = "qdrant" if src.kind == "chroma" else "chroma"
            engine.copy(src, replace(unsafe, kind=target_kind))
        else:
            with Store(unsafe):
                pytest.fail("A backend was opened through a symlink")
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize("alias", ["relative", "dotdot", "nested", "parent"])
def test_direct_path_aliases_rejected_before_backend_access(migration_source, tmp_path, monkeypatch, alias):
    src = migration_source
    monkeypatch.chdir(tmp_path)
    paths = {
        "relative": Path(src.path.name),
        "dotdot": src.path / ".." / src.path.name,
        "nested": Path(src.path.name) / "new-store",
        "parent": src.path / "..",
    }
    dst = StoreURI("qdrant" if src.kind == "chroma" else "chroma", paths[alias], "docs")
    before = tree_state(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Aliased paths reached snapshot/backend access")

    monkeypatch.setattr(engine, "read_store", forbidden)
    with pytest.raises(ParityError, match="separate"):
        engine.copy(src, dst)
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize("name", ["../outside", "absolute"])
@pytest.mark.parametrize("entry", [
    "copy-source", "copy-target", "dry-run-source", "dry-run-target",
    "verify-source", "verify-target", "inspect", "read-store", "check-directory", "store",
])
def test_direct_collection_traversal_refused_before_backend_access(
    migration_source, tmp_path, monkeypatch, entry, name,
):
    src = migration_source
    dst = StoreURI("qdrant" if src.kind == "chroma" else "chroma", tmp_path / "target", "docs")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel").write_text("must stay unchanged")
    name = str(outside) if name == "absolute" else name
    before = tree_state(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid collection name reached snapshot/backend access")

    monkeypatch.setattr(engine, "read_store", forbidden)
    monkeypatch.setattr(stores.shutil, "copytree", forbidden)
    monkeypatch.setattr("chromadb.PersistentClient", forbidden)
    monkeypatch.setattr("qdrant_client.QdrantClient", forbidden)
    with pytest.raises(ParityError, match="Collection name"):
        if entry.endswith("target"):
            dst = replace(dst, collection=name)
        else:
            src = StoreURI(src.kind, src.path, name)
        if entry.startswith(("copy-", "dry-run-")):
            engine.copy(src, dst, dry_run=entry.startswith("dry-run-"))
        elif entry.startswith("verify-"):
            engine.verify(src, dst)
        elif entry == "inspect":
            engine.inspect(src)
        elif entry == "read-store":
            with read_store(src):
                pytest.fail("Invalid collection name was accepted")
        elif entry == "check-directory":
            check_directory(src, required=True)
        else:
            with Store(src):
                pytest.fail("Invalid collection name was accepted")
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize("recognized", [False, True])
@pytest.mark.parametrize("entry", ["file-link", "directory-link", "dangling-link", "loop-link", "fifo"])
def test_direct_store_rejects_unsafe_descendants_before_backend_access(
    migration_source, tmp_path, monkeypatch, entry, recognized,
):
    src = migration_source
    uri = src if recognized else replace(src, path=tmp_path / "unrecognized")
    nested = uri.path / "nested" / "deeper"
    nested.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("must stay unchanged")
    unsafe = nested / "unsafe"
    if entry == "fifo":
        os.mkfifo(unsafe)
    else:
        targets = {"file-link": sentinel, "directory-link": outside,
                   "dangling-link": outside / "missing", "loop-link": unsafe}
        unsafe.symlink_to(targets[entry], target_is_directory=entry == "directory-link")
    before = tree_state(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Unsafe descendant reached metadata audit/backend construction")

    monkeypatch.setattr("chromadb.PersistentClient", forbidden)
    monkeypatch.setattr("qdrant_client.QdrantClient", forbidden)
    monkeypatch.setattr("ragwrench.migrate.safety.audit_qdrant", forbidden)
    with pytest.raises(ParityError, match="symlink or special file"):
        with Store(uri):
            pytest.fail("Unsafe descendant was accepted")
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize("layout", ["missing", "missing-parents", "empty", "existing"])
def test_direct_store_can_create_safe_collections(migration_source, tmp_path, layout):
    src = migration_source
    path = src.path if layout == "existing" else tmp_path / "target"
    if layout == "missing-parents":
        path = path / "parent" / "store"
    elif layout == "empty":
        path.mkdir()
    uri = StoreURI(src.kind, path, "new-docs")
    record = Record("one", [1., 2.], {"key": "value"}, "text")

    with Store(uri) as store:
        assert store.collection is None
        store.create(Schema(2, "l2", 0))
        store.write([record])
    with Store(uri) as store:
        assert store.schema() == Schema(2, "l2", 1)
        restored = next(store.batches(1))[0]
        assert (restored.id, restored.vector, restored.metadata, restored.document) == (
            record.id, record.vector, record.metadata, record.document,
        )
    if layout == "existing":
        with Store(src) as store:
            assert store.schema().count > 0  # The original collection remains readable.


@pytest.mark.parametrize("entry", [
    "copy-source", "copy-target", "dry-run-source", "dry-run-target",
    "verify-source", "verify-target", "inspect", "read-store", "check-directory", "store",
])
def test_direct_nul_paths_refused_before_filesystem_or_backend_access(
    migration_source, tmp_path, monkeypatch, entry,
):
    src = migration_source
    dst = StoreURI("qdrant" if src.kind == "chroma" else "chroma", tmp_path / "target", "docs")
    (tmp_path / "outside-sentinel").write_text("must stay unchanged")
    before = tree_state(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("NUL path reached filesystem/snapshot/backend access")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "lstat", forbidden)
        guard.setattr(stores.shutil, "copytree", forbidden)
        guard.setattr("chromadb.PersistentClient", forbidden)
        guard.setattr("qdrant_client.QdrantClient", forbidden)
        guard.setattr("ragwrench.migrate.safety.audit_qdrant", forbidden)
        with pytest.raises(ParityError, match="local path.*NUL"):
            if entry.endswith("target"):
                dst = replace(dst, path=tmp_path / "invalid\x00path")
            else:
                src = StoreURI(src.kind, tmp_path / "invalid\x00path", "docs")
            if entry.startswith(("copy-", "dry-run-")):
                engine.copy(src, dst, dry_run=entry.startswith("dry-run-"))
            elif entry.startswith("verify-"):
                engine.verify(src, dst)
            elif entry == "inspect":
                engine.inspect(src)
            elif entry == "read-store":
                with read_store(src):
                    pytest.fail("NUL path accepted")
            elif entry == "check-directory":
                check_directory(src, required=True)
            else:
                with Store(src):
                    pytest.fail("NUL path accepted")
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize("path", [Path("invalid\x00path"), Path("bad\x00parent/../store")])
def test_checked_store_path_rejects_nul_before_lstat(monkeypatch, path):
    def forbidden(*args, **kwargs):
        pytest.fail("NUL path reached filesystem access")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "lstat", forbidden)
        with pytest.raises(ParityError, match="local path.*NUL"):
            checked_store_path(path)


@pytest.mark.parametrize("kind", ["chroma", "qdrant"])
def test_cli_nul_path_refusal_is_json_error_without_mutation(tmp_path, capsys, kind):
    (tmp_path / "sentinel").write_text("must stay unchanged")
    before = tree_state(tmp_path)
    # An OS process argv cannot contain NUL; exercise the Python CLI entry directly.
    assert main(["inspect", f"{kind}:{tmp_path}/invalid\x00path#docs", "--json"]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "local paths" in json.loads(output.err)["error"]
    assert tree_state(tmp_path) == before


@pytest.fixture
def unknown_home_path():
    import pwd

    username = "vecparity_no_such_user_r7_9f72a1"
    with pytest.raises(KeyError):
        pwd.getpwnam(username)
    return Path(f"~{username}/store")


@pytest.mark.parametrize("entry", [
    "copy-source", "copy-target", "dry-run-source", "dry-run-target",
    "verify-source", "verify-target", "inspect", "read-store", "optional-read-store",
    "check-directory", "store",
])
def test_unknown_home_refused_before_snapshot_or_backend_access(
    migration_source, tmp_path, monkeypatch, unknown_home_path, entry,
):
    src = migration_source
    dst = StoreURI("qdrant" if src.kind == "chroma" else "chroma", tmp_path / "target", "docs")
    if entry.endswith("target"):
        dst = replace(dst, path=unknown_home_path)
    else:
        src = replace(src, path=unknown_home_path)
    (tmp_path / "sentinel").write_text("must stay unchanged")
    before = tree_state(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Unresolvable home reached snapshot/backend access")

    with monkeypatch.context() as guard:
        guard.setattr(stores, "TemporaryDirectory", forbidden)
        guard.setattr(stores.shutil, "copytree", forbidden)
        guard.setattr("chromadb.PersistentClient", forbidden)
        guard.setattr("qdrant_client.QdrantClient", forbidden)
        guard.setattr("ragwrench.migrate.safety.audit_qdrant", forbidden)
        with pytest.raises(ParityError, match="local path.*resolvable home directory"):
            if entry.startswith(("copy-", "dry-run-")):
                engine.copy(src, dst, dry_run=entry.startswith("dry-run-"))
            elif entry.startswith("verify-"):
                engine.verify(src, dst)
            elif entry == "inspect":
                engine.inspect(src)
            elif entry.endswith("read-store"):
                with read_store(src, optional=entry.startswith("optional-")):
                    pytest.fail("Unresolvable home accepted")
            elif entry == "check-directory":
                check_directory(src, required=False)
            else:
                with Store(src):
                    pytest.fail("Unresolvable home accepted")
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize("kind", ["chroma", "qdrant"])
@pytest.mark.parametrize("entry", ["inspect", "copy-source", "copy-target", "verify-source", "verify-target"])
def test_cli_unknown_home_is_specific_input_error(
    tmp_path, capsys, unknown_home_path, kind, entry,
):
    (tmp_path / "sentinel").write_text("must stay unchanged")
    before = tree_state(tmp_path)
    invalid = f"{kind}:{unknown_home_path}#docs"
    valid = f"{'qdrant' if kind == 'chroma' else 'chroma'}:{tmp_path}/target#docs"
    command = entry.split("-")[0]
    if command == "inspect":
        args = [command, invalid]
    elif entry.endswith("target"):
        args = [command, valid, invalid]
    else:
        args = [command, invalid, valid]

    assert main(args + ["--json"]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err) == {"error": "Store path must be a local path with a resolvable home directory"}
    assert tree_state(tmp_path) == before


def test_unknown_home_rejected_before_filesystem_probe(monkeypatch, unknown_home_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Unresolvable home reached filesystem probe")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "lstat", forbidden)
        with pytest.raises(ParityError, match="local path.*resolvable home directory"):
            checked_store_path(unknown_home_path)


def test_runtime_error_after_home_expansion_is_not_misclassified(monkeypatch, tmp_path):
    def failed_absolute(*args, **kwargs):
        raise RuntimeError("unrelated absolute-path failure")

    monkeypatch.setattr(Path, "absolute", failed_absolute)
    with pytest.raises(RuntimeError, match="unrelated absolute-path failure") as caught:
        checked_store_path(tmp_path)
    assert not isinstance(caught.value, ParityError)


def test_current_username_home_remains_supported():
    import pwd

    user = pwd.getpwuid(os.getuid())
    assert checked_store_path(Path(f"~{user.pw_name}")) == Path(user.pw_dir).resolve()


def test_home_paths_and_missing_parent_targets_still_work(migration_source, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    src = replace(migration_source, path=Path("~") / migration_source.path.name)
    dst = StoreURI("qdrant" if src.kind == "chroma" else "chroma", Path("~/missing-parent/store"), "docs")
    before = tree_state(migration_source.path)

    assert engine.copy(src, dst, dry_run=True)["would_copy"] == engine.inspect(src)["count"]
    assert not (tmp_path / "missing-parent").exists()
    assert engine.copy(src, dst)["copied"] > 0
    assert engine.verify(src, dst, top_k=1)["ok"]
    assert tree_state(migration_source.path) == before


@pytest.mark.parametrize("entry", [
    "copy-source", "copy-target", "dry-run-source", "dry-run-target",
    "verify-source", "verify-target", "inspect", "read-store", "optional-read-store",
    "check-directory", "store",
])
@pytest.mark.parametrize("origin", ["literal", "expanded-home"])
def test_unencodable_path_refused_before_snapshot_or_backend_access(
    migration_source, tmp_path, monkeypatch, entry, origin,
):
    src = migration_source
    dst = StoreURI("qdrant" if src.kind == "chroma" else "chroma", tmp_path / "target", "docs")
    invalid = tmp_path / "invalid\ud800path"
    if origin == "expanded-home":
        # The OS environment cannot carry high surrogates. Inject the Python
        # mapping to exercise validation of text contributed by expansion.
        monkeypatch.setattr(os, "environ", dict(os.environ, HOME=str(invalid)))
        invalid = Path("~/store")
    (tmp_path / "sentinel").write_text("must stay unchanged")
    before = tree_state(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Unencodable path reached snapshot/backend access")

    with monkeypatch.context() as guard:
        guard.setattr(stores, "TemporaryDirectory", forbidden)
        guard.setattr(stores.shutil, "copytree", forbidden)
        guard.setattr("chromadb.PersistentClient", forbidden)
        guard.setattr("qdrant_client.QdrantClient", forbidden)
        guard.setattr("ragwrench.migrate.safety.audit_qdrant", forbidden)
        with pytest.raises(ParityError, match="local path encodable by the filesystem"):
            if entry.endswith("target"):
                dst = replace(dst, path=invalid)
            else:
                src = StoreURI(src.kind, invalid, "docs")
            if entry.startswith(("copy-", "dry-run-")):
                engine.copy(src, dst, dry_run=entry.startswith("dry-run-"))
            elif entry.startswith("verify-"):
                engine.verify(src, dst)
            elif entry == "inspect":
                engine.inspect(src)
            elif entry.endswith("read-store"):
                with read_store(src, optional=entry.startswith("optional-")):
                    pytest.fail("Unencodable path accepted")
            elif entry == "check-directory":
                check_directory(src, required=False)
            else:
                with Store(src):
                    pytest.fail("Unencodable path accepted")
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize("text", ["invalid\ud800path", "bad\udc00parent/../store"])
def test_unencodable_path_text_rejected_before_filesystem_probe(monkeypatch, tmp_path, text):
    def forbidden(*args, **kwargs):
        pytest.fail("Unencodable path reached filesystem probe")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "lstat", forbidden)
        for path in (Path(text), tmp_path / text):
            with pytest.raises(ParityError, match="local path encodable by the filesystem"):
                checked_store_path(path)


def test_unencodable_expanded_home_rejected_before_filesystem_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", dict(os.environ, HOME=str(tmp_path / "invalid\ud800home")))

    def forbidden(*args, **kwargs):
        pytest.fail("Unencodable home reached filesystem probe")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "lstat", forbidden)
        with pytest.raises(ParityError, match="local path encodable by the filesystem"):
            checked_store_path(Path("~/store"))


@pytest.mark.parametrize("kind", ["chroma", "qdrant"])
@pytest.mark.parametrize("entry", ["inspect", "copy-source", "copy-target", "verify-source", "verify-target"])
def test_cli_unencodable_path_is_specific_input_error(tmp_path, capsys, kind, entry):
    before = tree_state(tmp_path)
    invalid = f"{kind}:{tmp_path}/invalid\ud800path#docs"
    valid = f"{'qdrant' if kind == 'chroma' else 'chroma'}:{tmp_path}/target#docs"
    command = entry.split("-")[0]
    args = [command, invalid] if command == "inspect" else (
        [command, valid, invalid] if entry.endswith("target") else [command, invalid, valid]
    )
    # OS argv cannot carry a high surrogate; call the Python CLI entry directly.
    assert main(args + ["--json"]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err) == {"error": "Store path must be a local path encodable by the filesystem"}
    assert tree_state(tmp_path) == before


def test_unicode_encode_error_after_path_validation_is_not_misclassified(monkeypatch, tmp_path):
    unrelated = UnicodeEncodeError("utf-8", "\ud800", 0, 1, "unrelated filesystem failure")

    def failed_lstat(*args, **kwargs):
        raise unrelated

    monkeypatch.setattr(Path, "lstat", failed_lstat)
    with pytest.raises(UnicodeEncodeError) as caught:
        checked_store_path(tmp_path)
    assert caught.value is unrelated


@pytest.mark.parametrize("flavor", ["unicode", "surrogateescape"])
def test_filesystem_encodable_source_paths_still_work(migration_source, tmp_path, flavor):
    name = "données-文書" if flavor == "unicode" else os.fsdecode(b"source-\xff")
    path = tmp_path / name
    assert os.fsdecode(os.fsencode(path)) == str(path)
    migration_source.path.rename(path)
    src = StoreURI.parse(f"{migration_source.kind}:{path}#docs")
    assert src.path == checked_store_path(path) == path
    dst = StoreURI("qdrant" if src.kind == "chroma" else "chroma", tmp_path / "新しい-parent" / "store", "docs")
    before = tree_state(path)

    count = engine.inspect(src)["count"]
    assert engine.copy(src, dst, dry_run=True)["would_copy"] == count
    assert not dst.path.parent.exists()
    assert engine.copy(src, dst)["copied"] == count
    assert engine.verify(src, dst, top_k=1)["ok"]
    assert tree_state(path) == before


@pytest.mark.parametrize("entry", [
    "copy-source", "copy-target", "dry-run-source", "dry-run-target",
    "verify-source", "verify-target", "inspect", "read-store", "optional-read-store",
    "check-directory", "store",
])
def test_non_path_input_refused_before_filesystem_or_backend_access(
    migration_source, tmp_path, monkeypatch, entry,
):
    src = migration_source
    dst = StoreURI("qdrant" if src.kind == "chroma" else "chroma", tmp_path / "target", "docs")
    (tmp_path / "sentinel").write_text("must stay unchanged")
    before = tree_state(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Non-Path input reached filesystem/snapshot/backend access")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "lstat", forbidden)
        guard.setattr(stores, "TemporaryDirectory", forbidden)
        guard.setattr(stores.shutil, "copytree", forbidden)
        guard.setattr("chromadb.PersistentClient", forbidden)
        guard.setattr("qdrant_client.QdrantClient", forbidden)
        guard.setattr("ragwrench.migrate.safety.audit_qdrant", forbidden)
        with pytest.raises(ParityError, match=r"^Store path must be a local pathlib\.Path object$"):
            # Every entry expression fails while building its URI, before access.
            if entry.endswith("target"):
                dst = replace(dst, path="./store")
            else:
                src = StoreURI(src.kind, "./store", "docs")
            if entry.startswith(("copy-", "dry-run-")):
                engine.copy(src, dst, dry_run=entry.startswith("dry-run-"))
            elif entry.startswith("verify-"):
                engine.verify(src, dst)
            elif entry == "inspect":
                engine.inspect(src)
            elif entry.endswith("read-store"):
                with read_store(src, optional=entry.startswith("optional-")):
                    pytest.fail("Non-Path input accepted")
            elif entry == "check-directory":
                check_directory(src, required=False)
            else:
                with Store(src):
                    pytest.fail("Non-Path input accepted")
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize("path", ["./store", b"./store", None])
def test_non_path_checker_input_refused_before_filesystem_probe(monkeypatch, path):
    def forbidden(*args, **kwargs):
        pytest.fail("Non-Path input reached filesystem probe")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "lstat", forbidden)
        with pytest.raises(ParityError, match=r"^Store path must be a local pathlib\.Path object$"):
            checked_store_path(path)


def test_path_subclass_remains_supported(migration_source, tmp_path, monkeypatch):
    class LocalPath(type(tmp_path)):
        pass

    monkeypatch.chdir(tmp_path)
    src = replace(migration_source, path=LocalPath(migration_source.path.name))
    dst = StoreURI("qdrant" if src.kind == "chroma" else "chroma", LocalPath("missing-parent/store"), "docs")
    assert isinstance(src.path, LocalPath) and isinstance(dst.path, LocalPath)
    before = tree_state(migration_source.path)
    count = engine.inspect(src)["count"]
    with read_store(src) as reader:
        assert reader.schema().count == count
    assert engine.copy(src, dst, dry_run=True)["would_copy"] == count
    assert not dst.path.parent.exists()
    assert engine.copy(src, dst)["copied"] == count
    assert engine.verify(src, dst, top_k=1)["ok"]
    with Store(dst) as store:
        assert store.schema().count == count
    assert tree_state(migration_source.path) == before
