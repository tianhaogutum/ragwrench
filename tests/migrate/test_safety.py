from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path

import pytest

from ragwrench.migrate import engine
from ragwrench.migrate.model import ParityError, StoreURI
from ragwrench.migrate.safety import audit_pickle


class _Trap:
    def __init__(self, sentinel):
        self.sentinel = sentinel

    def __reduce__(self):
        # If accidentally deserialized by the backend, this would create the sentinel.
        return (Path.touch, (self.sentinel,))


def test_pickle_hook_is_never_executed(qdrant_store, tmp_path):
    src = qdrant_store(points=[{"id": 1, "vector": [1., 2., 3., 4.], "payload": {}}])
    sentinel = tmp_path / "must-not-exist"
    malicious = pickle.dumps(_Trap(sentinel))
    db = src.path / "collection" / "docs" / "storage.sqlite"
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE points SET point = ?", (malicious,))
    dst = StoreURI("chroma", tmp_path / "not-created", "docs")
    with pytest.raises(ParityError, match="Unsafe|unsupported"):
        engine.copy(src, dst)
    assert not sentinel.exists()
    assert not dst.path.exists()


def test_meta_collection_traversal_refused(qdrant_store, tmp_path):
    src = qdrant_store()
    meta_path = src.path / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["collections"][str(tmp_path / "outside")] = meta["collections"]["docs"]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ParityError, match="Collection name"):
        engine.inspect(src)
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("blob", [b"not pickle", pickle.dumps({"plain": "object"}), pickle.dumps(Path("/tmp/unsupported"))])
def test_malformed_or_other_pickle_refused(blob):
    with pytest.raises(ParityError):
        audit_pickle(blob)


def test_duplicate_physical_points_refused(qdrant_store):
    src = qdrant_store(points=[{"id": 1, "vector": [1., 2., 3., 4.], "payload": {}}])
    db = src.path / "collection" / "docs" / "storage.sqlite"
    with sqlite3.connect(db) as connection:
        blob = connection.execute("SELECT point FROM points").fetchone()[0]
        connection.execute("INSERT INTO points VALUES (?, ?)", ("different-key", blob))
    with pytest.raises(ParityError, match="Duplicate physical"):
        engine.inspect(src)


def test_sqlite_sidecar_is_refused(qdrant_store):
    src = qdrant_store()
    db = src.path / "collection" / "docs" / "storage.sqlite"
    db.with_name(db.name + "-wal").touch()
    with pytest.raises(ParityError, match="WAL/journal"):
        engine.inspect(src)


def test_alias_is_refused(qdrant_store):
    from dataclasses import replace
    src = qdrant_store()
    meta_path = src.path / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["aliases"]["alias"] = "docs"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ParityError, match="aliases"):
        engine.inspect(replace(src, collection="alias"))
