import json
import pickle
import sqlite3

import pytest
from qdrant_client import models

from ragwrench.sync.core import SyncError
from ragwrench.sync.safe_store import check_point_blob
from ragwrench.sync.store import local_client


def test_point_pickle_data_only():
    safe = models.PointStruct(id=1, vector=[0.5, 0.1], payload={"nested": [1, None, {"safe": True}]})
    check_point_blob(pickle.dumps(safe))
    with pytest.raises(SyncError, match="serialization"):
        check_point_blob(b"not pickle")


def test_executable_pickle_refused_before_backend_opens(tmp_path):
    side_effect = tmp_path / "must-not-execute"

    class Malicious:
        def __reduce__(self):
            return eval, (f"open({str(side_effect)!r}, 'w').write('bad')",)

    db_root = tmp_path / "qd"
    collection = db_root / "collection" / "kb"
    collection.mkdir(parents=True)
    (db_root / "meta.json").write_text(json.dumps({"collections": {"kb": {"vectors": {"size": 2, "distance": "Dot"}}}, "aliases": {}}))
    with sqlite3.connect(collection / "storage.sqlite") as db:
        db.execute("CREATE TABLE points (id TEXT PRIMARY KEY, point BLOB)")
        db.execute("INSERT INTO points VALUES (?, ?)", ("1", pickle.dumps(Malicious())))
    with pytest.raises(SyncError, match="pickle"):
        with local_client(str(db_root)):
            pytest.fail("unsafe database opened")
    assert not side_effect.exists()
    assert not (db_root / ".lock").exists()


def test_existing_collection_metadata_path_traversal(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"collections": {"../../escape": {}}, "aliases": {}}))
    with pytest.raises(SyncError, match="collection must"):
        with local_client(str(tmp_path)):
            pass
    assert not (tmp_path / ".lock").exists()


def test_legacy_and_alias_stores_refused(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"collections": {"kb": {"vectors": {"size": 2, "distance": "Dot"}}}, "aliases": {}}))
    with pytest.raises(SyncError, match="legacy"):
        with local_client(str(tmp_path)):
            pass
    (tmp_path / "meta.json").write_text(json.dumps({"collections": {}, "aliases": {"alias": "kb"}}))
    with pytest.raises(SyncError, match="aliases"):
        with local_client(str(tmp_path)):
            pass


@pytest.mark.parametrize("config", [{"vectors": {"size": 999999999, "distance": "Dot"}}, {"vectors": "invalid"}])
def test_unsafe_vector_config_refused_before_loading(tmp_path, config):
    (tmp_path / "meta.json").write_text(json.dumps({"collections": {"kb": config}, "aliases": {}}))
    with pytest.raises(SyncError, match="configuration"):
        with local_client(str(tmp_path)):
            pass
    assert not (tmp_path / ".lock").exists()
