"""Preflight qdrant-client 1.16.2 stores before its pickle-based loader runs.

This adapter accepts only the PointStruct serialization emitted by this tool.
It is a single-writer preflight, not protection against concurrent filesystem
replacement. We do not support legacy dbm migration or arbitrary Qdrant stores.
"""
import io
import pickle
import pickletools
from pathlib import Path
import sqlite3

from qdrant_client import models

from .core import MAX_CHUNKS, MAX_JSON_BYTES, SyncError, canonical, load_json
from .embedders import checked_vector


class _PointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("qdrant_client.http.models.models", "PointStruct"):
            return models.PointStruct
        raise SyncError("unsafe Qdrant pickle global refused")


def check_point_blob(blob, dimension=None):
    if not isinstance(blob, bytes) or len(blob) > MAX_JSON_BYTES:
        raise SyncError("unsupported Qdrant point blob size/type")
    try:
        for opcode, _, _ in pickletools.genops(blob):
            if opcode.name in {"REDUCE", "INST", "OBJ", "NEWOBJ_EX", "EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID"}:
                raise SyncError("executable Qdrant pickle opcode refused")
        point = _PointUnpickler(io.BytesIO(blob)).load()
        if type(point) is not models.PointStruct:
            raise SyncError("unsupported Qdrant point serialization")
        if dimension is not None:
            checked_vector(point.vector, dimension)
            if not isinstance(point.payload, dict):
                raise SyncError("unsupported Qdrant payload")
            canonical(point.payload)
    except (pickle.UnpicklingError, EOFError, ValueError, TypeError, AttributeError, OverflowError) as exc:
        if isinstance(exc, SyncError):
            raise
        raise SyncError("invalid Qdrant point serialization") from exc


def collection_config(raw_config):
    """Validate supported catalog configuration without touching point storage."""
    try:
        config = models.CreateCollection.model_validate(raw_config)
    except ValueError as exc:
        raise SyncError("invalid Qdrant collection configuration") from exc
    vector = config.vectors
    if (not isinstance(vector, models.VectorParams) or not 1 <= vector.size <= 4096
            or vector.distance != models.Distance.DOT or vector.multivector_config is not None
            or config.sparse_vectors):
        raise SyncError("unsupported Qdrant vector configuration")
    return config


def preflight(root, validate_name, *, metadata=None):
    root = Path(root)
    meta_file = root / "meta.json"
    if metadata is None and not meta_file.exists():
        # Never allow the backend to discover/migrate untracked legacy data.
        if root.exists() and any(root.iterdir()):
            raise SyncError("nonempty Qdrant directory has no meta.json")
        return
    meta = load_json(meta_file) if metadata is None else metadata
    if not isinstance(meta, dict) or not isinstance(meta.get("collections"), dict) or meta.get("aliases") != {}:
        raise SyncError("unsupported Qdrant metadata or aliases")
    total, count = 0, 0
    for name, raw_config in meta["collections"].items():
        validate_name(name)
        vector = collection_config(raw_config).vectors
        database = root / "collection" / name / "storage.sqlite"
        if not database.is_file():
            raise SyncError("missing SQLite store; legacy Qdrant migration is unsupported")
        try:
            # Fixed SQL, read-only file URI, extensions disabled by default.
            con = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
            try:
                for (blob,) in con.execute("SELECT point FROM points"):
                    count += 1
                    total += len(blob) if isinstance(blob, bytes) else MAX_JSON_BYTES + 1
                    if count > MAX_CHUNKS or total > MAX_JSON_BYTES * 4:
                        raise SyncError("Qdrant preflight size/count limit exceeded")
                    check_point_blob(blob, vector.size)
            finally:
                con.close()
        except sqlite3.Error as exc:
            raise SyncError("invalid Qdrant SQLite store") from exc
