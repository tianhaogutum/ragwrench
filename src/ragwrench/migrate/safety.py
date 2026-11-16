"""Audit Qdrant's local pickle records before its backend can deserialize them.

No third-party source is copied here. The allowlist is intentionally narrow:
modern dense PointStruct state and primitive data, with no callable reductions.
"""
from __future__ import annotations

import io
import json
import pickle
import pickletools
import sqlite3
from contextlib import closing
from pathlib import Path

from .model import ParityError, StoreURI, json_value, valid_qdrant_id

MAX_PICKLE_BYTES = 64 * 1024 * 1024


class _InertPoint:
    """Never invokes Pydantic or any user-defined serialization hooks."""


class _PointReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("qdrant_client.http.models.models", "PointStruct"):
            return _InertPoint
        raise ParityError("Unsafe or unsupported Qdrant pickle class; only dense PointStruct is accepted")

    def persistent_load(self, pid):
        raise ParityError("Persistent references in Qdrant pickle are not supported")


def audit_pickle(blob: bytes) -> dict:
    if not isinstance(blob, bytes) or len(blob) > MAX_PICKLE_BYTES:
        raise ParityError("Qdrant point pickle exceeds the 64 MiB per-point limit or is not a blob")
    try:
        for opcode, _, _ in pickletools.genops(blob):
            if opcode.name in {"REDUCE", "INST", "OBJ", "NEWOBJ_EX", "EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID"}:
                raise ParityError("Unsafe or unsupported callable opcode in Qdrant pickle")
        obj = _PointReader(io.BytesIO(blob)).load()
        if type(obj) is not _InertPoint:
            raise ParityError("Qdrant point must be a serialized PointStruct")
        state = vars(obj)
        allowed = {"__dict__", "__pydantic_extra__", "__pydantic_fields_set__", "__pydantic_private__", "__fields_set__"}
        if (set(state) - allowed or state.get("__pydantic_extra__") is not None
                or state.get("__pydantic_private__") is not None):
            raise ParityError("Unsupported Qdrant PointStruct pickle state")
        data = state.get("__dict__")
        if type(data) is not dict or set(data) != {"id", "vector", "payload"}:
            raise ParityError("Unsupported Qdrant PointStruct fields")
        # No nested instances, tuples, functions or custom state may reach the real loader.
        json_value(data, depth=-4)
        field_set = state.get("__pydantic_fields_set__", state.get("__fields_set__", set()))
        if type(field_set) is not set or not field_set.issubset(data):
            raise ParityError("Unsupported Qdrant PointStruct field-set state")
        return data
    except ParityError:
        raise
    except (ValueError, TypeError, AttributeError, EOFError, pickle.UnpicklingError, RecursionError) as exc:
        raise ParityError("Malformed or unsupported Qdrant local point pickle") from exc


def audit_qdrant(path: Path) -> dict:
    try:
        meta = json.loads((path / "meta.json").read_text())
        if (type(meta) is not dict or type(meta.get("collections")) is not dict
                or type(meta.get("aliases")) is not dict):
            raise ParityError("Invalid Qdrant meta.json")
        for name in meta["collections"]:
            # Validate every loaded collection, not just the collection selected by the URI.
            StoreURI.parse(f"qdrant:{path}#{name}")
            db = path / "collection" / name / "storage.sqlite"
            if not db.exists():
                if list(db.parent.glob("storage.dbm*")):
                    raise ParityError("Legacy Qdrant dbm storage is not supported")
                # Empty, newly created collections need not have a persisted database yet.
                continue
            if any(db.with_name(db.name + suffix).exists() for suffix in ("-wal", "-journal")):
                raise ParityError("Qdrant SQLite WAL/journal sidecars are not supported; close and checkpoint the store first")
            with closing(sqlite3.connect(db.as_uri() + "?mode=ro&immutable=1", uri=True)) as connection:
                connection.execute("PRAGMA trusted_schema=OFF")
                seen = set()
                for (blob,) in connection.execute("SELECT point FROM points"):
                    data = audit_pickle(blob)
                    qid = valid_qdrant_id(data["id"])
                    if qid in seen:
                        raise ParityError("Duplicate physical Qdrant IDs in SQLite storage; refusing automatic deduplication")
                    seen.add(qid)
        return meta
    except (json.JSONDecodeError, sqlite3.Error, UnicodeError) as exc:
        raise ParityError("Invalid Qdrant local store metadata or SQLite point storage") from exc
