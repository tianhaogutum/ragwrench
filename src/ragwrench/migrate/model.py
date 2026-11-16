"""Canonical records, reversible IDs and a versioned metadata codec."""
from __future__ import annotations

import json
import math
import os
import re
import stat
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


class ParityError(ValueError):
    """An unsupported input or unsafe operation; displayed without a traceback."""


# Fixed forever: changing this value would change persistent point IDs.
ID_NAMESPACE = uuid.UUID("9b33df64-cd30-52b9-9328-3d8ab7287f76")
ENVELOPE = "__vecparity__"
RESERVED = {ENVELOPE, "__id__", "document"}
DISTANCES = {"l2": "Euclid", "cosine": "Cosine", "ip": "Dot"}


def _validate_path_text(path: Path) -> None:
    # Direct Python callers must supply a concrete filesystem Path, as annotated.
    if not isinstance(path, Path):
        raise ParityError("Store path must be a local pathlib.Path object")
    # Path accepts NUL but filesystem operations raise a raw ValueError for it.
    if "\x00" in str(path):
        raise ParityError("Store path must be a local path without NUL bytes")
    try:
        # Use the OS codec/error handler: Linux surrogateescape names are valid.
        os.fsencode(path)
    except UnicodeEncodeError as exc:
        raise ParityError("Store path must be a local path encodable by the filesystem") from exc


def checked_store_path(path: Path) -> Path:
    """Reject links before canonicalization can hide them, even above a missing store."""
    _validate_path_text(path)
    try:
        path = path.expanduser()
    except RuntimeError as exc:
        raise ParityError("Store path must be a local path with a resolvable home directory") from exc
    path = path.absolute()
    # HOME and the working directory can contribute text absent from the input.
    _validate_path_text(path)
    # Keep '..' until after checking: link/../store still traverses a symlink.
    # lstat also detects dangling links, which exists() would treat as absent.
    for entry in (*reversed(path.parents), path):
        try:
            mode = entry.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ParityError("Store path or ancestor is a symlink; refusing to follow it")
        if not stat.S_ISDIR(mode):
            raise ParityError("Store path and its existing ancestors must be directories")
    return path.resolve()


@dataclass(frozen=True)
class StoreURI:
    kind: str
    path: Path
    collection: str

    def __post_init__(self) -> None:
        # Validate direct Python construction and dataclasses.replace as well as parse().
        if not isinstance(self.kind, str) or self.kind not in {"chroma", "qdrant"}:
            raise ParityError("Store backend must be chroma or qdrant")
        # Qdrant local uses collection names in filesystem paths. Do not allow traversal.
        if not isinstance(self.collection, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{1,126}[A-Za-z0-9]", self.collection
        ):
            raise ParityError("Collection name must be 3–128 ASCII letters/digits/._-, with alphanumeric ends")
        if ".." in self.collection:
            raise ParityError("Collection names containing '..' are not supported")
        _validate_path_text(self.path)

    @classmethod
    def parse(cls, value: str) -> StoreURI:
        kind, sep, rest = value.partition(":")
        path, marker, name = rest.partition("#")
        if kind not in {"chroma", "qdrant"} or not sep or not marker or not path:
            raise ParityError("URI must be chroma:PATH#collection or qdrant:PATH#collection")
        if "://" in path or "\x00" in path or "#" in name:
            raise ParityError("Only local paths are supported; URI escaping is not supported")
        uri = cls(kind, Path(path), name)
        return replace(uri, path=checked_store_path(uri.path))


@dataclass
class Record:
    id: str
    vector: list[float]
    metadata: dict[str, Any] | None
    document: str | None = None
    qdrant_id: int | str | None = None


@dataclass
class Schema:
    dimension: int | None
    distance: str
    count: int


def physical_id(value: str) -> int | str:
    if not isinstance(value, str) or not value:
        raise ParityError("IDs must be nonempty strings")
    # Only canonical decimal strings can be kept without ambiguous integer aliases.
    if len(value) <= 20 and re.fullmatch(r"0|[1-9][0-9]*", value) and int(value) <= 2**64 - 1:
        return int(value)
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return str(uuid.uuid5(ID_NAMESPACE, value))


def valid_qdrant_id(value: Any) -> int | str:
    if type(value) is int and 0 <= value <= 2**64 - 1:
        return value
    if isinstance(value, str):
        try:
            return str(uuid.UUID(value))
        except ValueError:
            pass
    raise ParityError("Invalid stored Qdrant ID (expected UUID or unsigned 64-bit integer)")


def json_value(value: Any, depth: int = 0) -> None:
    if depth > 32:
        raise ParityError("Metadata nesting exceeds 32 levels")
    if type(value) is str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ParityError("Strings must be valid Unicode without unpaired surrogates") from exc
        return
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for v in value:
            json_value(v, depth + 1)
        return
    if type(value) is dict and all(isinstance(k, str) for k in value):
        for k, v in value.items():
            json_value(k, depth + 1)
            json_value(v, depth + 1)
        return
    raise ParityError("Metadata must contain only finite JSON values and string object keys")


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_record(record: Record, dimension: int | None, distance: str) -> None:
    if not isinstance(record.id, str) or not record.id:
        raise ParityError("IDs must be nonempty strings")
    if not record.vector or (dimension is not None and len(record.vector) != dimension):
        raise ParityError("Vector dimension mismatch or empty vector")
    if not all(type(v) in {int, float} and math.isfinite(v) for v in record.vector):
        raise ParityError("Vectors must contain finite numeric values")
    if distance == "cosine" and not any(record.vector):
        raise ParityError("Zero vectors have undefined cosine similarity")
    if record.metadata is not None and type(record.metadata) is not dict:
        raise ParityError("Metadata must be a JSON object or null")
    json_value(record.metadata)
    json_value(record.id)
    json_value(record.document)
    if record.document is not None and not isinstance(record.document, str):
        raise ParityError("Documents must be strings or null")
    if record.qdrant_id is not None:
        valid_qdrant_id(record.qdrant_id)


def encode_metadata(record: Record, kind: str) -> dict[str, Any]:
    """Record key renames and JSON encodings instead of guessing on restore."""
    result: dict[str, Any] = {}
    keys: dict[str, str] = {}
    encoded: list[str] = []
    metadata = record.metadata or {}
    for key, value in metadata.items():
        stored = key
        if kind == "chroma" or key in RESERVED:
            stored = "vp_m_" + key.encode("utf-8").hex()
            while stored in metadata or stored in result or stored in RESERVED:
                stored = "_" + stored
        keys[stored] = key
        if kind == "chroma" and not (
            type(value) in {str, bool, float} or (type(value) is int and -(2**63) <= value < 2**63)
        ):
            value = dumps(value)
            encoded.append(stored)
        result[stored] = value
    envelope = {"version": 1, "keys": keys, "json": encoded, "metadata_none": record.metadata is None,
                "qdrant_id": record.qdrant_id}
    result[ENVELOPE] = dumps(envelope) if kind == "chroma" else envelope
    if kind == "qdrant":
        result["__id__"] = record.id
        if record.document is not None:
            result["document"] = record.document
    return result


def decode_metadata(raw: dict[str, Any] | None, kind: str) -> tuple[dict[str, Any] | None, int | str | None]:
    if raw is None:
        return None, None
    if ENVELOPE not in raw:
        return {k: v for k, v in raw.items() if kind != "qdrant" or k not in RESERVED}, None
    try:
        env = json.loads(raw[ENVELOPE]) if kind == "chroma" else raw[ENVELOPE]
        if (not isinstance(env, dict) or type(env.get("version")) is not int or env.get("version") != 1
                or set(env) != {"version", "keys", "json", "metadata_none", "qdrant_id"}
                or type(env.get("keys")) is not dict or type(env.get("json")) is not list
                or type(env.get("metadata_none")) is not bool):
            raise ValueError()
        keys = env["keys"]
        if (not all(isinstance(k, str) and isinstance(v, str) for k, v in keys.items())
                or len(set(keys.values())) != len(keys) or not set(env["json"]).issubset(keys)
                or set(keys) & RESERVED or (env["metadata_none"] and keys)):
            raise ValueError()
        allowed = set(keys) | ({ENVELOPE} if kind == "chroma" else RESERVED)
        if set(raw) - allowed:
            raise ValueError()
        metadata = {original: json.loads(raw[k]) if k in env["json"] else raw[k]
                    for k, original in keys.items()}
        qid = env.get("qdrant_id")
        if qid is not None:
            qid = valid_qdrant_id(qid)
        json_value(metadata)
        return (None if env["metadata_none"] else metadata), qid
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        raise ParityError("Invalid __vecparity__ mapping envelope; reserved key or corrupt mapping") from exc


def normalized(vector: list[float]) -> list[float]:
    norm = math.hypot(*vector)
    return [x / norm for x in vector] if norm else list(vector)
