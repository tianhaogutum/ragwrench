"""Versioned, atomically replaced local migration progress (no record contents)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .model import ParityError, Record, StoreURI, checked_store_path, dumps


def _validate_path_text(path: Path) -> None:
    if "\x00" in str(path):
        raise ParityError("Checkpoint path must be a local path without NUL bytes")
    try:
        # Check the filename too, using the OS codec's surrogateescape support.
        os.fsencode(path)
    except UnicodeEncodeError as exc:
        raise ParityError("Checkpoint path must be a local path encodable by the filesystem") from exc


def checkpoint_path(value: Path | str, source: StoreURI, target: StoreURI, *, resume: bool) -> Path:
    # Enforce the Python API contract before Path can invoke conversion hooks.
    if not isinstance(value, (str, Path)):
        raise ParityError("Checkpoint path must be a local str or pathlib.Path object")
    raw = Path(value)
    _validate_path_text(raw)
    try:
        raw = raw.expanduser()
    except RuntimeError as exc:
        raise ParityError("Checkpoint path must be a local path with a resolvable home directory") from exc
    raw = raw.absolute()
    # HOME and the working directory can contribute text absent from the input.
    _validate_path_text(raw)
    path = checked_store_path(raw.parent) / raw.name
    for store in (source, target):
        root = checked_store_path(store.path)
        if path == root or root in path.parents:
            raise ParityError("Checkpoint must be outside both store directories")
    if not path.parent.is_dir():
        raise ParityError("Checkpoint parent directory must already exist")
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        if resume:
            raise ParityError("Checkpoint does not exist; start without --resume") from None
    else:
        if not stat.S_ISREG(mode):
            raise ParityError("Checkpoint must be a regular file, not a symlink or special file")
        if not resume:
            raise ParityError("Checkpoint already exists; use --resume or choose a new checkpoint")
    return path


def uri_identity(uri: StoreURI) -> dict:
    return {"kind": uri.kind, "path": str(checked_store_path(uri.path)), "collection": uri.collection}


def record_hash(record: Record) -> str:
    return hashlib.sha256(dumps(asdict(record)).encode("utf-8")).hexdigest()


def load(path: Path) -> dict[str, Any]:
    def finite_number(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite checkpoint number")
        return number

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        # Reject constants and float overflow at every depth before using state.
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                          parse_constant=finite_number, parse_float=finite_number)
        if (type(data) is not dict or set(data) != {"version", "manifest", "completed_ids", "parity"}
                or type(data["version"]) is not int or data["version"] != 1
                or type(data["manifest"]) is not dict or type(data["completed_ids"]) is not list
                or type(data["manifest"].get("source")) is not dict
                or type(data["manifest"].get("target")) is not dict
                or type(data["manifest"].get("schema")) is not dict
                or type(data["manifest"].get("records")) is not dict
                or not all(type(rid) is str for rid in data["completed_ids"])
                or len(set(data["completed_ids"])) != len(data["completed_ids"])
                or (data["parity"] is not None and type(data["parity"]) is not dict)):
            raise ValueError()
        return data
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ParityError("Invalid or unsupported checkpoint") from exc


def save(path: Path, data: dict[str, Any]) -> None:
    """A failed replace leaves the previous acknowledged ID set intact."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(dumps(data) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp.unlink(missing_ok=True)
