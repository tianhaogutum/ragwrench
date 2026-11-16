"""Strict, data-only input adapters. No framework deserialization or imports."""
import json
import math
import os
import re
from pathlib import Path
import tempfile

from .model import Chunk, Config, InputError

MAX_BYTES = 32 * 1024 * 1024
MAX_CHUNKS = 10000


def read_bytes(path):
    path = Path(path)
    if not path.is_file():
        raise InputError(f"{path}: input must be a regular file")
    if path.stat().st_size > MAX_BYTES:
        raise InputError(f"{path}: exceeds {MAX_BYTES} byte input limit")
    # Bounded binary read preserves CRLF/CR and guards growing files and special devices.
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise InputError(f"{path}: exceeds {MAX_BYTES} byte input limit")
    return raw


def read_text(path):
    return read_bytes(path).decode("utf-8-sig")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _constant(value):
    raise InputError(f"invalid JSON constant {value}")


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise InputError("JSON number exceeds finite floating-point range")
    return number


def _decode(raw):
    value = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant, parse_float=_float)
    # Escaped lone surrogates are accepted by Python's JSON parser but cannot be
    # written as UTF-8. Reject them at the input line, including metadata and keys.
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            item.encode("utf-8")
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return value


def load_chunks(path, adapter="auto"):
    return parse_chunks(read_text(path), adapter, str(path))


def parse_chunks(raw, adapter="auto", label="chunks"):
    """Parse an already-read snapshot, so hashing and adapting see identical bytes."""
    path = label
    records = []
    if raw.lstrip().startswith("["):
        try:
            data = _decode(raw)
        except (ValueError, RecursionError) as exc:
            raise InputError(f"{path}: invalid JSON array: {exc}") from exc
        records = [(i + 1, item) for i, item in enumerate(data)]
    else:
        for line, value in enumerate(re.split(r"\r\n|\r|\n", raw), 1):
            if not value.strip():
                continue
            try:
                records.append((line, _decode(value)))
            except (ValueError, RecursionError) as exc:
                raise InputError(f"{path}: line {line}: invalid JSON: {exc}") from exc
    if not records:
        raise InputError(f"{path}: empty chunk set")
    if len(records) > MAX_CHUNKS:
        raise InputError(f"{path}: exceeds {MAX_CHUNKS} chunk limit")
    chunks, seen = [], set()
    for line, item in records:
        try:
            if not isinstance(item, dict):
                raise InputError("chunk must be an object")
            selected = adapter
            if selected == "auto":
                selected = "langchain" if "page_content" in item or "kwargs" in item else "generic"
            body = item
            if selected == "langchain" and "kwargs" in item:
                if item.get("type") != "constructor" or not isinstance(item["kwargs"], dict):
                    raise InputError("unsupported LangChain envelope (expected constructor with kwargs)")
                body = item["kwargs"]
            text = body.get("page_content") if selected == "langchain" else body.get("text")
            if not isinstance(text, str) or not text.strip():
                raise InputError("chunk text must be a non-empty string")
            cid = body.get("id", body.get("id_", f"c{len(chunks) + 1}"))
            if selected == "langchain" and cid is None:
                cid = f"c{len(chunks) + 1}"
            if not isinstance(cid, (str, int)) or isinstance(cid, bool) or str(cid) == "":
                raise InputError("id must be a non-empty string or integer")
            cid = str(cid)
            if cid in seen:
                raise InputError(f"duplicate chunk id {cid!r}")
            metadata = body.get("metadata", {})
            if not isinstance(metadata, dict):
                raise InputError("metadata must be an object")
            start, end = body.get("start_char_idx"), body.get("end_char_idx")
            if (start is None) != (end is None):
                raise InputError("start_char_idx and end_char_idx must be supplied together")
            if start is not None and (type(start) is not int or type(end) is not int or not 0 <= start < end):
                raise InputError("character hints must be integers with 0 <= start < end")
            seen.add(cid)
            chunks.append(Chunk(cid, text, metadata, start, end))
        except InputError as exc:
            raise InputError(f"{path}: line/item {line}: {exc}") from exc
    return chunks


def load_config(path=None):
    try:
        import tomllib
    except ImportError:  # Python 3.10
        import tomli as tomllib
    explicit = path is not None
    path = Path(path) if explicit else Path("chunklint.toml")
    if not path.exists():
        if explicit:
            raise InputError(f"config does not exist: {path}")
        return Config()
    try:
        data = tomllib.loads(read_text(path))
    except ValueError as exc:
        raise InputError(f"{path}: invalid TOML: {exc}") from exc
    if set(data) != {"chunklint"} or not isinstance(data["chunklint"], dict):
        raise InputError("config must contain only a [chunklint] table")
    unknown = set(data["chunklint"]) - set(Config.__dataclass_fields__)
    if unknown:
        raise InputError(f"unknown config keys: {', '.join(sorted(unknown))}")
    return Config(**data["chunklint"]).validate()


def atomic_write(path, text):
    """Write only the explicit file; no archive extraction, shell, or path interpolation."""
    path = Path(path)
    if path.is_symlink():
        raise InputError(f"refusing symlink output: {path}")
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temp = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
