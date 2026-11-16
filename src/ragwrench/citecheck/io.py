"""Bounded local JSON I/O. Export fields are data, never paths or programs."""
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

from .model import InputError, MAX_INPUT_BYTES


MAX_RECORDS = 1000


def read_bytes(path: str) -> bytes:
    if path == "-":
        data = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    else:
        # Refuse FIFOs/devices instead of hanging on a local 'file'.
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK), "rb", buffering=0) as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise InputError("input must be a regular file or '-' for stdin")
            data = stream.read(MAX_INPUT_BYTES + 1)
    if len(data) > MAX_INPUT_BYTES:
        raise InputError(f"input exceeds {MAX_INPUT_BYTES} bytes")
    return data


def decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise InputError("input must be UTF-8") from None


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError("duplicate JSON object key; ambiguous payload")
        result[key] = value
    return result


def _constant(value):
    raise InputError("non-finite JSON numbers are unsupported")


def loads(text: str):
    # A fixed depth bound is independent of Python's JSON implementation and
    # recursion settings. Brackets inside strings do not affect nesting.
    depth = 0
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            if depth > 64:
                raise InputError("JSON nesting exceeds 64 levels")
        elif char in "]}":
            depth -= 1
    try:
        return json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
    except json.JSONDecodeError as exc:
        raise InputError(f"invalid JSON at line {exc.lineno}, column {exc.colno}") from None
    except (RecursionError, ValueError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError("JSON nesting or numeric value exceeds parser limits") from None


def read_payloads(path: str, jsonl: bool):
    text = decode(read_bytes(path))
    if not jsonl:
        return [loads(text)]
    lines = [(n, line) for n, line in enumerate(text.splitlines(), 1) if line.strip()]
    if not lines:
        raise InputError("JSONL input must contain at least one record")
    if len(lines) > MAX_RECORDS:
        raise InputError(f"JSONL supports at most {MAX_RECORDS} records")
    result = []
    for n, line in lines:
        try:
            result.append(loads(line))
        except InputError as exc:
            raise InputError(f"JSONL line {n}: {exc}") from None
    return result


def write_output(path: str, content: str, *, force: bool = False, inputs=()):
    """Write only below cwd, atomically; no symlinks, input clobber or implicit dirs."""
    raw = Path(path)
    target = raw.absolute()
    root = Path.cwd().resolve()
    resolved = target.resolve()
    if not resolved.is_relative_to(root):
        raise InputError("output must stay within the current directory")
    # Check each existing path component. This is a local CLI, not a multi-user
    # hostile-filesystem sandbox; callers must control their output directory.
    for part in (target, *target.parents):
        if part == root:
            break
        if part.is_symlink():
            raise InputError("output path must not contain symlinks")
    for source in inputs:
        if source != "-" and Path(source).resolve() == resolved:
            raise InputError("output must not overwrite an input file")
    if not target.parent.is_dir():
        raise InputError("output parent directory does not exist; create it explicitly")
    if target.exists() and not force:
        raise InputError("output already exists; use --force to replace a report")
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as stream:
            temp_name = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if force:
            os.replace(temp_name, target)
        else:
            # link is atomic and refuses an existing target, unlike replace.
            os.link(temp_name, target)
            os.unlink(temp_name)
        temp_name = None
    finally:
        if temp_name is not None:
            os.unlink(temp_name)
