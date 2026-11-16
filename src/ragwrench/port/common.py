from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


class PortError(Exception):
    """An actionable error safe to show without upstream response bodies."""


class TransientAPIError(PortError):
    """Transport/429/5xx failure; a mutation may already have committed."""

    def __init__(self, message, *, retry_after=0):
        super().__init__(message)
        self.retry_after = retry_after


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def content_hash(text):
    return digest(text.encode("utf-8"))


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise PortError("API returned an unsupported identifier (expected 1..128 safe ASCII characters)")
    return value


def safe_file(path):
    path = Path(path)
    if path.is_symlink():
        raise PortError(f"Refusing symlink: {path.name}")
    return path


def read_json(path, limit=256 * 1024 * 1024):
    return read_json_snapshot(path, limit)[0]


def read_json_snapshot(path, limit=256 * 1024 * 1024):
    """Parse and retain the exact bytes from one file read, including whitespace."""
    path = safe_file(path)
    if path.stat().st_size > limit:
        raise PortError(f"JSON file exceeds {limit} byte safety limit")
    try:
        data = path.read_bytes()
        value = json.loads(data.decode("utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        return value, data
    except (ValueError, UnicodeError) as exc:
        raise PortError(f"Invalid JSON in {path.name}") from exc


def atomic_bytes(path, data):
    path = safe_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".kbport-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, value):
    blob = (dumps(value) + "\n").encode()
    if len(blob) > 256 * 1024 * 1024:
        raise PortError("JSON state exceeds 256 MiB safety limit")
    atomic_bytes(path, blob)


@contextmanager
def locked(path):
    path = safe_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PortError("Another ragwrench port process holds this state/output lock") from exc
        yield
    finally:
        os.close(fd)
