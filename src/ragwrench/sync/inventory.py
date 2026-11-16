"""Bounded, metadata-only policy inventory for offline local stores.

Never open a Qdrant client or SQLite. Recovery here is an in-memory catalog
selection, not permission to repair disk metadata without point preflight.
"""
import json
import os
from pathlib import Path
import tempfile

from .core import MAX_JSON_BYTES, SyncError, load_json
from .recovery import SIDECAR, recovery_catalog
from .safe_store import collection_config
from .store import policy_inspection, validate_collection_name

MAX_COLLECTIONS = 1000


def archive_path(path, store):
    """Validate an external archive before reading the inventory; never mkdir.

    Like catalog reads, this assumes no concurrent filesystem replacement.
    Reject even external symlinks rather than publishing through an alias.
    """
    if "\0" in str(path):
        raise SyncError("policy inventory output path contains a null byte")
    target = Path(path).absolute()
    for entry in (target, *target.parents):
        if entry.is_symlink():
            raise SyncError("policy inventory output path may not traverse symlinks")
    target = target.resolve()
    if target.is_relative_to(Path(store).resolve()):
        raise SyncError("policy inventory output must be outside the Qdrant store")
    if not target.parent.is_dir():
        raise SyncError("policy inventory output requires an existing parent directory")
    if target.exists() and not target.is_file():
        raise SyncError("policy inventory output must be a regular file")
    return target


def archive_inventory(target, encoded):
    """Publish exact UTF-8 report bytes by a same-directory atomic replace.

    All writing/syncing/closing precedes replace, the final fallible filesystem
    operation. No directory fsync: atomic visibility, not power-loss durability.
    """
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent,
                                         prefix=".chunksync-inventory-", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass  # Preserve the publication error if cleanup also fails.
        raise


def inventory_json(report):
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n"
    if len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
        raise SyncError("policy inventory output exceeds 64 MiB limit")
    return encoded


def _catalog(root, repair_metadata):
    primary = root / "meta.json"
    try:
        return load_json(primary), "primary"
    except SyncError as exc:
        if primary.exists() and not isinstance(exc.__cause__, (json.JSONDecodeError, UnicodeError)):
            raise
        if not repair_metadata:
            raise SyncError("primary metadata repair required; reopen without --no-repair "
                            "to attempt validated recovery") from exc
    return recovery_catalog(load_json(root / SIDECAR)), "recovery"


def _row(name, raw_config):
    row = {"collection": name, "management_state": "unknown", "status": "unavailable",
           "policy": None, "diagnostic": None}

    def diagnostic(status, code, message):
        row.update(status=status, diagnostic={"code": code, "message": message})
        return row

    try:
        validate_collection_name(name)
        config = collection_config(raw_config)
    except SyncError as exc:
        return diagnostic("unavailable", "unavailable_collection_metadata", str(exc))
    state = (config.metadata or {}).get("chunksync")
    if state is None:
        row["management_state"] = "unmanaged"
        return diagnostic("unmanaged", "unmanaged_collection", "collection has no management state")
    if not isinstance(state, dict) or type(state.get("version")) is not int or state["version"] != 1:
        row["management_state"] = "unsupported"
        return diagnostic("unsupported", "unsupported_management_state", "unsupported collection management state")
    row["management_state"] = "managed"
    try:
        row["policy"] = policy_inspection(name, state)
    except SyncError as exc:
        return diagnostic("invalid", "invalid_policy_metadata", str(exc))
    row["status"] = "ok"
    return row


def list_policies(path, *, repair_metadata=True):
    """Return sorted policies/diagnostics, reading only primary/fallback catalogs.

    Existing catalog required; no locks or writes. Caller must exclude concurrent
    writers/replacement. This does not validate point storage or full journals.
    """
    root = Path(path).absolute()
    for entry in (root, *root.parents, root / "meta.json", root / SIDECAR):
        if entry.is_symlink():
            raise SyncError("Qdrant catalog path may not traverse symlinks")
    if (str(path) == ":memory:" or not root.is_dir()
            or not any((root / name).is_file() for name in ("meta.json", SIDECAR))):
        raise SyncError("list-policies requires an existing persistent Qdrant store")
    metadata, source = _catalog(root, repair_metadata)
    if (not isinstance(metadata, dict) or not isinstance(metadata.get("collections"), dict)
            or metadata.get("aliases") != {}):
        raise SyncError("unsupported Qdrant metadata or aliases")
    collections = metadata["collections"]
    if len(collections) > MAX_COLLECTIONS:
        raise SyncError("policy inventory exceeds 1000 collection limit")
    report = {"kind": "chunksync.policy_inventory", "version": 1, "catalog_source": source,
              "collections": [_row(name, collections[name]) for name in sorted(collections)]}
    inventory_json(report)  # Enforce the same output bound for Python callers.
    return report
