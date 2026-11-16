"""Atomic write-ahead catalog for the pinned local Qdrant metadata writer.

Qdrant 1.16.2 truncates meta.json in place. Save its entire intended catalog
before that write, including all collection configs, management state, journals
and review receipts. Point data remains exclusively in Qdrant's SQLite stores.
"""
import json
from pathlib import Path

from qdrant_client._pydantic_compat import to_dict

from .core import SyncError, digest, load_json, write_json
from .safe_store import preflight

SIDECAR = "chunksync-recovery.json"


def require_apply_ready(client):
    """Failed persistent saves leave live metadata ahead of disk; reopen first."""
    if getattr(client, "_chunksync_metadata_save_failed", False):
        raise SyncError("persistent metadata save failed; close and reopen the client before applying again")


def recover_metadata(root, validate_name, *, repair_metadata=True):
    """Prefer valid primary JSON; repair only missing or undecodable metadata.

    Never fall back from a semantic/preflight error, oversized input, or an I/O
    error other than a missing file. Validate the fallback and SQLite blobs
    before repairing metadata, so recovery cannot bypass safe-store admission.
    """
    root = Path(root)
    meta_file, sidecar = root / "meta.json", root / SIDECAR
    if not repair_metadata:
        # Do not even read the fallback: inspection must use the primary or
        # refuse before Qdrant can create a catalog or acquire its store lock.
        try:
            load_json(meta_file)
        except SyncError as exc:
            if not meta_file.exists() or isinstance(exc.__cause__, (json.JSONDecodeError, UnicodeError)):
                raise SyncError("primary metadata repair required; reopen without --no-repair "
                                "to attempt validated recovery") from exc
            raise
        preflight(root, validate_name)
        return
    if not sidecar.exists():
        preflight(root, validate_name)
        return
    try:
        load_json(meta_file)
    except SyncError as exc:
        if meta_file.exists() and not isinstance(exc.__cause__, (json.JSONDecodeError, UnicodeError)):
            raise
    else:
        preflight(root, validate_name)
        return

    metadata = recovery_catalog(load_json(sidecar))
    preflight(root, validate_name, metadata=metadata)
    write_json(meta_file, metadata, durable=True)


def recovery_catalog(saved):
    """Decode/checksum a sidecar; callers must validate its intended use.

    This checks the envelope, not configurations, point blobs or journal state.
    Persistent repair must still pass the complete safe-store preflight.
    """
    if (not isinstance(saved, dict) or set(saved) != {"kind", "version", "metadata", "fingerprint"}
            or saved["kind"] != "chunksync.recovery" or type(saved["version"]) is not int
            or saved["version"] != 1):
        raise SyncError("unsupported recovery sidecar schema/version")
    body = {key: saved[key] for key in ("kind", "version", "metadata")}
    if saved["fingerprint"] != digest(body):
        raise SyncError("recovery sidecar fingerprint mismatch")
    if not isinstance(saved["metadata"], dict):
        raise SyncError("invalid recovery sidecar metadata")
    return saved["metadata"]


def protect_metadata(client, root):
    """Install before any mutation, using the pinned backend's catalog format.

    This hook also protects collection creation and preserves other collections
    when Qdrant rewrites the whole catalog. Sync the temporary file, rename,
    then sync the directory before entering Qdrant's truncating writer. Any
    failure stops that save; after rename the new sidecar may be published.
    Latch any save failure, including a directory fsync error, so apply_plan
    requires close/reopen rather than trusting potentially unsaved live state.
    In-memory clients and clients opened outside local_client are unaffected.
    """
    backend = client._client
    original = backend._save

    def save():
        if backend.closed:
            return original()  # preserve Qdrant's closed-client refusal
        try:
            body = {"kind": "chunksync.recovery", "version": 1,
                    "metadata": {"collections": {name: to_dict(collection.config)
                                                 for name, collection in backend.collections.items()},
                                 "aliases": backend.aliases}}
            write_json(Path(root) / SIDECAR, {**body, "fingerprint": digest(body)}, durable=True)
            return original()
        except BaseException:
            # An interrupted save is ambiguous too. Preserve its original error;
            # reads/close remain usable, but no later apply may trust live state.
            client._chunksync_metadata_save_failed = True
            raise

    backend._save = save
