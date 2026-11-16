"""Compare complete archived inventories, without accessing either source store."""
import json
from pathlib import Path

from .core import MAX_JSON_BYTES, SyncError
from .inventory import MAX_COLLECTIONS
from .store import policy_inspection, validate_collection_name


_ROW_FIELDS = {"collection", "management_state", "status", "policy", "diagnostic"}
_POLICY_FIELDS = {"kind", "version", "collection", "max_deletion_fraction",
                  "policy_source", "pending_plan_fingerprint", "policy_override"}
_DIAGNOSTICS = {
    "unmanaged": ("unmanaged", "unmanaged_collection", "collection has no management state"),
    "unsupported": ("unsupported", "unsupported_management_state", "unsupported collection management state"),
    "invalid": ("managed", "invalid_policy_metadata", None),
    "unavailable": ("unknown", "unavailable_collection_metadata", None),
}


def comparison_archive_path(path, before, after):
    """Refuse aliases of either input before reading; assume stable filesystem.

    Inspect lexical ancestors before resolving so even symlink/.. traversal is
    refused. Input symlinks remain supported, but cannot disguise an output alias.
    """
    try:
        if "\0" in str(path):
            raise SyncError("policy comparison output path contains a null byte")
        target = Path(path).absolute()
        for entry in (target, *target.parents):
            if entry.is_symlink():
                raise SyncError("policy comparison output path may not traverse symlinks")
        target = target.resolve()
        if not target.parent.is_dir():
            raise SyncError("policy comparison output requires an existing parent directory")
        if target.exists() and not target.is_file():
            raise SyncError("policy comparison output must be a regular file")
        for source in (Path(before), Path(after)):
            if target == source.resolve() or (target.exists() and target.samefile(source)):
                raise SyncError("policy comparison output must be distinct from both input archives")
        return target
    except SyncError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise SyncError(f"cannot validate policy comparison output: {exc}") from exc


def comparison_json(value):
    """Serialize completely before stdout, including the newline in the byte bound."""
    try:
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        if len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
            raise SyncError("policy comparison JSON exceeds 64 MiB limit")
        return encoded
    except SyncError:
        raise
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise SyncError("invalid policy comparison JSON") from exc


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SyncError("duplicate JSON object key in policy inventory")
        result[key] = value
    return result


def _invalid_constant(value):
    raise SyncError(f"invalid JSON constant: {value}")


def load_inventory(path):
    """Bounded strict JSON read; duplicate keys must not hide malformed fields."""
    try:
        path = Path(path)
        if not path.is_file():
            raise SyncError("policy inventory input must be a regular file")
        if path.stat().st_size > MAX_JSON_BYTES:
            raise SyncError("policy inventory input exceeds 64 MiB limit")
        with path.open("rb") as stream:
            raw = stream.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise SyncError("policy inventory input exceeds 64 MiB limit")
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant)
    except SyncError:
        raise
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise SyncError(f"cannot read policy inventory: {exc}") from exc


def _shape(value, fields, label):
    if not isinstance(value, dict) or set(value) != fields:
        raise SyncError(f"invalid policy inventory {label} fields")


def _row(row):
    _shape(row, _ROW_FIELDS, "row")
    name, status = row["collection"], row["status"]
    if not isinstance(name, str) or not isinstance(status, str):
        raise SyncError("invalid policy inventory collection or status")
    if status != "unavailable":
        validate_collection_name(name)
    if status == "ok":
        if row["management_state"] != "managed" or row["diagnostic"] is not None:
            raise SyncError("inconsistent healthy policy inventory row")
        policy = row["policy"]
        _shape(policy, _POLICY_FIELDS, "policy")
        if (policy["kind"] != "chunksync.policy_inspection" or type(policy["version"]) is not int
                or policy["version"] != 1 or policy["collection"] != name):
            raise SyncError("invalid policy inventory inspection identity/version")
        # Reconstruct only the reported state to share live inspection validators.
        # A retained exception may precede tightening or a different pending plan.
        state = {"version": 1, "pending": policy["pending_plan_fingerprint"],
                 "last_policy_override": policy["policy_override"]}
        limit = {"max_deletion_fraction": policy["max_deletion_fraction"]}
        source = policy["policy_source"]
        if source == "explicit":
            state["review_policy"] = limit
        elif source == "legacy-review":
            state["last_review"] = {"policy": limit}
        elif source != "default" or type(policy["max_deletion_fraction"]) not in (int, float) or policy["max_deletion_fraction"] != 1:
            raise SyncError("invalid policy inventory policy source/default limit")
        # The archive schema has a fingerprint, not the live journal object.
        if state["pending"] is not None and not isinstance(state["pending"], str):
            raise SyncError("invalid policy inventory pending fingerprint")
        return {**row, "policy": policy_inspection(name, state)}
    if status not in _DIAGNOSTICS:
        raise SyncError("unsupported policy inventory row status")
    management, code, message = _DIAGNOSTICS[status]
    if row["management_state"] != management or row["policy"] is not None:
        raise SyncError("inconsistent diagnostic policy inventory row")
    diagnostic = row["diagnostic"]
    _shape(diagnostic, {"code", "message"}, "diagnostic")
    if (diagnostic["code"] != code or not isinstance(diagnostic["message"], str)
            or not diagnostic["message"].strip()
            or (message is not None and diagnostic["message"] != message)):
        raise SyncError("invalid policy inventory diagnostic")
    return row


def validate_inventory(report):
    """Validate every row, returning a map with normalized policy numbers."""
    _shape(report, {"kind", "version", "catalog_source", "collections"}, "report")
    if (report["kind"] != "chunksync.policy_inventory" or type(report["version"]) is not int
            or report["version"] != 1 or report["catalog_source"] not in ("primary", "recovery")):
        raise SyncError("unsupported policy inventory kind/version/catalog source")
    rows = report["collections"]
    if not isinstance(rows, list) or len(rows) > MAX_COLLECTIONS:
        raise SyncError("policy inventory requires a list of at most 1000 collections")
    comparison_json(report)  # Python inputs share the serialized byte/JSON bounds.
    result = {}
    for row in rows:
        row = _row(row)
        if row["collection"] in result:
            raise SyncError("duplicate collection identity in policy inventory")
        result[row["collection"]] = row
    return result


def compare_policies(before, after):
    """Validate both archives before comparing all row fields by exact identity."""
    old, new = validate_inventory(before), validate_inventory(after)
    added = [new[name] for name in sorted(new.keys() - old.keys())]
    removed = [old[name] for name in sorted(old.keys() - new.keys())]
    changed = [{"collection": name, "before": old[name], "after": new[name]}
               for name in sorted(old.keys() & new.keys()) if old[name] != new[name]]
    result = {"kind": "chunksync.policy_comparison", "version": 1,
              "identical": not (added or removed or changed),
              "catalog_source": {"before": before["catalog_source"], "after": after["catalog_source"]},
              "added": added, "removed": removed, "changed": changed}
    comparison_json(result)
    return result
