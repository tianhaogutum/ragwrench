"""Read-only summaries of durable import mappings, independent of remote state."""
import base64
import binascii
import json
import re

from jsonschema import Draft202012Validator

from .api import DEFAULT_PROFILES
from .bundle import fingerprint, load_bundle
from .common import PortError, digest, dumps, read_json, read_json_snapshot
from .operations import chunk_key, make_plan


CHUNK_STATUSES = ("completed", "pending", "unvisited", "skipped")
PENDING_STAGES = ("document_creation", "bootstrap_cleanup", "chunk_creation", "enabled_flag")


_ID = {"type": "string", "pattern": r"^[A-Za-z0-9_-]{1,128}\Z"}
_NULLABLE_ID = {"anyOf": [_ID, {"type": "null"}]}
_STATE_SCHEMA = {
    "type": "object", "required": ["identity", "dataset", "dataset_pending", "documents"],
    "properties": {
        "identity": {
            "type": "object",
            "required": ["version", "bundle", "system", "base_url", "dataset_name", "reuse_dataset",
                         "allow_lossy", "max_chunk_bytes", "on_error", "indexing"],
            "properties": {
                "version": {"type": "integer", "const": 1},
                "bundle": {"type": "string", "pattern": r"^[a-f0-9]{64}\Z"},
                "system": {"enum": ["dify", "ragflow"]},
                "base_url": {"type": "string", "minLength": 1},
                "api_profile": {"type": "string"},
                "dataset_name": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
                "reuse_dataset": _NULLABLE_ID,
                "allow_lossy": {"type": "boolean"},
                "max_chunk_bytes": {"type": "integer", "minimum": 1, "maximum": 1048576},
                "on_error": {"enum": ["abort", "skip"]},
                "indexing": {"enum": ["high_quality", "economy"]},
            },
        },
        "dataset": _NULLABLE_ID,
        "dataset_pending": {"type": "boolean"},
        "documents": {
            "type": "object", "propertyNames": _ID,
            "additionalProperties": {
                "type": "object", "required": ["id", "phase", "chunks", "pending"],
                "properties": {
                    "id": _NULLABLE_ID,
                    "phase": {"enum": ["new", "creating", "clearing", "chunks", "complete"]},
                    "chunks": {"type": "object", "propertyNames": _ID, "additionalProperties": _ID},
                    "pending": {"anyOf": [
                        {"type": "null"},
                        {"type": "object", "required": ["source", "target"],
                         "properties": {"source": _ID, "target": _NULLABLE_ID}},
                    ]},
                },
            },
        },
    },
}
_VALIDATOR = Draft202012Validator(_STATE_SCHEMA)


def _invalid():
    # Do not include private state values or a schema validation error in logs.
    raise PortError("Invalid import state: expected a consistent version 1 migration checkpoint")


def import_status(state_path, bundle=None, *, document=None, chunk=None, list_chunks=False,
                  page_size=None, cursor=None, chunk_status=None, summary_only=False, pending_counts=False,
                  pending_stage=None, document_page_size=None, document_cursor=None,
                  list_skipped_chunks=False, skip_page_size=None, skip_cursor=None,
                  list_bundle_chunks=False, bundle_chunk_status=None,
                  bundle_chunk_page_size=None, bundle_chunk_cursor=None):
    """Take one atomic-file snapshot without credentials, a writer lock or writes.

    An optional matching bundle supplies the saved preflight plan's totals and
    exclusions. A document selector narrows only bundle detail rows, after full
    validation. Chunk selection or listing adds document-scoped checkpoint detail.
    Summary-only omits detail arrays after the same full validation. Optional
    pending counts aggregate the full snapshot's saved intents in that view.
    Pending-stage filtering and paging narrow document arrays after complete validation.
    Bundle-wide skip listing and paging add saved-preflight exclusions only.
    Bundle chunk listing/filtering/paging adds local rows across all documents.
    No mode checks remote existence or authorizes write replay.
    """
    if (list_bundle_chunks or bundle_chunk_status is not None or
            bundle_chunk_page_size is not None or bundle_chunk_cursor is not None):
        if not list_bundle_chunks or bundle is None:
            raise PortError("Bundle chunk listing, filtering and pagination require a bundle and list-bundle-chunks")
        if (summary_only or pending_counts or pending_stage is not None or document is not None or
                chunk is not None or list_chunks or chunk_status is not None or page_size is not None or
                cursor is not None or document_page_size is not None or document_cursor is not None or
                list_skipped_chunks or skip_page_size is not None or skip_cursor is not None):
            raise PortError("Bundle chunk listing cannot be combined with summary, counts, pending stage, document/chunk selection, other listing, filtering or pagination")
    if bundle_chunk_status is not None and (not isinstance(bundle_chunk_status, str) or bundle_chunk_status not in CHUNK_STATUSES):
        raise PortError("Bundle chunk status must be completed, pending, unvisited or skipped")
    if bundle_chunk_page_size is not None and (type(bundle_chunk_page_size) is not int or bundle_chunk_page_size <= 0):
        raise PortError("Bundle chunk page size must be a positive integer")
    if bundle_chunk_cursor is not None and bundle_chunk_page_size is None:
        raise PortError("Bundle chunk continuation requires a bundle chunk page size")
    if list_skipped_chunks or skip_page_size is not None or skip_cursor is not None:
        if not list_skipped_chunks or bundle is None:
            raise PortError("Skipped chunk listing and pagination require a bundle and list-skipped-chunks")
        if (summary_only or pending_counts or pending_stage is not None or document is not None or
                chunk is not None or list_chunks or chunk_status is not None or page_size is not None or
                cursor is not None or document_page_size is not None or document_cursor is not None):
            raise PortError("Skipped chunk listing cannot be combined with summary, counts, pending stage, document/chunk selection, listing, filtering or other pagination")
    if skip_page_size is not None and (type(skip_page_size) is not int or skip_page_size <= 0):
        raise PortError("Skip page size must be a positive integer")
    if skip_cursor is not None and skip_page_size is None:
        raise PortError("Skip continuation requires a skip page size")
    if pending_stage is not None:
        if not isinstance(pending_stage, str) or pending_stage not in PENDING_STAGES:
            raise PortError("Pending stage must be document_creation, bootstrap_cleanup, chunk_creation or enabled_flag")
        if (summary_only or pending_counts or document is not None or chunk is not None or list_chunks or
                chunk_status is not None or page_size is not None or cursor is not None):
            raise PortError("Pending stage cannot be combined with summary, counts, document/chunk selection, listing, filtering or pagination")
    if pending_counts and not summary_only:
        raise PortError("Pending counts require summary-only")
    if summary_only and (document is not None or chunk is not None or list_chunks or
                         chunk_status is not None or page_size is not None or cursor is not None):
        raise PortError("Summary-only cannot be combined with document/chunk selection, listing, filtering or pagination")
    if chunk_status is not None:
        if not isinstance(chunk_status, str) or chunk_status not in CHUNK_STATUSES:
            raise PortError("Chunk status must be completed, pending, unvisited or skipped")
        if not list_chunks:
            raise PortError("Chunk status filtering requires chunk listing")
    if list_chunks:
        if chunk is not None:
            raise PortError("Chunk listing cannot be combined with chunk selection")
        if bundle is None or document is None:
            raise PortError("Chunk listing requires a bundle and source document")
    if chunk is not None and (bundle is None or document is None):
        raise PortError("Chunk selection requires a bundle and source document")
    if document is not None and bundle is None:
        raise PortError("Document selection requires a bundle")
    if page_size is not None and (type(page_size) is not int or page_size <= 0):
        raise PortError("Chunk page size must be a positive integer")
    if (page_size is not None or cursor is not None) and not list_chunks:
        raise PortError("Chunk pagination requires chunk listing")
    if cursor is not None and page_size is None:
        raise PortError("Chunk continuation requires a page size")
    if document_page_size is not None and (type(document_page_size) is not int or document_page_size <= 0):
        raise PortError("Document page size must be a positive integer")
    if (document_page_size is not None or document_cursor is not None) and pending_stage is None:
        raise PortError("Document pagination requires a pending stage")
    if document_cursor is not None and document_page_size is None:
        raise PortError("Document continuation requires a document page size")
    state_digest = None
    try:
        if (page_size is None and document_page_size is None and skip_page_size is None and
                bundle_chunk_page_size is None):
            state = read_json(state_path)
        else:
            state, data = read_json_snapshot(state_path)
            state_digest = digest(data)
        valid = _VALIDATOR.is_valid(state)
    except RecursionError as exc:
        raise PortError("Invalid import state: excessive nesting") from exc
    if not valid:
        _invalid()
    identity = state["identity"]
    system = identity["system"]
    profile = identity.get("api_profile", DEFAULT_PROFILES[system])
    profiles = {"dify": {"dify-f8fc371a"}, "ragflow": {"ragflow-1.0", "ragflow-0.27.3"}}
    if profile not in profiles[system]:
        _invalid()
    if bool(identity["dataset_name"]) == bool(identity["reuse_dataset"]):
        _invalid()
    if identity["reuse_dataset"] and (state["dataset"] != identity["reuse_dataset"] or state["dataset_pending"]):
        _invalid()
    if ((state["dataset_pending"] and state["dataset"] is not None) or
            (state["documents"] and state["dataset"] is None)):
        _invalid()

    documents, targets = [], set()
    for source, doc in sorted(state["documents"].items()):
        target, phase, chunks, pending = doc["id"], doc["phase"], doc["chunks"], doc["pending"]
        if (target is None) != (phase in ("new", "creating")):
            _invalid()
        if phase in ("new", "creating", "clearing") and (chunks or pending):
            _invalid()
        if phase == "complete" and pending:
            _invalid()
        if (target is not None and target in targets) or len(set(chunks.values())) != len(chunks):
            _invalid()
        if target is not None:
            targets.add(target)
        if pending and (pending["source"] in chunks or
                        (pending["target"] is not None and pending["target"] in chunks.values())):
            _invalid()
        stage = {"creating": "document_creation", "clearing": "bootstrap_cleanup"}.get(phase)
        if pending:
            stage = "enabled_flag" if pending["target"] is not None else "chunk_creation"
        documents.append({
            "source_id": source, "target_id": target, "phase": phase, "completed_chunks": len(chunks),
            "pending_stage": stage,
            "pending_chunk": {"source_id": pending["source"], "target_id": pending["target"]} if pending else None,
        })
    result = {
        "state": str(state_path), "target_system": system, "api_profile": profile,
        "dataset_id": state["dataset"], "dataset_pending": state["dataset_pending"],
        "mapped_documents": len(targets), "completed_chunks": sum(d["completed_chunks"] for d in documents),
        "documents": documents,
    }
    if bundle is not None:
        result["bundle_progress"] = _bundle_progress(state, result, bundle, document=document, chunk=chunk,
                                                     list_chunks=list_chunks, page_size=page_size,
                                                     cursor=cursor, state_digest=state_digest, chunk_status=chunk_status,
                                                     list_bundle_chunks=list_bundle_chunks,
                                                     bundle_chunk_status=bundle_chunk_status,
                                                     bundle_chunk_page_size=bundle_chunk_page_size,
                                                     bundle_chunk_cursor=bundle_chunk_cursor)
        if document is not None:
            rows = [row for row in result["bundle_progress"]["documents"] if row["source_id"] == document]
            if not rows:
                raise PortError("Selected source document is not in the bundle")
            result["bundle_progress"]["documents"] = rows
    if list_skipped_chunks:
        # Reuse the fully validated saved-preflight reasons, including unvisited
        # documents. Both document arrays and every global count remain intact.
        progress = result["bundle_progress"]
        rows = [{"document_source_id": doc["source_id"], **item}
                for doc in progress["documents"] for item in doc["skipped_chunks"]]
        if skip_page_size is not None:
            sources = [[row["document_source_id"], row["source_id"]] for row in rows]
            binding = {"version": 1, "kind": "bundle-skipped-chunks",
                       "state": state_digest, "bundle": progress["manifest_fingerprint"]}
            start = (_cursor_start(skip_cursor, binding, sources, kind="skip")
                     if skip_cursor is not None else 0)
            rows = rows[start:start + skip_page_size]
            progress["next_skip_cursor"] = (_encode_cursor({**binding, "after": sources[start + len(rows) - 1]})
                                            if start + len(rows) < len(sources) else None)
        progress["skipped_chunk_details"] = rows
    if summary_only:
        # Output-only projection: never bypass validation of omitted details.
        if pending_counts:
            counts = {"dataset_creation": int(state["dataset_pending"]), **dict.fromkeys(PENDING_STAGES, 0)}
            for row in documents:
                if row["pending_stage"] is not None:
                    counts[row["pending_stage"]] += 1
            result["pending_counts"] = counts
        del result["documents"]
        if bundle is not None:
            del result["bundle_progress"]["documents"]
    if pending_stage is not None:
        # Project only after validating every state/bundle detail and saved preflight.
        rows = [row for row in documents if row["pending_stage"] == pending_stage]
        if document_page_size is not None:
            sources = [row["source_id"] for row in rows]
            binding = {"version": 1, "kind": "pending-stage-documents",
                       "scope": "bundle" if bundle is not None else "state",
                       "bundle": result["bundle_progress"]["manifest_fingerprint"] if bundle is not None else None,
                       "state": state_digest, "pending_stage": pending_stage}
            start = (_cursor_start(document_cursor, binding, sources, kind="document")
                     if document_cursor is not None else 0)
            rows = rows[start:start + document_page_size]
            result["next_document_cursor"] = (_encode_cursor({**binding, "after": rows[-1]["source_id"]})
                                              if start + len(rows) < len(sources) else None)
        result["documents"] = rows
        if bundle is not None:
            sources = {row["source_id"] for row in result["documents"]}
            result["bundle_progress"]["documents"] = [
                row for row in result["bundle_progress"]["documents"] if row["source_id"] in sources
            ]
    return result


def _bundle_progress(state, snapshot, bundle, *, document=None, chunk=None, list_chunks=False,
                     page_size=None, cursor=None, state_digest=None, chunk_status=None,
                     list_bundle_chunks=False, bundle_chunk_status=None,
                     bundle_chunk_page_size=None, bundle_chunk_cursor=None):
    manifest, documents, chunks = load_bundle(bundle)
    identity = state["identity"]
    bundle_id = fingerprint(manifest)
    if identity["bundle"] != bundle_id:
        raise PortError("Import state belongs to a different bundle")
    plan, projected, _ = make_plan(
        manifest, documents, chunks, identity["system"],
        allow_lossy=identity["allow_lossy"], max_chunk_bytes=identity["max_chunk_bytes"],
        on_error=identity["on_error"],
    )
    scope = {doc["id"]: set() for doc in documents}
    skipped = {source: [] for source in scope}
    for record in projected.values():
        scope[record["doc_id"]].add(record["id"])
    for record in chunks:
        reason = plan["errors"].get(chunk_key(record))
        if reason is not None:
            skipped[record["doc_id"]].append({"source_id": record["id"], "reason": reason})
    completed_documents = pending_chunks = 0
    for source, doc in state["documents"].items():
        if source not in scope or not set(doc["chunks"]) <= scope[source]:
            _invalid()
        if doc["pending"]:
            if doc["pending"]["source"] not in scope[source]:
                _invalid()
            pending_chunks += 1
        if doc["phase"] == "complete":
            if set(doc["chunks"]) != scope[source]:
                _invalid()
            completed_documents += 1
    breakdown = []
    for source in sorted(scope):
        doc = state["documents"].get(source)
        completed = len(doc["chunks"]) if doc else 0
        pending = int(bool(doc and doc["pending"]))
        remaining = len(scope[source]) - completed
        breakdown.append({
            "source_id": source, "visited": doc is not None,
            "complete": bool(doc and doc["phase"] == "complete"),
            "planned_chunks": len(scope[source]), "completed_chunks": completed,
            "remaining_chunks": remaining, "pending_chunks": pending,
            "unvisited_chunks": remaining - pending,
            "skipped_chunks": sorted(skipped[source], key=lambda item: item["source_id"]),
        })
    remaining_chunks = plan["chunks"] - snapshot["completed_chunks"]
    result = {
        "bundle": str(bundle), "manifest_fingerprint": bundle_id,
        "planned_documents": plan["documents"], "planned_chunks": plan["chunks"],
        "skipped_chunks": plan["skipped"], "warnings": plan["warnings"],
        "completed_documents": completed_documents,
        "remaining_documents": plan["documents"] - completed_documents,
        "remaining_chunks": remaining_chunks, "pending_chunks": pending_chunks,
        "unvisited_documents": len(scope.keys() - state["documents"].keys()),
        "unvisited_chunks": remaining_chunks - pending_chunks,
        "documents": breakdown,
    }
    if list_bundle_chunks:
        # Enumerate original chunks with the same helper as document-scoped
        # details, only after every state/bundle record and saved plan is valid.
        rows = []
        for source in sorted(scope):
            reasons = {item["source_id"]: item["reason"] for item in skipped[source]}
            doc = state["documents"].get(source)
            for chunk_id in sorted(scope[source] | reasons.keys()):
                row = _chunk_progress(doc, source, chunk_id, reasons.get(chunk_id))
                if bundle_chunk_status is None or row["status"] == bundle_chunk_status:
                    rows.append(row)
        if bundle_chunk_page_size is not None:
            sources = [[row["document_source_id"], row["source_id"]] for row in rows]
            binding = {"version": 1, "kind": "bundle-chunks", "state": state_digest,
                       "bundle": bundle_id, "bundle_chunk_status": bundle_chunk_status}
            start = (_cursor_start(bundle_chunk_cursor, binding, sources, kind="bundle chunk")
                     if bundle_chunk_cursor is not None else 0)
            rows = rows[start:start + bundle_chunk_page_size]
            result["next_bundle_chunk_cursor"] = (_encode_cursor({**binding, "after": sources[start + len(rows) - 1]})
                                                  if start + len(rows) < len(sources) else None)
        result["chunk_details"] = rows
    if chunk is not None or list_chunks:
        # All bundle/preflight/state checks above apply even outside the selection.
        if document not in scope:
            raise PortError("Selected source document is not in the bundle")
        reasons = {item["source_id"]: item["reason"] for item in skipped[document]}
        doc = state["documents"].get(document)
        if list_chunks:
            sources = sorted(scope[document] | reasons.keys())
            if chunk_status is not None:
                sources = [source for source in sources
                           if _chunk_progress(doc, document, source, reasons.get(source))["status"] == chunk_status]
            if page_size is not None:
                binding = {"version": 1, "document": document, "bundle": bundle_id, "state": state_digest}
                if chunk_status is not None:
                    binding.update(version=2, chunk_status=chunk_status)
                start = _cursor_start(cursor, binding, sources) if cursor is not None else 0
                page = sources[start:start + page_size]
                result["next_cursor"] = (_encode_cursor({**binding, "after": page[-1]})
                                         if start + len(page) < len(sources) else None)
                sources = page
            result["chunks"] = [
                _chunk_progress(doc, document, source, reasons.get(source))
                for source in sources
            ]
        else:
            reason = reasons.get(chunk)
            if chunk not in scope[document] and reason is None:
                raise PortError("Selected source chunk is not in the selected bundle document")
            result["chunk"] = _chunk_progress(doc, document, chunk, reason)
    return result


def _encode_cursor(value):
    return base64.urlsafe_b64encode(dumps(value).encode("utf-8")).decode("ascii").rstrip("=")


def _cursor_start(cursor, binding, sources, *, kind="chunk"):
    # Tokens are opaque continuation data, not authenticated resume permissions.
    # Enforce one bounded, canonical representation and never echo token contents.
    try:
        if not isinstance(cursor, str) or not 1 <= len(cursor) <= 2048 or not re.fullmatch(r"[A-Za-z0-9_-]+", cursor):
            raise ValueError
        raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
        value = json.loads(raw.decode("utf-8"))
        if (not isinstance(value, dict) or set(value) != set(binding) | {"after"} or
                type(value["version"]) is not int or
                any(value[key] != expected for key, expected in binding.items()) or
                _encode_cursor(value) != cursor):
            raise ValueError
        after = value["after"]
        if kind in ("skip", "bundle chunk"):
            if not isinstance(after, list) or len(after) != 2 or not all(isinstance(part, str) for part in after):
                raise ValueError
        elif not isinstance(after, str):
            raise ValueError
        start = sources.index(after) + 1
        if start == len(sources):
            raise ValueError
        return start
    except (ValueError, TypeError, UnicodeError, binascii.Error, RecursionError) as exc:
        flag = {"document": "--document-cursor", "skip": "--skip-cursor",
                "bundle chunk": "--bundle-chunk-cursor"}.get(kind, "--cursor")
        raise PortError(f"Invalid {kind} continuation cursor or snapshot mismatch; restart without {flag}") from exc


def _chunk_progress(doc, document, chunk, reason):
    status, target, stage = "unvisited", None, None
    if reason is not None:
        status = "skipped"
    elif doc:
        if chunk in doc["chunks"]:
            status, target = "completed", doc["chunks"][chunk]
        elif doc["pending"] and doc["pending"]["source"] == chunk:
            status, target = "pending", doc["pending"]["target"]
            stage = "enabled_flag" if target is not None else "chunk_creation"
    return {"document_source_id": document, "source_id": chunk, "status": status,
            "target_id": target, "pending_stage": stage, "reason": reason}
