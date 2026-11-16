"""Single-writer, resumable local Qdrant transactions."""
from __future__ import annotations

import math
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import re
import struct

from qdrant_client import QdrantClient, models

from .core import SyncError, digest, payload, validate_plan, validate_snapshot
from .embedders import checked_vector
from .journal import ApplyJournal, point_fingerprint
from .review import require_allowed, review_plan, validate_review
from .recovery import protect_metadata, recover_metadata, require_apply_ready
from .policy import admit_policy, collection_policy, stored_policy_override, validate_policy_override


def _open_client(path, *, repair_metadata=True):
    if path == ":memory:":
        return QdrantClient(location=":memory:")
    # qdrant-client uses collection names as directories. Disallow traversal
    # names separately and refuse existing symlinks anywhere below the store.
    root = Path(path).absolute()
    for parent in (root, *root.parents):
        if parent.is_symlink():
            raise SyncError("Qdrant path may not traverse symlinks")
    if root.exists():
        for entry in root.rglob("*"):
            if entry.is_symlink():
                raise SyncError("Qdrant store contains a symlink")
    recover_metadata(root, validate_collection_name, repair_metadata=repair_metadata)
    client = QdrantClient(path=str(root))
    protect_metadata(client, root)
    return client


@contextmanager
def local_client(path, *, repair_metadata=True):
    """Open with safe-store checks; False refuses missing/undecodable metadata.

    This controls opening-time repair only, not later client mutations or lock
    handling. The default retains validated recovery; memory is unaffected.
    """
    client = _open_client(path, repair_metadata=repair_metadata)
    try:
        yield client
    finally:
        client.close()


def validate_collection_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", name):
        raise SyncError("collection must use 1-128 ASCII letters, digits, underscores or hyphens")


def stored_payload(chunk):
    return {**payload(chunk), "payload_hash": chunk["payload_hash"]}


def all_points(client, collection):
    points, offset = {}, None
    while True:
        rows, offset = client.scroll(collection, limit=256, offset=offset, with_vectors=True, with_payload=True)
        for row in rows:
            points[str(row.id)] = row
        if offset is None:
            return points


def data_fingerprint(points):
    return digest([{ "id": pid, "payload": p.payload, "vector_float32": struct.pack(f"<{len(p.vector)}f", *p.vector).hex()}
                   for pid, p in sorted(points.items())])


def check_snapshot_points(points, snapshot):
    expected = {c["id"]: c for c in snapshot["chunks"]}
    if set(points) != set(expected):
        raise SyncError("collection ids differ from snapshot")
    for pid, chunk in expected.items():
        if points[pid].payload != stored_payload(chunk):
            raise SyncError(f"collection payload differs for {pid}")


def _state(client, collection):
    return deepcopy((client.get_collection(collection).config.metadata or {}).get("chunksync"))


def _write_state(client, collection, state):
    # The local client's metadata may otherwise retain our mutable dictionaries.
    # An unacknowledged later checkpoint must not mutate an earlier saved one.
    client.update_collection(collection, metadata={"chunksync": deepcopy(state)})


def inspect_policy(client, collection):
    """Read effective policy and recorded audit context, without admitting work.

    This checks the reported fields, not point integrity or the full journal.
    Opening a persistent local_client may already have repaired damaged metadata
    unless repair_metadata=False was supplied when opening it.
    """
    validate_collection_name(collection)
    if not client.collection_exists(collection):
        raise SyncError("collection does not exist")
    state = _state(client, collection)
    return policy_inspection(collection, state)


def policy_inspection(collection, state):
    """Decode the reported policy fields from metadata without adopting policy."""
    validate_collection_name(collection)
    if not isinstance(state, dict) or type(state.get("version")) is not int or state["version"] != 1:
        raise SyncError("unsupported or missing collection management state")
    policy = collection_policy(state)
    source = ("explicit" if "review_policy" in state else
              "legacy-review" if state.get("last_review") is not None else "default")
    pending = state.get("pending")
    if pending is not None:
        pending = pending.get("plan_fingerprint") if isinstance(pending, dict) else pending
        if not isinstance(pending, str) or not re.fullmatch(r"[0-9a-f]{64}", pending):
            raise SyncError("invalid pending plan fingerprint")
    return {"kind": "chunksync.policy_inspection", "version": 1, "collection": collection,
            "max_deletion_fraction": policy["max_deletion_fraction"], "policy_source": source,
            "pending_plan_fingerprint": pending,
            "policy_override": stored_policy_override(collection, state.get("last_policy_override"))}


def _check_dimension(client, collection, dimension):
    vectors = client.get_collection(collection).config.params.vectors
    if not isinstance(vectors, models.VectorParams) or vectors.size != dimension or vectors.distance != models.Distance.DOT:
        raise SyncError("collection vector dimension/distance mismatch (requires one dense DOT vector)")


def _assert_committed(points, state, snapshot):
    check_snapshot_points(points, snapshot)
    if data_fingerprint(points) != state.get("data_fingerprint"):
        raise SyncError("collection fingerprint mismatch: vectors or payloads changed outside chunksync")


def apply_plan(client, collection, plan, embedder, *, review=None, policy_override=None):
    require_apply_ready(client)
    validate_collection_name(collection)
    validate_plan(plan)
    review = review_plan(plan) if review is None else validate_review(plan, review)
    require_allowed(review)
    if policy_override is not None:
        validate_policy_override(plan, collection, policy_override, review["policy"])
    if not 1 <= embedder.dimension <= 4096:
        raise SyncError("dimension must be between 1 and 4096")
    old, new = plan["old"], plan["new"]
    if not client.collection_exists(collection):
        if policy_override is not None:
            raise SyncError("policy override requires an existing collection review policy")
        if old["chunks"]:
            raise SyncError("collection missing but plan expects an existing snapshot")
        client.create_collection(collection, vectors_config=models.VectorParams(size=embedder.dimension, distance=models.Distance.DOT))
    _check_dimension(client, collection, embedder.dimension)
    points = all_points(client, collection)
    state = _state(client, collection)
    if state is None:
        if policy_override is not None:
            raise SyncError("policy override requires an existing collection review policy")
        if points or old["chunks"]:
            raise SyncError("unmanaged/nonempty collection; fingerprint unavailable")
        state = {"version": 1, "fingerprint": old["fingerprint"], "data_fingerprint": data_fingerprint(points),
                 "embedder_signature": embedder.signature, "pending": None}
        _write_state(client, collection, state)
    if not isinstance(state, dict) or state.get("version") != 1:
        raise SyncError("unsupported collection management state")
    if state.get("embedder_signature") != embedder.signature:
        raise SyncError("collection embedder fingerprint mismatch; use the original embedder")
    pending = state.get("pending")
    if isinstance(pending, dict):
        if pending.get("plan_fingerprint") != plan["fingerprint"]:
            raise SyncError("a different plan is pending; apply journal binding mismatch")
        if "review_fingerprint" in pending and pending["review_fingerprint"] != review["fingerprint"]:
            raise SyncError("apply journal review binding mismatch; reuse the original review receipt")
        if "policy_override_fingerprint" in pending:
            expected = policy_override["fingerprint"] if policy_override is not None else None
            if pending["policy_override_fingerprint"] != expected:
                raise SyncError("apply journal policy override binding mismatch; reuse the original policy override")
    replay = pending is None and state.get("fingerprint") == new["fingerprint"]
    policy = admit_policy(state, review, policy_override, replay=replay)
    if replay:
        _assert_committed(points, state, new)
        return {"status": "already_applied", "embedded": 0, "points": len(points),
                "review": state.get("last_review"), "review_policy": policy,
                "policy_override": state.get("last_policy_override")}
    if state.get("fingerprint") != old["fingerprint"]:
        raise SyncError("collection fingerprint mismatch: plan is based on a different snapshot")
    pending = state.get("pending")
    if isinstance(pending, str):
        if pending != plan["fingerprint"]:
            raise SyncError("a different plan is pending; resume that plan first")
        # Upgrade legacy pending markers using their original admission checks.
        # Their earlier vector history was not journaled and cannot be recovered.
        allowed = {}
        for c in old["chunks"] + new["chunks"]:
            allowed.setdefault(c["id"], []).append(stored_payload(c))
        for pid, p in points.items():
            if pid not in allowed or p.payload not in allowed[pid]:
                raise SyncError("unexpected point/payload in pending collection")
            checked_vector(p.vector, embedder.dimension)
        # Unchanged source vectors are still required; re-embedding them on
        # resume would conceal data loss.
        for c in plan["unchanged"]:
            if c["id"] not in points:
                raise SyncError("unchanged point missing from pending collection")
    elif pending is None:
        _assert_committed(points, state, old)

    # Stable ordering: all writes precede per-point retirement. A cursor records
    # the acknowledged prefix, and a write-ahead intent covers the next call.
    operations = [("upsert", c, None) for c in plan["upsert"]]
    operations += [("moved", m["after"], m["before"]) for m in plan["moved"]]
    operations += [("payload_only", c, None) for c in plan["payload_only"]]
    operations += [("delete", c, None) for c in plan["delete"]]
    operations += [("delete", m["before"], None) for m in plan["moved"] if m["before"]["id"] != m["after"]["id"]]

    def save_journal(entry):
        state["pending"] = entry
        state["last_review"] = review
        state["review_policy"] = policy
        if policy_override is not None:
            state["last_policy_override"] = policy_override
        _write_state(client, collection, state)

    binding = {"plan_fingerprint": plan["fingerprint"], "old_fingerprint": old["fingerprint"],
               "new_fingerprint": new["fingerprint"], "source_data_fingerprint": state["data_fingerprint"],
               "embedder_signature": embedder.signature}
    # Existing version-1 journals have no review binding. Keep their recovery
    # format, while new transactions bind policy and override provenance too.
    if not isinstance(pending, dict) or "review_fingerprint" in pending:
        binding["review_fingerprint"] = review["fingerprint"]
    if not isinstance(pending, dict) or "policy_override_fingerprint" in pending:
        binding["policy_override_fingerprint"] = policy_override["fingerprint"] if policy_override is not None else None
    elif policy_override is not None:
        raise SyncError("legacy pending journal cannot admit a new policy override")
    journal = ApplyJournal(None if isinstance(pending, str) else pending, binding,
                           [c["id"] for _, c, _ in operations], points, save_journal)

    embedded = 0

    def matches(chunk):
        p = points.get(chunk["id"])
        return p is not None and p.payload == stored_payload(chunk)

    for kind, chunk, before in operations[journal.cursor:]:
        pid = chunk["id"]
        point = None
        if kind != "delete":
            if matches(chunk):  # successful writes from a legacy pending apply
                vector = points[pid].vector
            elif kind == "upsert":
                vector = embedder.embed(chunk, new["embed_model"])
                embedded += 1
            else:
                source = points.get(before["id"] if before else pid)
                if source is None:
                    raise SyncError("move/payload-only source missing; cannot reuse its vector")
                vector = source.vector
            vector = checked_vector(vector, embedder.dimension)
            point = models.PointStruct(id=pid, vector=vector, payload=stored_payload(chunk))
        if not journal.prepare(pid, point_fingerprint(point)):
            continue
        if kind == "delete":
            client.delete(collection, points_selector=models.PointIdsList(points=[pid]), wait=True)
            points.pop(pid, None)
        else:
            if kind == "payload_only":
                client.overwrite_payload(collection, payload=point.payload, points=[pid], wait=True)
            else:
                client.upsert(collection, points=[point], wait=True)
            # The database is authoritative (float32 conversion happens here).
            points[pid] = client.retrieve(collection, [pid], with_vectors=True)[0]
        journal.complete(pid, point_fingerprint(points.get(pid)))
    points = all_points(client, collection)
    check_snapshot_points(points, new)
    state.update(fingerprint=new["fingerprint"], data_fingerprint=data_fingerprint(points), pending=None,
                 last_review=review)
    _write_state(client, collection, state)
    return {"status": "applied", "embedded": embedded, "points": len(points), "review": review,
            "review_policy": collection_policy(state), "policy_override": state.get("last_policy_override")}


def verify_snapshot(client, collection, snapshot, embedder, tolerance=1e-6):
    validate_collection_name(collection)
    validate_snapshot(snapshot)
    if not client.collection_exists(collection):
        raise SyncError("collection does not exist")
    _check_dimension(client, collection, embedder.dimension)
    state = _state(client, collection)
    if not state or state.get("pending") is not None or state.get("fingerprint") != snapshot["fingerprint"]:
        raise SyncError("collection fingerprint mismatch or transaction pending")
    if state.get("embedder_signature") != embedder.signature:
        raise SyncError("collection embedder fingerprint mismatch")
    points = all_points(client, collection)
    _assert_committed(points, state, snapshot)
    # Independent from-scratch vectors, not hashes copied out of the plan.
    for chunk in snapshot["chunks"]:
        expected = checked_vector(embedder.embed(chunk, snapshot["embed_model"]), embedder.dimension)
        actual = checked_vector(points[chunk["id"]].vector, embedder.dimension)
        if any(not math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance) for a, b in zip(actual, expected)):
            raise SyncError(f"vector differs from from-scratch embedding for {chunk['id']}")
    return {"status": "verified", "points": len(points), "vectors_recomputed": len(points), "tolerance": tolerance}
