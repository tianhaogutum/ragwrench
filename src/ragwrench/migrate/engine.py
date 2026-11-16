"""Preflight migration and parity checks shared by CLI and Python callers."""
from __future__ import annotations

import random
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .model import (ParityError, Record, StoreURI, checked_store_path, dumps, normalized, physical_id,
                    validate_record)
from .stores import Store, check_directory, read_store
from . import checkpoints


def records(store: Store, batch_size: int):
    schema = store.schema()
    for batch in store.batches(batch_size):
        for record in batch:
            validate_record(record, schema.dimension, schema.distance)
            yield record


def destination_id(record: Record, kind: str):
    return (record.qdrant_id if record.qdrant_id is not None else physical_id(record.id)) if kind == "qdrant" else record.id


def disjoint(source: StoreURI, target: StoreURI) -> None:
    source_path = checked_store_path(source.path)
    target_path = checked_store_path(target.path)
    if source_path == target_path or source_path in target_path.parents or target_path in source_path.parents:
        raise ParityError("Source and target store directories must be separate and must not contain each other")
    if source.kind == target.kind:
        raise ParityError("Copy requires different backends (Chroma <-> Qdrant)")


def _validate_top_k(top_k: int) -> None:
    # bool is an int subclass, but is not a retrieval cutoff.
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ParityError("top_k must be a positive integer")


def copy(source: StoreURI, target: StoreURI, *, batch_size: int = 256,
         dry_run: bool = False, vector_name: str | None = None,
         checkpoint: Path | str | None = None, resume: bool = False,
         top_k: int = 10) -> dict[str, Any]:
    _validate_top_k(top_k)
    if not 1 <= batch_size <= 4096:
        raise ParityError("batch_size must be in [1,4096]")
    if resume and checkpoint is None:
        raise ParityError("--resume requires --checkpoint")
    if dry_run and checkpoint is not None:
        raise ParityError("--checkpoint cannot be combined with --dry-run")
    disjoint(source, target)
    progress_path = checkpoints.checkpoint_path(checkpoint, source, target, resume=resume) if checkpoint is not None else None
    progress = checkpoints.load(progress_path) if resume else None
    completed = set(progress["completed_ids"]) if progress else set()
    with ExitStack() as stack:
        src = stack.enter_context(read_store(source, vector_name))
        schema = src.schema()
        if not schema.count:
            raise ParityError("Empty source collection: no points to migrate")
        target_snapshot = stack.enter_context(read_store(target, vector_name, optional=True))
        if target_snapshot:
            other = target_snapshot.schema()
            if other.distance != schema.distance or (other.dimension is not None and other.dimension != schema.dimension):
                raise ParityError("Target dimension/distance differs from source")
        manifest_records = {}
        completed_source = {}
        ids: set[str] = set()
        mapped: dict[int | str, str] = {}
        stats = {"null_metadata_records": 0, "null_metadata_values": 0, "null_documents": 0,
                 "uuid5_mapped": 0, "json_encoded_values": 0}
        from .model import encode_metadata, ENVELOPE
        import json
        for r in records(src, batch_size):
            key = destination_id(r, target.kind)
            if r.id in ids or key in mapped:
                raise ParityError("Duplicate logical ID or collision after ID mapping; nothing was written")
            ids.add(r.id)
            mapped[key] = r.id
            if progress_path is not None:
                manifest_records[r.id] = {"target_id": key, "sha256": checkpoints.record_hash(r)}
                if r.id in completed:
                    completed_source[r.id] = r
            stats["null_metadata_records"] += r.metadata is None
            stats["null_documents"] += r.document is None
            stats["null_metadata_values"] += sum(v is None for v in (r.metadata or {}).values())
            if target.kind == "qdrant":
                import pickle
                from qdrant_client import models
                from .safety import MAX_PICKLE_BYTES
                point = models.PointStruct(id=key, vector=r.vector, payload=encode_metadata(r, "qdrant"))
                if len(pickle.dumps(point)) > MAX_PICKLE_BYTES:
                    raise ParityError("Encoded Qdrant point exceeds the 64 MiB per-point limit")
            if target.kind == "qdrant" and r.qdrant_id is None:
                import uuid
                if isinstance(key, str):
                    try:
                        uuid.UUID(r.id)
                    except ValueError:
                        stats["uuid5_mapped"] += 1
            if target.kind == "chroma":
                env = json.loads(encode_metadata(r, "chroma")[ENVELOPE])
                stats["json_encoded_values"] += len(env["json"])
        if len(ids) != schema.count:
            raise ParityError("Source count changed or pagination was incomplete")
        if progress_path is not None:
            manifest = {"source": checkpoints.uri_identity(source), "target": checkpoints.uri_identity(target),
                        "schema": asdict(schema), "vector_name": vector_name, "records": manifest_records}
            if progress is not None:
                if dumps(progress["manifest"]) != dumps(manifest) or not completed.issubset(ids):
                    raise ParityError("Checkpoint does not match source contents, ID mapping, target or vector selection")
            else:
                progress = {"version": 1, "manifest": manifest, "completed_ids": [], "parity": None}
        checked_completed = set()
        if target_snapshot:
            existing_ids: set[str] = set()
            for r in records(target_snapshot, batch_size):
                if r.id in existing_ids:
                    raise ParityError("Target contains duplicate logical IDs")
                existing_ids.add(r.id)
                key = destination_id(r, target.kind)
                if r.id in completed:
                    expected = completed_source[r.id]
                    av = normalized(expected.vector) if schema.distance == "cosine" else expected.vector
                    bv = normalized(r.vector) if schema.distance == "cosine" else r.vector
                    if (len(av) != len(bv) or any(abs(a-b) > 1e-5 for a, b in zip(av, bv))
                            or dumps(expected.metadata) != dumps(r.metadata) or expected.document != r.document
                            or destination_id(expected, "qdrant") != destination_id(r, "qdrant")):
                        raise ParityError("Checkpoint completed target record changed; start a new checkpoint to recopy")
                    checked_completed.add(r.id)
                if key in mapped and mapped[key] != r.id:
                    raise ParityError("ID mapping would overwrite a different target logical ID")
                if r.id in ids and mapped.get(key) != r.id:
                    raise ParityError("Target logical ID has a conflicting physical ID")
        if completed != checked_completed:
            raise ParityError("Checkpoint completed target record is missing; start a new checkpoint to recopy")
        report = {"operation": "copy", "dry_run": dry_run, "source_count": schema.count,
                  "copied": 0, "would_copy": len(ids), "dimension": schema.dimension,
                  "distance": schema.distance, "mapping": stats}
        if dry_run:
            return report
        if progress_path is not None:
            report.update(checkpoint=str(progress_path), resumed=resume, skipped=len(completed))
            progress["parity"] = None
            checkpoints.save(progress_path, progress)
        # All source data, mapping collisions and target compatibility were checked on snapshots.
        check_directory(target, required=False)
        with Store(target, vector_name) as dst:
            if dst.collection is None:
                dst.create(schema)
            else:
                current = dst.schema()
                if current.distance != schema.distance or (current.dimension is not None and current.dimension != schema.dimension):
                    raise ParityError("Target schema changed during preflight")
            for batch in src.batches(batch_size):
                pending = [r for r in batch if r.id not in completed]
                if not pending:
                    continue
                dst.write(pending)
                report["copied"] += len(pending)
                if progress_path is not None:
                    completed.update(r.id for r in pending)
                    progress["completed_ids"] = sorted(completed)
                    checkpoints.save(progress_path, progress)
        if progress_path is not None:
            # Close the writer before taking read-only snapshots for parity.
            report["parity"] = verify(source, target, vector_name=vector_name, top_k=top_k, batch_size=batch_size)
            progress["parity"] = report["parity"]
            checkpoints.save(progress_path, progress)
        return report


def inspect(uri: StoreURI, *, batch_size: int = 256, vector_name: str | None = None) -> dict[str, Any]:
    if not 1 <= batch_size <= 4096:
        raise ParityError("batch_size must be in [1,4096]")
    with read_store(uri, vector_name) as store:
        schema = store.schema()
        types: dict[str, set[str]] = {}
        nulls = 0
        docs = 0
        seen: set[str] = set()
        for r in records(store, batch_size):
            if r.id in seen:
                raise ParityError("Duplicate logical IDs in collection")
            seen.add(r.id)
            for k, v in (r.metadata or {}).items():
                types.setdefault(k, set()).add("null" if v is None else type(v).__name__)
            nulls += r.metadata is None
            docs += r.document is not None
        return {"operation": "inspect", "backend": uri.kind, "collection": uri.collection,
                **asdict(schema), "payload_key_types": {k: sorted(v) for k, v in sorted(types.items())},
                "null_metadata_records": nulls, "documents": docs}


def verify(source: StoreURI, target: StoreURI, *, batch_size: int = 256,
           vector_name: str | None = None, tolerance: float = 1e-5,
           sample_queries: int = 10, top_k: int = 10, min_overlap: float = 1.0,
           seed: int = 0) -> dict[str, Any]:
    import math
    _validate_top_k(top_k)
    if not 1 <= batch_size <= 4096 or sample_queries < 0:
        raise ParityError("batch_size must be in [1,4096], sample_queries nonnegative")
    if not math.isfinite(tolerance) or tolerance < 0 or not 0 <= min_overlap <= 1:
        raise ParityError("Tolerance must be finite and nonnegative; min_overlap must be in [0,1]")
    # Reject an invalid target path before opening even the source snapshot.
    checked_store_path(source.path)
    checked_store_path(target.path)
    with read_store(source, vector_name) as src, read_store(target, vector_name) as dst:
        ss, ds = src.schema(), dst.schema()
        if not ss.count:
            raise ParityError("Empty source collection: retrieval parity is undefined")
        left: dict[str, Record] = {}
        right: dict[str, Record] = {}
        for store, index in [(src, left), (dst, right)]:
            for r in records(store, batch_size):
                if r.id in index:
                    raise ParityError("Duplicate logical IDs prevent parity verification")
                index[r.id] = r
        missing = sorted(left.keys() - right.keys())
        extra = sorted(right.keys() - left.keys())
        schema_ok = ss.dimension == ds.dimension and ss.distance == ds.distance
        mismatches: dict[str, list[str]] = {k: [] for k in ("vectors", "metadata", "documents", "id_mapping")}
        max_diff = 0.0
        totals = {k: 0 for k in mismatches}
        for rid in sorted(left.keys() & right.keys()):
            a, b = left[rid], right[rid]
            av = normalized(a.vector) if ss.distance == "cosine" else a.vector
            bv = normalized(b.vector) if ds.distance == "cosine" else b.vector
            diff = max((abs(x-y) for x, y in zip(av, bv)), default=0.0)
            max_diff = max(max_diff, diff)
            bad = {"vectors": len(av) != len(bv) or diff > tolerance,
                   "metadata": dumps(a.metadata) != dumps(b.metadata), "documents": a.document != b.document,
                   "id_mapping": False}
            if source.kind != target.kind:
                cr, qr = (a, b) if source.kind == "chroma" else (b, a)
                bad["id_mapping"] = destination_id(cr, "qdrant") != qr.qdrant_id
            for key, failed in bad.items():
                if failed:
                    totals[key] += 1
                    if len(mismatches[key]) < 20:
                        mismatches[key].append(rid)
        overlaps = []
        if sample_queries and schema_ok and left and right:
            chosen = random.Random(seed).sample(sorted(left), min(sample_queries, len(left)))
            k = min(top_k, len(left))
            for rid in chosen:
                a = set(src.query(left[rid].vector, k))
                b = set(dst.query(left[rid].vector, min(k, len(right))))
                overlaps.append(len(a & b) / k)
        mean = sum(overlaps)/len(overlaps) if overlaps else None
        retrieval_ok = not sample_queries or (mean is not None and mean >= min_overlap)
        count_ok = ss.count == ds.count == len(left) == len(right)
        ok = schema_ok and count_ok and not missing and not extra and not any(totals.values()) and retrieval_ok
        return {"operation": "verify", "ok": ok, "count_ok": count_ok,
                "source_count": ss.count, "target_count": ds.count, "schema_ok": schema_ok,
                "ids_ok": not missing and not extra, "missing_count": len(missing), "extra_count": len(extra),
                "missing_ids": missing[:20], "extra_ids": extra[:20],
                "mismatch_counts": totals, "mismatch_ids": mismatches, "max_abs_diff": max_diff,
                "tolerance": tolerance, "cosine_normalized": ss.distance == "cosine",
                "top_k": {"checked": bool(overlaps), "k": top_k, "samples": len(overlaps),
                          "seed": seed, "overlap_mean": mean, "overlaps": overlaps,
                          "min_overlap": min_overlap, "ok": retrieval_ok}}
