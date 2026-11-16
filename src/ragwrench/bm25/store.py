"""Validated JSONL ingestion and local-only Qdrant integration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import warnings
from collections import Counter
from copy import deepcopy
from pathlib import Path

from qdrant_client import QdrantClient, models

from .encoder import Encoder, token_hash
from .explanations import explain_scores
from .identity import id_key, point_id
from .tokenizers import MAX_TEXT_CHARS, normalize

VECTOR_NAME = "bm25"
MAX_DOCUMENTS = 10_000
MAX_CORPUS_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 4 * 1024 * 1024
MAX_ENCODER_CONFIG_BYTES = 1024 * 1024
MAX_SNAPSHOT_BYTES = 128 * 1024 * 1024


def validate_collection_name(name: str) -> None:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError(
            "collection must be 1–64 ASCII letters/digits/underscore/hyphen, starting alphanumeric"
        )


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON number: {value}")


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_documents(documents: list[dict]) -> None:
    if not documents:
        raise ValueError("corpus contains no documents")
    if len(documents) > MAX_DOCUMENTS:
        raise ValueError(f"corpus exceeds {MAX_DOCUMENTS} documents (local-mode limit)")
    seen = set()
    for line, doc in enumerate(documents, 1):
        prefix = f"record {line}: "
        if not isinstance(doc, dict) or set(doc) - {"id", "text", "metadata"}:
            raise ValueError(prefix + "expected object with id, text and optional metadata only")
        source_id = doc.get("id")
        if (
            type(source_id) not in (str, int)
            or (isinstance(source_id, str) and (not source_id or len(source_id) > 256))
            or (type(source_id) is int and not 0 <= source_id < 2**64)
        ):
            raise ValueError(
                prefix + "id must be a nonempty string (<=256 chars) or uint64 integer"
            )
        if not isinstance(doc.get("text"), str) or len(doc["text"]) > MAX_TEXT_CHARS:
            raise ValueError(
                prefix + f"text must be a string of at most {MAX_TEXT_CHARS} characters"
            )
        if not isinstance(doc.get("metadata", {}), dict):
            raise ValueError(prefix + "metadata must be an object")
        # Also protects Python API ingestion from arbitrary/non-JSON payloads and NaN.
        json.dumps(doc, ensure_ascii=False, allow_nan=False).encode("utf-8")
        key = id_key(source_id)
        if key in seen:
            raise ValueError(prefix + f"duplicate id: {source_id!r}")
        seen.add(key)


def read_corpus(path: str | Path) -> list[dict]:
    documents = []
    total = 0
    with Path(path).open("rb") as stream:
        line_number = 0
        while True:
            raw = stream.readline(MAX_LINE_BYTES + 1)
            if not raw:
                break
            line_number += 1
            total += len(raw)
            if len(raw) > MAX_LINE_BYTES or total > MAX_CORPUS_BYTES:
                raise ValueError("corpus exceeds 4 MiB/line or 64 MiB total")
            if not raw.strip():
                continue
            try:
                doc = json.loads(
                    raw.decode("utf-8"),
                    parse_constant=_reject_constant,
                    object_pairs_hook=_unique_object,
                )
            except (ValueError, UnicodeError, RecursionError) as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            documents.append(doc)
            if len(documents) > MAX_DOCUMENTS:
                raise ValueError(f"corpus exceeds {MAX_DOCUMENTS} documents")
    validate_documents(documents)
    return documents


def read_document(path: str | Path) -> dict:
    """Read one bounded strict JSON source document (not a JSONL corpus)."""
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_LINE_BYTES + 1)
    if len(raw) > MAX_LINE_BYTES:
        raise ValueError("document file exceeds 4 MiB")
    try:
        document = json.loads(
            raw.decode("utf-8"), parse_constant=_reject_constant, object_pairs_hook=_unique_object
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        # Do not echo caller text, metadata or duplicate key names.
        raise ValueError("invalid document JSON") from exc
    validate_documents([document])
    return document


def read_encoder_config(path: str | Path) -> Encoder:
    """Read a bounded, strict, runtime-compatible Encoder.config object."""
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_ENCODER_CONFIG_BYTES + 1)
    if len(raw) > MAX_ENCODER_CONFIG_BYTES:
        raise ValueError("encoder config file exceeds 1 MiB")
    try:
        config = json.loads(
            raw.decode("utf-8"), parse_constant=_reject_constant, object_pairs_hook=_unique_object
        )
        # Also reject overflowed JSON floats and escaped lone surrogates.
        json.dumps(config, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return Encoder.from_config(config)
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError) as exc:
        raise ValueError(f"invalid or incompatible encoder config: {exc}") from exc


def _encode_points(documents: list[dict], encoder: Encoder) -> list[models.PointStruct]:
    vectors = encoder.encode_documents(doc["text"] for doc in documents)
    return [
        models.PointStruct(
            id=point_id(doc["id"]),
            vector={VECTOR_NAME: models.SparseVector(**vector.as_dict())},
            payload={
                "source_id": doc["id"],
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
            },
        )
        for doc, vector in zip(documents, vectors)
        if vector.indices
    ]


def _snapshot_digest(snapshot: dict) -> str:
    content = {key: value for key, value in snapshot.items() if key != "sha256"}
    raw = json.dumps(content, ensure_ascii=False, allow_nan=False, sort_keys=True).encode("utf-8")
    if len(raw) > MAX_SNAPSHOT_BYTES:
        raise ValueError("intended-corpus snapshot exceeds 128 MiB")
    return hashlib.sha256(raw).hexdigest()


def _pending_manifest(collection: str, documents: list[dict], ready: dict, state: str) -> dict:
    snapshot = {"schema": 1, "collection": collection, "documents": documents, "manifest": ready}
    snapshot["sha256"] = _snapshot_digest(snapshot)
    return {**ready, "state": state, "recovery": snapshot}


def _sync_metadata(client: QdrantClient) -> None:
    """Flush the pinned local client's atomically replaced metadata before points.

    In-memory clients have no durability. Use the public constructor options to
    locate the local directory; never change Qdrant's file format or contents.
    """
    path = client.init_options.get("path")
    if path is not None and str(path) != ":memory:":
        directory = Path(path)
        with (directory / "meta.json").open("rb") as stream:
            os.fsync(stream.fileno())
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _finalize(client: QdrantClient, collection: str, ready: dict) -> None:
    if client.get_collection(collection).points_count != ready["indexed"]:
        raise RuntimeError("point count differs from expected; collection remains incomplete")
    # qbm25 is replaced as a whole: the pending snapshot disappears atomically
    # with the ready marker. There is no separate journal-cleanup crash window.
    client.update_collection(collection, metadata={"qbm25": ready})
    _sync_metadata(client)


def build_collection(
    client: QdrantClient,
    collection: str,
    documents: list[dict],
    *,
    tokenizer: str = "cjk-bigram",
    k1: float = 1.2,
    b: float = 0.75,
    avgdl: float | None = None,
    stopwords: tuple[str, ...] = (),
) -> dict:
    validate_collection_name(collection)
    validate_documents(documents)
    if client.collection_exists(collection):
        raise ValueError(
            "collection already exists; use a new collection to rebuild or update to add/delete"
        )
    probe = Encoder(
        tokenizer, k1=k1, b=b, avgdl=avgdl if avgdl is not None else 100, stopwords=stopwords
    )
    lengths = [len(probe.tokenizer(doc["text"])) for doc in documents]
    nonempty_lengths = [length for length in lengths if length]
    if not nonempty_lengths:
        warnings.warn(
            "all documents are empty or stopword-only; skipped", UserWarning, stacklevel=2
        )
        return {"indexed": 0, "skipped": len(documents), "created": False}
    encoder = Encoder(
        tokenizer,
        k1=k1,
        b=b,
        avgdl=avgdl if avgdl is not None else sum(nonempty_lengths) / len(nonempty_lengths),
        stopwords=stopwords,
    )
    points = _encode_points(documents, encoder)
    manifest = {
        "state": "ready",
        "corpus": {
            "schema": 1,
            "avgdl_mode": "auto" if avgdl is None else "fixed",
            "total_tokens": sum(nonempty_lengths),
        },
        "encoder": encoder.config,
        "indexed": len(points),
        "skipped": len(documents) - len(points),
        "collision_stats": encoder.collision_stats,
    }
    client.create_collection(
        collection,
        vectors_config={},
        sparse_vectors_config={
            VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
        metadata={
            "qbm25": _pending_manifest(
                collection,
                [doc for doc, length in zip(documents, lengths) if length],
                manifest,
                "building",
            )
        },
    )
    _sync_metadata(client)
    # Interrupted builds remain explicitly unusable; no metadata point pollutes IDF.
    for start in range(0, len(points), 128):
        client.upsert(collection, points=points[start : start + 128], wait=True)
    _finalize(client, collection, manifest)
    return {"created": True, **manifest}


def load_encoder(client: QdrantClient, collection: str, **overrides) -> Encoder:
    validate_collection_name(collection)
    if not client.collection_exists(collection):
        raise ValueError(f"collection does not exist: {collection}")
    info = client.get_collection(collection)
    return _ready_encoder(info, **overrides)


def _ready_encoder(info, **overrides) -> Encoder:
    """Validate one saved ready collection/config read, shared with verification."""
    sparse = info.config.params.sparse_vectors or {}
    if VECTOR_NAME not in sparse or sparse[VECTOR_NAME].modifier != models.Modifier.IDF:
        raise ValueError("collection must contain named sparse vector 'bm25' with Modifier.IDF")
    manifest = (info.config.metadata or {}).get("qbm25")
    if not isinstance(manifest, dict) or manifest.get("state") != "ready":
        raise ValueError(
            "missing qbm25 metadata or incomplete build/update; use recover with a saved snapshot or rebuild"
        )
    if manifest.get("indexed") != info.points_count:
        raise ValueError(
            "collection point count differs from managed manifest; external edits are unsupported"
        )
    encoder = Encoder.from_config(manifest.get("encoder", {}))
    for key, value in overrides.items():
        if key not in {"tokenizer", "k1", "b", "avgdl", "stopwords"}:
            raise ValueError(f"unsupported query override: {key}")
        if key == "stopwords" and value is not None:
            value = sorted({normalize(word) for word in value})
        if value is not None and encoder.config[key] != value:
            raise ValueError(
                f"query {key} differs from collection config; refusing mismatched encoding"
            )
    return encoder


def validate_delete_ids(ids: list[str | int]) -> None:
    if not isinstance(ids, list):
        raise ValueError("delete IDs must be a JSON list")
    if ids:
        validate_documents([{"id": source_id, "text": ""} for source_id in ids])


def read_delete_ids(path: str | Path) -> list[str | int]:
    """Read a bounded, strict JSON array, preserving integer versus string IDs."""
    return _read_source_ids(path, "delete IDs")


def read_export_ids(path: str | Path) -> list[str | int]:
    """Read a nonempty selection with the same strict typed-ID rules as deletion."""
    ids = _read_source_ids(path, "export IDs")
    if not ids:
        raise ValueError("export IDs must be a nonempty JSON list")
    return ids


def _read_source_ids(path: str | Path, label: str) -> list[str | int]:
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_LINE_BYTES + 1)
    if len(raw) > MAX_LINE_BYTES:
        raise ValueError(f"{label} file exceeds 4 MiB")
    try:
        ids = json.loads(
            raw.decode("utf-8"), parse_constant=_reject_constant, object_pairs_hook=_unique_object
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(ids, list):
        raise ValueError(f"{label} must be a JSON list")
    validate_delete_ids(ids)
    return ids


def _stored_document(record) -> dict:
    """Decode only supported source fields; omitted metadata normalizes to {}."""
    payload = record.payload
    if not isinstance(payload, dict):
        raise ValueError("invalid stored source payload")
    doc = {
        "id": payload.get("source_id"),
        "text": payload.get("text"),
        "metadata": payload.get("metadata", {}),
    }
    validate_documents([doc])
    if str(record.id) != point_id(doc["id"]):
        raise ValueError("stored source ID differs from point ID; rebuild collection")
    return doc


def export_documents(
    client: QdrantClient, collection: str, source_ids: list[str | int]
) -> list[dict]:
    """Reconstruct selected stored sources in ID order, or raise without a result.

    No vectors, encoding, corpus scan or source authenticity guarantee. Enforce
    ingestion limits on the JSONL representation too, before CLI publication.
    """
    if not isinstance(source_ids, list) or not source_ids:
        raise ValueError("export IDs must be a nonempty JSON list")
    validate_delete_ids(source_ids)
    _verification_encoder(client, collection)
    documents = []
    total = 0
    for source_id in source_ids:
        selected = point_id(source_id)
        records = client.retrieve(collection, ids=[selected], with_payload=True, with_vectors=False)
        if not records:
            raise ValueError(f"source ID does not exist: {source_id!r}")
        if len(records) != 1 or str(records[0].id) != selected:
            raise ValueError("unexpected selected point identity or count")
        doc = _stored_document(records[0])
        if id_key(doc["id"]) != id_key(source_id):
            raise ValueError("stored source ID differs from selected ID")
        line = (json.dumps(doc, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        total += len(line)
        if len(line) > MAX_LINE_BYTES or total > MAX_CORPUS_BYTES:
            raise ValueError("export exceeds 4 MiB/line or 64 MiB total")
        documents.append(doc)
    # Return plain independent JSON values, as stored by ingestion.
    return json.loads(json.dumps(documents, ensure_ascii=False, allow_nan=False))


def _stored_documents(client: QdrantClient, collection: str, indexed: int) -> list[dict]:
    documents = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection, offset=offset, limit=128, with_payload=True, with_vectors=False
        )
        for record in records:
            doc = _stored_document(record)
            documents.append(doc)
            if len(documents) > MAX_DOCUMENTS:
                raise ValueError(f"corpus exceeds {MAX_DOCUMENTS} documents")
        if offset is None:
            break
    if len(documents) != indexed:
        raise ValueError("collection changed during update; rebuild collection")
    if documents:
        validate_documents(documents)
    return documents


def update_collection(
    client: QdrantClient,
    collection: str,
    *,
    additions: list[dict] | None = None,
    delete_ids: list[str | int] | None = None,
) -> dict:
    """Apply an add/delete batch using the collection's exact encoder config.

    Deleting and adding the same source ID explicitly replaces that document. All
    validation and encoding precede writes. Once writes start, any failure leaves
    an ``updating`` marker that blocks both search and further updates. This is a
    single-writer local operation. The intended corpus is saved before point writes
    so explicit recovery can replay it after interruption; this is not a transaction.
    """
    additions = [] if additions is None else additions
    delete_ids = [] if delete_ids is None else delete_ids
    if not isinstance(additions, list):
        raise ValueError("additions must be a list of documents")
    if additions:
        validate_documents(additions)
    validate_delete_ids(delete_ids)
    if not additions and not delete_ids:
        raise ValueError("update requires at least one addition or deletion")
    old_encoder = load_encoder(client, collection)
    manifest = deepcopy(client.get_collection(collection).config.metadata["qbm25"])
    corpus = manifest.get("corpus")
    if (
        not isinstance(corpus, dict)
        or corpus.get("schema") != 1
        or corpus.get("avgdl_mode") not in {"auto", "fixed"}
        or type(corpus.get("total_tokens")) is not int
        or corpus["total_tokens"] < 0
    ):
        raise ValueError("missing/invalid corpus statistics; rebuild before incremental updates")
    existing = _stored_documents(client, collection, manifest["indexed"])
    old_lengths = [len(old_encoder.tokenizer(doc["text"])) for doc in existing]
    if (
        any(length == 0 for length in old_lengths)
        or sum(old_lengths) != corpus["total_tokens"]
        or (
            corpus["avgdl_mode"] == "auto"
            and old_encoder.avgdl != (sum(old_lengths) / len(existing) if existing else 1.0)
        )
    ):
        raise ValueError("stored corpus statistics differ from documents; rebuild collection")
    existing_ids = {id_key(doc["id"]) for doc in existing}
    deleting = {id_key(source_id) for source_id in delete_ids}
    for source_id in delete_ids:
        if id_key(source_id) not in existing_ids:
            raise ValueError(f"delete ID does not exist: {source_id!r}")
    retained_ids = existing_ids - deleting
    for doc in additions:
        if id_key(doc["id"]) in retained_ids:
            raise ValueError(f"addition ID already exists: {doc['id']!r}; delete it to replace")
    surviving = [doc for doc in existing if id_key(doc["id"]) not in deleting]
    documents = surviving + additions
    if documents:
        validate_documents(documents)
    lengths = [len(old_encoder.tokenizer(doc["text"])) for doc in documents]
    total_tokens = sum(lengths)
    indexed = sum(length > 0 for length in lengths)
    config = old_encoder.config
    if corpus["avgdl_mode"] == "auto":
        # A positive placeholder keeps an empty collection usable for a later add.
        config["avgdl"] = total_tokens / indexed if indexed else 1.0
    encoder = Encoder.from_config(config)
    points = _encode_points(documents, encoder)
    added = indexed - len(surviving)
    ready = {
        **manifest,
        "state": "ready",
        "encoder": encoder.config,
        "corpus": {**corpus, "total_tokens": total_tokens},
        "indexed": indexed,
        "skipped": len(additions) - added,
        "collision_stats": encoder.collision_stats,
    }
    # All existing TF weights depend on avgdl. If unchanged, only new documents
    # need writing; encode the live corpus regardless to refresh collisions.
    if encoder.avgdl == old_encoder.avgdl:
        adding = {point_id(doc["id"]) for doc in additions}
        points = [point for point in points if point.id in adding]
    pending = _pending_manifest(
        collection, [doc for doc, length in zip(documents, lengths) if length], ready, "updating"
    )
    client.update_collection(collection, metadata={"qbm25": pending})
    _sync_metadata(client)
    if delete_ids:
        client.delete(
            collection,
            points_selector=models.PointIdsList(points=[point_id(i) for i in delete_ids]),
            wait=True,
        )
    for start in range(0, len(points), 128):
        client.upsert(collection, points=points[start : start + 128], wait=True)
    _finalize(client, collection, ready)
    return {**ready, "added": added, "deleted": len(delete_ids)}


def _validate_ready_statistics(ready: dict, encoder: Encoder) -> None:
    """Check saved aggregates without scanning or retokenizing a ready corpus."""
    indexed = ready.get("indexed")
    if type(indexed) is not int or not 0 <= indexed <= MAX_DOCUMENTS:
        raise ValueError("invalid ready collection statistics: indexed count")
    if "corpus" not in ready:
        return  # Pre-update manifests have no saved corpus-policy statistics.
    corpus = ready["corpus"]
    if (
        not isinstance(corpus, dict)
        or type(corpus.get("schema")) is not int
        or corpus["schema"] != 1
        or corpus.get("avgdl_mode") not in ("auto", "fixed")
        or type(corpus.get("total_tokens")) is not int
        or corpus["total_tokens"] < indexed
        or (indexed == 0 and corpus["total_tokens"] != 0)
    ):
        raise ValueError("invalid ready collection statistics: corpus policy or token totals")
    if corpus["avgdl_mode"] == "auto":
        try:
            expected_avgdl = corpus["total_tokens"] / indexed if indexed else 1.0
        except OverflowError as exc:
            raise ValueError("invalid ready collection statistics: token total overflow") from exc
        if encoder.avgdl != expected_avgdl:
            raise ValueError("invalid ready collection statistics: automatic average length")


def status_collection(
    client: QdrantClient,
    collection: str,
    *,
    require_ready: bool = False,
    expect_documents: int | None = None,
) -> dict:
    """Read managed readiness and saved aggregates, without auditing recovery.

    Ready collections reuse search's config/count checks and recovery's saved
    aggregate checks. Pending manifests describe the intended corpus, so their
    live point count need not match. A recovery key's presence says nothing about
    its validity: use recover(dry_run=True) for full validation and replay counts.
    With require_ready=True, valid pending states raise a state-specific
    ValueError after the same validation; ready summaries are unchanged.
    expect_documents requires an exact nonnegative int (None disables the gate),
    validated before client access. After validation and the readiness gate,
    compare the saved intended count, not the live count; mismatch raises ValueError.
    Count agreement is not a source-integrity check.
    No point scan, snapshot checksum/encoding, fsync or mutation is performed.
    """
    if expect_documents is not None and (type(expect_documents) is not int or expect_documents < 0):
        raise ValueError("expect_documents must be a nonnegative integer or None")
    validate_collection_name(collection)
    if not client.collection_exists(collection):
        raise ValueError(f"collection does not exist: {collection}")
    info = client.get_collection(collection)
    sparse = info.config.params.sparse_vectors or {}
    if VECTOR_NAME not in sparse or sparse[VECTOR_NAME].modifier != models.Modifier.IDF:
        raise ValueError("collection must contain named sparse vector 'bm25' with Modifier.IDF")
    manifest = (info.config.metadata or {}).get("qbm25")
    if not isinstance(manifest, dict):
        raise ValueError("missing/invalid qbm25 metadata; cannot report status")
    state = manifest.get("state")
    if state not in ("ready", "building", "updating"):
        raise ValueError("invalid collection state; cannot report status")
    try:
        encoder = (
            load_encoder(client, collection)
            if state == "ready"
            else Encoder.from_config(manifest.get("encoder", {}))
        )
    except OverflowError as exc:
        raise ValueError("invalid collection encoder config: numeric overflow") from exc
    _validate_ready_statistics(manifest, encoder)
    if require_ready and state != "ready":
        raise ValueError(f"collection is not ready: state={state}")
    if expect_documents is not None and manifest["indexed"] != expect_documents:
        raise ValueError(
            f"document count mismatch: expected={expect_documents} actual={manifest['indexed']}"
        )
    corpus = manifest.get("corpus", {})
    # Select only summary fields; snapshots and collision vocabulary contain text.
    return {
        "schema": 1,
        "collection": collection,
        "state": state,
        "config": deepcopy(manifest["encoder"]),
        "corpus": {
            "document_count": manifest["indexed"],
            "avgdl_mode": corpus.get("avgdl_mode"),
            "total_tokens": corpus.get("total_tokens"),
        },
        "points": {"current": info.points_count},
        "recovery_snapshot_present": "recovery" in manifest,
    }


def verify_document(client: QdrantClient, collection: str, document: dict) -> dict:
    """Compare one caller-supplied source with its selected stored point, without writes.

    Uses saved encoding, including avgdl; does not recompute corpus statistics.
    This is neither document authentication nor a whole-corpus integrity audit.
    Missing points are mismatches; unavailable/malformed vectors are errors.
    """
    validate_documents([document])
    # Match ingestion's JSON payload conversion without modifying caller objects.
    document = json.loads(json.dumps(document, ensure_ascii=False, allow_nan=False))
    encoder = _verification_encoder(client, collection)
    return _compare_document(client, collection, document, encoder)


def _verification_encoder(
    client: QdrantClient, collection: str, *, supplied_count: int | None = None
) -> Encoder:
    validate_collection_name(collection)
    if not client.collection_exists(collection):
        raise ValueError(f"collection does not exist: {collection}")
    info = client.get_collection(collection)
    try:
        encoder = _ready_encoder(info)
    except OverflowError as exc:
        raise ValueError("invalid collection encoder config: numeric overflow") from exc
    _validate_ready_statistics(info.config.metadata["qbm25"], encoder)
    indexed = info.config.metadata["qbm25"]["indexed"]
    if supplied_count is not None and supplied_count != indexed:
        raise ValueError(f"document count mismatch: expected={indexed} actual={supplied_count}")
    return encoder


def verify_documents(
    client: QdrantClient,
    collection: str,
    documents: list[dict],
    *,
    require_complete: bool = False,
) -> dict:
    """Compare a bounded nonempty source batch, returning all results or raising.

    Validate the full input before client access, and saved readiness/config once.
    Results follow input order and reuse single-document comparison semantics.
    No corpus scan, statistic recomputation, writes or partial result stream.
    File byte bounds belong to read_corpus, as with ingestion's Python API.
    require_complete must be an exact bool. When true, require the supplied count
    to equal the validated saved indexed count before any selected comparison.
    Empty batches remain invalid, including for a ready zero-indexed collection.
    Count agreement is not source authentication or a global-statistics audit.
    """
    if type(require_complete) is not bool:
        raise ValueError("require_complete must be a boolean")
    validate_documents(documents)
    documents = json.loads(json.dumps(documents, ensure_ascii=False, allow_nan=False))
    encoder = _verification_encoder(
        client, collection, supplied_count=len(documents) if require_complete else None
    )
    results = [_compare_document(client, collection, doc, encoder) for doc in documents]
    mismatched = sum(not result["matches"] for result in results)
    counts = Counter(category for result in results for category in result["mismatches"])
    return {
        "schema": 1,
        "collection": collection,
        "matches": mismatched == 0,
        "totals": {
            "documents": len(results),
            "matched": len(results) - mismatched,
            "mismatched": mismatched,
            "missing": counts["point_missing"],
        },
        "mismatch_counts": {
            category: counts[category]
            for category in (
                "point_missing",
                "source_id",
                "text",
                "metadata",
                "sparse_indices",
                "sparse_values",
            )
        },
        "results": results,
    }


def _compare_document(
    client: QdrantClient, collection: str, document: dict, encoder: Encoder
) -> dict:
    """Compare one validated source with an already validated saved encoder."""
    with warnings.catch_warnings():
        # Collision warnings contain tokens; the report must not disclose text.
        warnings.simplefilter("ignore", UserWarning)
        expected = encoder.encode_documents([document["text"]])[0]
    selected = point_id(document["id"])
    records = client.retrieve(
        collection, ids=[selected], with_payload=True, with_vectors=[VECTOR_NAME]
    )
    report = {
        "schema": 1,
        "collection": collection,
        "id": document["id"],
        "point_present": bool(records),
        "payload": None,
        "sparse_vector": None,
        "matches": False,
        "mismatches": [],
    }
    if not records:
        report["mismatches"] = ["point_missing"]
        return report
    if len(records) != 1 or str(records[0].id) != selected:
        raise ValueError("unexpected selected point identity/cardinality; cannot verify document")
    record = records[0]
    vector = record.vector.get(VECTOR_NAME) if isinstance(record.vector, dict) else None
    if (
        not isinstance(vector, models.SparseVector)
        or len(vector.indices) != len(vector.values)
        or any(type(index) is not int or not 0 <= index < 2**32 for index in vector.indices)
        or len(set(vector.indices)) != len(vector.indices)
        or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in vector.values
        )
    ):
        raise ValueError("missing/invalid stored bm25 sparse vector; cannot verify document")
    payload = record.payload or {}
    if not isinstance(payload, dict):
        raise ValueError("invalid stored payload; cannot verify document")
    expected_payload = {
        "source_id": document["id"],
        "text": document["text"],
        "metadata": document.get("metadata", {}),
    }
    try:
        fields = {
            field: field in payload
            and json.dumps(payload[field], sort_keys=True, ensure_ascii=False, allow_nan=False)
            == json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
            for field, value in expected_payload.items()
        }
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid stored JSON payload; cannot verify document") from exc
    report["payload"] = {**fields, "matches": all(fields.values())}
    stored_weights = dict(zip(vector.indices, vector.values))
    expected_weights = dict(zip(expected.indices, expected.values))
    indices_match = stored_weights.keys() == expected_weights.keys()
    # Pinned local Qdrant preserves sparse doubles across persistence. Compare
    # exact weights, not rounded query scores, and ignore array ordering.
    values_match = stored_weights == expected_weights
    report["sparse_vector"] = {
        "indices_match": indices_match,
        "values_match": values_match,
        "matches": indices_match and values_match,
    }
    report["mismatches"] = [field for field, matches in fields.items() if not matches]
    if not indices_match:
        report["mismatches"].append("sparse_indices")
    if not values_match:
        report["mismatches"].append("sparse_values")
    report["matches"] = not report["mismatches"]
    return report


def _recovery_summary(
    collection: str, state: str, ready: dict, current: int, remove: int, rewrite: int
) -> dict:
    # Select fields explicitly: snapshots and collision vocabularies contain text.
    corpus = ready.get("corpus") or {}
    return {
        "schema": 1,
        "collection": collection,
        "dry_run": True,
        "state": state,
        "pending": state != "ready",
        "config": deepcopy(ready["encoder"]),
        "corpus": {
            "document_count": ready["indexed"],
            "avgdl_mode": corpus.get("avgdl_mode"),
            "total_tokens": corpus.get("total_tokens"),
        },
        "points": {"current": current, "remove": remove, "rewrite": rewrite},
    }


def recover_collection(client: QdrantClient, collection: str, *, dry_run: bool = False) -> dict:
    """Replay a validated intended corpus, restarting safely after any point batch.

    No input files or partial live payloads are used to infer intent. Legacy
    interrupted collections without a snapshot cannot be recovered this way.
    A ready managed collection is a read-only, idempotent no-op. With dry_run,
    validate and plan the same replay, then return a schema-1 summary before any
    fsync or mutation. Counts describe operations, including all target upserts,
    not a diff of payloads/vectors. No document text or metadata is returned.
    """
    validate_collection_name(collection)
    if not client.collection_exists(collection):
        raise ValueError(f"collection does not exist: {collection}")
    info = client.get_collection(collection)
    sparse = info.config.params.sparse_vectors or {}
    if VECTOR_NAME not in sparse or sparse[VECTOR_NAME].modifier != models.Modifier.IDF:
        raise ValueError("collection must contain named sparse vector 'bm25' with Modifier.IDF")
    pending = (info.config.metadata or {}).get("qbm25")
    if not isinstance(pending, dict):
        raise ValueError("missing qbm25 metadata; cannot recover")
    if pending.get("state") == "ready":
        encoder = load_encoder(client, collection)
        _validate_ready_statistics(pending, encoder)
        if dry_run:
            return _recovery_summary(collection, "ready", pending, info.points_count, 0, 0)
        return {**pending, "recovered": False}
    if pending.get("state") not in {"building", "updating"}:
        raise ValueError("invalid collection state; cannot recover")
    snapshot = pending.get("recovery")
    if not isinstance(snapshot, dict):
        raise ValueError("missing intended-corpus snapshot; rebuild into a new collection")
    if (
        set(snapshot) != {"schema", "collection", "documents", "manifest", "sha256"}
        or type(snapshot["schema"]) is not int
        or snapshot["schema"] != 1
        or snapshot["collection"] != collection
        or snapshot["sha256"] != _snapshot_digest(snapshot)
    ):
        raise ValueError("invalid intended-corpus snapshot schema, collection or checksum")
    ready = snapshot["manifest"]
    documents = snapshot["documents"]
    if (
        not isinstance(ready, dict)
        or ready.get("state") != "ready"
        or "recovery" in ready
        or {k: v for k, v in pending.items() if k not in {"state", "recovery"}}
        != {k: v for k, v in ready.items() if k != "state"}
        or not isinstance(documents, list)
    ):
        raise ValueError("invalid intended-corpus snapshot manifest")
    if documents:
        validate_documents(documents)
    encoder = Encoder.from_config(ready.get("encoder", {}))
    corpus = ready.get("corpus")
    lengths = [len(encoder.tokenizer(doc["text"])) for doc in documents]
    if (
        not isinstance(corpus, dict)
        or corpus.get("schema") != 1
        or corpus.get("avgdl_mode") not in {"auto", "fixed"}
        or type(corpus.get("total_tokens")) is not int
        or corpus["total_tokens"] != sum(lengths)
        or any(length == 0 for length in lengths)
        or type(ready.get("indexed")) is not int
        or ready["indexed"] != len(documents)
        or type(ready.get("skipped")) is not int
        or not 0 <= ready["skipped"] <= MAX_DOCUMENTS
        or (
            corpus["avgdl_mode"] == "auto"
            and encoder.avgdl != (sum(lengths) / len(lengths) if lengths else 1.0)
        )
    ):
        raise ValueError("intended-corpus snapshot statistics differ from documents")
    points = _encode_points(documents, encoder)
    if ready.get("collision_stats") != encoder.collision_stats:
        raise ValueError("intended-corpus snapshot collision statistics differ from documents")
    # Collect obsolete IDs before deleting so scroll offsets cannot shift. Full
    # upserts repair replacements and avgdl rewrites, even if a prior batch was
    # only partly committed. All validation and encoding precede these writes.
    wanted = {point.id for point in points}
    obsolete = []
    count, offset = 0, None
    while True:
        records, offset = client.scroll(
            collection, offset=offset, limit=128, with_payload=False, with_vectors=False
        )
        count += len(records)
        if count > MAX_DOCUMENTS:
            raise ValueError(f"corpus exceeds {MAX_DOCUMENTS} documents; cannot recover")
        obsolete.extend(record.id for record in records if record.id not in wanted)
        if offset is None:
            break
    if dry_run:
        return _recovery_summary(
            collection, pending["state"], ready, count, len(obsolete), len(points)
        )
    _sync_metadata(client)
    for start in range(0, len(obsolete), 128):
        client.delete(
            collection,
            points_selector=models.PointIdsList(points=obsolete[start : start + 128]),
            wait=True,
        )
    for start in range(0, len(points), 128):
        client.upsert(collection, points=points[start : start + 128], wait=True)
    _finalize(client, collection, ready)
    return {**ready, "recovered": True}


def search_collection(
    client: QdrantClient, collection: str, query: str, *, top_k: int = 10, **overrides
) -> list[dict]:
    if type(top_k) is not int or not 1 <= top_k <= 1000:
        raise ValueError("top-k must be between 1 and 1000")
    encoder = load_encoder(client, collection, **overrides)
    vector = encoder.encode_query(query)
    if not vector.indices:
        return []
    hits = client.query_points(
        collection,
        query=models.SparseVector(**vector.as_dict()),
        using=VECTOR_NAME,
        limit=top_k,
        with_payload=True,
    ).points
    results = [
        {
            "id": hit.payload["source_id"],
            "score": float(hit.score),
            "text": hit.payload["text"],
            "metadata": hit.payload.get("metadata", {}),
        }
        for hit in hits
        if hit.score > 0
    ]
    # Stable order within the returned tie set; Qdrant chooses ties at the cutoff.
    return sorted(results, key=lambda hit: (-hit["score"], id_key(hit["id"])))


def explain_collection(
    client: QdrantClient,
    collection: str,
    query: str,
    *,
    top_k: int = 10,
    source_id: str | int | None = None,
    **overrides,
) -> dict:
    """Explain actual search results using live stored sparse-vector statistics.

    This opt-in, read-only operation scans the full bounded local collection.
    Results retain Qdrant's scores and cutoff choices; explanations do not rerank.
    An optional typed source_id adds selected_document, even outside top-k or at
    zero score. None disables selection. Unknown IDs fail rather than imply zero.
    As with updates, use a single writer and do not externally mutate the store.
    """
    if source_id is not None:
        validate_documents([{"id": source_id, "text": ""}])
    encoder = load_encoder(client, collection, **overrides)
    hits = search_collection(client, collection, query, top_k=top_k, **overrides)
    expected = client.get_collection(collection).points_count
    if expected > MAX_DOCUMENTS:
        raise ValueError(f"corpus exceeds {MAX_DOCUMENTS} documents (local-mode limit)")
    wanted = {point_id(hit["id"]) for hit in hits}
    selected = None
    if source_id is not None:
        selected_point = point_id(source_id)
        records = client.retrieve(
            collection, ids=[selected_point], with_payload=True, with_vectors=False
        )
        if not records:
            raise ValueError(f"source ID does not exist: {source_id!r}")
        payload = records[0].payload or {}
        doc = {
            "id": payload.get("source_id"),
            "text": payload.get("text"),
            "metadata": payload.get("metadata", {}),
        }
        validate_documents([doc])
        if str(records[0].id) != selected_point or id_key(doc["id"]) != id_key(source_id):
            raise ValueError("stored source ID differs from point ID; rebuild collection")
        selected = {**doc, "score": 0.0}
        wanted.add(selected_point)
    query_indices = {token_hash(token) for token in encoder.tokenizer(query)}
    frequencies: Counter[int] = Counter()
    vectors = {}
    count = 0
    offset = None
    while True:
        records, offset = client.scroll(
            collection, offset=offset, limit=128, with_payload=False, with_vectors=[VECTOR_NAME]
        )
        for record in records:
            count += 1
            if count > MAX_DOCUMENTS:
                raise ValueError(f"corpus exceeds {MAX_DOCUMENTS} documents (local-mode limit)")
            vector = (record.vector or {}).get(VECTOR_NAME)
            if not isinstance(vector, models.SparseVector) or not vector.indices:
                raise ValueError("missing stored bm25 sparse vector; rebuild collection")
            frequencies.update(set(vector.indices) & query_indices)
            if str(record.id) in wanted:
                vectors[str(record.id)] = vector
        if offset is None:
            break
    if count != expected or set(vectors) != wanted:
        raise ValueError("collection changed during explanation; retry with a single writer")
    if selected is not None and query_indices:
        # The filter selects one point without narrowing Qdrant's global IDF corpus.
        # Retain its actual float32 score, just like ordinary search results.
        scored = client.query_points(
            collection,
            query=models.SparseVector(**encoder.encode_query(query).as_dict()),
            using=VECTOR_NAME,
            query_filter=models.Filter(must=[models.HasIdCondition(has_id=[selected_point])]),
            limit=1,
            with_payload=False,
        ).points
        if scored and scored[0].score > 0:
            selected["score"] = float(scored[0].score)
    report = explain_scores(
        encoder,
        query,
        hits + ([selected] if selected is not None else []),
        count,
        frequencies,
        vectors,
    )
    if selected is not None:
        report["selected_document"] = report["results"].pop()
    return report


def open_local(path: str | Path, *, must_exist: bool = False) -> QdrantClient:
    raw = str(path)
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:/", raw) or raw == ":memory:":
        raise ValueError("--path must be a local directory; URLs and :memory: are unsupported")
    directory = Path(path).expanduser().resolve()
    if must_exist and not (directory / "meta.json").is_file():
        raise ValueError(f"Qdrant local store does not exist: {directory}")
    return QdrantClient(path=str(directory))
