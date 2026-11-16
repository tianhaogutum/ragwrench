"""Versioned deterministic JSONL bundles; all hashes are over exact UTF-8 bytes."""
from importlib.resources import files
import json

from jsonschema import Draft202012Validator

from .common import PortError, atomic_bytes, atomic_json, content_hash, digest, dumps, identifier, read_json, safe_file

MAX_FILE = 256 * 1024 * 1024
MAX_LINE = 2 * 1024 * 1024


def schema():
    return json.loads(files("ragwrench.port").joinpath("bundle.schema.json").read_text())


def check_schema(value):
    errors = sorted(Draft202012Validator(schema()).iter_errors(value), key=lambda e: str(e.path))
    if errors:
        error = errors[0]
        # Do not echo the rejected instance (it may contain private content).
        raise PortError(f"Bundle schema violation at {'/'.join(map(str, error.absolute_path))}: {error.validator}")


def write_bundle(path, source, settings, documents, chunks):
    path = safe_file(path)
    path.mkdir(parents=True, exist_ok=True)
    documents = sorted(documents, key=lambda d: d["id"])
    chunks = sorted(chunks, key=lambda c: (c["doc_id"], c["position"], c["id"]))
    blobs = {name: "".join(dumps(row) + "\n" for row in rows).encode("utf-8")
             for name, rows in (("documents.jsonl", documents), ("chunks.jsonl", chunks))}
    for name, blob in blobs.items():
        if len(blob) > MAX_FILE or any(len(line) > MAX_LINE for line in blob.splitlines()):
            raise PortError(f"{name} exceeds portable file/line byte limits")
    manifest = {"format": "kbbundle/1", "source": source, "settings": settings,
                "counts": {"documents": len(documents), "chunks": len(chunks)},
                "sha256": {name: digest(blob) for name, blob in blobs.items()}}
    validate_records(manifest, documents, chunks)
    for name, blob in blobs.items():
        atomic_bytes(path / name, blob)
    atomic_json(path / "manifest.json", manifest)  # commit marker is written last
    return manifest


def load_bundle(path):
    path = safe_file(path)
    manifest = read_json(path / "manifest.json")
    check_schema({"manifest": manifest, "documents": [], "chunks": []})
    rows = {}
    for name in ("documents.jsonl", "chunks.jsonl"):
        file = safe_file(path / name)
        if file.stat().st_size > MAX_FILE:
            raise PortError(f"{name} exceeds 256 MiB safety limit")
        blob = file.read_bytes()
        if digest(blob) != manifest["sha256"][name]:
            raise PortError(f"SHA256 mismatch: {name}")
        result = []
        for line in blob.splitlines():
            if not line or len(line) > MAX_LINE:
                raise PortError(f"Empty or oversized JSONL line: {name}")
            try:
                result.append(json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError())))
            except (ValueError, UnicodeError) as exc:
                raise PortError(f"Invalid JSONL: {name}") from exc
        rows[name] = result
    docs, chunks = rows["documents.jsonl"], rows["chunks.jsonl"]
    validate_records(manifest, docs, chunks)
    return manifest, docs, chunks


def validate_records(manifest, documents, chunks):
    check_schema({"manifest": manifest, "documents": documents, "chunks": chunks})
    if manifest["counts"] != {"documents": len(documents), "chunks": len(chunks)}:
        raise PortError("Manifest counts do not match JSONL records")
    identifier(manifest["source"]["dataset_id"])
    ids = {identifier(doc["id"]) for doc in documents}
    if len(ids) != len(documents):
        raise PortError("Duplicate document ID")
    chunk_ids, positions = set(), set()
    for chunk in chunks:
        identifier(chunk["id"])
        identifier(chunk["doc_id"])
        if chunk["doc_id"] not in ids:
            raise PortError("Chunk references an unknown document")
        key = (chunk["doc_id"], chunk["id"])
        pos = (chunk["doc_id"], chunk["position"])
        if key in chunk_ids or pos in positions:
            raise PortError("Duplicate chunk ID or position within a document")
        chunk_ids.add(key)
        positions.add(pos)
        if content_hash(chunk["content"]) != chunk["content_sha256"]:
            raise PortError("Chunk content_sha256 mismatch")
        if not chunk["content"].strip():
            raise PortError("Whitespace-only chunks cannot be imported")
    if documents != sorted(documents, key=lambda d: d["id"]):
        raise PortError("Documents are not in canonical ID order")
    if chunks != sorted(chunks, key=lambda c: (c["doc_id"], c["position"], c["id"])):
        raise PortError("Chunks are not in canonical document/position/ID order")


def fingerprint(manifest):
    return digest(dumps(manifest).encode())
