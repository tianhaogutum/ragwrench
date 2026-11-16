"""Portable, deterministic snapshots and self-contained change plans."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unicodedata
import uuid
from collections import Counter, defaultdict, deque

VERSION = 1
NAMESPACE = "e3ef9618-c611-52aa-b041-c6c945dfb9c3"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_CHUNKS = 100_000


class SyncError(ValueError):
    """Invalid input or unsafe operation; safe to display to a CLI user."""


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def normalize(text):
    if not isinstance(text, str):
        raise SyncError("text must be a string")
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def load_json(path):
    try:
        if not Path(path).is_file():
            raise SyncError("JSON input must be a regular file")
        if Path(path).stat().st_size > MAX_JSON_BYTES:
            raise SyncError("JSON input exceeds 64 MiB limit")
        return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda x: _invalid(x))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SyncError(f"cannot read JSON {path}: {exc}") from exc


def _invalid(value):
    raise SyncError(f"invalid JSON constant: {value}")


def write_json(path, data, *, durable=False):
    """Atomic replacement; recovery writes also sync file and directory on Linux.

    A post-rename error leaves the complete new file published but does not
    acknowledge durability. Never suppress an unsupported/failed fsync.
    Ordinary exported JSON keeps its portable, atomic-only behavior.
    """
    path = Path(path)
    encoded = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
        raise SyncError("output exceeds 64 MiB limit; split the dataset")
    directory_fd, tmp = None, None
    try:
        if durable:
            if sys.platform != "linux":
                raise SyncError("durable recovery writes require a local Linux filesystem with file/directory fsync")
            # Open before publishing anything; fail closed if unavailable.
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write(encoded)
            if durable:
                out.flush()
                os.fsync(out.fileno())
        os.replace(tmp, path)
        if durable:
            os.fsync(directory_fd)
    finally:
        try:
            if tmp is not None and os.path.exists(tmp):
                os.unlink(tmp)
        finally:
            if directory_fd is not None:
                os.close(directory_fd)


def chunk_text(text, method="cdc", min_size=128, target_size=256, max_size=512):
    """Character sizes; CDC anchors use a rolling 64-character content window.

    Prefer paragraph/sentence/word boundaries; max_size is a hard cap even for
    unbroken input. The final chunk may be shorter than min_size.
    """
    if method not in {"fixed", "cdc"}:
        raise SyncError("chunker must be fixed or cdc")
    if not all(type(x) is int for x in (min_size, target_size, max_size)) or not 1 <= min_size <= target_size <= max_size:
        raise SyncError("require 1 <= min-size <= target-size <= max-size")
    text = normalize(text)
    if not text:
        return []
    if method == "fixed":
        if (len(text) + target_size - 1) // target_size > MAX_CHUNKS:
            raise SyncError("chunk count exceeds 100,000 limit")
        return [part for i in range(0, len(text), target_size) if (part := normalize(text[i:i + target_size]))]
    result, start = [], 0
    for end in range(1, len(text) + 1):
        size = end - start
        if size < min_size:
            continue
        # A sliding content window (not an offset) determines each anchor. A
        # local insertion therefore stops influencing anchors after 64 chars.
        paragraph = text[max(0, end - 2):end] == "\n\n"
        sentence = end > 1 and text[end - 1].isspace() and text[end - 2] in ".!?。！？"
        word = text[end - 1].isspace()
        boundary = paragraph or sentence or word
        anchor = False
        if boundary:
            h = int.from_bytes(hashlib.blake2b(text[max(0, end - 64):end].encode(), digest_size=8).digest(), "big")
            # Prefer aligned boundaries while giving ordinary word breaks
            # enough chance to anchor prose lacking punctuation.
            weight = 64 if paragraph else 32 if sentence else 6
            anchor = h % target_size < weight
        if size >= max_size or anchor:
            part = normalize(text[start:end])
            if part:
                result.append(part)
                if len(result) > MAX_CHUNKS:
                    raise SyncError("chunk count exceeds 100,000 limit")
            start = end
    tail = normalize(text[start:])
    if tail:
        result.append(tail)
    if len(result) > MAX_CHUNKS:
        raise SyncError("chunk count exceeds 100,000 limit")
    return result


def _object(value, name):
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise SyncError(f"{name} must be a JSON object")
    try:
        canonical(value)
    except (ValueError, TypeError) as exc:
        raise SyncError(f"invalid {name}: {exc}") from exc
    return value


def content_hash(text, embed_metadata, embed_model):
    return digest({"text": text, "embed_metadata": embed_metadata, "embed_model": embed_model})


def point_id(namespace, doc_id, content, occurrence):
    return str(uuid.uuid5(uuid.UUID(namespace), canonical([doc_id, content, occurrence])))


def payload(chunk):
    return {k: chunk[k] for k in ("doc_id", "text", "metadata", "embed_metadata", "index", "content_hash", "occurrence")}


def build_snapshot(records, *, embed_model="hash-v1", namespace=NAMESPACE, chunking=None):
    try:
        namespace = str(uuid.UUID(namespace))
    except (ValueError, AttributeError, TypeError) as exc:
        raise SyncError("namespace must be a UUID") from exc
    if not isinstance(embed_model, str) or not embed_model.strip():
        raise SyncError("embed_model must be a nonempty string")
    counts, positions, chunks = Counter(), Counter(), []
    total_bytes = 0
    for record in records:
        if len(chunks) >= MAX_CHUNKS:
            raise SyncError("chunk count exceeds 100,000 limit")
        if not isinstance(record, dict) or set(record) - {"doc_id", "text", "metadata", "embed_metadata"}:
            raise SyncError("chunk rows accept only doc_id, text, metadata, embed_metadata")
        doc = record.get("doc_id")
        if not isinstance(doc, str) or not doc.strip():
            raise SyncError("doc_id must be a nonempty string")
        text = normalize(record.get("text"))
        if not text:
            raise SyncError("chunk text must not be empty")
        meta = _object(record.get("metadata", {}), "metadata")
        emeta = _object(record.get("embed_metadata", {}), "embed_metadata")
        total_bytes += len(canonical(record).encode("utf-8"))
        if total_bytes > MAX_JSON_BYTES:
            raise SyncError("chunk input exceeds 64 MiB limit")
        chash = content_hash(text, emeta, embed_model)
        occurrence = counts[doc, chash]
        chunk = {"doc_id": doc, "text": text, "metadata": meta, "embed_metadata": emeta,
                 "index": positions[doc], "content_hash": chash, "occurrence": occurrence,
                 "id": point_id(namespace, doc, chash, occurrence)}
        chunk["payload_hash"] = digest(payload(chunk))
        chunks.append(chunk)
        counts[doc, chash] += 1
        positions[doc] += 1
    chunks.sort(key=lambda c: (c["doc_id"], c["index"]))
    manifest = {"kind": "chunksync.snapshot", "version": VERSION, "namespace": namespace,
                "embed_model": embed_model, "chunking": chunking or {"method": "provided"}, "chunks": chunks}
    manifest["fingerprint"] = digest(manifest)
    return manifest


def snapshot_dir(directory, *, method="cdc", min_size=128, target_size=256, max_size=512, **kwargs):
    chunk_text("", method, min_size, target_size, max_size)
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise SyncError("--dir must be a real directory, not a symlink")
    root = root.resolve()
    records = []
    total_bytes = 0
    # No symlinks, special files or unsupported parsers: never follow input
    # paths to outside data or silently index binary formats.
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise SyncError(f"symlink input refused: {path.relative_to(root)}")
        if path.is_dir():
            continue
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            raise SyncError(f"unsupported input (only UTF-8 .md/.txt): {path.relative_to(root)}")
        total_bytes += path.stat().st_size
        if total_bytes > MAX_JSON_BYTES:
            raise SyncError("document set exceeds 64 MiB limit")
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise SyncError(f"cannot read {path.name}: {exc}") from exc
        for part in chunk_text(text, method, min_size, target_size, max_size):
            records.append({"doc_id": path.relative_to(root).as_posix(), "text": part})
            if len(records) > MAX_CHUNKS:
                raise SyncError("chunk count exceeds 100,000 limit")
    return build_snapshot(records, chunking={"method": method, "min_size": min_size,
                                            "target_size": target_size, "max_size": max_size}, **kwargs)


def snapshot_jsonl(path, **kwargs):
    if not Path(path).is_file():
        raise SyncError("JSONL input must be a regular file")
    if Path(path).stat().st_size > MAX_JSON_BYTES:
        raise SyncError("JSONL input exceeds 64 MiB limit")

    def rows():
        with Path(path).open(encoding="utf-8") as source:
            for n, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line, parse_constant=lambda x: _invalid(x))
                except json.JSONDecodeError as exc:
                    raise SyncError(f"invalid JSONL at line {n}: {exc}") from exc
    return build_snapshot(rows(), **kwargs)


def validate_snapshot(value):
    try:
        required = {"kind", "version", "namespace", "embed_model", "chunking", "chunks", "fingerprint"}
        if not isinstance(value, dict) or set(value) != required or value["kind"] != "chunksync.snapshot" or type(value["version"]) is not int or value["version"] != VERSION:
            raise SyncError("invalid snapshot schema/version (expected version 1)")
        _object(value["chunking"], "chunking")
        if not isinstance(value["chunks"], list):
            raise SyncError("snapshot chunks must be an array")
        records = []
        for c in value["chunks"]:
            if set(c) != {"id", "doc_id", "text", "metadata", "embed_metadata", "index", "content_hash", "occurrence", "payload_hash"}:
                raise SyncError("invalid chunk schema")
            if type(c["index"]) is not int or type(c["occurrence"]) is not int:
                raise SyncError("chunk index and occurrence must be integers")
            records.append({k: c[k] for k in ("doc_id", "text", "metadata", "embed_metadata")})
        expected = build_snapshot(records, embed_model=value["embed_model"], namespace=value["namespace"], chunking=value["chunking"])
        if canonical(value) != canonical(expected):
            raise SyncError("snapshot hash, id, normalization or ordering mismatch")
        return value
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        if isinstance(exc, SyncError):
            raise
        raise SyncError(f"invalid snapshot: {exc}") from exc


def make_plan(old, new):
    validate_snapshot(old)
    validate_snapshot(new)
    if old["namespace"] != new["namespace"]:
        raise SyncError("snapshot namespaces differ")
    previous = {c["id"]: c for c in old["chunks"]}
    current = {c["id"]: c for c in new["chunks"]}
    unchanged, payload_only, moved, upsert = [], [], [], []
    removed = defaultdict(deque)
    # Deterministic one-to-one matching permits rename/reorder without embedding.
    for c in old["chunks"]:
        if c["id"] not in current:
            removed[c["content_hash"]].append(c)
    for c in new["chunks"]:
        before = previous.get(c["id"])
        if before is not None:
            if before["payload_hash"] == c["payload_hash"]:
                unchanged.append(c)
            elif all(before[k] == c[k] for k in ("text", "metadata", "embed_metadata")):
                moved.append({"before": before, "after": c})
            else:
                payload_only.append(c)
        elif removed[c["content_hash"]]:
            moved.append({"before": removed[c["content_hash"]].popleft(), "after": c})
        else:
            upsert.append(c)
    deleted = sorted((c for queue in removed.values() for c in queue), key=lambda c: c["id"])
    avoided = len(new["chunks"]) - len(upsert)
    summary = {"upsert": len(upsert), "delete": len(deleted), "unchanged": len(unchanged),
               "payload_only": len(payload_only), "moved": len(moved), "new_chunks": len(new["chunks"]),
               "embeddings_avoided": avoided, "embed_model_changed": old["embed_model"] != new["embed_model"]}
    plan = {"kind": "chunksync.plan", "version": VERSION, "old": old, "new": new,
            "upsert": upsert, "delete": deleted, "unchanged": unchanged, "payload_only": payload_only,
            "moved": moved, "summary": summary}
    plan["fingerprint"] = digest(plan)
    return plan


def validate_plan(value):
    try:
        if not isinstance(value, dict) or value.get("kind") != "chunksync.plan" or type(value.get("version")) is not int or value["version"] != VERSION:
            raise SyncError("invalid plan schema/version")
        if canonical(make_plan(value["old"], value["new"])) != canonical(value):
            raise SyncError("plan hash or operations mismatch; regenerate it")
        return value
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, SyncError):
            raise
        raise SyncError(f"invalid plan: {exc}") from exc


def has_changes(plan):
    return plan["old"]["fingerprint"] != plan["new"]["fingerprint"]
