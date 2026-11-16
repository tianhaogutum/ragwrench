"""Explicitly selected embedders; manifests and plans never contain commands."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

from .core import MAX_JSON_BYTES, SyncError, canonical, digest


def checked_vector(vector, dimension):
    if not isinstance(vector, list) or len(vector) != dimension:
        raise SyncError(f"vector dimension mismatch: expected {dimension}")
    if any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1e30 for x in vector):
        raise SyncError("vectors must contain finite numbers with magnitude <= 1e30")
    return [float(x) for x in vector]


class HashEmbedder:
    """A reproducible test embedding, deliberately not a semantic ML model."""
    def __init__(self, dimension=32):
        self.dimension = dimension
        self.signature = digest({"type": "hash-v1", "dimension": dimension})
        self.calls = 0

    def embed(self, chunk, model):
        self.calls += 1
        seed = canonical([chunk["text"], chunk["embed_metadata"], model]).encode()
        values = []
        for index in range(self.dimension):
            raw = hashlib.sha256(seed + index.to_bytes(4, "big")).digest()
            values.append(int.from_bytes(raw[:4], "big") / (2**31) - 1)
        norm = math.sqrt(sum(v * v for v in values)) or 1
        return [v / norm for v in values]


class CommandEmbedder:
    def __init__(self, argv, dimension=32, timeout=30):
        if not isinstance(argv, list) or not argv or any(not isinstance(s, str) or not s or "\0" in s for s in argv):
            raise SyncError("--embed-command must be a JSON array of nonempty argv strings")
        if not math.isfinite(timeout) or timeout <= 0:
            raise SyncError("embedder timeout must be positive and finite")
        self.argv, self.dimension, self.timeout = argv, dimension, timeout
        self.signature = digest({"type": "command-v1", "argv": argv, "dimension": dimension})
        self.calls = 0

    def embed(self, chunk, model):
        self.calls += 1
        row = {"id": chunk["id"], "text": chunk["text"], "embed_metadata": chunk["embed_metadata"], "embed_model": model}
        # Disk-backed bounded reads avoid buffering arbitrary subprocess output
        # in memory. Never echo stderr, which may contain credentials.
        with tempfile.TemporaryFile(dir=".") as out, tempfile.TemporaryFile(dir=".") as err:
            try:
                result = subprocess.run(self.argv, input=(canonical(row) + "\n").encode(),
                                        stdout=out, stderr=err, timeout=self.timeout, shell=False, check=False)
            except subprocess.TimeoutExpired as exc:
                raise SyncError(f"embedder exceeded {self.timeout:g}s timeout") from exc
            except OSError as exc:
                raise SyncError(f"cannot start embedder: {exc.strerror}") from exc
            if result.returncode:
                raise SyncError(f"embedder exited with code {result.returncode}; apply can be resumed")
            if out.tell() > 1024 * 1024:
                raise SyncError("embedder output exceeds 1 MiB per chunk")
            out.seek(0)
            try:
                lines = out.read().decode("utf-8").splitlines()
                if len(lines) != 1:
                    raise SyncError("embedder must return exactly one JSONL vector row per input")
                response = json.loads(lines[0])
                if not isinstance(response, dict) or set(response) != {"id", "vector"} or response["id"] != chunk["id"]:
                    raise SyncError("embedder response must contain matching id and vector only")
                return checked_vector(response["vector"], self.dimension)
            except (ValueError, UnicodeError) as exc:
                if isinstance(exc, SyncError):
                    raise
                raise SyncError("invalid JSONL from embedder") from exc


class PrecomputedEmbedder:
    def __init__(self, path, dimension=32):
        self.dimension, self.calls = dimension, 0
        self.signature = digest({"type": "precomputed-v1", "dimension": dimension})
        if not Path(path).is_file():
            raise SyncError("vectors input must be a regular file")
        if Path(path).stat().st_size > MAX_JSON_BYTES:
            raise SyncError("vectors file exceeds 64 MiB limit")
        self.vectors = {}
        with Path(path).open(encoding="utf-8") as source:
            for n, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or set(row) != {"id", "vector"} or not isinstance(row["id"], str):
                        raise SyncError(f"invalid vector row {n}")
                    if row["id"] in self.vectors:
                        raise SyncError(f"duplicate vector id at row {n}")
                    self.vectors[row["id"]] = checked_vector(row["vector"], dimension)
                except (ValueError, KeyError) as exc:
                    if isinstance(exc, SyncError):
                        raise
                    raise SyncError(f"invalid vector row {n}") from exc

    def embed(self, chunk, model):
        self.calls += 1
        try:
            return self.vectors[chunk["id"]]
        except KeyError as exc:
            raise SyncError(f"precomputed vector missing for {chunk['id']}") from exc
