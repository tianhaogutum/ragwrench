"""The only backend-specific code. No remote clients or embedding functions."""
from __future__ import annotations

import shutil
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

from .model import (DISTANCES, ParityError, Record, Schema, StoreURI, decode_metadata,
                    checked_store_path, encode_metadata, physical_id, valid_qdrant_id)


def check_directory(uri: StoreURI, *, required: bool) -> bool:
    path = checked_store_path(uri.path)
    marker = "chroma.sqlite3" if uri.kind == "chroma" else "meta.json"
    if not path.exists():
        if required:
            raise ParityError("Store directory does not exist")
        return False
    if not path.is_dir():
        raise ParityError("Store path must be a directory")
    for entry in path.rglob("*"):
        if entry.is_symlink() or (not entry.is_file() and not entry.is_dir()):
            raise ParityError("Store contains a symlink or special file; refusing to follow it")
    if not (path / marker).is_file():
        if required or any(path.iterdir()):
            raise ParityError(f"Not a recognized {uri.kind} local store (missing {marker})")
        return False
    return True


@contextmanager
def read_store(uri: StoreURI, vector_name: str | None = None, *, optional: bool = False):
    """Open a disposable snapshot: even backend lock/migration writes stay off the input."""
    uri = replace(uri, path=checked_store_path(uri.path))
    if not check_directory(uri, required=not optional):
        yield None
        return
    with TemporaryDirectory(prefix="vecparity-") as temp:
        # Our own fresh snapshot directory: canonicalize it so a symlinked system
        # temp root (macOS /var -> /private/var, or a symlinked TMPDIR) does not
        # trip the store-path symlink refusal. User-supplied paths stay strict.
        path = Path(temp).resolve() / "store"
        shutil.copytree(uri.path, path)
        with Store(replace(uri, path=path), vector_name) as store:
            if store.collection is None and not optional:
                raise ParityError("Collection does not exist")
            yield store if store.collection is not None else None


class Store:
    def __init__(self, uri: StoreURI, vector_name: str | None = None):
        if vector_name == "":
            raise ParityError("--vector-name must be nonempty")
        self.uri = uri
        self.vector_name = vector_name
        self.client = None
        self.collection = None

    def __enter__(self):
        try:
            # Python callers may construct StoreURI directly, bypassing parse().
            self.uri = replace(self.uri, path=checked_store_path(self.uri.path))
            # Apply the engine's descendant/format preflight before any backend I/O.
            check_directory(self.uri, required=False)
            if self.uri.kind == "chroma":
                import chromadb
                from chromadb.config import Settings
                from chromadb.errors import NotFoundError
                self.client = chromadb.PersistentClient(
                    path=str(self.uri.path), settings=Settings(anonymized_telemetry=False))
                try:
                    self.collection = self.client.get_collection(self.uri.collection, embedding_function=None)
                except NotFoundError:
                    pass
            else:
                from qdrant_client import QdrantClient
                from .safety import audit_qdrant
                if (self.uri.path / "meta.json").is_file():
                    meta = audit_qdrant(self.uri.path)
                    if self.uri.collection in meta["aliases"]:
                        raise ParityError("Collection aliases are not supported; use a real collection name")
                self.client = QdrantClient(path=str(self.uri.path))
                if self.client.collection_exists(self.uri.collection):
                    self.collection = self.client.get_collection(self.uri.collection)
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.client is not None:
            self.client.close()

    def schema(self) -> Schema:
        if self.collection is None:
            raise ParityError("Collection does not exist")
        if self.uri.kind == "chroma":
            # Read the serialized index settings without rebuilding a stored embedding
            # function: constructing one may download a model or require credentials.
            conf = self.collection.get_model().configuration_json
            if not conf.get("hnsw"):
                raise ParityError("Only dense HNSW Chroma collections are supported")
            distance = conf["hnsw"]["space"]
            if distance not in DISTANCES:
                raise ParityError("Unsupported Chroma distance")
            return Schema(self.collection.get_model().dimension, distance, self.collection.count())
        params = self.collection.config.params
        if params.sparse_vectors:
            raise ParityError("Sparse vectors are not supported")
        vectors = params.vectors
        if isinstance(vectors, dict):
            if not self.vector_name:
                raise ParityError("Named Qdrant vectors require --vector-name")
            if len(vectors) != 1:
                raise ParityError("Multiple vector fields are outside v1 scope; refusing to discard fields")
            if self.vector_name not in vectors:
                raise ParityError("--vector-name does not exist in the Qdrant collection")
            vectors = vectors[self.vector_name]
        elif self.vector_name:
            raise ParityError("--vector-name supplied for an unnamed Qdrant collection")
        if vectors is None or vectors.multivector_config is not None:
            raise ParityError("Sparse-only and multi-vector collections are not supported")
        inverse = {v: k for k, v in DISTANCES.items()}
        if vectors.distance.value not in inverse:
            raise ParityError("Unsupported Qdrant distance (only Euclid, Cosine, Dot)")
        return Schema(vectors.size, inverse[vectors.distance.value], self.collection.points_count)

    def create(self, schema: Schema) -> None:
        if schema.dimension is None:
            raise ParityError("Cannot create a collection with unknown dimension")
        if self.uri.kind == "chroma":
            self.collection = self.client.create_collection(
                self.uri.collection, embedding_function=None,
                configuration={"hnsw": {"space": schema.distance, "num_threads": 1}})
        else:
            from qdrant_client import models
            config = models.VectorParams(size=schema.dimension, distance=models.Distance(DISTANCES[schema.distance]))
            self.client.create_collection(self.uri.collection,
                                          vectors_config={self.vector_name: config} if self.vector_name else config)
            self.collection = self.client.get_collection(self.uri.collection)

    def batches(self, size: int) -> Iterator[list[Record]]:
        if self.uri.kind == "chroma":
            offset = 0
            while True:
                data = self.collection.get(limit=size, offset=offset,
                                           include=["embeddings", "metadatas", "documents", "uris"])
                if not data["ids"]:
                    break
                records = []
                for i, rid in enumerate(data["ids"]):
                    if data.get("uris") and data["uris"][i] is not None:
                        raise ParityError("Chroma URI/image data is not supported")
                    metadata, qid = decode_metadata(data["metadatas"][i], "chroma")
                    records.append(Record(rid, data["embeddings"][i].tolist(), metadata,
                                          data["documents"][i], qid))
                yield records
                offset += len(records)
        else:
            offset = None
            while True:
                points, offset = self.client.scroll(self.uri.collection, limit=size, offset=offset,
                                                    with_payload=True, with_vectors=True)
                records = []
                for point in points:
                    payload = point.payload or {}
                    rid = payload.get("__id__", str(point.id))
                    metadata, _ = decode_metadata(payload, "qdrant")
                    vector = point.vector.get(self.vector_name) if isinstance(point.vector, dict) else point.vector
                    if not isinstance(vector, list) or (vector and isinstance(vector[0], list)):
                        raise ParityError("Missing dense vector or unsupported multi-vector point")
                    records.append(Record(rid, vector, metadata, payload.get("document"), valid_qdrant_id(point.id)))
                if records:
                    yield records
                if offset is None:
                    break

    def write(self, records: list[Record]) -> None:
        if self.uri.kind == "chroma":
            # Delete first so absent documents replace old documents, rather than leaving stale values.
            # Copy is deliberately resumable, not transactional across backends.
            ids = [r.id for r in records]
            self.collection.delete(ids=ids)
            # Chroma accepts null document entries when embeddings are explicit.
            self.collection.upsert(ids=ids, embeddings=[r.vector for r in records],
                                   metadatas=[encode_metadata(r, "chroma") for r in records],
                                   documents=[r.document for r in records])
        else:
            from qdrant_client import models
            self.client.upsert(self.uri.collection, wait=True, points=[
                models.PointStruct(id=r.qdrant_id if r.qdrant_id is not None else physical_id(r.id),
                                   vector={self.vector_name: r.vector} if self.vector_name else r.vector,
                                   payload=encode_metadata(r, "qdrant")) for r in records])

    def query(self, vector: list[float], top_k: int) -> list[str]:
        if self.uri.kind == "chroma":
            return self.collection.query(query_embeddings=[vector], n_results=top_k, include=[])["ids"][0]
        points = self.client.query_points(self.uri.collection, query=vector, using=self.vector_name,
                                          limit=top_k, with_payload=True).points
        return [p.payload.get("__id__", str(p.id)) for p in points]
