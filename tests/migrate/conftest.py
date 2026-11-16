from __future__ import annotations

import json
from contextlib import closing
import socket
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings
from qdrant_client import QdrantClient, models

from ragwrench.migrate.model import StoreURI


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Tests must not access the network")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)


@pytest.fixture
def fixture_data():
    return json.loads((Path(__file__).parent / "fixtures/records.json").read_text())


@pytest.fixture
def chroma_store(tmp_path, fixture_data):
    def make(name="chroma", *, distance="cosine", rows=None, dimension=None):
        path = tmp_path / name
        client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        try:
            col = client.create_collection("docs", embedding_function=None,
                configuration={"hnsw": {"space": distance, "num_threads": 1}})
            for r in fixture_data["records"] if rows is None else rows:
                col.add(ids=[r["id"]], embeddings=[r["vector"][:dimension] if dimension else r["vector"]],
                        metadatas=[r["metadata"]] if r.get("metadata") is not None else None,
                        documents=[r["document"]] if r.get("document") is not None else None)
        finally:
            client.close()
        return StoreURI("chroma", path, "docs")
    return make


@pytest.fixture
def qdrant_store(tmp_path):
    def make(name="qdrant", *, distance="Cosine", dimension=4, points=None,
             vector_names=None, sparse=False, multi=False):
        path = tmp_path / name
        with closing(QdrantClient(path=str(path))) as client:
            config = models.VectorParams(size=dimension, distance=models.Distance(distance),
                multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM) if multi else None)
            client.create_collection("docs", vectors_config={n: config for n in vector_names} if vector_names else config,
                sparse_vectors_config={"sparse": models.SparseVectorParams()} if sparse else None)
            if points:
                client.upsert("docs", points=[models.PointStruct(**r) for r in points])
        return StoreURI("qdrant", path, "docs")
    return make
