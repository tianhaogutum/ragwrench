from __future__ import annotations

import hashlib
import json
from contextlib import closing
from dataclasses import replace

import chromadb
import pytest
from chromadb.config import Settings
from qdrant_client import QdrantClient, models

from ragwrench.migrate import engine
from ragwrench.migrate.cli import main
from ragwrench.migrate.model import ENVELOPE, ParityError, StoreURI, physical_id
from ragwrench.migrate.stores import read_store


def fingerprint(path):
    return {str(f.relative_to(path)): hashlib.sha256(f.read_bytes()).hexdigest() for f in path.rglob("*") if f.is_file()}


def uristr(uri):
    return f"{uri.kind}:{uri.path}#{uri.collection}"


@pytest.mark.parametrize("distance,qdistance", [("cosine", "Cosine"), ("l2", "Euclid"), ("ip", "Dot")])
def test_real_roundtrip(chroma_store, tmp_path, distance, qdistance):
    src = chroma_store(distance=distance)
    dst = StoreURI("qdrant", tmp_path / "target", "docs")
    restored = StoreURI("chroma", tmp_path / "restored", "docs")
    before = fingerprint(src.path)
    result = engine.copy(src, dst, batch_size=2)
    assert result["copied"] == 5
    assert result["mapping"]["uuid5_mapped"] == 3
    assert engine.verify(src, dst, top_k=2, batch_size=2)["ok"]
    with closing(QdrantClient(path=str(dst.path))) as client:
        assert client.get_collection("docs").config.params.vectors.distance.value == qdistance
        p = client.retrieve("docs", ids=[physical_id("guide.md::12")])[0]
        assert p.payload["__id__"] == "guide.md::12"
        assert p.payload["document"] == "A synthetic installation guide."
    assert engine.copy(dst, restored, batch_size=2)["copied"] == 5
    report = engine.verify(src, restored, top_k=2)
    assert report["ok"]
    assert report["top_k"]["samples"] == 5
    assert report["top_k"]["overlap_mean"] == 1.0
    assert fingerprint(src.path) == before
    assert engine.copy(src, dst, batch_size=1)["copied"] == 5
    assert engine.copy(dst, restored, batch_size=1)["copied"] == 5
    assert engine.verify(src, dst)["target_count"] == 5
    assert engine.verify(dst, restored)["ok"]


def test_native_nested_qdrant_roundtrip(qdrant_store, tmp_path):
    metadata = {"nested": {"x": [1, None, {"a": True}]}, "tags": ["a", "b"], "nullable": None,
                "__vecparity_user": "allowed", "number": 2**70, "": {"empty-key": []}}
    src = qdrant_store(distance="Dot", points=[
        {"id": 7, "vector": [1., 2., 3., 4.], "payload": {**metadata, "document": "Doc"}},
        {"id": "550e8400-e29b-41d4-a716-446655440000", "vector": [4., 3., 2., 1.], "payload": {"__id__": "original", "document": ""}},
    ])
    dst = StoreURI("chroma", tmp_path / "chroma", "docs")
    result = engine.copy(src, dst, batch_size=1)
    assert result["mapping"]["json_encoded_values"] >= 4
    assert result["mapping"]["null_metadata_values"] == 1
    assert engine.verify(src, dst, top_k=1)["ok"]
    with read_store(dst) as store:
        data = {r.id: r for batch in store.batches(1) for r in batch}
    assert data["7"].metadata == metadata
    assert data["7"].document == "Doc"
    assert data["original"].qdrant_id == "550e8400-e29b-41d4-a716-446655440000"
    back = StoreURI("qdrant", tmp_path / "back", "docs")
    engine.copy(dst, back)
    assert engine.verify(src, back, top_k=1)["ok"]
    with closing(QdrantClient(path=str(back.path))) as client:
        points, _ = client.scroll("docs", with_payload=True)
        assert {p.id for p in points} == {7, "550e8400-e29b-41d4-a716-446655440000"}


@pytest.mark.parametrize("corruption,expected", [("vector", "vectors"), ("payload", "metadata"),
    ("document", "documents"), ("missing", None), ("extra", None), ("mapping", "id_mapping")])
def test_detects_corruption(chroma_store, tmp_path, corruption, expected):
    src = chroma_store()
    dst = StoreURI("qdrant", tmp_path / "qdrant", "docs")
    engine.copy(src, dst)
    rid = physical_id("guide.md::12")
    with closing(QdrantClient(path=str(dst.path))) as client:
        if corruption == "vector":
            client.update_vectors("docs", points=[models.PointVectors(id=rid, vector=[0., 1., 0., 0.])])
        elif corruption in {"payload", "document"}:
            client.set_payload("docs", payload={"page": 999} if corruption == "payload" else {"document": "tampered"}, points=[rid])
        elif corruption == "missing":
            client.delete("docs", points_selector=models.PointIdsList(points=[rid]))
        elif corruption == "extra":
            client.upsert("docs", points=[models.PointStruct(id=999, vector=[1., 0., 0., 0.], payload={})])
        else:
            point = client.retrieve("docs", ids=[rid], with_vectors=True)[0]
            client.delete("docs", points_selector=models.PointIdsList(points=[rid]))
            client.upsert("docs", points=[models.PointStruct(id=999, vector=point.vector, payload=point.payload)])
    report = engine.verify(src, dst, top_k=2)
    assert not report["ok"]
    if expected:
        assert report["mismatch_counts"][expected] == 1
    else:
        assert not report["ids_ok"]
        assert not report["count_ok"]
    assert main(["verify", uristr(src), uristr(dst), "--json"]) == 1


@pytest.mark.parametrize("distance,dimension", [("Cosine", 3), ("Dot", 4)])
def test_incompatible_target_preflight(chroma_store, qdrant_store, distance, dimension):
    src = chroma_store()
    dst = qdrant_store(distance=distance, dimension=dimension)
    before = fingerprint(dst.path)
    with pytest.raises(ParityError, match="dimension/distance"):
        engine.copy(src, dst)
    assert fingerprint(dst.path) == before


def test_empty_and_missing(chroma_store, tmp_path):
    src = chroma_store(rows=[])
    dst = StoreURI("qdrant", tmp_path / "no-create", "docs")
    assert engine.inspect(src)["dimension"] is None
    with pytest.raises(ParityError, match="Empty"):
        engine.copy(src, dst)
    assert not dst.path.exists()
    with pytest.raises(ParityError, match="Collection does not exist"):
        engine.copy(replace(src, collection="missing"), dst)
    with pytest.raises(ParityError, match="does not exist"):
        engine.copy(replace(src, path=tmp_path / "missing"), dst)


def test_collision_does_not_create_target(chroma_store, tmp_path):
    rows = [{"id": "550e8400-e29b-41d4-a716-446655440000", "vector": [1., 2.]},
            {"id": "550E8400-E29B-41D4-A716-446655440000", "vector": [2., 1.]}]
    src = chroma_store(rows=rows)
    dst = StoreURI("qdrant", tmp_path / "no-create", "docs")
    with pytest.raises(ParityError, match="collision"):
        engine.copy(src, dst, batch_size=1)
    assert not dst.path.exists()


def test_existing_target_collision(chroma_store, qdrant_store):
    src = chroma_store(rows=[{"id": "doc", "vector": [1., 0., 0., 0.]}])
    dst = qdrant_store(points=[{"id": physical_id("doc"), "vector": [1., 0., 0., 0.], "payload": {}}])
    before = fingerprint(dst.path)
    with pytest.raises(ParityError, match="overwrite"):
        engine.copy(src, dst)
    assert fingerprint(dst.path) == before


def test_dry_run_is_read_only(chroma_store, tmp_path):
    src = chroma_store()
    dst = StoreURI("qdrant", tmp_path / "absent", "docs")
    before = fingerprint(src.path)
    assert engine.copy(src, dst, dry_run=True)["would_copy"] == 5
    assert not dst.path.exists()
    assert fingerprint(src.path) == before
    engine.copy(src, dst)
    target_before = fingerprint(dst.path)
    assert engine.copy(src, dst, dry_run=True)["copied"] == 0
    assert fingerprint(src.path) == before
    assert fingerprint(dst.path) == target_before


def test_named_vectors(qdrant_store, tmp_path):
    src = qdrant_store(vector_names=["dense"], points=[{"id": 1, "vector": {"dense": [1., 2., 3., 4.]}, "payload": {}}])
    dst = StoreURI("chroma", tmp_path / "restored", "docs")
    with pytest.raises(ParityError, match="--vector-name"):
        engine.copy(src, dst)
    with pytest.raises(ParityError, match="does not exist"):
        engine.copy(src, dst, vector_name="other")
    engine.copy(src, dst, vector_name="dense")
    assert engine.verify(src, dst, vector_name="dense")["ok"]
    back = StoreURI("qdrant", tmp_path / "back", "docs")
    engine.copy(dst, back, vector_name="dense")
    assert engine.verify(src, back, vector_name="dense")["ok"]


@pytest.mark.parametrize("kwargs,message", [({"vector_names": ["a", "b"]}, "Multiple"),
    ({"sparse": True}, "Sparse"), ({"multi": True}, "multi-vector"), ({"distance": "Manhattan"}, "distance")])
def test_unsupported_vector_collections(qdrant_store, kwargs, message):
    src = qdrant_store(**kwargs)
    with pytest.raises(ParityError, match=message):
        engine.inspect(src, vector_name="a" if "vector_names" in kwargs else None)


def test_same_and_nested_path_refused(chroma_store):
    src = chroma_store()
    for path in (src.path, src.path / "nested", src.path.parent):
        with pytest.raises(ParityError, match="separate"):
            engine.copy(src, StoreURI("qdrant", path, "docs"))


def test_symlink_refused(chroma_store, tmp_path):
    src = chroma_store()
    (src.path / "link").symlink_to(tmp_path / "outside")
    with pytest.raises(ParityError, match="symlink"):
        engine.inspect(src)


def test_stale_document_is_cleared(qdrant_store, chroma_store):
    src = qdrant_store(points=[{"id": 42, "vector": [1., 2., 3., 4.], "payload": {}}])
    dst = chroma_store(rows=[{"id": "42", "vector": [1., 2., 3., 4.], "document": "old"}])
    engine.copy(src, dst)
    assert engine.verify(src, dst)["ok"]


def test_inspect_and_cli(chroma_store, tmp_path, capsys):
    src = chroma_store()
    dst = StoreURI("qdrant", tmp_path / "dst", "docs")
    assert main(["inspect", uristr(src), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["count"] == 5
    assert report["payload_key_types"]["page"] == ["int"]
    assert main(["copy", uristr(src), uristr(dst), "--dry-run"]) == 0
    assert "would copy 5" in capsys.readouterr().out
    assert not dst.path.exists()
    assert main(["copy", uristr(src), uristr(dst)]) == 0
    assert "copied 5" in capsys.readouterr().out
    assert main(["verify", uristr(src), uristr(dst), "--top-k", "2"]) == 0
    assert "PARITY OK" in capsys.readouterr().out
    assert main(["copy", "chroma:/missing#docs", uristr(dst), "--json"]) == 2
    assert "error" in json.loads(capsys.readouterr().err)
    assert main(["verify", uristr(src), uristr(dst), "--tolerance", "nan"]) == 2
    assert main(["verify", uristr(src), uristr(dst), "--sample-queries", "-1"]) == 2
    with pytest.raises(SystemExit) as result:
        main(["inspect", uristr(src), "--batch-size", "0"])
    assert result.value.code == 2


def test_generated_id_collision(chroma_store, tmp_path):
    src = chroma_store(rows=[{"id": "doc", "vector": [1., 2.]},
                             {"id": str(physical_id("doc")), "vector": [2., 1.]}])
    dst = StoreURI("qdrant", tmp_path / "no-create", "docs")
    with pytest.raises(ParityError, match="collision"):
        engine.copy(src, dst)
    assert not dst.path.exists()


def test_duplicate_logical_qdrant_ids(qdrant_store, tmp_path):
    src = qdrant_store(points=[{"id": i, "vector": [1., 2., 3., 4.], "payload": {"__id__": "same"}} for i in [1, 2]])
    dst = StoreURI("chroma", tmp_path / "no-create", "docs")
    with pytest.raises(ParityError, match="Duplicate"):
        engine.copy(src, dst, batch_size=1)
    assert not dst.path.exists()


def test_unknown_payload_types_refused_before_write(qdrant_store, tmp_path):
    src = qdrant_store(points=[{"id": 1, "vector": [1., 2., 3., 4.], "payload": {"document": {"unexpected": "object"}}}])
    dst = StoreURI("chroma", tmp_path / "no-create", "docs")
    with pytest.raises(ParityError, match="Documents"):
        engine.copy(src, dst)
    assert not dst.path.exists()


def test_tolerance_and_sampling(chroma_store, tmp_path):
    src = chroma_store(distance="l2")
    dst = StoreURI("qdrant", tmp_path / "qdrant", "docs")
    engine.copy(src, dst)
    with closing(QdrantClient(path=str(dst.path))) as client:
        client.update_vectors("docs", points=[models.PointVectors(id=physical_id("guide.md::12"), vector=[1.0001, 0.2, 0.1, 0.])])
    strict = engine.verify(src, dst, tolerance=1e-5, top_k=2, sample_queries=3, seed=17)
    assert strict["mismatch_counts"]["vectors"] == 1
    assert not strict["ok"]
    loose = engine.verify(src, dst, tolerance=1e-3, top_k=2, sample_queries=3, seed=17)
    assert loose["ok"]
    assert loose["top_k"]["samples"] == 3
    assert loose["top_k"]["overlap_mean"] == 1.0
    assert engine.verify(src, dst, tolerance=1e-3, top_k=2, sample_queries=3, seed=17) == loose
    skipped = engine.verify(src, dst, tolerance=1e-3, sample_queries=0)
    assert skipped["ok"] and not skipped["top_k"]["checked"]


def test_unrelated_target_points_are_retained(chroma_store, qdrant_store):
    src = chroma_store()
    dst = qdrant_store(points=[{"id": 999, "vector": [1., 0., 0., 0.], "payload": {"document": "extra"}}])
    engine.copy(src, dst)
    report = engine.verify(src, dst)
    assert not report["ok"] and report["extra_ids"] == ["999"]
    assert report["target_count"] == 6


def test_inspect_verify_do_not_change_either_store(chroma_store, tmp_path):
    src = chroma_store()
    dst = StoreURI("qdrant", tmp_path / "qdrant", "docs")
    engine.copy(src, dst)
    before = (fingerprint(src.path), fingerprint(dst.path))
    engine.inspect(src)
    engine.inspect(dst)
    assert engine.verify(src, dst, top_k=2)["ok"]
    assert before == (fingerprint(src.path), fingerprint(dst.path))


def test_stored_embedding_function_is_never_reconstructed(tmp_path, monkeypatch):
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    src = StoreURI("chroma", tmp_path / "source", "docs")
    client = chromadb.PersistentClient(path=str(src.path), settings=Settings(anonymized_telemetry=False))
    try:
        col = client.create_collection("docs", configuration={"hnsw": {"num_threads": 1}})
        col.add(ids=["doc"], embeddings=[[1., 2.]], documents=["supplied embedding"])
    finally:
        client.close()

    def denied(*args, **kwargs):
        raise AssertionError("Stored embedding functions must not be rebuilt")
    monkeypatch.setattr(DefaultEmbeddingFunction, "build_from_config", denied)
    assert engine.inspect(src)["dimension"] == 2
    dst = StoreURI("qdrant", tmp_path / "target", "docs")
    assert engine.copy(src, dst)["copied"] == 1
    assert engine.verify(src, dst)["ok"]
