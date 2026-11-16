"""Create a tiny real Chroma store, offline; refuses to overwrite an existing path."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python examples/migrate/make_fixture.py OUTPUT_DIR")
    root = Path(sys.argv[1]).resolve()
    if root.exists():
        raise SystemExit("OUTPUT_DIR already exists; choose a new directory")
    fixture = json.loads((Path(__file__).resolve().parents[2] / "tests/migrate/fixtures/records.json").read_text())
    root.mkdir(parents=True)
    client = chromadb.PersistentClient(path=str(root / "chroma"), settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.create_collection("docs", embedding_function=None,
            configuration={"hnsw": {"space": fixture["distance"], "num_threads": 1}})
        for r in fixture["records"]:
            collection.add(ids=[r["id"]], embeddings=[r["vector"]],
                           metadatas=[r["metadata"]] if r["metadata"] is not None else None,
                           documents=[r["document"]] if r["document"] is not None else None)
    finally:
        client.close()
    print(f"Created {len(fixture['records'])} synthetic points in chroma:{root}/chroma#docs")


if __name__ == "__main__":
    main()
