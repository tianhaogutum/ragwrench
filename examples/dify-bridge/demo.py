"""Build disposable synthetic Qdrant data, serve it over HTTP, check, shut down."""
import json
import tempfile
import threading
from pathlib import Path

from qdrant_client import QdrantClient, models as m

from ragwrench.difybridge.check import check_endpoint, request
from ragwrench.difybridge.config import load_config
from ragwrench.difybridge.embedding import hash_embed
from ragwrench.difybridge.service import Application, make_server


def main():
    rows = [json.loads(line) for line in Path(__file__).with_name("chunks.jsonl").read_text().splitlines()]
    with tempfile.TemporaryDirectory(prefix="difykb-demo-") as tmp:
        root = Path(tmp)
        # TemporaryDirectory is owned exclusively by this demo and removed at exit.
        client = QdrantClient(path=str(root / "qdrant"))
        try:
            client.create_collection("handbook", vectors_config={"text": m.VectorParams(size=64, distance=m.Distance.COSINE)})
            client.upsert("handbook", points=[m.PointStruct(id=i, vector={"text": hash_embed(row["content"])}, payload=row) for i, row in enumerate(rows)])
        finally:
            client.close()
        config_path = root / "bridge.toml"
        config_path.write_text('''[knowledge.handbook]
backend = "qdrant"
path = "qdrant"
collection = "handbook"
vector_name = "text"
metadata_allowlist = ["lang", "page", "team", "updated", "tags"]
[knowledge.handbook.embedder]
kind = "hash"
dimensions = 64
''')
        app = Application(load_config(config_path), "local-demo")
        server = make_server(app, port=0)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_port}"
            payload = json.loads(Path(__file__).with_name("request.json").read_text())
            status, body = request(endpoint, payload, "local-demo")
            assert status == 200 and body["records"], body
            assert all(r["metadata"]["lang"] == "en" and "internal_secret" not in r["metadata"] for r in body["records"])
            print(json.dumps(body, indent=2))
            report = check_endpoint(endpoint, "local-demo", "handbook", "vacation policy")
            print(json.dumps(report, indent=2))
            assert report["passed"], "Conformance failed"
            print(f"DEMO PASS: {len(report['checks'])} HTTP conformance probes; mapped named vectors and metadata filter verified")
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
            app.close()


if __name__ == "__main__":
    main()
