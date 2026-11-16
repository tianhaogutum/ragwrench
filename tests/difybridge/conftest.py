import json
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models as m

from ragwrench.difybridge.config import Config, Knowledge, load_config
from ragwrench.difybridge.embedding import hash_embed
from ragwrench.difybridge.service import Application, make_server

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def rows():
    return [json.loads(line) for line in (ROOT / 'examples/dify-bridge/chunks.jsonl').read_text().splitlines()]


@contextmanager
def running(app):
    server = make_server(app, port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        app.close()
        assert not thread.is_alive()


@pytest.fixture
def json_app():
    app = Application(load_config(ROOT / 'examples/dify-bridge/bridge.toml'), 'test-key')
    yield app
    app.close()


@pytest.fixture
def json_endpoint(json_app):
    with running(json_app) as url:
        yield url


@pytest.fixture
def qdrant_app(tmp_path, rows):
    path = tmp_path / 'db'
    client = QdrantClient(path=str(path))
    try:
        client.create_collection('handbook', vectors_config={'text': m.VectorParams(size=64, distance=m.Distance.COSINE)})
        client.upsert('handbook', points=[m.PointStruct(id=i, vector={'text': hash_embed(row['content'])}, payload=row) for i, row in enumerate(rows)])
    finally:
        client.close()
    k = Knowledge(backend='qdrant', path=path, collection='handbook', vector_name='text',
                  metadata_allowlist=('lang', 'page', 'team', 'updated', 'tags'), embedder={'kind': 'hash', 'dimensions': 64})
    app = Application(Config({'handbook': k}), 'test-key')
    yield app
    app.close()


@pytest.fixture
def body():
    return {"knowledge_id": "handbook", "query": "vacation policy", "retrieval_setting": {"top_k": 3, "score_threshold": 0.0}}


def retrieve(app, body):
    return app.retrieve('Bearer test-key', json.dumps(body).encode())['records']


@pytest.fixture(autouse=True)
def deny_external_network(monkeypatch):
    """Keep all test network traffic on loopback; fail accidental external calls."""
    import socket
    connect = socket.socket.connect

    def local_connect(sock, address):
        if isinstance(address, tuple) and address[0] not in {'127.0.0.1', '::1', 'localhost'}:
            raise AssertionError('Tests must not connect outside loopback')
        return connect(sock, address)
    monkeypatch.setattr(socket.socket, 'connect', local_connect)
