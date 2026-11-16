import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ragwrench.difybridge.check import check_endpoint, record_errors
from ragwrench.difybridge.embedding import embed, hash_embed
from ragwrench.difybridge.errors import BridgeError


@contextmanager
def fake_endpoint(reply):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get('Content-Length', '0')))
            status, headers, body = reply(self.path, self.headers, raw)
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01})
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_http_embedding(monkeypatch):
    monkeypatch.setenv('TEST_EMBED_KEY', 'test-secret')
    seen = []
    def reply(path, headers, raw):
        seen.append((path, headers['Authorization'], json.loads(raw)))
        return 200, {}, {'data': [{'index': 0, 'embedding': [0.3, 0.4]}]}
    with fake_endpoint(reply) as endpoint:
        assert embed('vacation', {'kind': 'http', 'url': endpoint + '/v1/embeddings', 'model': 'local', 'dimensions': 2, 'api_key_env': 'TEST_EMBED_KEY'}) == [0.3, 0.4]
    assert seen == [('/v1/embeddings', 'Bearer test-secret', {'model': 'local', 'input': 'vacation'})]


@pytest.mark.parametrize('response', [None, {}, {'data': []}, {'data': [{'embedding': [1]}]},
    {'data': [{'embedding': [1, float('nan')]}]}, {'data': [{'index': 1, 'embedding': [1, 2]}]},
    {'data': [{'embedding': ['1', 2]}]}])
def test_invalid_embedding(response):
    with fake_endpoint(lambda *args: (200, {}, response)) as endpoint:
        with pytest.raises(BridgeError) as exc:
            embed('vacation', {'kind': 'http', 'url': endpoint, 'model': 'x', 'dimensions': 2})
        assert exc.value.status == 502


def test_no_credential_redirect(monkeypatch):
    received = []
    monkeypatch.setenv('EMBEDDING_API_KEY', 'never-forward')
    with fake_endpoint(lambda *args: (received.append(args) or (200, {}, {'data': [{'embedding': [1, 2]}]}))) as target:
        with fake_endpoint(lambda *args: (307, {'Location': target}, {})) as source:
            with pytest.raises(BridgeError):
                embed('vacation', {'kind': 'http', 'url': source, 'model': 'x', 'dimensions': 2})
    assert received == []


def test_hash_deterministic_and_normalized():
    assert hash_embed('Vacation POLICY') == hash_embed('vacation policy')
    assert sum(x*x for x in hash_embed('a b c')) == pytest.approx(1)
    assert hash_embed('!!!') == [0.0]*64


def test_checker_rejects_broken_endpoint():
    # Broken in multiple independent ways: no auth, no errors, invalid score/metadata.
    record = {'content': 'x', 'title': 'x', 'score': 1.2, 'metadata': None}
    with fake_endpoint(lambda *args: (200, {}, {'records': [record]*4})) as endpoint:
        report = check_endpoint(endpoint, 'key', 'source', 'hello')
    assert not report['passed']
    failures = {c['name']: c['errors'] for c in report['checks']}
    assert 'records exceeds top_k' in failures['top_k=1']
    assert 'metadata must be a non-null object' in failures['retrieval shape and ordering']
    assert failures['wrong authorization'] and failures['malformed JSON']


def test_checker_detects_ordering_threshold():
    records = [{'content': 'a', 'title': 'a', 'score': s, 'metadata': {}} for s in [0.2, 0.8]]
    assert set(record_errors({'records': records}, 1, 0.9)) == {
        'records exceeds top_k', 'score below score_threshold', 'scores must be descending'}


def test_checker_empty_is_not_silent(json_endpoint):
    assert not check_endpoint(json_endpoint, 'test-key', 'handbook', 'zzzzmissing')['passed']
    report = check_endpoint(json_endpoint, 'test-key', 'handbook', 'zzzzmissing', allow_empty=True)
    assert report['passed'] and any('vacuous' in text for text in report['limitations'])


def test_qdrant_server_transport(qdrant_app, body, tmp_path, monkeypatch):
    """Use real SDK HTTP transport against synthetic Qdrant protocol responses."""
    from dataclasses import replace
    from ragwrench.difybridge.config import Config
    from ragwrench.difybridge.service import Application
    from .conftest import retrieve
    local = qdrant_app.backends.items['handbook']
    info = local.client.get_collection('handbook').model_dump(mode='json')
    seen = []

    def reply(path, headers, raw):
        if path == '/':
            return 200, {}, {'title': 'qdrant', 'version': '1.15.1'}
        if path == '/collections/handbook':
            return 200, {}, {'result': info, 'status': 'ok', 'time': 0.001}
        if path == '/collections/handbook/points/query':
            seen.append((headers.get('api-key'), json.loads(raw)))
            return 200, {}, {'result': {'points': [{'id': 1, 'version': 0, 'score': 0.8,
                'payload': {'content': 'remote vacation', 'title': 'remote.md', 'metadata': {'lang': 'en', 'private': 'excluded'}}}]}, 'status': 'ok', 'time': 0.001}
        return 404, {}, {'status': {'error': 'unknown route'}}

    monkeypatch.setenv('MOCK_QDRANT_KEY', 'synthetic-key')
    with fake_endpoint(reply) as endpoint:
        config = replace(local.config, path=None, url=endpoint, api_key_env='MOCK_QDRANT_KEY')
        # An insecure-channel warning is expected for a loopback HTTP fixture.
        with pytest.warns(UserWarning, match='insecure'):
            app = Application(Config({'handbook': config}), 'test-key')
        try:
            body['metadata_condition'] = {'conditions': [{'name': 'lang', 'comparison_operator': 'is', 'value': 'en'}]}
            result = retrieve(app, body)
            assert result == [{'content': 'remote vacation', 'title': 'remote.md', 'score': 0.9, 'metadata': {'lang': 'en'}}]
        finally:
            app.close()
    assert seen[0][0] == 'synthetic-key'
    query = seen[0][1]
    assert query['using'] == 'text' and len(query['query']['nearest']) == 64
    assert query['filter']['must'][0]['key'] == 'metadata.lang'
    assert query['filter']['must'][0]['match']['value'] == 'en'


def test_http_embedder_to_qdrant_integration(qdrant_app, body):
    from dataclasses import replace
    from .conftest import retrieve
    calls = []
    def reply(path, headers, raw):
        calls.append(json.loads(raw))
        return 200, {}, {'data': [{'index': 0, 'embedding': hash_embed(json.loads(raw)['input'])}]}
    backend = qdrant_app.backends.items['handbook']
    with fake_endpoint(reply) as endpoint:
        backend.config = replace(backend.config, embedder={'kind': 'http', 'url': endpoint + '/v1/embeddings', 'model': 'synthetic', 'dimensions': 64})
        assert retrieve(qdrant_app, body)[0]['title'] == 'handbook.md'
    assert calls == [{'model': 'synthetic', 'input': 'vacation policy'}]
