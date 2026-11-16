"""Read-only snapshot status contract: one configured JSONL ID per request."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from ragwrench.difybridge.config import Config, Knowledge
from ragwrench.difybridge.service import Application
from .conftest import running


def post_status(endpoint, payload, authorization='Bearer test-key'):
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    headers = {'Content-Type': 'application/json'}
    if authorization is not None:
        headers['Authorization'] = authorization
    req = Request(endpoint + '/snapshot/status', data=raw, headers=headers)
    try:
        response = urlopen(req, timeout=3)
    except HTTPError as exc:
        response = exc
    with response:
        assert response.headers['Cache-Control'] == 'no-store'
        return response.status, json.load(response)


def publish(path, count):
    candidate = path.with_suffix('.next')
    candidate.write_text('{"content":"apple"}\n' * count, encoding='utf-8')
    candidate.replace(path)


@pytest.fixture
def snapshot_app(tmp_path):
    path = tmp_path / 'private-source.jsonl'
    publish(path, 2)
    config = Knowledge(backend='jsonl', path=path)
    app = Application(Config({'first': config, 'alias': config}), 'test-key')
    yield app
    app.close()


def expected(knowledge='first', version=1, count=2):
    return 200, {'knowledge_id': knowledge, 'version': version, 'record_count': count}


def test_status_tracks_only_published_snapshots_per_id(snapshot_app):
    app = snapshot_app
    path = app.backends.items['first'].config.path
    with running(app) as endpoint:
        assert post_status(endpoint, {'knowledge_id': 'first'}) == expected()
        publish(path, 3)
        # A replacement file is not active until this particular ID reloads.
        assert post_status(endpoint, {'knowledge_id': 'first'}) == expected()
        assert app.reload('Bearer test-key', b'{"knowledge_id":"first"}') == expected(version=2, count=3)[1]
        assert post_status(endpoint, {'knowledge_id': 'first'}) == expected(version=2, count=3)
        assert post_status(endpoint, {'knowledge_id': 'alias'}) == expected('alias')
        path.write_text('{broken}\n')
        status, error = app_error(lambda: app.reload('Bearer test-key', b'{"knowledge_id":"first"}'))
        assert (status, error['error_code']) == (503, 3001)
        assert post_status(endpoint, {'knowledge_id': 'first'}) == expected(version=2, count=3)
        publish(path, 0)
        app.reload('Bearer test-key', b'{"knowledge_id":"first"}')
        assert post_status(endpoint, {'knowledge_id': 'first'}) == expected(version=3, count=0)
        # An unchanged successful reload still consumes a version; status does not.
        app.reload('Bearer test-key', b'{"knowledge_id":"first"}')
        assert post_status(endpoint, {'knowledge_id': 'first'}) == expected(version=4, count=0)
        assert post_status(endpoint, {'knowledge_id': 'alias'}) == expected('alias')


def app_error(action):
    from ragwrench.difybridge.errors import BridgeError
    with pytest.raises(BridgeError) as caught:
        action()
    return caught.value.status, caught.value.body()


def test_status_never_loads_or_retrieves_and_works_without_source(snapshot_app, monkeypatch):
    backend = snapshot_app.backends.items['first']
    backend.config.path.unlink()

    def forbidden(*args, **kwargs):
        pytest.fail('Status must use the active snapshot without loading or retrieving')

    monkeypatch.setattr(backend, '_load_snapshot', forbidden)
    monkeypatch.setattr(backend, 'retrieve', forbidden)
    with running(snapshot_app) as endpoint:
        for _ in range(2):
            assert post_status(endpoint, {'knowledge_id': 'first'}) == expected()
        assert post_status(endpoint, {'knowledge_id': 'alias'}) == expected('alias')


def test_status_remains_available_during_reload_build(snapshot_app, monkeypatch):
    backend = snapshot_app.backends.items['first']
    publish(backend.config.path, 3)
    entered, release = threading.Event(), threading.Event()
    load = backend._load_snapshot

    def paused_load(version):
        candidate = load(version)
        entered.set()
        assert release.wait(5), 'reload was not released'
        return candidate

    monkeypatch.setattr(backend, '_load_snapshot', paused_load)
    with running(snapshot_app) as endpoint, ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(backend.reload)
        try:
            assert entered.wait(5), 'reload did not begin'
            assert post_status(endpoint, {'knowledge_id': 'first'}) == expected()
        finally:
            release.set()
        assert pending.result(timeout=5) == {'version': 2, 'record_count': 3}
        assert post_status(endpoint, {'knowledge_id': 'first'}) == expected(version=2, count=3)


def test_status_captures_one_coherent_snapshot_across_publication(snapshot_app, monkeypatch):
    backend = snapshot_app.backends.items['first']
    entered, release = threading.Event(), threading.Event()
    get_snapshot = backend._get_snapshot
    captures = []

    def paused_capture():
        snapshot = get_snapshot()
        if threading.current_thread().name.startswith('status-reader'):
            captures.append(snapshot.version)
            entered.set()
            assert release.wait(5), 'status reader was not released'
        return snapshot

    monkeypatch.setattr(backend, '_get_snapshot', paused_capture)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='status-reader') as pool:
        pending = pool.submit(backend.snapshot_status)
        try:
            assert entered.wait(5), 'status did not capture a snapshot'
            publish(backend.config.path, 3)
            assert backend.reload() == {'version': 2, 'record_count': 3}
        finally:
            release.set()
        assert pending.result(timeout=5) == {'version': 1, 'record_count': 2}
    assert captures == [1]
    info = backend.snapshot_status()
    assert info == {'version': 2, 'record_count': 3}
    info['version'] = 999
    assert backend.snapshot_status() == {'version': 2, 'record_count': 3}


@pytest.mark.parametrize('payload,authorization,status,code', [
    (b'{broken', None, 401, 1001),
    (b'{broken', 'Basic test-key', 401, 1001),
    (b'{broken', 'Bearer wrong-secret', 403, 1002),
    ({'knowledge_id': 'unknown'}, 'Bearer test-key', 404, 2001),
    ({'knowledge_id': ''}, 'Bearer test-key', 400, 2002),
    ({'knowledge_id': 'a' * 257}, 'Bearer test-key', 400, 2002),
    ({'knowledge_id': None}, 'Bearer test-key', 400, 2002),
    ({'knowledge_id': True}, 'Bearer test-key', 400, 2002),
    ({'knowledge_id': ['handbook']}, 'Bearer test-key', 400, 2002),
    ({'knowledge_id': 'handbook', 'path': '/private-source'}, 'Bearer test-key', 400, 2002),
    ({}, 'Bearer test-key', 400, 2002),
    ([], 'Bearer test-key', 400, 2002),
    (b'{broken', 'Bearer test-key', 400, 2002),
    (b'{"knowledge_id":"handbook","knowledge_id":"other"}', 'Bearer test-key', 400, 2002),
])
def test_status_auth_and_validation_are_redacted(json_app, payload, authorization, status, code, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Rejected status requests must not inspect a snapshot')

    monkeypatch.setattr(json_app.backends.items['handbook'], '_get_snapshot', forbidden)
    with running(json_app) as endpoint:
        actual, error = post_status(endpoint, payload, authorization)
        assert actual == status
        assert set(error) == {'error_code', 'error_msg'}
        assert error['error_code'] == code
        assert isinstance(error['error_msg'], str) and error['error_msg']
        assert all(secret not in json.dumps(error) for secret in ('wrong-secret', '/private-source', 'other'))


def test_status_rejects_qdrant_without_backend_calls(qdrant_app, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Snapshot status must not call Qdrant')

    backend = qdrant_app.backends.items['handbook']
    monkeypatch.setattr(backend, 'retrieve', forbidden)
    monkeypatch.setattr(backend.client, 'get_collection', forbidden)
    monkeypatch.setattr(backend.client, 'query_points', forbidden)
    with running(qdrant_app) as endpoint:
        status, error = post_status(endpoint, {'knowledge_id': 'handbook'})
        assert (status, error['error_code']) == (400, 2002)
        assert set(error) == {'error_code', 'error_msg'}
