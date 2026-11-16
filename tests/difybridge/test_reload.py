import json
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from ragwrench.difybridge.backends import JsonlBackend
from ragwrench.difybridge.check import request
from ragwrench.difybridge.config import Config, Knowledge
from ragwrench.difybridge.errors import ConfigError
from ragwrench.difybridge.filters import Predicate
from ragwrench.difybridge.service import Application
from .conftest import running


OLD = [
    {'content': 'apple apple', 'title': 'old-first', 'metadata': {'tags': ['old']}},
    {'content': 'apple pear pear pear', 'title': 'old-second'},
]
NEW = [
    {'content': 'apple', 'title': 'new-first', 'metadata': {'tags': ['new']}},
    {'content': 'apple apple pear', 'title': 'new-second'},
    {'content': 'pear', 'title': 'new-third'},
]


def publish(path, rows):
    candidate = path.with_suffix('.next')
    candidate.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    candidate.replace(path)


def query(backend, predicate=None):
    return backend.retrieve('apple', predicate or Predicate(None, ()), 100, 0)


def post_reload(endpoint, payload, key='test-key'):
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    headers = {'Content-Type': 'application/json'}
    if key is not None:
        headers['Authorization'] = 'Bearer ' + key
    req = Request(endpoint + '/reload', data=raw, headers=headers)
    try:
        response = urlopen(req, timeout=3)
    except HTTPError as exc:
        response = exc
    with response:
        return response.status, json.load(response)


@pytest.fixture
def local_config(tmp_path):
    path = tmp_path / 'corpus.jsonl'
    publish(path, OLD)
    return Knowledge(backend='jsonl', path=path, metadata_allowlist=('tags',))


def test_http_reload_publishes_version_and_retains_last_good(local_config):
    app = Application(Config({'handbook': local_config}), 'test-key')
    body = {'knowledge_id': 'handbook', 'query': 'apple',
            'retrieval_setting': {'top_k': 100, 'score_threshold': 0}}
    with running(app) as endpoint:
        old = request(endpoint, body, 'test-key')
        publish(local_config.path, NEW)
        # Merely editing the file must not change the active version.
        assert request(endpoint, body, 'test-key') == old
        assert post_reload(endpoint, {'knowledge_id': 'handbook'}) == (
            200, {'knowledge_id': 'handbook', 'version': 2, 'record_count': 3})
        new = request(endpoint, body, 'test-key')
        assert new == (200, {'records': query(JsonlBackend(local_config))})
        assert {r['title'] for r in new[1]['records']} == {'new-first', 'new-second'}
        local_config.path.write_text('{"content":"apple partial"}\n{"content":"secret", "metadata":null}\n')
        status, error = post_reload(endpoint, {'knowledge_id': 'handbook'})
        assert (status, error['error_code']) == (503, 3001)
        assert 'secret' not in json.dumps(error)
        assert request(endpoint, body, 'test-key') == new
        publish(local_config.path, OLD)
        assert post_reload(endpoint, {'knowledge_id': 'handbook'}) == (
            200, {'knowledge_id': 'handbook', 'version': 3, 'record_count': 2})
        assert request(endpoint, body, 'test-key') == old


def test_reload_is_per_knowledge_id_even_when_paths_are_shared(local_config):
    app = Application(Config({'first': local_config, 'second': local_config}), 'test-key')
    try:
        first, second = app.backends.items.values()
        old = query(first)
        publish(local_config.path, NEW)
        assert app.reload('Bearer test-key', b'{"knowledge_id":"first"}') == {
            'knowledge_id': 'first', 'version': 2, 'record_count': 3}
        assert {r['title'] for r in query(first)} == {'new-first', 'new-second'}
        assert query(second) == old
        assert app.reload('Bearer test-key', b'{"knowledge_id":"second"}') == {
            'knowledge_id': 'second', 'version': 2, 'record_count': 3}
        assert query(second) == query(first)
    finally:
        app.close()


def test_inflight_query_keeps_records_and_bm25_statistics(local_config):
    backend = JsonlBackend(local_config)
    old = query(backend)
    entered, release = threading.Event(), threading.Event()

    class PausedPredicate:
        def matches(self, metadata):
            entered.set()
            assert release.wait(5), 'query was not released'
            return True

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(query, backend, PausedPredicate())
        try:
            assert entered.wait(5), 'query did not begin'
            publish(local_config.path, NEW)
            assert backend.reload() == {'version': 2, 'record_count': 3}
            assert query(backend) == query(JsonlBackend(local_config))
        finally:
            release.set()
        # Exact scores also prove df, n and average length came from the old version.
        assert pending.result(timeout=5) == old


def test_concurrent_reload_builds_serialize_without_blocking_queries(local_config, monkeypatch):
    from ragwrench.difybridge import backends

    backend = JsonlBackend(local_config)
    old = query(backend)
    publish(local_config.path, NEW)
    new = query(JsonlBackend(local_config))
    first_entered, first_release = threading.Event(), threading.Event()
    second_started, second_entered, second_release = (threading.Event() for _ in range(3))
    tokenize = backends.tokens

    def paused_tokens(text):
        if text == 'apple apple pear':
            first_entered.set()
            assert first_release.wait(5), 'first reload was not released'
        if text == 'apple final':
            second_entered.set()
            assert second_release.wait(5), 'second reload was not released'
        return tokenize(text)

    def second_reload():
        second_started.set()
        return backend.reload()

    monkeypatch.setattr(backends, 'tokens', paused_tokens)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(backend.reload)
        try:
            assert first_entered.wait(5)
            assert pool.submit(query, backend).result(timeout=2) == old
            # The first build keeps its open file, even after another rename.
            publish(local_config.path, [{'content': 'apple final', 'title': 'final'}])
            second = pool.submit(second_reload)
            assert second_started.wait(5)
            first_release.set()
            assert first.result(timeout=5) == {'version': 2, 'record_count': 3}
            assert second_entered.wait(5)
            assert pool.submit(query, backend).result(timeout=2) == new
        finally:
            first_release.set()
            second_release.set()
        assert second.result(timeout=5) == {'version': 3, 'record_count': 1}
    assert [r['title'] for r in query(backend)] == ['final']


def test_inplace_file_mutation_during_build_retains_previous_corpus(local_config, monkeypatch):
    from ragwrench.difybridge import backends

    backend = JsonlBackend(local_config)
    old = query(backend)
    publish(local_config.path, NEW)
    tokenize = backends.tokens

    def mutate_file(text):
        if text == 'apple apple pear':
            with local_config.path.open('ab') as stream:
                stream.write(b'{"content":"apple appended"}\n')
        return tokenize(text)

    monkeypatch.setattr(backends, 'tokens', mutate_file)
    with pytest.raises(ConfigError):
        backend.reload()
    assert query(backend) == old
    publish(local_config.path, OLD)
    assert backend.reload() == {'version': 2, 'record_count': 2}


@pytest.mark.parametrize('invalid', [
    b'{broken}\n', b'{"content":"x","content":"y"}\n',
    b'{"content":"x","metadata":null}\n', b'{"content":42}\n',
    b'{"content":"x","metadata":{"tags":[NaN]}}\n', b'\xff\n',
    b' ' * (1024 * 1024 + 1), None, 'oversize', 'too-many',
], ids=['syntax', 'duplicate-key', 'null-metadata', 'non-string-content', 'nan',
        'utf8', 'long-line', 'missing-file', 'large-file', 'record-limit'])
def test_invalid_reload_is_rollback_not_partial_replacement(local_config, invalid):
    backend = JsonlBackend(local_config)
    old = query(backend)
    if invalid is None:
        local_config.path.unlink()
    elif invalid == 'oversize':
        with local_config.path.open('wb') as stream:
            stream.truncate(64 * 1024 * 1024 + 1)
    elif invalid == 'too-many':
        local_config.path.write_bytes(b'{"content":""}\n' * 100001)
    else:
        local_config.path.write_bytes(b'{"content":"apple partial"}\n' + invalid)
    with pytest.raises(ConfigError):
        backend.reload()
    assert query(backend) == old
    publish(local_config.path, NEW)
    assert backend.reload() == {'version': 2, 'record_count': 3}


def test_empty_reload_and_result_metadata_cannot_mutate_snapshot(local_config):
    backend = JsonlBackend(local_config)
    baseline = query(backend)
    query(backend)[0]['metadata']['tags'].append('injected')
    assert query(backend) == baseline
    publish(local_config.path, [])
    assert backend.reload() == {'version': 2, 'record_count': 0}
    assert query(backend) == []
    publish(local_config.path, NEW)
    assert backend.reload() == {'version': 3, 'record_count': 3}
    assert query(backend)


@pytest.mark.parametrize('payload,key,status,code', [
    ({'knowledge_id': 'handbook'}, None, 401, 1001),
    ({'knowledge_id': 'handbook'}, 'wrong', 403, 1002),
    ({'knowledge_id': 'unknown'}, 'test-key', 404, 2001),
    ({'knowledge_id': ''}, 'test-key', 400, 2002),
    ({'knowledge_id': 42}, 'test-key', 400, 2002),
    ({'knowledge_id': 'handbook', 'path': '/tmp/untrusted'}, 'test-key', 400, 2002),
    ({}, 'test-key', 400, 2002), ([], 'test-key', 400, 2002),
    (b'{"knowledge_id":"handbook","knowledge_id":"other"}', 'test-key', 400, 2002),
])
def test_reload_auth_and_request_validation(json_endpoint, payload, key, status, code):
    actual, error = post_reload(json_endpoint, payload, key)
    assert (actual, error['error_code']) == (status, code)


def test_reload_rejects_qdrant_without_touching_it(qdrant_app, body):
    with running(qdrant_app) as endpoint:
        before = request(endpoint, body, 'test-key')
        status, error = post_reload(endpoint, {'knowledge_id': 'handbook'})
        assert (status, error['error_code']) == (400, 2002)
        assert request(endpoint, body, 'test-key') == before
