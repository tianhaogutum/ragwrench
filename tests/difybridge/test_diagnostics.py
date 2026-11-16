"""Request-scoped diagnostics over the real bridge HTTP/SDK boundaries."""
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace

import pytest
from qdrant_client import models as m

from .conftest import running
from ragwrench.difybridge.embedding import hash_embed
from .test_http_cli_boundaries import raw_http

PATH = '/retrieval/diagnostics'


def post(endpoint, body, *, auth='Bearer test-key', path=PATH):
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    headers = [('Content-Type', 'application/json'), ('Content-Length', str(len(raw)))]
    if auth is not None:
        headers.append(('Authorization', auth))
    return raw_http(endpoint, path=path, headers=headers, data=raw)


@pytest.fixture
def ranked_qdrant(qdrant_app, body):
    backend = qdrant_app.backends.items['handbook']
    vector = hash_embed(body['query'])
    axis = next(i for i, value in enumerate(vector) if value == 0)
    backend.client.delete('handbook', points_selector=m.PointIdsList(points=[0, 1, 2, 3]))
    points = []
    for i in range(40):
        v = vector.copy()
        v[axis] = i / 10
        points.append(m.PointStruct(id=i, vector={'text': v}, payload={
            'content': f'candidate {i}', 'title': str(i),
            'metadata': {'team': 'keep' if i == 30 else 'discard', 'internal_secret': 'never-export'},
        }))
    backend.client.upsert('handbook', points=points)
    backend.config = replace(backend.config, max_candidates=40, overfetch=1)
    body['retrieval_setting']['top_k'] = 1
    body['metadata_condition'] = {'conditions': [
        {'name': 'team', 'comparison_operator': 'start with', 'value': 'keep'}]}
    return qdrant_app, backend, body


def test_paginated_diagnostics_and_budget_http(ranked_qdrant, monkeypatch):
    app, backend, body = ranked_qdrant
    calls = []
    original = backend.client.query_points
    def query_points(**kwargs):
        calls.append((kwargs['offset'], kwargs['limit'], kwargs['with_payload']))
        return original(**kwargs)
    monkeypatch.setattr(backend.client, 'query_points', query_points)
    with running(app) as endpoint:
        status, result = post(endpoint, body)
        assert status == 200
        assert calls == [(0, 16, True), (16, 16, True)]
        assert result['records'][0]['title'] == '30'
        d = result['diagnostics']
        assert d == {
            'backend': 'qdrant', 'query_characters': len(body['query']),
            'top_k': 1, 'score_threshold': 0.0,
            'filter': {'logical_operator': 'and', 'condition_count': 1, 'native_condition_count': 0},
            'candidate_budget': 40, 'page_size': 16, 'pages_fetched': 2,
            'candidates_fetched': 32, 'candidates_evaluated': 31,
            'filter_rejected': 30, 'threshold_rejected': 0, 'lexical_misses': 0,
            'eligible_count': 1, 'returned_count': 1, 'budget_probe_points': 0,
            'stop_reason': 'top_k',
        }
        assert 'never-export' not in json.dumps(result)
        assert post(endpoint, body, path='/retrieval') == (200, {'records': result['records']})
        backend.config = replace(backend.config, max_candidates=16)
        calls.clear()
        status, failure = post(endpoint, body)
        assert calls == [(0, 16, True), (16, 1, False)]
        assert (status, failure['error_code']) == (422, 2003)
        assert 'records' not in failure
        d = failure['diagnostics']
        assert (d['pages_fetched'], d['candidates_fetched'], d['candidates_evaluated']) == (1, 17, 16)
        assert d['budget_probe_points'] == 1 and d['stop_reason'] == 'candidate_budget'
        assert d['filter_rejected'] == 16 and d['returned_count'] == 0
        ordinary = post(endpoint, body, path='/retrieval')
        assert ordinary == (422, {k: v for k, v in failure.items() if k != 'diagnostics'})


def test_native_filter_and_or_fallback(qdrant_app, body):
    body['metadata_condition'] = {'conditions': [
        {'name': 'lang', 'comparison_operator': 'is', 'value': 'en'},
        {'name': 'team', 'comparison_operator': 'start with', 'value': 'people'}]}
    with running(qdrant_app) as endpoint:
        status, result = post(endpoint, body)
        assert status == 200
        assert result['diagnostics']['filter'] == {
            'logical_operator': 'and', 'condition_count': 2, 'native_condition_count': 1}
        assert result['diagnostics']['stop_reason'] == 'source_exhausted'
        body['metadata_condition']['logical_operator'] = 'or'
        status, result = post(endpoint, body)
        assert status == 200
        assert result['diagnostics']['filter']['native_condition_count'] == 0


@pytest.mark.parametrize('budget,threshold,reason,probe', [
    (40, 0, 'source_exhausted', 0), (41, 0, 'source_exhausted', 0),
    (40, 0.99, 'score_threshold', 0), (16, 0.77, 'score_threshold', 1),
])
def test_pagination_terminal_reasons(ranked_qdrant, budget, threshold, reason, probe):
    app, backend, body = ranked_qdrant
    backend.config = replace(backend.config, max_candidates=budget)
    body['retrieval_setting']['score_threshold'] = threshold
    body['metadata_condition']['conditions'][0]['value'] = 'absent'
    # At 0.77, points 0..15 clear the threshold; probe point 16 is below it.
    with running(app) as endpoint:
        status, result = post(endpoint, body)
        assert status == 200
        assert result['records'] == []
        assert result['diagnostics']['stop_reason'] == reason
        assert result['diagnostics']['budget_probe_points'] == probe


def test_jsonl_counts_and_request_isolation(json_app, body):
    body['query'] = 'vacation'
    body['metadata_condition'] = {'conditions': [
        {'name': 'lang', 'comparison_operator': 'is', 'value': 'en'}]}
    no_hits = deepcopy(body)
    no_hits['query'] = 'zzzznomatch'
    with running(json_app) as endpoint:
        status, result = post(endpoint, body)
        assert status == 200 and len(result['records']) == 2
        d = result['diagnostics']
        assert d == {
            'backend': 'jsonl', 'query_characters': 8, 'top_k': 3, 'score_threshold': 0.0,
            'filter': {'logical_operator': 'and', 'condition_count': 1, 'native_condition_count': 0},
            'candidate_budget': None, 'page_size': None, 'pages_fetched': 0,
            'candidates_fetched': 4, 'candidates_evaluated': 4,
            'filter_rejected': 1, 'threshold_rejected': 0, 'lexical_misses': 1,
            'eligible_count': 2, 'returned_count': 2, 'budget_probe_points': 0,
            'stop_reason': 'corpus_scanned', 'snapshot_version': 1,
        }
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(lambda b: post(endpoint, b), [body, no_hits] * 4))
        for index, (status, response) in enumerate(responses):
            assert status == 200
            assert response['diagnostics']['eligible_count'] == (0 if index % 2 else 2)
            assert response['diagnostics']['lexical_misses'] == (3 if index % 2 else 1)
        body['retrieval_setting']['score_threshold'] = 1
        status, response = post(endpoint, body)
        assert status == 200 and response['records'] == []
        assert response['diagnostics']['threshold_rejected'] == 2


@pytest.mark.parametrize('auth,status,code', [
    (None, 401, 1001), ('Basic private-token', 401, 1001), ('Bearer wrong-private-token', 403, 1002),
])
def test_redacted_auth_before_parse_or_retrieval(json_app, monkeypatch, capsys, auth, status, code):
    def never_retrieve(*args, **kwargs):
        pytest.fail('Unauthenticated request reached backend')
    monkeypatch.setattr(json_app.backends.items['handbook'], 'retrieve', never_retrieve)
    with running(json_app) as endpoint:
        actual, result = post(endpoint, b'{private-query-and-filter', auth=auth)
        assert (actual, result['error_code']) == (status, code)
        assert set(result) == {'error_code', 'error_msg'}
        assert 'private' not in json.dumps(result)
    captured = capsys.readouterr()
    assert 'private' not in captured.out + captured.err


def test_diagnostics_invalid_and_backend_errors(json_app, body, monkeypatch):
    with running(json_app) as endpoint:
        invalid = deepcopy(body)
        invalid['metadata_condition'] = {'conditions': [
            {'name': 'private-field', 'comparison_operator': 'is', 'value': 'private-value'}]}
        status, result = post(endpoint, invalid)
        assert (status, result['error_code']) == (400, 2002)
        assert set(result) == {'error_code', 'error_msg'} and 'private' not in json.dumps(result)
        def broken(*args, **kwargs):
            raise RuntimeError('private-provider-key private-query')
        monkeypatch.setattr(json_app.backends.items['handbook'], 'retrieve', broken)
        status, result = post(endpoint, body)
        assert (status, result['error_code']) == (500, 3002)
        assert set(result) == {'error_code', 'error_msg'} and 'private' not in json.dumps(result)


def test_budget_does_not_expose_partial_records(ranked_qdrant):
    app, backend, body = ranked_qdrant
    backend.config = replace(backend.config, max_candidates=32)
    body['retrieval_setting']['top_k'] = 2
    with running(app) as endpoint:
        status, result = post(endpoint, body)
        assert (status, result['error_code']) == (422, 2003)
        assert 'records' not in result
        d = result['diagnostics']
        assert d['eligible_count'] == 1 and d['returned_count'] == 0
        assert (d['candidates_fetched'], d['candidates_evaluated'], d['filter_rejected']) == (33, 32, 31)


def test_diagnostics_qdrant_http_protocol(ranked_qdrant, monkeypatch):
    """Pinned qdrant-client 1.15.1 transport, two HTTP boundaries, original fixtures."""
    from ragwrench.difybridge.config import Config
    from ragwrench.difybridge.service import Application
    from .test_embedding_checker import fake_endpoint
    app, backend, body = ranked_qdrant
    info = backend.client.get_collection('handbook').model_dump(mode='json')
    seen = []
    def reply(path, headers, raw):
        if path == '/':
            return 200, {}, {'title': 'qdrant', 'version': '1.15.1'}
        if path == '/collections/handbook':
            return 200, {}, {'result': info, 'status': 'ok', 'time': 0.001}
        if path == '/collections/handbook/points/query':
            query = json.loads(raw)
            seen.append(query)
            offset, limit = query.get('offset', 0), query['limit']
            points = [{'id': i, 'version': 0, 'score': 1 - i / 100,
                       'payload': {'content': f'candidate {i}', 'title': str(i),
                                   'metadata': {'team': 'keep' if i == 30 else 'discard'}}}
                      for i in range(offset, min(offset + limit, 40))]
            return 200, {}, {'result': {'points': points}, 'status': 'ok', 'time': 0.001}
        return 404, {}, {'status': {'error': 'unknown route'}}
    # A dedicated local protocol fixture supplies deterministic upstream errors too.
    with fake_endpoint(reply) as upstream:
        remote = Application(Config({'handbook': replace(backend.config, path=None, url=upstream)}), 'test-key')
        with running(remote) as endpoint:
            status, result = post(endpoint, body)
            assert status == 200 and result['records'][0]['title'] == '30'
            assert result['diagnostics']['pages_fetched'] == 2
            assert result['diagnostics']['filter_rejected'] == 30
            assert [(q['offset'], q['limit']) for q in seen] == [(0, 16), (16, 16)]
            assert all(q['using'] == 'text' and q.get('filter') is None for q in seen)
            def broken(*args, **kwargs):
                raise RuntimeError('private-upstream-url private-token')
            monkeypatch.setattr(remote.backends.items['handbook'].client, 'query_points', broken)
            status, result = post(endpoint, body)
            assert (status, result['error_code']) == (503, 3001)
            assert set(result) == {'error_code', 'error_msg'}
            assert 'private' not in json.dumps(result)
