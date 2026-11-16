import json
from urllib.request import urlopen

import pytest

from ragwrench.difybridge.check import check_endpoint, request
from ragwrench.difybridge.errors import BridgeError
from .conftest import running, retrieve


def test_jsonl_http_fixture(json_endpoint, body):
    status, result = request(json_endpoint, body, 'test-key')
    assert status == 200
    assert len(result['records']) == 2
    assert result['records'][0]['title'] == 'handbook.md'
    assert all(isinstance(r['metadata'], dict) and 'internal_secret' not in r['metadata'] for r in result['records'])
    scores = [r['score'] for r in result['records']]
    assert scores == sorted(scores, reverse=True)
    body['retrieval_setting'] = {'top_k': 1, 'score_threshold': scores[0]}
    assert request(json_endpoint, body, 'test-key')[1]['records'] == result['records'][:1]
    body['retrieval_setting']['score_threshold'] = 1
    assert request(json_endpoint, body, 'test-key')[1] == {'records': []}
    with urlopen(json_endpoint + '/healthz') as response:
        assert json.load(response) == {'status': 'ok'}


@pytest.mark.parametrize('auth,status,code', [(None, 401, 1001), ('', 401, 1001), ('Basic x', 401, 1001),
    ('Bearer ', 401, 1001), ('Bearer x y', 401, 1001), ('Bearer wrong', 403, 1002), ('Bearer 秘密', 403, 1002)])
def test_auth(json_app, body, auth, status, code):
    with pytest.raises(BridgeError) as e:
        json_app.retrieve(auth, json.dumps(body).encode())
    assert (e.value.status, e.value.code) == (status, code)
    assert '秘密' not in e.value.message


@pytest.mark.parametrize('payload', [b'{', b'null', b'[]', b'{}', b'\xff', b'{"a":NaN}', b'[[[['])
def test_malformed(json_endpoint, payload):
    status, result = request(json_endpoint, payload, 'test-key')
    assert status == 400
    assert result['error_code'] == 2002 and result['error_msg']


@pytest.mark.parametrize('setting', [{}, {'top_k': 0, 'score_threshold': 0}, {'top_k': -1, 'score_threshold': 0},
    {'top_k': True, 'score_threshold': 0}, {'top_k': 101, 'score_threshold': 0}, {'top_k': 1.5, 'score_threshold': 0},
    {'top_k': 1, 'score_threshold': -0.1}, {'top_k': 1, 'score_threshold': 2}, {'top_k': 1, 'score_threshold': True}])
def test_settings(json_endpoint, body, setting):
    body['retrieval_setting'] = setting
    status, result = request(json_endpoint, body, 'test-key')
    assert (status, result['error_code']) == (400, 2002)


def test_unknown_and_bad_filters(json_endpoint, body):
    body['knowledge_id'] = 'missing'
    assert request(json_endpoint, body, 'test-key') == (404, {'error_code': 2001, 'error_msg': 'Knowledge base not found'})
    body['knowledge_id'] = 'handbook'
    for raw in [None, {}, {'conditions': [], 'logical_operator': 'xor'}, {'conditions': [{'name': 'lang', 'comparison_operator': 'exec'}]},
                {'conditions': [{'name': 'internal_secret', 'comparison_operator': 'is', 'value': 'excluded'}]}]:
        body['metadata_condition'] = raw
        assert request(json_endpoint, body, 'test-key')[0] == 400


def test_no_lexical_hits(json_app, body):
    body['query'] = 'zzzzzznomatch'
    assert retrieve(json_app, body) == []


def test_conformance_jsonl(json_endpoint):
    report = check_endpoint(json_endpoint, 'test-key', 'handbook', 'vacation')
    assert report['passed'], report
    assert len(report['checks']) == 13


def test_qdrant_http(qdrant_app, body):
    with running(qdrant_app) as endpoint:
        status, result = request(endpoint, body, 'test-key')
        assert status == 200 and result['records'][0]['title'] == 'handbook.md'
        assert check_endpoint(endpoint, 'test-key', 'handbook', 'vacation')['passed']
        body['metadata_condition'] = {'conditions': [{'name': 'lang', 'comparison_operator': 'is', 'value': 'de'}]}
        result = request(endpoint, body, 'test-key')[1]['records']
        assert len(result) == 1 and result[0]['title'] == 'handbuch.md'
