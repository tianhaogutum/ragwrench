"""Complete saved-request CLI contracts; no probes on local input failures."""
import copy
import io
import json

import pytest

from ragwrench.difybridge import check
from ragwrench.difybridge.cli import main
from ragwrench.difybridge.errors import BridgeError
from .test_checker_filters import FILTER, NO_HITS
from .test_embedding_checker import fake_endpoint

LIMIT = 1024 * 1024
BASE = {'knowledge_id': 'handbook', 'query': 'vacation policy',
        'retrieval_setting': {'top_k': 3, 'score_threshold': 0.2}, 'metadata_condition': FILTER}


def invoke(path, endpoint='http://127.0.0.1:1', flags=()):
    try:
        return main(['check', '--endpoint', endpoint, '--request-file', str(path), *flags])
    except SystemExit as exc:
        return exc.code


def saved(tmp_path, body=BASE):
    path = tmp_path / 'secret-path-marker.json'
    path.write_text(json.dumps(body), encoding='utf-8')
    return path


def assert_local_error(path, flags, monkeypatch, capsys):
    monkeypatch.setenv('BRIDGE_API_KEY', 'secret-key-marker')
    seen = []
    monkeypatch.setattr(check, 'request', lambda *a, **k: seen.append(a))
    assert invoke(path, flags=flags) == 2
    output = capsys.readouterr()
    assert not output.out and 'Configuration' in output.err
    assert all(s not in output.err for s in [str(path), 'secret-path-marker',
        'secret-content-marker', 'secret-key-marker', 'secret-inline-marker'])
    assert not seen


@pytest.mark.parametrize('backend', ['json_app', 'qdrant_app'])
def test_saved_filtered_baseline_exact_bytes(backend, request, tmp_path, monkeypatch, capsys):
    app = request.getfixturevalue(backend)
    body = copy.deepcopy(BASE)
    body["retrieval_setting"]["top_k"] = 2
    path = saved(tmp_path, body)
    # Distinguish byte preservation from reconstruction (order, whitespace, 2e-1).
    raw = b' \n' + path.read_bytes().replace(b'0.2', b'2e-1') + b'\n'
    path.write_bytes(raw)
    seen = []
    def reply(route, headers, data):
        assert route == '/retrieval'
        seen.append(data)
        try:
            return 200, {}, app.retrieve(headers.get('Authorization'), data)
        except BridgeError as exc:
            return exc.status, {}, exc.body()
    monkeypatch.setenv('FILE_CHECK_KEY', 'test-key')
    with fake_endpoint(reply) as endpoint:
        assert invoke(path, endpoint, ['--key-env', 'FILE_CHECK_KEY', '--expect-count', '2',
            '--expect-title', 'handbook.md', '--expect-title', 'leave.md']) == 0
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert not output.err and report['passed'] and len(report['checks']) == 13
    assert seen[0] == raw and path.read_bytes() == raw
    parsed = [json.loads(data) for data in seen if data != b'{']
    assert all(b['metadata_condition'] == FILTER for b in parsed[:8])
    assert parsed[1]['retrieval_setting'] == {'top_k': 1, 'score_threshold': 0.0}
    assert parsed[2]['retrieval_setting'] == {'top_k': 3, 'score_threshold': 0.9}
    assert parsed[3]['retrieval_setting'] == {'top_k': 3, 'score_threshold': 1.0}
    assert parsed[-1]['metadata_condition']['conditions'][0]['comparison_operator'] == 'exec'
    assert all(s not in output.out for s in ['test-key', str(path), 'vacation policy', 'handbook.md', 'leave.md'])


@pytest.mark.parametrize('condition,flags,code', [
    (FILTER, ['--expect-count', '1'], 1),
    (FILTER, ['--expect-title', 'secret-content-marker'], 1),
    (NO_HITS, [], 1),
    (NO_HITS, ['--expect-count', '0'], 0),
    (NO_HITS, ['--allow-empty'], 0),
    (NO_HITS, ['--expect-count', '1', '--allow-empty'], 1),
    (NO_HITS, ['--expect-title', 'secret-content-marker', '--allow-empty'], 1),
    (None, ['--expect-count', '2'], 0),  # Omitted filter stays omitted.
])
def test_saved_assertions(json_endpoint, tmp_path, monkeypatch, capsys, condition, flags, code):
    body = copy.deepcopy(BASE)
    if condition is None:
        del body['metadata_condition']
        body['retrieval_setting']['score_threshold'] = 0
    else:
        body['metadata_condition'] = condition
    path = saved(tmp_path, body)
    monkeypatch.setenv('BRIDGE_API_KEY', 'test-key')
    assert invoke(path, json_endpoint, flags) == code
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report['passed'] is (code == 0) and not output.err
    assert len(report['checks']) == 13
    assert all(s not in output.out for s in ['secret-content-marker', str(path), 'test-key', 'vacation policy'])
    if '--expect-count' in flags and '0' in flags:
        assert any('Empty baseline expected' in s for s in report['limitations'])


INVALID = [None, [], {}, {'knowledge_id': 'x'},
    {**BASE, 'unknown': 'secret-content-marker'}, {**BASE, 'metadata_condition': None}]
for field in ['knowledge_id', 'query', 'retrieval_setting']:
    INVALID.append({k: v for k, v in BASE.items() if k != field})
for field, values in [('knowledge_id', ['', 'x' * 257, False, 4]),
                      ('query', ['', ' ', 'x' * 16385, None, []]),
                      ('retrieval_setting', [None, [], {}, {'top_k': 3}, {'score_threshold': 0},
                          {**BASE['retrieval_setting'], 'extra': True}])]:
    INVALID.extend({**BASE, field: value} for value in values)
for field, values in [('top_k', [0, 101, True, 3.0, '3']),
                      ('score_threshold', [-1, 1.1, True, None, '0'])]:
    INVALID.extend({**BASE, 'retrieval_setting': {**BASE['retrieval_setting'], field: v}} for v in values)
for condition in [[], {}, {'conditions': [], 'extra': True}, {'logical_operator': [], 'conditions': []},
    {'conditions': [FILTER['conditions'][0]] * 65},
    {'conditions': [{'name': 'lang', 'comparison_operator': 'exec', 'value': 'secret-content-marker'}]},
    {'conditions': [{'name': 'page', 'comparison_operator': '=', 'value': True}]},
    {'conditions': [{'name': 'lang', 'comparison_operator': 'in', 'value': [1]}]},
    {'conditions': [{'name': 'updated', 'comparison_operator': 'before', 'value': 'secret-content-marker'}]},
    {'conditions': [{'conditions': []}]}]:
    INVALID.append({**BASE, 'metadata_condition': condition})


@pytest.mark.parametrize('body', INVALID)
def test_invalid_saved_schema(body, tmp_path, monkeypatch, capsys):
    assert_local_error(saved(tmp_path, body), [], monkeypatch, capsys)


@pytest.mark.parametrize('raw', [b'', b'{secret-content-marker', b'\xff', b'{} {}',
    b'[' * 1500 + b']' * 1500,
    json.dumps(BASE).replace('0.2', 'NaN').encode(),
    json.dumps(BASE).replace('0.2', 'Infinity').encode(),
    json.dumps(BASE).replace('0.2', '-Infinity').encode(),
    json.dumps(BASE).replace('0.2', '1e999').encode(),
    json.dumps(BASE).replace('"top_k": 3', '"top_k": 3, "top_k": 2').encode(),
    json.dumps(BASE).replace('"query":', '"query":"secret-content-marker", "query":').encode(),
    json.dumps(BASE).replace('"value":', '"value":"secret-content-marker", "value":').encode(),
    # Even values ignored by an empty comparison must be finite JSON.
    json.dumps({**BASE, 'metadata_condition': {'conditions': [
        {'name': 'lang', 'comparison_operator': 'empty', 'value': float('inf')}]}}).encode(),
])
def test_invalid_saved_json(raw, tmp_path, monkeypatch, capsys):
    path = saved(tmp_path)
    path.write_bytes(raw)
    assert_local_error(path, [], monkeypatch, capsys)


@pytest.mark.parametrize('kind', ['missing', 'directory', 'oversized', 'unreadable'])
def test_file_errors(kind, tmp_path, monkeypatch, capsys):
    path = tmp_path / 'secret-path-marker'
    if kind == 'directory':
        path.mkdir()
    elif kind == 'oversized':
        path.write_bytes(b' ' * (LIMIT + 1))
    elif kind == 'unreadable':
        path.write_bytes(b'{}')
        def denied(*a, **kw):
            raise PermissionError('secret-path-marker secret-content-marker')
        monkeypatch.setattr('builtins.open', denied)
    assert_local_error(path, [], monkeypatch, capsys)


@pytest.mark.parametrize('flags', [
    ['--knowledge-id', 'secret-inline-marker'], ['--query', 'secret-inline-marker'],
    ['--metadata-condition', '{}'], ['--top-k', '3'], ['--score-threshold', '0'],
    ['--expect-count', '4'], ['--expect-count', '0', '--expect-title', 'secret-inline-marker'],
    ['--top-k', 'secret-inline-marker'], ['--request-file'], ['--token', 'secret-inline-marker'],
])
def test_file_conflicts_and_bad_arguments(tmp_path, monkeypatch, capsys, flags):
    assert_local_error(saved(tmp_path), flags, monkeypatch, capsys)


@pytest.mark.parametrize('flags', [[], ['--knowledge-id', 'handbook'], ['--query', 'secret-inline-marker']])
def test_inline_requires_id_and_query(flags, monkeypatch, capsys):
    monkeypatch.setenv('BRIDGE_API_KEY', 'test-key')
    seen = []
    monkeypatch.setattr(check, 'request', lambda *a, **kw: seen.append(a))
    try:
        code = main(['check', '--endpoint', 'http://127.0.0.1:1', *flags])
    except SystemExit as exc:
        code = exc.code
    assert code == 2 and not seen
    output = capsys.readouterr()
    assert not output.out and 'secret-inline-marker' not in output.err


@pytest.mark.parametrize('size', [LIMIT, LIMIT + 1])
def test_bounded_single_read(size, tmp_path, monkeypatch, capsys):
    raw = json.dumps(BASE).encode()
    raw += b' ' * (size - len(raw))
    class Tracked(io.BytesIO):
        def read(self, n=-1):
            reads.append(n)
            return super().read(n)
    reads, opens, calls = [], [], []
    def opened(*args, **kwargs):
        opens.append((args, kwargs))
        return Tracked(raw)
    monkeypatch.setattr('builtins.open', opened)
    monkeypatch.setenv('BRIDGE_API_KEY', 'test-key')
    def response(endpoint, payload, *args, **kwargs):
        calls.append(payload)
        return 400, {'error_code': 2002, 'error_msg': 'invalid'}
    monkeypatch.setattr(check, 'request', response)
    assert invoke(tmp_path / 'secret-path-marker') == (1 if size == LIMIT else 2)
    assert reads == [LIMIT + 1] and len(opens) == 1
    assert len(calls) == (13 if size == LIMIT else 0)
    if calls:
        assert calls[0] == raw
    capsys.readouterr()


def test_utf8_request_not_limited_by_reserialized_escapes(json_endpoint, tmp_path, monkeypatch, capsys):
    body = copy.deepcopy(BASE)
    # Valid UTF-8 under 1 MiB; escaping to ASCII would exceed it. Derived
    # probes must also fit the real HTTP server's byte limit.
    body['metadata_condition'] = {'conditions': [{'name': 'lang', 'comparison_operator': 'in',
        'value': ['\u4e2d' * 4096] * 60}]}
    raw = json.dumps(body, ensure_ascii=False).encode('utf-8')
    assert len(raw) < LIMIT < len(json.dumps(body).encode())
    path = saved(tmp_path)
    path.write_bytes(raw)
    monkeypatch.setenv('BRIDGE_API_KEY', 'test-key')
    assert invoke(path, json_endpoint, ['--expect-count', '0']) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)['passed'] and not output.err
