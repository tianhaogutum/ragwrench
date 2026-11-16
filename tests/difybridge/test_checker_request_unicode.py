"""Unicode scalar preflight is checker-local and preserves saved request bytes."""
import copy
import json

import pytest

from ragwrench.difybridge import check
from ragwrench.difybridge.cli import main
from ragwrench.difybridge.errors import ConfigError
from .conftest import ROOT
from .test_checker_request_file import BASE
from .test_checker_response_framing import framed_endpoint

INVALID = ['\ud800', '\udfff', '\ud800x', '\udfff\ud800']
LOCATIONS = ['knowledge_id', 'query', 'name', 'value', 'list-value',
             'ignored-value', 'ignored-key', 'filter-key', 'condition-key', 'title']


def request_body(text, location):
    body = copy.deepcopy(BASE)
    condition = body['metadata_condition']['conditions'][0]
    titles = []
    if location in ('knowledge_id', 'query'):
        body[location] = 'private-marker' + text
    elif location == 'name':
        condition['name'] = 'private-marker' + text
    elif location == 'value':
        condition['value'] = 'private-marker' + text
    elif location == 'list-value':
        condition.update(comparison_operator='in', value=['en', 'private-marker' + text])
    elif location.startswith('ignored-'):
        # empty/not empty allow arbitrary JSON values: visit even ignored data.
        condition['comparison_operator'] = 'not empty'
        condition['value'] = {'deeper': [None, {'private-marker' + text: True}]} if location == 'ignored-key' else {'deeper': [True, ['private-marker' + text]]}
    elif location == 'filter-key':
        body['metadata_condition']['private-marker' + text] = None
    elif location == 'condition-key':
        condition['private-marker' + text] = None
    else:
        titles = ['private-marker' + text]
    return body, titles


def flags_for(body, titles, mode, path):
    if mode == 'file':
        path.write_bytes(json.dumps(body).encode())
        flags = ['--request-file', str(path)]
    else:
        flags = ['--knowledge-id', body['knowledge_id'], '--query', body['query'],
                 '--metadata-condition', json.dumps(body['metadata_condition']),
                 '--top-k', str(body['retrieval_setting']['top_k']),
                 '--score-threshold', str(body['retrieval_setting']['score_threshold'])]
    return flags + [arg for title in titles for arg in ('--expect-title', title)]


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('location', LOCATIONS)
@pytest.mark.parametrize('text', INVALID)
def test_cli_rejects_surrogates_before_any_probe(mode, location, text, tmp_path, monkeypatch, capsys):
    body, titles = request_body(text, location)
    path = tmp_path / 'private-path.json'
    flags = flags_for(body, titles, mode, path)
    monkeypatch.setenv('UNICODE_KEY', 'test-key')
    seen = []
    monkeypatch.setattr(check, 'request', lambda *a, **k: seen.append(a))
    code = main(['check', '--endpoint', 'http://127.0.0.1:1', '--key-env', 'UNICODE_KEY', *flags])
    output = capsys.readouterr()
    assert code == 2 and output.out == '' and 'Configuration failure:' in output.err
    assert not seen
    assert all(s not in output.err for s in ('private', 'test-key', 'UNICODE_KEY', text, '\\ud800', '\\udfff'))


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('location', LOCATIONS)
@pytest.mark.parametrize('text', INVALID[:2])
def test_library_rejects_surrogates_with_config_error(mode, location, text, tmp_path, monkeypatch):
    body, titles = request_body(text, location)
    path = tmp_path / 'private-path.json'
    path.write_bytes(json.dumps(body).encode())
    seen = []
    monkeypatch.setattr(check, 'request', lambda *a, **k: seen.append(a))
    with pytest.raises(ConfigError) as exc:
        if mode == 'file':
            check.check_request_file('http://127.0.0.1:1', 'test-key', path, expect_titles=titles)
        else:
            check.check_endpoint('http://127.0.0.1:1', 'test-key', body['knowledge_id'], body['query'],
                                 metadata_condition=body['metadata_condition'], expect_titles=titles)
    assert not seen and 'private' not in str(exc.value) and text not in str(exc.value)


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('ensure_ascii', [True, False])
@pytest.mark.parametrize('text', ['😀', 'café 日本語', '\ufffd', r'\ud800\udfff', 'e\u0301'])
def test_valid_scalar_requests_preserve_bytes(mode, ensure_ascii, text, tmp_path, json_app, monkeypatch, capsys):
    body = copy.deepcopy(BASE)
    body['query'] += ' ' + text
    body['metadata_condition']['conditions'][0].update(comparison_operator='is not', value=text)
    raw = b' \n' + json.dumps(body, ensure_ascii=ensure_ascii, indent=1).encode().replace(b'0.2', b'2e-1') + b'\n'
    path = tmp_path / 'private-request.json'
    flags = flags_for(body, [], mode, path)
    path.write_bytes(raw)
    original = path.read_bytes(), path.stat().st_mtime_ns
    corpus = ROOT / 'examples/dify-bridge/chunks.jsonl'
    before = corpus.read_bytes(), corpus.stat().st_mtime_ns
    state = json_app.backends.items['handbook'].snapshot_status()
    monkeypatch.setenv('CHECK_KEY', 'test-key')
    monkeypatch.setenv('BRIDGE_API_KEY', 'wrong-default')
    with framed_endpoint(json_app, 'valid') as (endpoint, seen):
        code = main(['check', '--endpoint', endpoint, '--key-env', 'CHECK_KEY', *flags,
                     '--expect-count', '2', '--expect-title', 'handbook.md'])
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert code == 0 and report['passed'] and not output.err
    assert len(seen) == len(report['checks']) == 13
    assert json.loads(seen[0][1]) == body
    if mode == 'file':
        assert seen[0][1] == raw
    assert all(json.loads(raw)['metadata_condition'] == body['metadata_condition'] for _, raw in seen[:8])
    assert (path.read_bytes(), path.stat().st_mtime_ns) == original
    assert (corpus.read_bytes(), corpus.stat().st_mtime_ns) == before
    assert json_app.backends.items['handbook'].snapshot_status() == state
    assert all(s not in output.out for s in ('test-key', 'wrong-default', 'vacation policy', 'handbook.md', str(path)))


def test_library_visits_tuple_values_and_cyclic_inputs(monkeypatch):
    seen = []
    monkeypatch.setattr(check, 'request', lambda *a, **k: seen.append(a))
    cycle = []; cycle.append(cycle)
    for value in [('private\ud800',), cycle]:
        condition = {'conditions': [{'name': 'lang', 'comparison_operator': 'empty', 'value': value}]}
        with pytest.raises(ConfigError):
            check.check_endpoint('http://127.0.0.1:1', 'test-key', 'handbook', 'policy', metadata_condition=condition)
    assert not seen
