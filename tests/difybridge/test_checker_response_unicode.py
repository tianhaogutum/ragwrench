"""Checker response strings/keys must contain Unicode scalar values everywhere."""
import io
import json
from email.message import Message
from urllib.error import HTTPError
from urllib.request import OpenerDirector
from urllib.response import addinfourl

import pytest

from ragwrench.difybridge import check
from ragwrench.difybridge.cli import main
from .conftest import ROOT
from .test_checker_credentials import arguments
from .test_checker_response_framing import framed_endpoint

INVALID = [
    br'\ud800', br'\udbff', br'\udc00', br'\udfff',
    br'\udfff\ud800', br'\ud800x\udc00', br'\ud800\ud800',
    br'\udc00\udfff', br'\ud83d\ude00\ud800', br'\ud83d\ude00\udfff',
]
VALID = [
    br'\ud800\udc00', br'\udbff\udfff', br'\uD83D\uDE00',
    br'\ud83d\ude00\ud800\udc00', br'\ud7ff\ue000\ufffd',
    br'\\ud800', 'café 日本語 😀 \ufffd'.encode('utf-8'), b'',
]
LOCATIONS = ['primary', 'title', 'metadata-value', 'metadata-key',
             'ignored-value', 'ignored-key', 'nested-value', 'nested-key']
MARKER = 'upstream-private-unicode-marker'


def response_bytes(body, escaped, location):
    """Insert wire escapes without letting json.dumps re-escape the backslash."""
    if location == 'primary':
        if 'records' in body:
            for record in body['records']:
                record['content'] += MARKER
        else:
            body['error_msg'] += MARKER
    elif location == 'title' and 'records' in body:
        # Preserve baseline assertions and corrupt only a later record title.
        for record in body['records'][1:]:
            record['title'] += MARKER
    elif location.startswith('metadata') and 'records' in body:
        for record in body['records']:
            record['metadata'][MARKER if location.endswith('key') else 'private'] = (
                'private' if location.endswith('key') else MARKER)
    elif location in ('ignored-key', 'metadata-key'):
        body[MARKER] = None
    elif location == 'nested-key':
        body['extra'] = [None, True, 7, {'deeper': [{MARKER: 'private'}]}]
    elif location == 'nested-value':
        body['extra'] = [None, False, 7, {'deeper': [['private', MARKER]]}]
    else:
        body['extra'] = MARKER
    raw = json.dumps(body, ensure_ascii=False).encode('utf-8')
    if MARKER.encode() not in raw:
        # Empty/one-hit probes may have no target record: still corrupt an
        # ignored field so every deliberately modified response is exercised.
        body['extra'] = MARKER
        raw = json.dumps(body, ensure_ascii=False).encode('utf-8')
    return raw.replace(MARKER.encode(), escaped)


@pytest.mark.parametrize('status', [200, 400])
@pytest.mark.parametrize('escaped', INVALID + VALID)
@pytest.mark.parametrize('location', ['primary', 'metadata-value', 'metadata-key',
                                      'ignored-key', 'nested-value', 'nested-key'])
def test_decoded_response_unicode(status, escaped, location, monkeypatch):
    body = {'records': [{'content': 'private', 'title': 'private', 'metadata': {}}],
            'error_msg': 'private'}
    raw = response_bytes(body, escaped, location)
    headers = Message()
    headers['Content-Type'] = 'application/json'
    headers['Content-Length'] = str(len(raw))
    response = addinfourl(io.BytesIO(raw), headers, 'http://127.0.0.1/retrieval', status)

    def opened(*args, **kwargs):
        if status == 400:
            raise HTTPError(response.url, status, 'private', headers, response)
        return response

    monkeypatch.setattr(OpenerDirector, 'open', opened)
    if escaped in INVALID:
        with pytest.raises(ValueError):
            check.request('http://127.0.0.1', {})
    else:
        assert check.request('http://127.0.0.1', {}) == (status, json.loads(raw))


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('errors_only', [False, True])
@pytest.mark.parametrize('location', LOCATIONS)
@pytest.mark.parametrize('escaped', [br'\ud800', br'\udfff', br'\ud83d\ude00',
                                    'café 日本語 \ufffd'.encode('utf-8')])
def test_cli_response_unicode(mode, errors_only, location, escaped, tmp_path,
                              json_app, monkeypatch, capsys):
    monkeypatch.setenv('CHECK_KEY', 'test-key')
    monkeypatch.setenv('BRIDGE_API_KEY', 'wrong-default')
    flags = arguments(mode, tmp_path)
    saved = tmp_path / 'secret-path-marker.json'
    if mode == 'file':
        saved.write_bytes(b' \n' + saved.read_bytes().replace(b'0.2', b'2e-1') + b'\n')
    original = (saved.read_bytes(), saved.stat().st_mtime_ns) if mode == 'file' else None
    source = ROOT / 'examples/dify-bridge/chunks.jsonl'
    before = source.read_bytes(), source.stat().st_mtime_ns
    state = json_app.backends.items['handbook'].snapshot_status()
    with framed_endpoint(json_app, 'valid', errors_only=errors_only,
                         encode_body=lambda body: response_bytes(body, escaped, location)) as (endpoint, seen):
        code = main(['check', '--endpoint', endpoint, '--key-env', 'CHECK_KEY', *flags,
                     '--expect-count', '2', '--expect-title', 'handbook.md'])
    output = capsys.readouterr()
    report = json.loads(output.out)
    valid = escaped not in INVALID
    assert (code, report['passed']) == (0 if valid else 1, valid)
    assert not output.err and len(report['checks']) == len(seen) == 13
    for index, probe in enumerate(report['checks']):
        affected = not valid and (not errors_only or index >= 4)
        assert probe['passed'] is (not affected)
        if affected:
            assert probe['errors'] == ['request failed or response is not bounded valid JSON']
    assert all(route == '/bridge/retrieval' for route, _ in seen)
    if mode == 'file':
        assert seen[0][1] == original[0]
        assert (saved.read_bytes(), saved.stat().st_mtime_ns) == original
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before
    assert json_app.backends.items['handbook'].snapshot_status() == state
    assert all(s not in output.out + output.err for s in
               ('test-key', 'wrong-default', 'upstream-private', 'vacation policy', 'handbook.md',
                'café', '\\ud800', '\\udfff', endpoint, str(saved)))
