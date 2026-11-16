"""Decoded response numbers must stay finite, even in ignored fields."""
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

OVERFLOW = [b'1e999', b'-1e999', b'1.7976931348623159e308', b'-1.7976931348623159e308']
FINITE = [b'1.7976931348623157e308', b'-1.7976931348623157e308',
          b'5e-324', b'-1e-999', b'0.125', b'0', b'-7', b'9' * 400,
          b'true', b'false', b'null']
MARKER = 'upstream-private-number-marker'
LOCATIONS = ['ignored', 'nested-list', 'metadata']


def response_bytes(body, token, location):
    if location == 'root':
        return token
    if location == 'ignored':
        body['private'] = MARKER
    elif location == 'nested-list':
        body['private'] = [None, True, {'deeper': [False, [MARKER]]}]
    else:
        for record in body.get('records', []):
            record['metadata']['private'] = {'deeper': [MARKER]}
        if not body.get('records'):
            # Error/empty-record responses still exercise a nested number.
            body['private'] = {'deeper': [MARKER]}
    return json.dumps(body).encode().replace(json.dumps(MARKER).encode(), token)


@pytest.mark.parametrize('status', [200, 400])
@pytest.mark.parametrize('token', OVERFLOW + FINITE)
@pytest.mark.parametrize('location', ['root', *LOCATIONS])
def test_decoded_response_numbers(status, token, location, monkeypatch):
    raw = response_bytes({'records': [{'metadata': {}}]}, token, location)
    headers = Message()
    headers['Content-Type'] = 'application/json'
    headers['Content-Length'] = str(len(raw))
    response = addinfourl(io.BytesIO(raw), headers, 'http://127.0.0.1/retrieval', status)

    def opened(*args, **kwargs):
        if status == 400:
            raise HTTPError(response.url, status, 'private', headers, response)
        return response

    monkeypatch.setattr(OpenerDirector, 'open', opened)
    if token in OVERFLOW:
        with pytest.raises(ValueError):
            check.request('http://127.0.0.1', {})
    else:
        assert check.request('http://127.0.0.1', {}) == (status, json.loads(raw))


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('errors_only', [False, True])
@pytest.mark.parametrize('location', LOCATIONS)
@pytest.mark.parametrize('token', [b'1e999', b'-1e999', b'1.7976931348623157e308', b'9' * 400])
def test_cli_response_numbers(mode, errors_only, location, token, tmp_path,
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
                         encode_body=lambda body: response_bytes(body, token, location)) as (endpoint, seen):
        code = main(['check', '--endpoint', endpoint, '--key-env', 'CHECK_KEY', *flags,
                     '--expect-count', '2', '--expect-title', 'handbook.md'])
    output = capsys.readouterr()
    report = json.loads(output.out)
    valid = token not in OVERFLOW
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
               ('test-key', 'wrong-default', 'upstream-private', 'vacation policy',
                'handbook.md', '1e999', endpoint, str(saved)))
