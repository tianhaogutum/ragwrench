"""Unsupported or ambiguous content codings fail without reading/decompressing."""
import io
import json
from email.message import Message
from urllib.error import HTTPError
from urllib.request import OpenerDirector
from urllib.response import addinfourl

import pytest

from ragwrench.difybridge import check, snapshot
from ragwrench.difybridge.cli import main
from .conftest import ROOT
from .test_checker_credentials import arguments
from .test_checker_response_framing import framed_endpoint

VALID_CODINGS = [(), ('identity',), ('IDENTITY',), ('IdEnTiTy',)]
INVALID_CODINGS = [
    ('gzip',), ('br',), ('deflate',), ('upstream-private',), ('',),
    ('identity', 'identity'), ('identity', 'gzip'), ('gzip', 'identity'),
    ('identity', ''), ('identity, identity',), ('gzip, br',), ('identity, gzip',),
    ('identity; private=value',), ('identity ',), ('identity\t',), ('idéntity',),
]


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('errors_only', [False, True])
@pytest.mark.parametrize('codings', VALID_CODINGS + INVALID_CODINGS)
def test_checker_content_encoding(mode, errors_only, codings, tmp_path, json_app, monkeypatch, capsys):
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
                         content_encodings=codings) as (endpoint, seen):
        code = main(['check', '--endpoint', endpoint, '--key-env', 'CHECK_KEY', *flags,
                     '--expect-count', '2', '--expect-title', 'leave.md'])
    output = capsys.readouterr()
    report = json.loads(output.out)
    valid = codings in VALID_CODINGS
    assert (code, report['passed']) == (0 if valid else 1, valid)
    assert not output.err and len(report['checks']) == len(seen) == 13
    for index, probe in enumerate(report['checks']):
        passing = valid or (errors_only and index < 4)
        assert probe['passed'] is passing
        if not passing:
            assert probe['errors'] == ['request failed or response is not bounded valid JSON']
    assert all(route == '/bridge/retrieval' for route, _ in seen)
    if mode == 'file':
        assert seen[0][1] == original[0]
        assert (saved.read_bytes(), saved.stat().st_mtime_ns) == original
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before
    assert json_app.backends.items['handbook'].snapshot_status() == state
    assert all(s not in output.out + output.err for s in
               ('test-key', 'wrong-default', 'upstream-private', 'vacation policy', 'leave.md', endpoint, str(saved)))


@pytest.mark.parametrize('codings', VALID_CODINGS + INVALID_CODINGS)
def test_snapshot_content_encoding(codings, json_app, monkeypatch, capsys):
    monkeypatch.setenv('CHECK_KEY', 'test-key')
    monkeypatch.setenv('BRIDGE_API_KEY', 'wrong-default')
    with framed_endpoint(json_app, 'valid', content_encodings=codings) as (endpoint, seen):
        code = main(['snapshot-status', '--endpoint', endpoint, '--key-env', 'CHECK_KEY',
                     '--knowledge-id', 'handbook'])
    output = capsys.readouterr()
    assert len(seen) == 1
    if codings in VALID_CODINGS:
        assert code == 0 and not output.err
        assert json.loads(output.out) == {'knowledge_id': 'handbook', 'version': 1, 'record_count': 4}
    else:
        assert code == 1 and not output.out
        assert output.err == 'Snapshot status failed: request rejected, unavailable or invalid response\n'
    assert all(s not in output.out + output.err for s in ('test-key', 'wrong-default', 'private', endpoint))


@pytest.mark.parametrize('client', ['check-success', 'check-error', 'snapshot'])
@pytest.mark.parametrize('codings', INVALID_CODINGS + [(' identity',)])
def test_invalid_content_encoding_rejected_before_read(client, codings, monkeypatch):
    monkeypatch.setenv('CHECK_KEY', 'test-key')
    class Unreadable(io.BytesIO):
        def read(self, n=-1):
            pytest.fail('invalid content coding must fail before body read')
    headers = Message()
    headers['Content-Type'] = 'application/json'
    headers['Content-Length'] = '2'
    for index, value in enumerate(codings):
        headers['Content-Encoding' if index == 0 else 'content-encoding'] = value
    if client == 'check-error':
        response = HTTPError('http://127.0.0.1', 400, 'private', headers, Unreadable(b'{}'))
        def open_response(*args, **kwargs):
            raise response
    else:
        response = addinfourl(Unreadable(b'{}'), headers, 'http://127.0.0.1', 200)
        def open_response(*args, **kwargs):
            return response
    monkeypatch.setattr(OpenerDirector, 'open', open_response)
    with pytest.raises(snapshot.SnapshotStatusError if client == 'snapshot' else ValueError):
        if client == 'snapshot':
            snapshot.snapshot_status('http://127.0.0.1', 'handbook', key_env='CHECK_KEY')
        else:
            check.request('http://127.0.0.1', {})
    assert response.closed
