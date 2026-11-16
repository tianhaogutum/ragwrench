"""One unambiguous JSON response media type, on success and checker errors."""
import io
import json
from email.message import Message
from urllib.request import OpenerDirector
from urllib.response import addinfourl

import pytest

from ragwrench.difybridge import check, snapshot
from ragwrench.difybridge.cli import main
from .conftest import ROOT
from .test_checker_credentials import arguments
from .test_checker_response_framing import framed_endpoint

VALID_TYPES = [
    ('application/json',),
    ('Application/JSON',),
    ('application/json; charset=utf-8',),
    (' APPLICATION/JSON \t; Charset = "UTF-8"; profile="private, value; v=1" \t',),
    ('application/json; note="escaped\\\"quote,comma"',),
]
INVALID_TYPES = [
    ('application/json', 'text/plain'),
    ('application/json', 'application/json'),
    ('application/json', 'APPLICATION/JSON; charset=utf-8'),
    ('text/plain', 'application/json'),
    ('application/json', ''),
    ('application/json, text/plain',),
    ('application/json, application/json',),
    ('application/json; charset=utf-8, text/plain',),
    ('application/json; profile="private, value", text/plain',),
    (), ('',), ('text/plain',), ('application/json-seq',),
    ('application/json;',), ('application/json; charset',),
    ('application/json; charset="unterminated',),
]


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('types', VALID_TYPES + INVALID_TYPES)
def test_checker_content_type(mode, types, tmp_path, json_app, monkeypatch, capsys):
    monkeypatch.setenv('CHECK_KEY', 'test-key')
    monkeypatch.setenv('BRIDGE_API_KEY', 'wrong-default')
    flags = arguments(mode, tmp_path)
    path = tmp_path / 'secret-path-marker.json'
    if mode == 'file':
        path.write_bytes(b' \n' + path.read_bytes().replace(b'0.2', b'2e-1') + b'\n')
    source = ROOT / 'examples/dify-bridge/chunks.jsonl'
    before = source.read_bytes(), source.stat().st_mtime_ns
    state = json_app.backends.items['handbook'].snapshot_status()
    with framed_endpoint(json_app, 'valid', content_types=types) as (endpoint, seen):
        code = main(['check', '--endpoint', endpoint, '--key-env', 'CHECK_KEY', *flags,
                     '--expect-count', '2', '--expect-title', 'leave.md'])
    output = capsys.readouterr()
    report = json.loads(output.out)
    valid = types in VALID_TYPES
    assert code == (0 if valid else 1) and report['passed'] is valid
    assert len(report['checks']) == len(seen) == 13 and not output.err
    assert all(c['passed'] is valid for c in report['checks'])
    if not valid:
        assert all(c['errors'] == ['request failed or response is not bounded valid JSON']
                   for c in report['checks'])
    if mode == 'file':
        assert seen[0][1] == path.read_bytes()
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before
    assert json_app.backends.items['handbook'].snapshot_status() == state
    assert all(s not in output.out for s in
               ('test-key', 'wrong-default', 'private', 'vacation policy', 'leave.md', endpoint, str(path)))


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('types', VALID_TYPES + INVALID_TYPES[:9])
def test_checker_error_content_type(mode, types, tmp_path, json_app, monkeypatch, capsys):
    monkeypatch.setenv('CHECK_KEY', 'test-key')
    with framed_endpoint(json_app, 'valid', errors_only=True, content_types=types) as (endpoint, seen):
        code = main(['check', '--endpoint', endpoint, '--key-env', 'CHECK_KEY',
                     *arguments(mode, tmp_path)])
    output = capsys.readouterr()
    report = json.loads(output.out)
    valid = types in VALID_TYPES
    assert (code, report['passed']) == (0 if valid else 1, valid)
    assert len(report['checks']) == len(seen) == 13 and not output.err
    assert all(c['passed'] for c in report['checks'][:4])
    assert all(c['passed'] is valid for c in report['checks'][4:])
    if not valid:
        assert all(c['errors'] == ['request failed or response is not bounded valid JSON']
                   for c in report['checks'][4:])


@pytest.mark.parametrize('status', [200, 400])
@pytest.mark.parametrize('types', VALID_TYPES + INVALID_TYPES)
def test_snapshot_content_type(status, types, json_app, monkeypatch, capsys):
    monkeypatch.setenv('CHECK_KEY', 'test-key')
    monkeypatch.setenv('BRIDGE_API_KEY', 'wrong-default')
    with framed_endpoint(json_app, 'valid', status_override=status, content_types=types) as (endpoint, seen):
        code = main(['snapshot-status', '--endpoint', endpoint, '--key-env', 'CHECK_KEY',
                     '--knowledge-id', 'handbook'])
    output = capsys.readouterr()
    valid = status == 200 and types in VALID_TYPES
    assert code == (0 if valid else 1) and len(seen) == 1
    if valid:
        assert not output.err
        assert json.loads(output.out) == {'knowledge_id': 'handbook', 'version': 1, 'record_count': 4}
    else:
        assert not output.out
        assert output.err == 'Snapshot status failed: request rejected, unavailable or invalid response\n'
    assert all(s not in output.out + output.err for s in ('test-key', 'wrong-default', 'private', endpoint))


@pytest.mark.parametrize('client', ['check', 'snapshot'])
def test_duplicate_content_type_rejected_before_read(client, monkeypatch):
    monkeypatch.setenv('CHECK_KEY', 'test-key')
    class Unreadable(io.BytesIO):
        def read(self, n=-1):
            pytest.fail('ambiguous media type must be rejected before body read')
    headers = Message()
    headers['Content-Type'] = 'application/json'
    headers['content-type'] = 'text/plain'
    headers['Content-Length'] = '2'
    response = addinfourl(Unreadable(b'{}'), headers, 'http://127.0.0.1', 200)
    monkeypatch.setattr(OpenerDirector, 'open', lambda *a, **kw: response)
    with pytest.raises(ValueError if client == 'check' else snapshot.SnapshotStatusError):
        if client == 'check':
            check.request('http://127.0.0.1', {})
        else:
            snapshot.snapshot_status('http://127.0.0.1', 'handbook', key_env='CHECK_KEY')
