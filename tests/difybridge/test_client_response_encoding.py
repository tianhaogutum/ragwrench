"""CLI response bytes use strict UTF-8; error probes cannot bypass decoding."""
import json

import pytest

from ragwrench.difybridge.cli import main
from .conftest import ROOT
from .test_checker_credentials import arguments
from .test_checker_response_framing import framed_endpoint

VALID_ENCODINGS = ['utf-8', 'utf-8-charset']
INVALID_ENCODINGS = [
    'utf-8-sig', 'utf-16', 'utf-32', 'utf-16-le', 'utf-16-be', 'utf-32-le', 'utf-32-be',
    'invalid-byte', 'invalid-continuation', 'truncated', 'overlong', 'surrogate',
    'duplicate-key', 'nan', 'infinity',
]


def response_bytes(body, encoding):
    # Literal non-ASCII and literal U+FFFD are valid text, distinct from repairing
    # malformed bytes with errors='replace'. Keep status IDs/schema unchanged.
    if 'records' in body:
        for record in body['records']:
            record['content'] += ' upstream-private café \ufffd'
    elif 'error_msg' in body:
        body['error_msg'] += ' upstream-private café \ufffd'
    raw = json.dumps(body, ensure_ascii=False).encode('utf-8')
    if encoding in VALID_ENCODINGS:
        return raw
    if encoding in ('duplicate-key', 'nan', 'infinity'):
        if encoding == 'duplicate-key':
            return b'{"private":1,"private":2,' + raw[1:]
        value = b'NaN' if encoding == 'nan' else b'Infinity'
        return b'{"private":' + value + b',' + raw[1:]
    if encoding in ('invalid-byte', 'invalid-continuation', 'truncated', 'overlong', 'surrogate'):
        bad = {'invalid-byte': b'\xff', 'invalid-continuation': b'\xc3(',
               'truncated': b'\xe2\x82', 'overlong': b'\xc0\xaf', 'surrogate': b'\xed\xa0\x80'}[encoding]
        # A repairing decoder would produce valid JSON here. Exercise both a
        # malformed byte inside a string and an incomplete sequence at EOF.
        return raw + bad if encoding == 'truncated' else b'{"private":"upstream-private' + bad + b'",' + raw[1:]
    return raw.decode('utf-8').encode(encoding)


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('errors_only', [False, True])
@pytest.mark.parametrize('encoding', VALID_ENCODINGS + INVALID_ENCODINGS)
def test_checker_response_encoding(mode, errors_only, encoding, tmp_path, json_app, monkeypatch, capsys):
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
    # A charset parameter never enables UTF-16/32 or replacement decoding.
    types = ('application/json; charset=utf-16',) if encoding == 'utf-8-charset' else ('application/json',)
    with framed_endpoint(json_app, 'valid', errors_only=errors_only, content_types=types,
                         encode_body=lambda body: response_bytes(body, encoding)) as (endpoint, seen):
        code = main(['check', '--endpoint', endpoint, '--key-env', 'CHECK_KEY', *flags,
                     '--expect-count', '2', '--expect-title', 'leave.md'])
    output = capsys.readouterr()
    report = json.loads(output.out)
    valid = encoding in VALID_ENCODINGS
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


@pytest.mark.parametrize('encoding', VALID_ENCODINGS + INVALID_ENCODINGS)
def test_snapshot_response_encoding(encoding, json_app, monkeypatch, capsys):
    monkeypatch.setenv('CHECK_KEY', 'test-key')
    monkeypatch.setenv('BRIDGE_API_KEY', 'wrong-default')
    with framed_endpoint(json_app, 'valid', encode_body=lambda body: response_bytes(body, encoding)) as (endpoint, seen):
        code = main(['snapshot-status', '--endpoint', endpoint, '--key-env', 'CHECK_KEY',
                     '--knowledge-id', 'handbook'])
    output = capsys.readouterr()
    assert len(seen) == 1
    if encoding in VALID_ENCODINGS:
        assert code == 0 and not output.err
        assert json.loads(output.out) == {'knowledge_id': 'handbook', 'version': 1, 'record_count': 4}
    else:
        assert code == 1 and not output.out
        assert output.err == 'Snapshot status failed: request rejected, unavailable or invalid response\n'
