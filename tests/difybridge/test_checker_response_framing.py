"""Response framing applies to every checker probe, including HTTP errors."""
import io
import json
import threading
from contextlib import contextmanager
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import OpenerDirector
from urllib.response import addinfourl

import pytest

from ragwrench.difybridge import check, snapshot
from ragwrench.difybridge.cli import main
from .conftest import ROOT
from .test_checker_credentials import arguments, forwarding

LIMIT = 8 * 1024 * 1024
TRANSFER_HEADERS = {
    'transfer-length': ['chunked'],
    'transfer-empty-length': ['chunked'],
    'transfer-duplicate': ['chunked', 'chunked'],
    'transfer-conflicting': ['chunked', 'upstream-private'],
    'transfer-list': ['gzip, chunked'],
    'transfer-unsupported': ['upstream-private'],
    'transfer-identity': ['identity'],
    'transfer-empty': [''],
    'transfer-parameter': ['chunked; private=value'],
    'transfer-trailing-space': ['chunked '],
    'transfer-upper-chunked': ['CHUNKED'],
}
TRANSFER_ERRORS = [kind for kind in TRANSFER_HEADERS if kind != 'transfer-upper-chunked']
VALID_FRAMING = ['valid', 'leading-zeroes', 'absent', 'chunked', 'transfer-upper-chunked']
FRAMING_ERRORS = [
    'duplicate', 'conflicting', 'comma', 'empty', 'negative', 'plus',
    'nondecimal', 'nonascii', 'trailing-space', 'truncated-valid-json',
    'truncated-json', 'underdeclared',
] + TRANSFER_ERRORS


@contextmanager
def framed_endpoint(app, framing, *, errors_only=False, status_override=None,
                    content_types=('application/json',), encode_body=None, content_encodings=()):
    seen = []
    forward = forwarding(app, seen)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers['Content-Length']))
            if self.path == '/bridge/snapshot/status':
                seen.append((self.path, raw))
                status, body = 200, app.snapshot_status(self.headers.get('Authorization'), raw)
            else:
                status, _, body = forward(self.path, self.headers, raw)
            if status_override is not None:
                status = status_override
            data = (encode_body(body) if encode_body is not None and (not errors_only or status >= 400)
                    else json.dumps(body).encode())
            size = str(len(data))
            kind = framing if not errors_only or status >= 400 else 'valid'
            lengths = {**dict.fromkeys(TRANSFER_HEADERS, []),
                'transfer-length': [size], 'transfer-empty-length': [''],
                'valid': [size], 'leading-zeroes': ['000' + size], 'absent': [], 'chunked': [],
                'duplicate': [size, size], 'conflicting': [size, str(len(data) + 1)],
                'comma': [size + ', ' + size], 'empty': [''], 'negative': ['-1'],
                'plus': ['+' + size], 'nondecimal': ['upstream-private'], 'nonascii': ['é'],
                'trailing-space': [size + ' '], 'truncated-valid-json': [str(len(data) + 10)],
                'truncated-json': [size], 'underdeclared': [str(len(data) - 1)],
            }[kind]
            self.send_response(status)
            types = ('application/json',) if errors_only and status < 400 else content_types
            for value in types:
                self.send_header('Content-Type', value)
            if not errors_only or status >= 400:
                for value in content_encodings:
                    self.send_header('Content-Encoding', value)
            for length in lengths:
                self.send_header('Content-Length', length)
            transfers = TRANSFER_HEADERS.get(kind, ['chunked'] if kind == 'chunked' else [])
            for transfer in transfers:
                self.send_header('Transfer-Encoding', transfer)
            if transfers and transfers[0].lower() == 'chunked':
                data = f'{len(data):x}\r\n'.encode() + data + b'\r\n0\r\n\r\n'
            if kind == 'truncated-json':
                data = data[:-1]
            self.end_headers()
            self.wfile.write(data)
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01})
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/bridge/', seen
    finally:
        server.shutdown()
        thread.join(5)
        server.server_close()
        assert not thread.is_alive()


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('framing', FRAMING_ERRORS + VALID_FRAMING)
def test_cli_response_framing(mode, framing, tmp_path, json_app, monkeypatch, capsys):
    monkeypatch.setenv('CHECK_KEY', 'test-key')
    monkeypatch.setenv('BRIDGE_API_KEY', 'wrong-default')
    source = ROOT / 'examples/dify-bridge/chunks.jsonl'
    before = source.read_bytes(), source.stat().st_mtime_ns
    snapshot = json_app.backends.items['handbook'].snapshot_status()
    flags = arguments(mode, tmp_path)
    if mode == 'file':
        path = tmp_path / 'secret-path-marker.json'
        path.write_bytes(b' \n' + path.read_bytes().replace(b'0.2', b'2e-1') + b'\n')
    with framed_endpoint(json_app, framing) as (endpoint, seen):
        code = main(['check', '--endpoint', endpoint, '--key-env', 'CHECK_KEY', *flags,
                     '--expect-count', '2', '--expect-title', 'leave.md'])
    output = capsys.readouterr()
    report = json.loads(output.out)
    valid = framing not in FRAMING_ERRORS
    assert (code, report['passed']) == (0 if valid else 1, valid)
    assert not output.err and len(report['checks']) == len(seen) == 13
    assert all(c['passed'] is valid for c in report['checks'])
    if not valid:
        assert all(c['errors'] == ['request failed or response is not bounded valid JSON']
                   for c in report['checks'])
    assert all(route == '/bridge/retrieval' for route, _ in seen)
    if mode == 'file':
        assert seen[0][1] == path.read_bytes()
    assert json_app.backends.items['handbook'].snapshot_status() == snapshot
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before
    assert all(s not in output.out for s in
               ('test-key', 'wrong-default', 'vacation policy', 'leave.md', 'upstream-private', endpoint))


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('framing', ['duplicate', 'truncated-valid-json'] + TRANSFER_ERRORS)
def test_error_response_framing_is_not_a_passing_auth_probe(mode, framing, tmp_path,
                                                           json_app, monkeypatch, capsys):
    monkeypatch.setenv('CHECK_KEY', 'test-key')
    with framed_endpoint(json_app, framing, errors_only=True) as (endpoint, seen):
        code = main(['check', '--endpoint', endpoint, '--key-env', 'CHECK_KEY',
                     *arguments(mode, tmp_path)])
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert code == 1 and not report['passed'] and not output.err
    assert len(report['checks']) == len(seen) == 13
    assert all(c['passed'] for c in report['checks'][:4])
    assert all(not c['passed'] for c in report['checks'][4:])


def test_snapshot_status_rejects_same_duplicate_framing(json_app, monkeypatch, capsys):
    monkeypatch.setenv('CHECK_KEY', 'test-key')
    with framed_endpoint(json_app, 'duplicate') as (endpoint, seen):
        assert main(['snapshot-status', '--endpoint', endpoint, '--key-env', 'CHECK_KEY',
                     '--knowledge-id', 'handbook']) == 1
    output = capsys.readouterr()
    assert not output.out and 'Snapshot status failed' in output.err and len(seen) == 1


@pytest.mark.parametrize('framing', TRANSFER_ERRORS + VALID_FRAMING)
@pytest.mark.parametrize('status', [200, 400])
def test_snapshot_status_response_transfer_framing(framing, status, json_app, monkeypatch, capsys):
    monkeypatch.setenv('CHECK_KEY', 'test-key')
    monkeypatch.setenv('BRIDGE_API_KEY', 'wrong-default')
    source = ROOT / 'examples/dify-bridge/chunks.jsonl'
    before = source.read_bytes(), source.stat().st_mtime_ns
    snapshot = json_app.backends.items['handbook'].snapshot_status()
    with framed_endpoint(json_app, framing, status_override=status) as (endpoint, seen):
        code = main(['snapshot-status', '--endpoint', endpoint, '--key-env', 'CHECK_KEY',
                     '--knowledge-id', 'handbook'])
    output = capsys.readouterr()
    valid = status == 200 and framing in VALID_FRAMING
    assert code == (0 if valid else 1)
    assert len(seen) == 1 and seen[0][0] == '/bridge/snapshot/status'
    if valid:
        assert not output.err
        assert json.loads(output.out) == {'knowledge_id': 'handbook', 'version': 1, 'record_count': 4}
    else:
        assert not output.out
        assert output.err == 'Snapshot status failed: request rejected, unavailable or invalid response\n'
    assert all(s not in output.out + output.err for s in
               ('test-key', 'wrong-default', 'upstream-private', endpoint))
    assert json_app.backends.items['handbook'].snapshot_status() == snapshot
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before


@pytest.mark.parametrize('size', [LIMIT, LIMIT + 1])
def test_response_read_is_bounded_at_existing_limit(size, monkeypatch):
    reads = []
    class Tracked(io.BytesIO):
        def read(self, n=-1):
            reads.append(n)
            return super().read(n)
    headers = Message()
    headers['Content-Type'] = 'application/json'
    headers['Content-Length'] = str(size)
    raw = b'{}' + b' ' * (size - 2)
    response = addinfourl(Tracked(raw), headers, 'http://127.0.0.1/retrieval', 200)
    monkeypatch.setattr(OpenerDirector, 'open', lambda *a, **kw: response)
    if size == LIMIT:
        assert check.request('http://127.0.0.1', {}) == (200, {})
    else:
        with pytest.raises(ValueError):
            check.request('http://127.0.0.1', {})
    assert reads == [LIMIT + 1]


@pytest.mark.parametrize('length', ['٢', '9' * 5000, '1', '3'])
def test_request_rejects_invalid_or_mismatched_lengths(length, monkeypatch):
    headers = Message()
    headers['Content-Type'] = 'application/json'
    headers['Content-Length'] = length
    response = addinfourl(io.BytesIO(b'{}'), headers, 'http://127.0.0.1/retrieval', 200)
    monkeypatch.setattr(OpenerDirector, 'open', lambda *a, **kw: response)
    with pytest.raises(ValueError):
        check.request('http://127.0.0.1', {})


@pytest.mark.parametrize('status', [200, 400])
@pytest.mark.parametrize('raw,content_type', [
    (b'{"records":[],"records":[]}', 'application/json'),
    (b'{"records":[],"private":NaN}', 'application/json'),
    (b'{broken upstream-private', 'application/json'),
    (b'\xff', 'application/json'),
    (b'{"records":[]}', 'text/plain'),
])
def test_existing_response_validation_survives_length_checks(status, raw, content_type, monkeypatch):
    from urllib.error import HTTPError
    headers = Message()
    headers['Content-Type'] = content_type
    headers['Content-Length'] = str(len(raw))
    response = addinfourl(io.BytesIO(raw), headers, 'http://127.0.0.1/retrieval', status)
    def opened(*a, **kw):
        if status == 400:
            raise HTTPError(response.url, status, 'upstream-private', headers, response)
        return response
    monkeypatch.setattr(OpenerDirector, 'open', opened)
    with pytest.raises(ValueError):
        check.request('http://127.0.0.1', {})


def test_checker_redirect_does_not_forward_credentials(monkeypatch):
    from ragwrench.difybridge.check import check_endpoint
    from .test_embedding_checker import fake_endpoint
    traps = []
    with fake_endpoint(lambda *args: (traps.append(args) or (200, {}, {}))) as target:
        with fake_endpoint(lambda *args: (307, {'Location': target}, {'private': 'upstream-private'})) as source:
            report = check_endpoint(source, 'test-key', 'handbook', 'vacation')
    assert not traps and not report['passed'] and len(report['checks']) == 13
    assert all(not c['passed'] for c in report['checks'])
    assert all(s not in json.dumps(report) for s in ('test-key', 'upstream-private', target, source))


@pytest.mark.parametrize('client', ['check', 'snapshot'])
@pytest.mark.parametrize('overflow', [False, True])
def test_chunked_decoded_body_keeps_each_client_byte_bound(client, overflow, monkeypatch):
    monkeypatch.setenv('CHECK_KEY', 'test-key')
    limit = check.MAX_RESPONSE if client == 'check' else snapshot.MAX_RESPONSE
    body = {} if client == 'check' else {'knowledge_id': 'handbook', 'version': 1, 'record_count': 4}
    raw = json.dumps(body).encode()
    decoded = raw + b' ' * (limit + overflow - len(raw))
    reads = []
    class Tracked(io.BytesIO):
        def read(self, n=-1):
            reads.append(n)
            return super().read(n)
    headers = Message()
    headers['Content-Type'] = 'application/json'
    headers['Transfer-Encoding'] = 'chunked'
    # HTTPResponse delivers decoded bytes; exercise the exact bound on that stream.
    response = addinfourl(Tracked(decoded), headers, 'http://127.0.0.1', 200)
    monkeypatch.setattr(OpenerDirector, 'open', lambda *a, **kw: response)
    def call():
        if client == 'check':
            return check.request('http://127.0.0.1', {})[1]
        return snapshot.snapshot_status('http://127.0.0.1', 'handbook', key_env='CHECK_KEY')
    if overflow:
        with pytest.raises(ValueError if client == 'check' else snapshot.SnapshotStatusError):
            call()
    else:
        assert call() == body
    assert reads == [limit + 1]


@pytest.mark.parametrize('client', ['check', 'snapshot'])
def test_ambiguous_transfer_headers_fail_before_body_read(client, monkeypatch):
    monkeypatch.setenv('CHECK_KEY', 'test-key')
    class Unreadable(io.BytesIO):
        def read(self, n=-1):
            pytest.fail('ambiguous transfer framing must be rejected before reading')
    headers = Message()
    headers['Content-Type'] = 'application/json'
    headers['Transfer-Encoding'] = 'chunked'
    headers['Content-Length'] = '2'
    response = addinfourl(Unreadable(b'{}'), headers, 'http://127.0.0.1', 200)
    monkeypatch.setattr(OpenerDirector, 'open', lambda *a, **kw: response)
    with pytest.raises(ValueError if client == 'check' else snapshot.SnapshotStatusError):
        if client == 'check':
            check.request('http://127.0.0.1', {})
        else:
            snapshot.snapshot_status('http://127.0.0.1', 'handbook', key_env='CHECK_KEY')
