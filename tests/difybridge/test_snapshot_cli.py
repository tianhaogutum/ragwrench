"""CLI contract: exactly one read-only status POST; stdout is validated data only."""
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import OpenerDirector

import pytest

from ragwrench.difybridge.cli import main
from .conftest import running

SECRET = 'synthetic-status-token'
VALID = {'knowledge_id': 'handbook', 'version': 1, 'record_count': 4}


def invoke(endpoint, capsys, monkeypatch, *, knowledge='handbook', key=SECRET, extra=()):
    # A custom variable must be used even if the default holds a different key.
    monkeypatch.setenv('BRIDGE_API_KEY', 'wrong-default-token')
    if key is None:
        monkeypatch.delenv('STATUS_TEST_KEY', raising=False)
    else:
        monkeypatch.setenv('STATUS_TEST_KEY', key)
    try:
        code = main(['snapshot-status', '--endpoint', endpoint, '--knowledge-id', knowledge,
                     '--key-env', 'STATUS_TEST_KEY', *extra])
    except SystemExit as exc:
        code = exc.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@contextmanager
def reply(raw=None, *, code=200, content_type='application/json', redirect=False, truncate=False):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append((self.command, self.path, self.headers.get('Authorization'),
                          self.headers.get('Content-Type'),
                          json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            data = raw if raw is not None else json.dumps(VALID).encode()
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data) + (10 if truncate else 0)))
            if redirect:
                self.send_header('Location', f'http://127.0.0.1:{self.server.server_port}/credential-trap')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01})
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', calls
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        assert not thread.is_alive()


def assert_redacted_failure(result, code):
    actual, out, err = result
    assert actual == code
    assert out == ''
    assert err.strip() and 'Traceback' not in err
    assert all(value not in err for value in (SECRET, 'upstream-private', 'wrong-default-token'))


def test_cli_active_status_is_read_only(json_app, capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('status CLI must never reload or retrieve')
    monkeypatch.setattr(json_app, 'reload', forbidden)
    monkeypatch.setattr(json_app, 'retrieve', forbidden)
    with running(json_app) as endpoint:
        for _ in range(2):
            code, out, err = invoke(endpoint, capsys, monkeypatch, key='test-key')
            assert (code, err) == (0, '')
            assert json.loads(out) == VALID
    assert json_app.backends.items['handbook'].snapshot_status() == {'version': 1, 'record_count': 4}


@pytest.mark.parametrize('knowledge,key', [('handbook', 'wrong-status-key'), ('unknown', 'test-key')])
def test_cli_auth_and_unknown_failures(json_endpoint, capsys, monkeypatch, knowledge, key):
    assert_redacted_failure(invoke(json_endpoint, capsys, monkeypatch, knowledge=knowledge, key=key), 1)


def test_cli_qdrant_failure(qdrant_app, capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('status CLI must not query Qdrant')
    monkeypatch.setattr(qdrant_app.backends.items['handbook'].client, 'get_collection', forbidden)
    with running(qdrant_app) as endpoint:
        assert_redacted_failure(invoke(endpoint, capsys, monkeypatch, key='test-key'), 1)


@pytest.mark.parametrize('value', [VALID, dict(VALID, version=19, record_count=0)])
def test_cli_exact_request_and_output(value, capsys, monkeypatch):
    with reply(json.dumps(value).encode()) as (endpoint, calls):
        code, out, err = invoke(endpoint + '/', capsys, monkeypatch)
    assert (code, err) == (0, '')
    assert json.loads(out) == value
    assert calls == [('POST', '/snapshot/status', 'Bearer ' + SECRET,
                      'application/json', {'knowledge_id': 'handbook'})]


@pytest.mark.parametrize('raw,content_type', [
    (b'{broken upstream-private', 'application/json'),
    (b'[]', 'application/json'),
    (b'null', 'application/json'),
    (json.dumps(dict(VALID, knowledge_id='upstream-private')).encode(), 'application/json'),
    (json.dumps(dict(VALID, version=True)).encode(), 'application/json'),
    (json.dumps(dict(VALID, version=1.0)).encode(), 'application/json'),
    (json.dumps(dict(VALID, version='1')).encode(), 'application/json'),
    (json.dumps(dict(VALID, version=0)).encode(), 'application/json'),
    (json.dumps(dict(VALID, record_count=False)).encode(), 'application/json'),
    (json.dumps(dict(VALID, record_count=-1)).encode(), 'application/json'),
    (json.dumps(dict(VALID, record_count=4.0)).encode(), 'application/json'),
    (json.dumps(dict(VALID, record_count=None)).encode(), 'application/json'),
    (json.dumps(dict(VALID, extra='upstream-private')).encode(), 'application/json'),
    (b'{"knowledge_id":"handbook","version":1}', 'application/json'),
    (b'{"knowledge_id":"handbook","version":1,"version":2,"record_count":4}', 'application/json'),
    (b'{"knowledge_id":"handbook","version":NaN,"record_count":4}', 'application/json'),
    (b'\xff', 'application/json'),
    (b'[' * 1500, 'application/json'),
    (b'x' * (64 * 1024 + 1), 'application/json'),
    (json.dumps(VALID).encode(), 'text/plain'),
], ids=lambda v: f'bytes-{len(v)}' if isinstance(v, bytes) else v)
def test_cli_rejects_invalid_responses(raw, content_type, capsys, monkeypatch):
    with reply(raw, content_type=content_type) as (endpoint, calls):
        assert_redacted_failure(invoke(endpoint, capsys, monkeypatch), 1)
    assert len(calls) == 1


@pytest.mark.parametrize('status', [201, 204, 301, 302, 303, 307, 308, 400, 401, 403, 404, 500, 503])
def test_cli_rejects_non_200_and_never_follows_redirects(status, capsys, monkeypatch):
    with reply(b'upstream-private ' + SECRET.encode(), code=status, redirect=True) as (endpoint, calls):
        assert_redacted_failure(invoke(endpoint, capsys, monkeypatch), 1)
    assert len(calls) == 1
    assert calls[0][1] == '/snapshot/status'


def test_cli_truncated_response(capsys, monkeypatch):
    with reply(truncate=True) as (endpoint, calls):
        assert_redacted_failure(invoke(endpoint, capsys, monkeypatch), 1)
    assert len(calls) == 1


@pytest.mark.parametrize('endpoint', [
    'file:///upstream-private', 'ftp://127.0.0.1', 'http://',
    'http://user:upstream-private@localhost', 'http://@localhost',
    'http://localhost?key=upstream-private', 'http://localhost#upstream-private',
    'http://localhost?', 'http://localhost#',
    'http://localhost:bad', 'http://localhost:65536', 'http://localhost:0',
    'http://[broken', 'http://[v1.invalid]', 'http://local host', ' http://localhost', 'http://localhost\n',
])
def test_cli_invalid_url_is_local(endpoint, capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('invalid local input must not open an HTTP request')
    monkeypatch.setattr(OpenerDirector, 'open', forbidden)
    assert_redacted_failure(invoke(endpoint, capsys, monkeypatch), 2)


@pytest.mark.parametrize('kwargs', [
    {'key': None}, {'key': ''}, {'key': 'bad\nkey'}, {'key': 'bad key'},
    {'key': 'é'}, {'key': 'x' * 4097}, {'key': 'bad\x7fkey'},
    {'extra': ('--key-env', '')}, {'extra': ('--key-env', 'invalid=name')},
    {'knowledge': ''}, {'knowledge': 'x' * 257},
    {'extra': ('--timeout', '0')}, {'extra': ('--timeout', 'nan')},
    {'extra': ('--timeout', 'inf')}, {'extra': ('--timeout', '121')},
    {'extra': ('--timeout', 'upstream-private')},
])
def test_cli_invalid_configuration_is_local(kwargs, capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('invalid local input must not open an HTTP request')
    monkeypatch.setattr(OpenerDirector, 'open', forbidden)
    assert_redacted_failure(invoke('http://127.0.0.1:1', capsys, monkeypatch, **kwargs), 2)


def test_cli_transport_failure_is_remote(capsys, monkeypatch):
    # The closed ephemeral listener supplies a loopback refused connection.
    with reply() as (endpoint, _):
        pass
    assert_redacted_failure(invoke(endpoint, capsys, monkeypatch), 1)


def test_cli_default_environment_and_prefixed_base(capsys, monkeypatch):
    monkeypatch.setenv('BRIDGE_API_KEY', SECRET)
    with reply() as (endpoint, calls):
        assert main(['snapshot-status', '--endpoint', endpoint + '/bridge/',
                     '--knowledge-id', 'handbook']) == 0
    out = capsys.readouterr()
    assert out.err == '' and json.loads(out.out) == VALID
    assert calls == [('POST', '/bridge/snapshot/status', 'Bearer ' + SECRET,
                      'application/json', {'knowledge_id': 'handbook'})]


def test_cli_timeout_is_remote_and_bounded(capsys, monkeypatch):
    calls = []
    def timed_out(self, req, *, timeout):
        calls.append((req.full_url, timeout))
        raise TimeoutError('upstream-private ' + SECRET)
    monkeypatch.setattr(OpenerDirector, 'open', timed_out)
    assert_redacted_failure(invoke('http://127.0.0.1:1', capsys, monkeypatch,
                                  extra=('--timeout', '0.25')), 1)
    assert calls == [('http://127.0.0.1:1/snapshot/status', 0.25)]


@pytest.mark.parametrize('extra', [('--api-key', SECRET), ('--unexpected', 'upstream-private')])
def test_cli_argument_errors_are_redacted(extra, capsys, monkeypatch):
    assert_redacted_failure(invoke('http://127.0.0.1:1', capsys, monkeypatch, extra=extra), 2)
