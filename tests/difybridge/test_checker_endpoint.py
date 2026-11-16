"""Local endpoint errors must abort all checker probes, in either input mode."""
import io
import json
from email.message import Message
from urllib.request import OpenerDirector
from urllib.response import addinfourl

import pytest

from ragwrench.difybridge import check
from ragwrench.difybridge.cli import main
from ragwrench.difybridge.errors import BridgeError, ConfigError
from .test_checker_credentials import arguments, forwarding
from .test_checker_request_file import BASE, saved
from .test_embedding_checker import fake_endpoint


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('prefix,suffix', [('', '\n'), (' ', '')], ids=['newline', 'leading-space'])
def test_unsafe_endpoint_never_contacts_service(mode, prefix, suffix, tmp_path, json_app, monkeypatch, capsys):
    monkeypatch.setenv('CHECK_KEY', 'test-key')
    seen = []
    with fake_endpoint(forwarding(json_app, seen)) as endpoint:
        code = main(['check', '--endpoint', prefix + endpoint + suffix, '--key-env', 'CHECK_KEY',
                     *arguments(mode, tmp_path)])
    assert (code, len(seen)) == (2, 0)
    output = capsys.readouterr()
    assert not output.out and 'Configuration' in output.err
    assert endpoint not in output.err and 'test-key' not in output.err


INVALID_ENDPOINTS = [
    'file:///private-url-marker', 'ftp://127.0.0.1', 'http://',
    'http://user:private-url-marker@localhost', 'http://@localhost',
    'http://localhost?key=private-url-marker', 'http://localhost#private-url-marker',
    'http://localhost?', 'http://localhost#',
    'http://localhost:bad', 'http://localhost:65536', 'http://localhost:0',
    'http://localhost:-1', 'http://[broken', 'http://[v1.invalid]',
    'http://local host', ' http://localhost', 'http://localhost\n',
    'http://local\thost', 'http://localhost/\rprivate-url-marker',
    'http://localhost/\x00private-url-marker', 'http://localhost/\x7fprivate-url-marker',
    'http://éxample.org', 'http://localhost/café', 'http://local_host',
    'http://local%68ost', 'http://localhost\\private-url-marker',
]


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('endpoint', INVALID_ENDPOINTS)
def test_cli_invalid_endpoint_is_local(mode, endpoint, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('CHECK_KEY', 'synthetic-private-token')
    calls = []
    monkeypatch.setattr(check, 'request', lambda *a, **kw: calls.append(a))
    assert main(['check', '--endpoint', endpoint, '--key-env', 'CHECK_KEY',
                 *arguments(mode, tmp_path)]) == 2
    assert not calls
    output = capsys.readouterr()
    assert not output.out and 'Configuration' in output.err
    assert all(value not in output.err for value in
               ('private-url-marker', 'synthetic-private-token', str(tmp_path), BASE['query'], endpoint))


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('endpoint', INVALID_ENDPOINTS + [None, 123, b'http://localhost'])
def test_library_invalid_endpoint_is_local(mode, endpoint, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(check, 'request', lambda *a, **kw: calls.append(a))
    with pytest.raises(ConfigError) as exc:
        if mode == 'file':
            check.check_request_file(endpoint, 'test-key', saved(tmp_path))
        else:
            check.check_endpoint(endpoint, 'test-key', 'handbook', 'vacation')
    assert not calls and 'private-url-marker' not in str(exc.value)


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('wrong_key', [False, True])
def test_prefixed_endpoint_retains_baseline_and_probes(mode, wrong_key, tmp_path, json_app,
                                                      monkeypatch, capsys):
    monkeypatch.setenv('CHECK_KEY', 'valid-but-wrong' if wrong_key else 'test-key')
    monkeypatch.setenv('BRIDGE_API_KEY', 'test-key' if wrong_key else 'wrong-default')
    seen = []
    with fake_endpoint(forwarding(json_app, seen)) as endpoint:
        code = main(['check', '--endpoint', endpoint + '/bridge/', '--key-env', 'CHECK_KEY',
                     *arguments(mode, tmp_path), '--expect-count', '2', '--expect-title', 'leave.md'])
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert code == int(wrong_key) and not output.err
    assert report['passed'] is (not wrong_key) and len(report['checks']) == 13
    assert len(seen) == 13 and all(route == '/bridge/retrieval' for route, _ in seen)
    assert json.loads(seen[0][1]) == BASE
    if mode == 'file':
        assert seen[0][1] == (tmp_path / 'secret-path-marker.json').read_bytes()
    assert 'test-key' not in output.out and 'valid-but-wrong' not in output.out


@pytest.mark.parametrize('mode', ['inline', 'file'])
def test_closed_valid_endpoint_remains_remote(mode, tmp_path, json_app, monkeypatch, capsys):
    monkeypatch.setenv('CHECK_KEY', 'test-key')
    with fake_endpoint(forwarding(json_app, [])) as endpoint:
        pass
    code = main(['check', '--endpoint', endpoint, '--key-env', 'CHECK_KEY',
                 *arguments(mode, tmp_path), '--timeout', '0.1'])
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert code == 1 and not output.err and not report['passed']
    assert len(report['checks']) == 13


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('endpoint', [
    'http://[::1]:1', 'https://[2001:db8::1]:65535/bridge/',
    'https://example.test:443/base%20prefix/', 'http://localhost',
    'http://127.0.0.1:65535',
])
def test_valid_url_syntax_preserves_http_request(mode, endpoint, tmp_path, json_app,
                                                monkeypatch, capsys):
    # Inspect real urllib Requests without DNS, IPv6 connectivity or TLS services.
    calls = []
    def open_request(self, req, *, timeout):
        calls.append((req.full_url, req.data, timeout))
        try:
            status, body = 200, json_app.retrieve(req.get_header('Authorization'), req.data)
        except BridgeError as exc:
            status, body = exc.status, exc.body()
        headers = Message()
        headers['Content-Type'] = 'application/json'
        return addinfourl(io.BytesIO(json.dumps(body).encode()), headers, req.full_url, status)
    monkeypatch.setattr(OpenerDirector, 'open', open_request)
    monkeypatch.setenv('CHECK_KEY', 'test-key')
    code = main(['check', '--endpoint', endpoint, '--key-env', 'CHECK_KEY',
                 *arguments(mode, tmp_path), '--timeout', '0.25'])
    output = capsys.readouterr()
    assert code == 0 and not output.err and json.loads(output.out)['passed']
    assert len(calls) == 13
    assert all(url == endpoint.rstrip('/') + '/retrieval' and timeout == 0.25
               for url, _, timeout in calls)
    assert json.loads(calls[0][1]) == BASE
    if mode == 'file':
        assert calls[0][1] == (tmp_path / 'secret-path-marker.json').read_bytes()
