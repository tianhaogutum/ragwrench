import http.client
import json
import os
import signal
import subprocess
import sys
from urllib.parse import urlsplit

import pytest

from ragwrench.difybridge.check import request
from .conftest import ROOT


def raw_http(endpoint, *, method='POST', path='/retrieval', headers=(), data=None):
    url = urlsplit(endpoint)
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=3)
    try:
        connection.putrequest(method, path)
        for name, value in headers:
            connection.putheader(name, value)
        connection.endheaders(data)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


@pytest.mark.parametrize('headers,status', [
    ([('Content-Type', 'application/json'), ('Content-Length', '1048577')], 413),
    ([('Content-Type', 'application/json'), ('Content-Length', '-1')], 400),
    ([('Content-Type', 'application/json'), ('Content-Length', '0'), ('Content-Length', '0')], 400),
    ([('Content-Type', 'application/json'), ('Transfer-Encoding', 'chunked')], 400),
    ([('Content-Length', '0'), ('Content-Type', 'text/plain')], 415),
])
@pytest.mark.parametrize('path', ['/retrieval', '/reload', '/retrieval/diagnostics', '/snapshot/status'])
def test_http_framing(json_endpoint, headers, status, path):
    code, body = raw_http(json_endpoint, path=path, headers=headers)
    assert code == status and type(body['error_code']) is int and body['error_msg']


@pytest.mark.parametrize('path', ['/retrieval', '/reload', '/retrieval/diagnostics', '/snapshot/status'])
def test_duplicate_auth_and_methods(json_endpoint, body, path):
    data = json.dumps(body).encode()
    status, result = raw_http(json_endpoint, path=path, headers=[('Content-Type', 'application/json'), ('Content-Length', str(len(data))),
        ('Authorization', 'Bearer test-key'), ('Authorization', 'Bearer test-key')], data=data)
    assert (status, result['error_code']) == (401, 1001)
    for method, path, status in [('PUT', '/retrieval', 405), ('TRACE', '/retrieval', 501), ('GET', '/missing', 404)]:
        actual, result = raw_http(json_endpoint, method=method, path=path)
        assert actual == status and result['error_code'] == 2004


def test_snapshot_status_route_is_post_only_and_exact(json_endpoint):
    for method, path, expected in [
        ('GET', '/snapshot/status', 404),
        ('PUT', '/snapshot/status', 405),
        ('POST', '/snapshot/status/', 404),
        ('POST', '/snapshot/status?knowledge_id=handbook', 404),
    ]:
        status, error = raw_http(json_endpoint, method=method, path=path)
        assert (status, error['error_code']) == (expected, 2004)


@pytest.mark.parametrize('data', [
    b'{"query":"x","query":"y"}',
    b'{"knowledge_id":"handbook","query":"vacation","retrieval_setting":{"top_k":1,"score_threshold":1e999}}',
    b'{"knowledge_id":"handbook","query":"vacation","retrieval_setting":{"top_k":1,"score_threshold":' + b'9'*1000 + b'}}',
])
def test_json_boundaries(json_endpoint, data):
    status, body = request(json_endpoint, data, 'test-key')
    assert status == 400 and body['error_code'] == 2002


def test_cli_start_check_shutdown():
    env = dict(os.environ, BRIDGE_API_KEY='cli-test-key')
    process = subprocess.Popen([sys.executable, '-m', 'ragwrench.difybridge.cli', 'serve', '--config', str(ROOT / 'examples/dify-bridge/bridge.toml'), '--port', '0'],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # select prevents an indefinitely blocked test if the startup path regresses.
        # Generous bound: cold imports on shared CI runners (notably macOS) can be slow.
        import select
        assert select.select([process.stdout], [], [], 30)[0], 'CLI did not start in 30 seconds'
        line = process.stdout.readline()
        assert '127.0.0.1:' in line, line
        endpoint = 'http://' + line.strip().split()[-1]
        result = subprocess.run([sys.executable, '-m', 'ragwrench.difybridge.cli', 'check', '--endpoint', endpoint,
                                 '--knowledge-id', 'handbook', '--query', 'vacation'],
                                env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr + result.stdout
        assert json.loads(result.stdout)['passed']
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=15) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


def test_cli_config_error_is_nonzero(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'ragwrench.difybridge.cli', 'serve', '--config', str(tmp_path / 'missing')], capture_output=True, text=True, timeout=5)
    assert result.returncode == 2 and 'Configuration' in result.stderr


def test_server_capacity_returns_structured_error(json_app):
    import threading
    from ragwrench.difybridge.service import make_server
    server = make_server(json_app, port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01})
    thread.start()
    for _ in range(16):
        assert server.slots.acquire(blocking=False)
    try:
        status, result = raw_http(f'http://127.0.0.1:{server.server_port}', method='GET', path='/healthz')
        assert status == 503 and result == {'error_code': 3003, 'error_msg': 'Bridge busy'}
    finally:
        for _ in range(16):
            server.slots.release()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
