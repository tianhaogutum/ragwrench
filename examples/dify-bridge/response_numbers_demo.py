"""Installed checker's finite response number contract on synthetic loopback only."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ragwrench.difybridge.config import load_config
from ragwrench.difybridge.errors import BridgeError
from ragwrench.difybridge.service import Application


def main():
    examples = Path(__file__).resolve().parent
    inputs = [examples / name for name in ('bridge.toml', 'chunks.jsonl', 'request.json')]
    original = [(p.read_bytes(), p.stat().st_mtime_ns) for p in inputs]
    app = Application(load_config(inputs[0]), 'local-demo')
    cli = str(Path(sys.executable).with_name('ragwrench'))
    seen, results = [], []
    token, location, errors_only = b'1e999', 'ignored', False

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers['Content-Length']))
            seen.append((self.path, raw))
            assert self.path == '/bridge/retrieval'
            try:
                body = app.retrieve(self.headers.get('Authorization'), raw)
                status = 200
            except BridgeError as exc:
                status, body = exc.status, exc.body()
            if not errors_only or status >= 400:
                marker = 'upstream-private-number-marker'
                if location == 'ignored':
                    body['extra'] = marker
                elif location == 'nested-list':
                    body['extra'] = [None, True, {'deeper': [False, [marker]]}]
                else:
                    for record in body.get('records', []):
                        record['metadata']['private'] = {'deeper': [marker]}
                    if not body.get('records'):
                        body['extra'] = {'deeper': [marker]}
                data = json.dumps(body).encode().replace(json.dumps(marker).encode(), token)
            else:
                data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix='checker-numbers-') as tmp:
            saved = Path(tmp) / 'private-request.json'
            saved.write_bytes(b' \n' + inputs[2].read_bytes().replace(b'0.2', b'2e-1') + b'\n')
            saved_before = saved.read_bytes(), saved.stat().st_mtime_ns
            body = json.loads(saved.read_bytes())
            endpoint = f'http://127.0.0.1:{server.server_port}/bridge/'
            inline = ['--knowledge-id', body['knowledge_id'], '--query', body['query'],
                      '--metadata-condition', json.dumps(body['metadata_condition']),
                      '--top-k', '3', '--score-threshold', '.2']

            def run(flags, expected, *, extra=(), key='local-demo', assertion_failure=False):
                before = len(seen)
                command = [cli, 'dify-bridge', 'check', '--endpoint', endpoint, '--key-env', 'CHECK_KEY', *flags, *extra]
                result = subprocess.run(command, capture_output=True, text=True, timeout=20,
                    env=dict(os.environ, CHECK_KEY=key,
                             BRIDGE_API_KEY='wrong-default' if key == 'local-demo' else 'local-demo'))
                calls = seen[before:]
                results.append(result.returncode)
                print(json.dumps({'command': shlex.join(command), 'responseNumber': token.decode(),
                                  'location': location, 'errorsOnly': errors_only,
                                  'exitCode': result.returncode, 'observedHTTPRequests': len(calls),
                                  'stdout': result.stdout, 'stderr': result.stderr}), flush=True)
                assert result.returncode == expected
                assert all(s not in result.stdout + result.stderr for s in
                           ('local-demo', 'wrong-default', 'private', 'vacation policy', 'handbook.md',
                            'leave.md', '1e999', str(saved), endpoint))
                assert len(calls) == (0 if expected == 2 else 13)
                if expected == 2:
                    assert not result.stdout and 'Configuration' in result.stderr
                    return
                report = json.loads(result.stdout)
                assert not result.stderr and len(report['checks']) == 13
                assert report['passed'] is (expected == 0)
                for i, probe in enumerate(report['checks']):
                    passing = expected == 0 or (errors_only and i < 4) or (assertion_failure and i > 0)
                    assert probe['passed'] is passing
                    if not passing and not assertion_failure:
                        assert probe['errors'] == ['request failed or response is not bounded valid JSON']
                assert all(route == '/bridge/retrieval' for route, _ in calls)
                assert json.loads(calls[0][1]) == body
                if '--request-file' in flags:
                    assert calls[0][1] == saved_before[0]
                assert all(json.loads(raw)['metadata_condition'] == body['metadata_condition']
                           for _, raw in calls[:8])

            for flags in (inline, ['--request-file', str(saved)]):
                assertions = ['--expect-count', '2', '--expect-title', 'leave.md']
                for location in ('ignored', 'nested-list', 'metadata'):
                    for token in (b'1e999', b'-1e999'):
                        for errors_only in (False, True):
                            run(flags, 1, extra=assertions)
                errors_only = False
                for token in (b'1.7976931348623157e308', b'-1.7976931348623157e308',
                              b'5e-324', b'-1e-999', b'0.125', b'0', b'-7', b'9' * 400,
                              b'true', b'false', b'null'):
                    run(flags, 0, extra=assertions)
                token = b'0.125'
                run(flags, 1, extra=['--expect-count', '1'], assertion_failure=True)
                run(flags, 1, extra=['--expect-title', 'private-missing-title'], assertion_failure=True)
                token = b'1e999'
                run(flags, 2, key='bad\nkey')
                run(flags, 2, extra=['--endpoint', ' ' + endpoint])
            run(['--request-file', str(saved), '--top-k', '3'], 2)
            assert (saved.read_bytes(), saved.stat().st_mtime_ns) == saved_before
        assert original == [(p.read_bytes(), p.stat().st_mtime_ns) for p in inputs]
        assert app.snapshot_status('Bearer local-demo', b'{"knowledge_id":"handbook"}') == {
            'knowledge_id': 'handbook', 'version': 1, 'record_count': 4}
        print(json.dumps({'result': 'RESPONSE NUMBERS DEMO PASS', 'cases': len(results),
                          'exit0': results.count(0), 'exit1': results.count(1), 'exit2': results.count(2),
                          'filesAndSnapshotUnchanged': True}))
    finally:
        server.shutdown()
        thread.join(5)
        server.server_close()
        app.close()
        assert not thread.is_alive()


if __name__ == '__main__':
    main()
