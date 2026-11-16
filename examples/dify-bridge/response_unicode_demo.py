"""Installed checker's response Unicode contract on synthetic loopback only."""
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
    text, location, errors_only, literal = '😀', 'primary', False, False

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
                if location == 'primary':
                    if 'error_msg' in body:
                        body['error_msg'] += text
                    else:
                        for record in body['records']:
                            record['content'] += text
                    # Also cover empty-hit responses.
                    body['extra'] = text
                elif location == 'metadata':
                    for record in body.get('records', []):
                        record['metadata']['private'] = [None, {'deeper': text}]
                    body['extra'] = [None, {'deeper': text}]
                elif location == 'nested-key':
                    body['extra'] = [True, 7, {'deeper': [{text: 'private'}]}]
                else:
                    body[text] = 'private'
            data = json.dumps(body, ensure_ascii=not literal).encode('utf-8')
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
        with tempfile.TemporaryDirectory(prefix='checker-unicode-') as tmp:
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
                print(json.dumps({'command': shlex.join(command), 'responseText': ascii(text),
                                  'location': location, 'errorsOnly': errors_only, 'literal': literal,
                                  'exitCode': result.returncode, 'observedHTTPRequests': len(calls),
                                  'stdout': result.stdout, 'stderr': result.stderr}), flush=True)
                assert result.returncode == expected
                assert all(s not in result.stdout + result.stderr for s in
                           ('local-demo', 'wrong-default', 'private', 'vacation policy', 'handbook.md',
                            'leave.md', '\\ud800', '\\udfff', 'café', str(saved), endpoint))
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
                for location in ('primary', 'metadata', 'nested-key', 'extra-key'):
                    for text in ('\ud800', '\udfff'):
                        for errors_only in (False, True):
                            run(flags, 1, extra=assertions)
                errors_only = False
                for text, literal in [('😀', False), ('café 日本語 😀 \ufffd', True), ('\\ud800', False)]:
                    run(flags, 0, extra=assertions)
                text, literal = '😀', False
                run(flags, 1, extra=['--expect-count', '1'], assertion_failure=True)
                run(flags, 1, extra=['--expect-title', 'private-missing-title'], assertion_failure=True)
                text = '\ud800'
                run(flags, 2, key='bad\nkey')
                run(flags, 2, extra=['--endpoint', ' ' + endpoint])
            run(['--request-file', str(saved), '--top-k', '3'], 2)
            assert (saved.read_bytes(), saved.stat().st_mtime_ns) == saved_before
        assert original == [(p.read_bytes(), p.stat().st_mtime_ns) for p in inputs]
        assert app.snapshot_status('Bearer local-demo', b'{"knowledge_id":"handbook"}') == {
            'knowledge_id': 'handbook', 'version': 1, 'record_count': 4}
        print(json.dumps({'result': 'RESPONSE UNICODE DEMO PASS', 'cases': len(results),
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
