"""Installed checker's Unicode request preflight; synthetic loopback only."""
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
        with tempfile.TemporaryDirectory(prefix='checker-unicode-') as tmp:
            saved = Path(tmp) / 'private-request.json'
            endpoint = f'http://127.0.0.1:{server.server_port}/bridge/'

            def run(body, mode, expected, *, titles=('handbook.md',), extra=(), key='local-demo', ascii_json=True):
                raw = b' \n' + json.dumps(body, ensure_ascii=ascii_json, indent=1).encode().replace(b'0.2', b'2e-1') + b'\n'
                saved.write_bytes(raw)
                before_file = saved.read_bytes(), saved.stat().st_mtime_ns
                flags = (['--request-file', str(saved)] if mode == 'file' else
                         ['--knowledge-id', body['knowledge_id'], '--query', body['query'],
                          '--metadata-condition', json.dumps(body['metadata_condition']),
                          '--top-k', '3', '--score-threshold', '.2'])
                flags += [arg for title in titles for arg in ('--expect-title', title)]
                command = [cli, 'dify-bridge', 'check', '--endpoint', endpoint, '--key-env', 'CHECK_KEY',
                           *flags, '--expect-count', '2', *extra]
                before = len(seen)
                result = subprocess.run(command, capture_output=True, text=True, timeout=20,
                    env=dict(os.environ, CHECK_KEY=key, BRIDGE_API_KEY='wrong-default'))
                calls = seen[before:]
                results.append(result.returncode)
                print(json.dumps({'command': shlex.join(command), 'mode': mode,
                    'exitCode': result.returncode, 'observedHTTPRequests': len(calls),
                    'stdout': result.stdout, 'stderr': result.stderr}), flush=True)
                assert result.returncode == expected
                assert all(s not in result.stdout + result.stderr for s in
                           ('local-demo', 'wrong-default', 'private', 'vacation policy', 'handbook.md',
                            '\\ud800', '\\udfff', str(saved), endpoint))
                assert len(calls) == (0 if expected == 2 else 13)
                if expected == 2:
                    assert not result.stdout and 'Configuration failure:' in result.stderr
                else:
                    report = json.loads(result.stdout)
                    assert not result.stderr and len(report['checks']) == 13
                    assert report['passed'] is (expected == 0)
                    assert all(c['passed'] for c in report['checks'][1:])
                    assert json.loads(calls[0][1]) == body
                    if mode == 'file':
                        assert calls[0][1] == raw
                    assert all(json.loads(data)['metadata_condition'] == body['metadata_condition']
                               for _, data in calls[:8])
                assert (saved.read_bytes(), saved.stat().st_mtime_ns) == before_file

            for mode in ('inline', 'file'):
                for text in ('\ud800', '\udfff'):
                    for location in ('value', 'list', 'ignored-value', 'ignored-key', 'name'):
                        body = json.loads(inputs[2].read_bytes())
                        condition = body['metadata_condition']['conditions'][0]
                        condition.update(comparison_operator='is not', value='private-marker' + text)
                        titles = ('handbook.md',)
                        if location == 'list':
                            condition.update(comparison_operator='not in', value=['private-marker' + text])
                        elif location == 'ignored-value':
                            condition.update(comparison_operator='not empty', value={'nested': [text]})
                        elif location == 'ignored-key':
                            condition.update(comparison_operator='not empty', value=[{'private' + text: None}])
                        elif location == 'name':
                            condition.update(name='private' + text, value='en')
                        run(body, mode, 2, titles=titles)
                    if mode == 'file':
                        for field in ('knowledge_id', 'query'):
                            body = json.loads(inputs[2].read_bytes())
                            body[field] = 'private' + text
                            run(body, mode, 2)
                for text in ('😀', 'café 日本語', '\ufffd', r'\ud800\udfff', 'e\u0301'):
                    body = json.loads(inputs[2].read_bytes())
                    body['query'] += ' ' + text
                    body['metadata_condition']['conditions'][0].update(comparison_operator='is not', value=text)
                    for ascii_json in (True, False):
                        run(body, mode, 0, ascii_json=ascii_json)
                run(body, mode, 1, titles=('private-missing-title',))
                run(body, mode, 2, key='bad\nkey')
                run(body, mode, 2, extra=['--endpoint', ' ' + endpoint])
                run(body, mode, 2, extra=['--expect-count', '4'])
            run(body, 'file', 2, extra=['--top-k', '3'])
        assert original == [(p.read_bytes(), p.stat().st_mtime_ns) for p in inputs]
        assert app.snapshot_status('Bearer local-demo', b'{"knowledge_id":"handbook"}') == {
            'knowledge_id': 'handbook', 'version': 1, 'record_count': 4}
        print(json.dumps({'result': 'REQUEST UNICODE DEMO PASS', 'cases': len(results),
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
