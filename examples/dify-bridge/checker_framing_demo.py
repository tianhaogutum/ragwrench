"""Installed checker: valid, ambiguous and truncated response framing on loopback."""
import json
import os
import shlex
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ragwrench.difybridge.config import load_config
from ragwrench.difybridge.errors import BridgeError
from ragwrench.difybridge.service import Application


def main():
    examples = Path(__file__).resolve().parent
    inputs = [examples / name for name in ('bridge.toml', 'chunks.jsonl', 'request.json')]
    original = [(p.read_bytes(), p.stat().st_mtime_ns) for p in inputs]
    app = Application(load_config(inputs[0]), 'local-demo')
    cli = str(Path(sys.executable).with_name('ragwrench'))
    framing, seen = 'valid', []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers['Content-Length']))
            seen.append((self.path, raw))
            assert self.path == '/bridge/retrieval'
            try:
                status, body = 200, app.retrieve(self.headers.get('Authorization'), raw)
            except BridgeError as exc:
                status, body = exc.status, exc.body()
            data = json.dumps(body).encode()
            size = str(len(data))
            lengths = {'valid': [size], 'duplicate': [size, size],
                       'conflicting': [size, str(len(data) + 1)], 'invalid': ['+' + size],
                       'nonascii': ['é'], 'truncated': [str(len(data) + 10)]}[framing]
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            for length in lengths:
                self.send_header('Content-Length', length)
            self.end_headers()
            self.wfile.write(data)
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01})
    thread.start()
    try:
        endpoint = f'http://127.0.0.1:{server.server_port}/bridge/'
        saved = inputs[2]
        body = json.loads(saved.read_bytes())
        inline = ['--knowledge-id', body['knowledge_id'], '--query', body['query'],
                  '--metadata-condition', json.dumps(body['metadata_condition']),
                  '--top-k', str(body['retrieval_setting']['top_k']),
                  '--score-threshold', str(body['retrieval_setting']['score_threshold'])]

        def run(flags, expected, *, extra=(), key='local-demo'):
            before = len(seen)
            env = dict(os.environ, CHECK_KEY=key,
                       BRIDGE_API_KEY='wrong-default' if expected == 0 else 'local-demo')
            command = [cli, 'dify-bridge', 'check', '--endpoint', endpoint, '--key-env', 'CHECK_KEY', *flags, *extra]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
            calls = seen[before:]
            print(json.dumps({'command': shlex.join(command), 'framing': framing,
                              'exitCode': result.returncode, 'observedHTTPRequests': len(calls),
                              'stdout': result.stdout, 'stderr': result.stderr}), flush=True)
            assert result.returncode == expected
            assert all(s not in result.stdout + result.stderr for s in
                       ('local-demo', 'wrong-default', 'valid-but-wrong', 'vacation policy',
                        'handbook.md', 'leave.md', str(saved)))
            assert len(calls) == (0 if expected == 2 else 13)
            if expected == 2:
                assert not result.stdout and 'Configuration' in result.stderr
            else:
                report = json.loads(result.stdout)
                assert not result.stderr and len(report['checks']) == 13
                assert report['passed'] is (expected == 0)
                if framing != 'valid':
                    assert all(not c['passed'] for c in report['checks'])
                assert json.loads(calls[0][1]) == body
                if '--request-file' in flags:
                    assert calls[0][1] == saved.read_bytes()
                condition = body['metadata_condition']
                assert all(json.loads(raw)['metadata_condition'] == condition for _, raw in calls[:8])

        for flags in (['--request-file', str(saved)], inline):
            for framing in ('valid', 'duplicate', 'conflicting', 'invalid', 'nonascii', 'truncated'):
                run(flags, 0 if framing == 'valid' else 1,
                    extra=['--expect-count', '2', '--expect-title', 'leave.md'])
            framing = 'valid'
            run(flags, 1, key='valid-but-wrong')
            run(flags, 1, extra=['--expect-count', '1'])
            # The framing fixture must never see locally invalid requests.
            framing = 'duplicate'
            run(flags, 2, key='bad\nkey')
            run(flags, 2, extra=['--endpoint', ' ' + endpoint])
        run(['--request-file', str(saved), '--top-k', '3'], 2)
        assert original == [(p.read_bytes(), p.stat().st_mtime_ns) for p in inputs]
        assert app.snapshot_status('Bearer local-demo', b'{"knowledge_id":"handbook"}') == {
            'knowledge_id': 'handbook', 'version': 1, 'record_count': 4}
        print('FRAMING DEMO PASS: 21 cases; both modes; malformed framing exit 1; '
              '13 redacted probes; local errors exit 2 with no requests; '
              'exact baseline/filters/assertions; files and snapshot unchanged')
    finally:
        server.shutdown()
        thread.join(5)
        server.server_close()
        app.close()
        assert not thread.is_alive()


if __name__ == '__main__':
    main()
