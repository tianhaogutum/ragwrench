"""Installed checker/status content-coding contract on synthetic loopback."""
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
    codings, errors_only, seen, results = (), False, [], []
    valid = ((), ('identity',), ('IDENTITY',))
    declarations = (*valid, ('gzip',), ('br',), ('deflate',), ('upstream-private',), ('',),
                    ('identity', 'identity'), ('identity', 'gzip'), ('gzip', 'identity'),
                    ('identity, identity',), ('gzip, br',), ('identity, gzip',),
                    ('identity; private=value',), ('identity ',), ('identity\t',))

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers['Content-Length']))
            seen.append((self.path, raw))
            try:
                if self.path == '/bridge/retrieval':
                    body = app.retrieve(self.headers.get('Authorization'), raw)
                elif self.path == '/bridge/snapshot/status':
                    body = app.snapshot_status(self.headers.get('Authorization'), raw)
                else:
                    raise AssertionError(self.path)
                status = 200
            except BridgeError as exc:
                status, body = exc.status, exc.body()
            if 'records' in body:
                for record in body['records']:
                    record['content'] += ' upstream-private café \ufffd'
            elif 'error_msg' in body:
                body['error_msg'] += ' upstream-private café \ufffd'
            data = json.dumps(body, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            for value in (() if errors_only and status < 400 else codings):
                self.send_header('Content-Encoding', value)
            self.send_header('Content-Length', str(len(data)))
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

        def run(subcommand, flags, expected, *, extra=(), key='local-demo', assertion_failure=False):
            before = len(seen)
            env = dict(os.environ, CHECK_KEY=key,
                       BRIDGE_API_KEY='wrong-default' if key == 'local-demo' else 'local-demo')
            command = [cli, 'dify-bridge', subcommand, '--endpoint', endpoint, '--key-env', 'CHECK_KEY', *flags, *extra]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
            calls = seen[before:]
            results.append(result.returncode)
            print(json.dumps({'command': shlex.join(command), 'contentEncoding': codings, 'bodyCoding': 'none; plain UTF-8 JSON',
                              'errorsOnly': errors_only, 'exitCode': result.returncode,
                              'observedHTTPRequests': len(calls), 'stdout': result.stdout,
                              'stderr': result.stderr}), flush=True)
            assert result.returncode == expected
            assert all(s not in result.stdout + result.stderr for s in
                       ('local-demo', 'wrong-default', 'private', 'vacation policy',
                        'handbook.md', 'leave.md', str(saved), endpoint))
            assert len(calls) == (0 if expected == 2 else 13 if subcommand == 'check' else 1)
            if expected == 2:
                assert not result.stdout and 'Configuration' in result.stderr
            elif subcommand == 'snapshot-status':
                assert calls == [('/bridge/snapshot/status', b'{"knowledge_id": "handbook"}')]
                if expected == 0:
                    assert not result.stderr
                    assert json.loads(result.stdout) == {'knowledge_id': 'handbook', 'version': 1, 'record_count': 4}
                else:
                    assert not result.stdout
                    assert result.stderr == 'Snapshot status failed: request rejected, unavailable or invalid response\n'
            else:
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
                    assert calls[0][1] == saved.read_bytes()
                assert all(json.loads(raw)['metadata_condition'] == body['metadata_condition'] for _, raw in calls[:8])

        for subcommand, flags in [('check', ['--request-file', str(saved)]), ('check', inline),
                                  ('snapshot-status', ['--knowledge-id', 'handbook'])]:
            assertions = ['--expect-count', '2', '--expect-title', 'leave.md'] if subcommand == 'check' else []
            for codings in declarations:
                run(subcommand, flags, 0 if codings in valid else 1, extra=assertions)
            if subcommand == 'check':
                errors_only = True
                for codings in declarations[len(valid):]:
                    run(subcommand, flags, 1, extra=assertions)
                errors_only = False
                codings = ()
                run(subcommand, flags, 1, extra=['--expect-count', '1'], assertion_failure=True)
            codings = ('gzip',)
            run(subcommand, flags, 2, key='bad\nkey')
            run(subcommand, flags, 2, extra=['--endpoint', ' ' + endpoint])
        run('check', ['--request-file', str(saved), '--top-k', '3'], 2)
        assert original == [(p.read_bytes(), p.stat().st_mtime_ns) for p in inputs]
        assert app.snapshot_status('Bearer local-demo', b'{"knowledge_id":"handbook"}') == {
            'knowledge_id': 'handbook', 'version': 1, 'record_count': 4}
        print(json.dumps({'result': 'CONTENT ENCODING DEMO PASS', 'cases': len(results),
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
