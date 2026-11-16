"""Library/CLI checker timeout contracts on synthetic loopback, including a wheel."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import ragwrench.difybridge as difykb_bridge
from ragwrench.difybridge import check
from ragwrench.difybridge.config import load_config
from ragwrench.difybridge.errors import BridgeError, ConfigError
from ragwrench.difybridge.service import Application


def main():
    examples = Path(__file__).resolve().parent
    inputs = [examples / name for name in ('bridge.toml', 'chunks.jsonl', 'request.json')]
    original = [(p.read_bytes(), p.stat().st_mtime_ns) for p in inputs]
    body = json.loads(inputs[2].read_bytes())
    app = Application(load_config(inputs[0]), 'local-demo')
    snapshot = app.snapshot_status('Bearer local-demo', b'{"knowledge_id":"handbook"}')
    cli = str(Path(sys.executable).with_name('ragwrench'))
    seen, exits, forwarded = [], [], []
    real_request = check.request

    def tracked_request(endpoint, payload, key, timeout, **kwargs):
        forwarded.append(timeout)
        return real_request(endpoint, payload, key, timeout, **kwargs)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers['Content-Length']))
            seen.append((self.path, raw))
            assert self.path == '/bridge/retrieval'
            try:
                status, response = 200, app.retrieve(self.headers.get('Authorization'), raw)
            except BridgeError as exc:
                status, response = exc.status, exc.body()
            data = json.dumps(response).encode()
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
    check.request = tracked_request
    try:
        with tempfile.TemporaryDirectory(prefix='checker-timeout-') as tmp:
            saved = Path(tmp) / 'private-request.json'
            raw = b' \n' + json.dumps(body, indent=1).encode().replace(b'0.2', b'2e-1') + b'\n'
            saved.write_bytes(raw)
            before_file = saved.read_bytes(), saved.stat().st_mtime_ns
            endpoint = f'http://127.0.0.1:{server.server_port}/bridge/'

            def library(mode, timeout):
                options = dict(timeout=timeout, expect_count=2, expect_titles=['handbook.md', 'leave.md'])
                if mode == 'file':
                    return check.check_request_file(endpoint, 'local-demo', saved, **options)
                return check.check_endpoint(endpoint, 'local-demo', body['knowledge_id'], body['query'],
                    metadata_condition=body['metadata_condition'], score_threshold=.2, **options)

            def verify_report(report, calls, mode, expected):
                assert report['passed'] is expected and len(report['checks']) == len(calls) == 13
                assert json.loads(calls[0][1]) == body
                if mode == 'file':
                    assert calls[0][1] == raw
                assert all(json.loads(data)['metadata_condition'] == body['metadata_condition']
                           for _, data in calls[:8])

            invalid = [0, -1, 121, float('nan'), float('inf'), -float('inf'),
                       True, False, None, '0.25', 'private-timeout-marker', b'0.25', [], {}]
            for mode in ('inline', 'file'):
                for timeout in invalid:
                    before, before_forwarded = len(seen), len(forwarded)
                    try:
                        library(mode, timeout)
                    except ConfigError as exc:
                        assert str(exc) == 'Checker timeout must be a finite int or float in (0,120]'
                        print(json.dumps({'entrypoint': mode, 'timeoutType': type(timeout).__name__,
                            'timeoutRepr': repr(timeout), 'exception': 'ConfigError',
                            'message': str(exc), 'observedHTTPRequests': len(seen)-before}), flush=True)
                    else:
                        raise AssertionError('Invalid library timeout accepted')
                    assert len(seen) == before and len(forwarded) == before_forwarded
                for timeout in (.25, 120):
                    before, before_forwarded = len(seen), len(forwarded)
                    report = library(mode, timeout)
                    verify_report(report, seen[before:], mode, True)
                    assert forwarded[before_forwarded:] == [timeout] * 13
                    print(json.dumps({'entrypoint': mode, 'timeout': timeout,
                        'observedHTTPRequests': len(seen)-before, 'report': report}), flush=True)

                flags = (['--request-file', str(saved)] if mode == 'file' else
                    ['--knowledge-id', body['knowledge_id'], '--query', body['query'],
                     '--metadata-condition', json.dumps(body['metadata_condition']), '--score-threshold', '.2'])
                cases = [(t, 2, 2, 'local-demo') for t in ('0', '-1', '121', 'nan', 'inf', '-inf', 'private-timeout-marker')]
                cases += [('.25', 0, 2, 'local-demo'), ('.25', 1, 1, 'local-demo'),
                          ('.25', 1, 2, 'valid-but-wrong')]
                for timeout, expected, count, key in cases:
                    command = [cli, 'dify-bridge', 'check', '--endpoint', endpoint, '--key-env', 'CHECK_KEY',
                        *flags, '--timeout=' + timeout, '--expect-count', str(count)]
                    before = len(seen)
                    result = subprocess.run(command, capture_output=True, text=True, timeout=20,
                        env=dict(os.environ, CHECK_KEY=key, BRIDGE_API_KEY='wrong-default'))
                    exits.append(result.returncode)
                    print(json.dumps({'command': shlex.join(command), 'exitCode': result.returncode,
                        'observedHTTPRequests': len(seen)-before,
                        'stdout': result.stdout, 'stderr': result.stderr}), flush=True)
                    assert result.returncode == expected and len(seen)-before == (0 if expected == 2 else 13)
                    assert all(s not in result.stdout+result.stderr for s in
                        ['local-demo', 'valid-but-wrong', 'wrong-default', 'private', body['query'], str(saved), endpoint])
                    if expected == 2:
                        assert not result.stdout and 'Configuration failure:' in result.stderr
                    else:
                        assert not result.stderr
                        verify_report(json.loads(result.stdout), seen[before:], mode, expected == 0)
            assert (saved.read_bytes(), saved.stat().st_mtime_ns) == before_file
        assert original == [(p.read_bytes(), p.stat().st_mtime_ns) for p in inputs]
        assert snapshot == app.snapshot_status('Bearer local-demo', b'{"knowledge_id":"handbook"}')
        print(json.dumps({'result': 'CHECKER TIMEOUT DEMO PASS', 'module': difykb_bridge.__file__,
            'libraryRejected': 28, 'libraryPassed': 4, 'cliCases': len(exits),
            'exit0': exits.count(0), 'exit1': exits.count(1), 'exit2': exits.count(2),
            'filesAndSnapshotUnchanged': True}))
    finally:
        check.request = real_request
        server.shutdown()
        thread.join(5)
        server.server_close()
        app.close()
        assert not thread.is_alive()


if __name__ == '__main__':
    main()
