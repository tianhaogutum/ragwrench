"""Exercise installed saved-request checks against a counted loopback service."""
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from ragwrench.difybridge.config import load_config
from ragwrench.difybridge.service import Application, make_server


def main():
    cli = str(Path(sys.executable).with_name('ragwrench'))
    examples = Path(__file__).resolve().parent
    inputs = [examples / name for name in ('bridge.toml', 'chunks.jsonl', 'request.json')]
    original = [(p.read_bytes(), p.stat().st_mtime_ns) for p in inputs]
    seen = []

    class ObservedApplication(Application):
        def retrieve(self, authorization, raw):
            seen.append(raw)
            return super().retrieve(authorization, raw)

    app = ObservedApplication(load_config(inputs[0]), 'local-demo')
    server = make_server(app, port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01})
    thread.start()
    try:
        endpoint = f'http://127.0.0.1:{server.server_port}'
        base = [cli, 'dify-bridge', 'check', '--endpoint', endpoint]
        body = json.loads(inputs[2].read_bytes())
        inline_flags = ['--knowledge-id', body['knowledge_id'], '--query', body['query'],
                        '--metadata-condition', json.dumps(body['metadata_condition']),
                        '--top-k', str(body['retrieval_setting']['top_k']),
                        '--score-threshold', str(body['retrieval_setting']['score_threshold'])]

        def run(path, flags, expected_code, *, key='local-demo',
                key_env='SAVED_REQUEST_DEMO_KEY', inline=False, endpoint_override=None):
            before = len(seen)
            env = dict(os.environ)
            # Success proves the named key is used despite a wrong default;
            # invalid/wrong named keys must not fall back to a valid default.
            env['BRIDGE_API_KEY'] = 'wrong-default-key' if expected_code == 0 else 'local-demo'
            env.pop('SAVED_REQUEST_DEMO_KEY', None)
            if key is not None:
                env['SAVED_REQUEST_DEMO_KEY'] = key
            request_flags = inline_flags if inline else ['--request-file', str(path)]
            command = base + ['--key-env', key_env, *request_flags, *flags]
            if endpoint_override is not None:
                command += ['--endpoint', endpoint_override]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
            assert result.returncode == expected_code
            assert all(value not in result.stdout + result.stderr for value in
                       ['local-demo', 'wrong-default-key', str(path), 'vacation policy',
                        'handbook.md', 'leave.md', 'secret-content-marker', 'secret-path-marker'])
            if key and len(key) > 4:
                assert key not in result.stdout + result.stderr
            assert len(seen) - before == (0 if expected_code == 2 else 13)
            print(json.dumps({'command': shlex.join(command), 'exitCode': result.returncode,
                              'observedHTTPRequests': len(seen) - before,
                              'stdout': result.stdout, 'stderr': result.stderr}), flush=True)
            if expected_code == 2:
                assert not result.stdout and 'Configuration' in result.stderr
                return
            report = json.loads(result.stdout)
            assert not result.stderr and report['passed'] is (expected_code == 0)
            assert len(report['checks']) == 13
            if expected_code == 0:
                assert all(c['passed'] for c in report['checks'])
            return report

        saved = inputs[2]
        run(saved, ['--expect-count', '2', '--expect-title', 'handbook.md', '--expect-title', 'leave.md'], 0)
        assert seen[0] == saved.read_bytes()
        condition = json.loads(saved.read_bytes())['metadata_condition']
        assert all(json.loads(raw)['metadata_condition'] == condition for raw in seen[:8])
        mismatch = run(saved, ['--expect-count', '1'], 1)
        assert 'baseline record count does not match --expect-count' in mismatch['checks'][0]['errors']
        mismatch = run(saved, ['--expect-title', 'secret-content-marker'], 1)
        assert 'baseline is missing an --expect-title' in mismatch['checks'][0]['errors']
        with tempfile.TemporaryDirectory(prefix='dify-saved-request-') as directory:
            path = Path(directory) / 'secret-path-marker.json'
            body = json.loads(saved.read_bytes())
            body['metadata_condition']['conditions'][0]['value'] = 'absent'
            path.write_text(json.dumps(body), encoding='utf-8')
            run(path, ['--expect-count', '0'], 0)
            run(path, ['--allow-empty', '--expect-count', '1'], 1)
            path.write_bytes(b' ' * (1024 * 1024 + 1))
            run(path, [], 2)
            path.write_bytes(b'{secret-content-marker')
            run(path, [], 2)
            path.write_bytes(saved.read_bytes().replace(b'"top_k":3', b'"top_k":3,"top_k":2'))
            run(path, [], 2)
            path.unlink()
            run(path, [], 2)
        for flag, value in [('--knowledge-id', 'handbook'), ('--query', 'vacation policy'),
                            ('--metadata-condition', '{}'), ('--top-k', '3'), ('--score-threshold', '0')]:
            run(saved, [flag, value], 2)
        for inline in [False, True]:
            run(saved, ['--expect-count', '2'], 0, inline=inline)
            run(saved, [], 1, key='valid-but-wrong-key', inline=inline)
            for key in [None, '', 'synthetic-private\nkey', 'synthetic-private\rkey',
                        'synthetic-private key', 'synthetic-private\tkey', '\x1f',
                        '\x7f', 'é', 'x' * 4097]:
                run(saved, [], 2, key=key, inline=inline)
            for name in ['', '1PRIVATE', 'PRIVATE-NAME', 'PRIVATE.NAME', 'éPRIVATE', 'PRIVATE=NAME']:
                run(saved, [], 2, key_env=name, inline=inline)
            for invalid_endpoint in [
                    ' ' + endpoint, endpoint + '\n', endpoint + '/\tprivate-url-marker',
                    'http://127.0.0.1:0', 'http://127.0.0.1:65536', 'http://127.0.0.1:bad',
                    'http://@127.0.0.1', endpoint + '?', endpoint + '#',
                    'http://local_host', 'http://éxample.org', endpoint + '/café',
                    endpoint + '\\private-url-marker']:
                run(saved, [], 2, inline=inline, endpoint_override=invalid_endpoint)
        assert original == [(p.read_bytes(), p.stat().st_mtime_ns) for p in inputs]
        assert app.snapshot_status('Bearer local-demo', b'{"knowledge_id":"handbook"}') == {
            'knowledge_id': 'handbook', 'version': 1, 'record_count': 4}
        print('SAVED REQUEST DEMO PASS: exact baseline; 13 probes; filters/assertions/zero hits; '
              'file/credential/URL failures send no probes; valid wrong keys exit 1; '
              'no source mutation or credential echo')
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        app.close()
        assert not thread.is_alive()


if __name__ == '__main__':
    main()
