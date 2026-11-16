"""Exercise installed-CLI JSONL reload and read-only status over loopback."""
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ragwrench.difybridge.check import request


def corpus_request(endpoint, route, key='local-demo'):
    req = Request(endpoint + route, data=b'{"knowledge_id":"handbook"}',
                  headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    try:
        response = urlopen(req, timeout=5)
    except HTTPError as exc:
        response = exc
    with response:
        return response.status, json.load(response)


def main():
    cli = str(Path(sys.executable).with_name('ragwrench'))
    with tempfile.TemporaryDirectory(prefix='difykb-reload-') as tmp:
        root = Path(tmp)
        corpus = root / 'chunks.jsonl'

        def publish(raw):
            candidate = root / 'chunks.next'
            candidate.write_text(raw, encoding='utf-8')
            candidate.replace(corpus)

        publish('{"content":"apple old","title":"old"}\n')
        config = root / 'bridge.toml'
        config.write_text('[knowledge.handbook]\nbackend="jsonl"\npath="chunks.jsonl"\n')
        env = dict(os.environ, BRIDGE_API_KEY='local-demo', DEMO_STATUS_KEY='local-demo')
        process = subprocess.Popen([cli, 'dify-bridge', 'serve', '--config', str(config), '--port', '0'],
                                   env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            assert select.select([process.stdout], [], [], 5)[0], 'CLI startup timed out'
            line = process.stdout.readline()
            assert '127.0.0.1:' in line, 'CLI did not announce its loopback endpoint'
            endpoint = 'http://' + line.strip().split()[-1]

            def status_cli(*, knowledge='handbook', key='local-demo'):
                result = subprocess.run(
                    [cli, 'dify-bridge', 'snapshot-status', '--endpoint', endpoint, '--knowledge-id', knowledge,
                     '--key-env', 'DEMO_STATUS_KEY'],
                    env=dict(env, DEMO_STATUS_KEY=key), capture_output=True, text=True, timeout=15)
                assert all(secret not in result.stdout + result.stderr
                           for secret in ('local-demo', 'wrong-demo-key', str(root)))
                return result

            def status(version, count):
                result = status_cli()
                assert (result.returncode, result.stderr) == (0, ''), result.stderr
                assert json.loads(result.stdout) == {
                    'knowledge_id': 'handbook', 'version': version, 'record_count': count}
                print('snapshot-status exit 0: ' + result.stdout.strip())

            status(1, 1)
            for options, expected_code in [({'key': 'wrong-demo-key'}, 1),
                                           ({'knowledge': 'unknown'}, 1), ({'key': ''}, 2)]:
                result = status_cli(**options)
                assert result.returncode == expected_code and result.stdout == '' and result.stderr.strip()
                print(f'snapshot-status exit {result.returncode}: ' + result.stderr.strip())
            body = {'knowledge_id': 'handbook', 'query': 'apple',
                    'retrieval_setting': {'top_k': 3, 'score_threshold': 0}}
            before = request(endpoint, body, 'local-demo')
            assert before[0] == 200 and before[1]['records'][0]['title'] == 'old'
            publish('{"content":"apple new","title":"new"}\n{"content":"pear"}\n')
            assert request(endpoint, body, 'local-demo') == before
            status(1, 1)
            assert corpus_request(endpoint, '/reload') == (
                200, {'knowledge_id': 'handbook', 'version': 2, 'record_count': 2})
            status(2, 2)
            after = request(endpoint, body, 'local-demo')
            assert after[0] == 200 and after[1]['records'][0]['title'] == 'new'
            publish('{"content":"apple partial"}\n{broken}\n')
            assert corpus_request(endpoint, '/reload') == (
                503, {'error_code': 3001, 'error_msg': 'JSONL reload failed; previous corpus retained'})
            assert request(endpoint, body, 'local-demo') == after
            status(2, 2)
            corpus.unlink()
            print('Source file deleted; active snapshot remains available:')
            status(2, 2)  # Status reads memory, even when the source is missing.
            # Failed publication did not consume version 3; an empty corpus is valid.
            publish('')
            assert corpus_request(endpoint, '/reload') == (
                200, {'knowledge_id': 'handbook', 'version': 3, 'record_count': 0})
            status(3, 0)
            status(3, 0)
            assert request(endpoint, body, 'local-demo') == (200, {'records': []})
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=5) == 0
            print('RELOAD/STATUS DEMO PASS: installed snapshot-status CLI; versions 1 -> 2 -> 3; read-only counts 1 -> 2 -> 0; malformed/missing source retained status; wrong key redacted')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()


if __name__ == '__main__':
    main()
