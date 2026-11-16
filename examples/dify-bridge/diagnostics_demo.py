"""Installed CLI: diagnose a paginated Dify filter query, budget error and auth."""
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

from qdrant_client import QdrantClient, models as m

from ragwrench.difybridge.embedding import hash_embed


def post(endpoint, payload, key='local-demo', path='/retrieval/diagnostics'):
    req = Request(endpoint + path, data=json.dumps(payload).encode(), headers={
        'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    try:
        response = urlopen(req, timeout=5)
    except HTTPError as exc:
        response = exc
    with response:
        return response.status, json.load(response)


def main():
    cli = str(Path(sys.executable).with_name('ragwrench'))
    body = {'knowledge_id': 'handbook', 'query': 'vacation policy',
            'retrieval_setting': {'top_k': 1, 'score_threshold': 0},
            'metadata_condition': {'conditions': [
                {'name': 'team', 'comparison_operator': 'start with', 'value': 'keep'}]}}
    with tempfile.TemporaryDirectory(prefix='difykb-diagnostics-') as tmp:
        root = Path(tmp)
        vector = hash_embed(body['query'])
        axis = next(i for i, value in enumerate(vector) if value == 0)
        # Synthetic ranked vectors: only the 31st candidate passes the prefix filter.
        client = QdrantClient(path=str(root / 'db'))
        try:
            client.create_collection('handbook', vectors_config={
                'text': m.VectorParams(size=64, distance=m.Distance.COSINE)})
            points = []
            for i in range(40):
                v = vector.copy()
                v[axis] = i / 10
                points.append(m.PointStruct(id=i, vector={'text': v}, payload={
                    'content': f'candidate {i}', 'title': str(i),
                    'metadata': {'team': 'keep' if i == 30 else 'discard', 'secret': 'excluded'}}))
            client.upsert('handbook', points=points)
        finally:
            client.close()
        config = root / 'bridge.toml'
        config.write_text('\n'.join(f'''[knowledge.{name}]
backend = "qdrant"
path = "db"
collection = "handbook"
vector_name = "text"
metadata_allowlist = ["team"]
max_candidates = {budget}
overfetch = 1
[knowledge.{name}.embedder]
kind = "hash"
dimensions = 64
''' for name, budget in [('handbook', 40), ('bounded', 16)]))
        process = subprocess.Popen([cli, 'dify-bridge', 'serve', '--config', str(config), '--port', '0'],
                                   env=dict(os.environ, BRIDGE_API_KEY='local-demo'),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            assert select.select([process.stdout], [], [], 5)[0], 'CLI startup timed out'
            line = process.stdout.readline()
            assert '127.0.0.1:' in line, 'CLI did not announce its endpoint'
            endpoint = 'http://' + line.strip().split()[-1]
            status, result = post(endpoint, body)
            assert status == 200 and result['records'][0]['title'] == '30'
            d = result['diagnostics']
            assert (d['pages_fetched'], d['candidates_fetched'], d['candidates_evaluated'],
                    d['filter_rejected'], d['stop_reason']) == (2, 32, 31, 30, 'top_k')
            assert post(endpoint, body, path='/retrieval') == (200, {'records': result['records']})
            body['knowledge_id'] = 'bounded'
            status, failure = post(endpoint, body)
            assert (status, failure['error_code']) == (422, 2003) and 'records' not in failure
            d = failure['diagnostics']
            assert (d['candidates_fetched'], d['candidates_evaluated'], d['budget_probe_points'],
                    d['stop_reason']) == (17, 16, 1, 'candidate_budget')
            assert post(endpoint, body, key='synthetic-wrong-key') == (
                403, {'error_code': 1002, 'error_msg': 'Authorization failed'})
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=5) == 0
            assert process.stderr.read() == ''
            print('DIAGNOSTICS DEMO PASS: installed CLI; 2 pages / 31 evaluated / 30 filtered; '
                  '16-candidate budget -> 422; redacted auth; ordinary retrieval unchanged')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()


if __name__ == '__main__':
    main()
