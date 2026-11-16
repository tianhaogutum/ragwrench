import json
import math
from dataclasses import replace

import pytest
from qdrant_client import QdrantClient, models as m

from ragwrench.difybridge.backends import normalize
from ragwrench.difybridge.config import Config, Knowledge
from ragwrench.difybridge.embedding import hash_embed
from ragwrench.difybridge.errors import BridgeError
from ragwrench.difybridge.service import Application
from .conftest import retrieve, running
from ragwrench.difybridge.check import request


@pytest.mark.parametrize('distance,raw,expected', [
    ('cosine', -1, 0), ('cosine', 0, 0.5), ('cosine', 1, 1), ('cosine', 1.000001, 1),
    ('dot', 0, 0.5), ('dot', math.log(3), 0.75), ('dot', -1000, 0), ('dot', 1000, 1),
    ('euclid', 0, 1), ('euclid', 3, 0.25), ('bm25', 0, 0), ('bm25', 3, 0.75),
])
def test_normalize(distance, raw, expected):
    assert normalize(raw, distance) == pytest.approx(expected)


@pytest.mark.parametrize('raw', [float('inf'), float('nan'), True, '1'])
def test_bad_scores(raw):
    with pytest.raises(BridgeError):
        normalize(raw, 'cosine')


@pytest.mark.parametrize('distance', ['cosine', 'dot', 'euclid'])
def test_real_distance_ordering(tmp_path, body, distance):
    path = tmp_path / distance
    vector = hash_embed(body['query'])
    client = QdrantClient(path=str(path))
    try:
        client.create_collection('c', vectors_config=m.VectorParams(size=64, distance=getattr(m.Distance, distance.upper())))
        client.upsert('c', points=[m.PointStruct(id=i, vector=[factor*x for x in vector], payload={'content': name, 'metadata': {}})
                                    for i, (factor, name) in enumerate([(1, 'same'), (-1, 'opposite')])])
    finally:
        client.close()
    app = Application(Config({'handbook': Knowledge(backend='qdrant', path=path, collection='c', distance=distance,
                                                   embedder={'kind': 'hash', 'dimensions': 64})}), 'test-key')
    try:
        records = retrieve(app, body)
        assert [r['content'] for r in records] == ['same', 'opposite']
        expected = {'cosine': [1, 0], 'dot': [1/(1+math.exp(-1)), 1/(1+math.exp(1))], 'euclid': [1, 1/3]}[distance]
        assert [r['score'] for r in records] == pytest.approx(expected)
        body['retrieval_setting'] = {'top_k': 1, 'score_threshold': 0.9}
        records = retrieve(app, body)
        assert len(records) == (0 if distance == 'dot' else 1)
    finally:
        app.close()


def test_scan_pagination_and_budget(qdrant_app, body):
    backend = qdrant_app.backends.items['handbook']
    client = backend.client
    vector = hash_embed(body['query'])
    orthogonal = next(i for i, x in enumerate(vector) if x == 0)
    points = []
    for i in range(40):
        v = vector.copy()
        v[orthogonal] = i / 10
        points.append(m.PointStruct(id=100+i, vector={'text': v}, payload={'content': f'candidate {i}', 'title': str(i), 'metadata': {'team': 'keep' if i == 30 else 'discard'}}))
    client.delete('handbook', points_selector=m.PointIdsList(points=[0, 1, 2, 3]))
    client.upsert('handbook', points=points)
    body['retrieval_setting']['top_k'] = 1
    body['metadata_condition'] = {'conditions': [{'name': 'team', 'comparison_operator': 'start with', 'value': 'keep'}]}
    backend.config = replace(backend.config, max_candidates=40, overfetch=1)
    assert retrieve(qdrant_app, body)[0]['content'] == 'candidate 30'
    backend.config = replace(backend.config, max_candidates=16)
    with pytest.raises(BridgeError) as exc:
        retrieve(qdrant_app, body)
    assert (exc.value.status, exc.value.code) == (422, 2003)
    # Budget must not falsely fail if the collection is fully exhausted.
    backend.config = replace(backend.config, max_candidates=40)
    body['metadata_condition']['conditions'][0]['value'] = 'no-match'
    assert retrieve(qdrant_app, body) == []


@pytest.mark.parametrize('payload', [
    {'content': 'invalid', 'metadata': None}, {'content': 'invalid', 'metadata': []},
    {'content': 42, 'metadata': {}}, {'content': 'invalid', 'title': None, 'metadata': {}},
])
def test_malformed_qdrant_payload(qdrant_app, body, payload):
    backend = qdrant_app.backends.items['handbook']
    backend.client.upsert('handbook', points=[m.PointStruct(id=99, vector={'text': hash_embed(body['query'])}, payload=payload)])
    with running(qdrant_app) as endpoint:
        status, error = request(endpoint, body, 'test-key')
        assert status == 502 and error['error_code'] == 3001
        assert 'object' in error['error_msg'] or 'strings' in error['error_msg']


def test_unavailable_and_incompatible(qdrant_app, body, monkeypatch):
    backend = qdrant_app.backends.items['handbook']
    backend.config = replace(backend.config, distance='euclid')
    with pytest.raises(BridgeError, match='distance') as exc:
        retrieve(qdrant_app, body)
    assert exc.value.status == 502

    def broken(*args, **kwargs):
        raise RuntimeError('secret-key https://private.invalid/?token=never-echo')
    monkeypatch.setattr(backend.client, 'get_collection', broken)
    with running(qdrant_app) as endpoint:
        status, error = request(endpoint, body, 'test-key')
        assert status == 503 and error['error_code'] == 3001
        assert 'secret' not in json.dumps(error) and 'private' not in json.dumps(error)


def test_collection_vector_mismatch(qdrant_app, body):
    backend = qdrant_app.backends.items['handbook']
    for updates in [{'vector_name': 'missing'}, {'embedder': {'kind': 'hash', 'dimensions': 16}}]:
        backend.config = replace(backend.config, **updates)
        with pytest.raises(BridgeError, match='Collection vector'):
            retrieve(qdrant_app, body)


def test_two_mappings_share_local_client(qdrant_app, body):
    config = qdrant_app.config
    qdrant_app.close()
    app = Application(Config({'handbook': config.knowledge['handbook'], 'alias': config.knowledge['handbook']}), 'test-key')
    try:
        a, b = app.backends.items.values()
        assert a.client is b.client and a.lock is b.lock
        body['knowledge_id'] = 'alias'
        assert retrieve(app, body)
    finally:
        app.close()
