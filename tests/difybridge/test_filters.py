import pytest

from ragwrench.difybridge.errors import BridgeError
from ragwrench.difybridge.filters import OPS, Predicate
from .conftest import retrieve

# Each documented operator has an independently enumerated expected fixture set.
CASES = [
    ('lang', 'contains', 'e', {'handbook.md', 'leave.md', 'handbuch.md', 'oncall.md'}),
    ('lang', 'not contains', 'e', set()),
    ('team', 'start with', 'peo', {'handbook.md', 'leave.md', 'handbuch.md'}),
    ('team', 'end with', 'ing', {'oncall.md'}),
    ('lang', 'is', 'de', {'handbuch.md'}),
    ('lang', 'is not', 'en', {'handbuch.md'}),
    ('lang', 'in', ['de'], {'handbuch.md'}),
    ('lang', 'not in', ['de'], {'handbook.md', 'leave.md', 'oncall.md'}),
    ('tags', 'empty', None, {'oncall.md'}),
    ('tags', 'not empty', None, {'handbook.md', 'leave.md', 'handbuch.md'}),
    ('page', '=', 4, {'handbook.md'}),
    ('page', '≠', 4, {'leave.md', 'handbuch.md', 'oncall.md'}),
    ('page', '>', 4, {'leave.md'}),
    ('page', '<', 3, {'oncall.md'}),
    ('page', '≥', 4, {'handbook.md', 'leave.md'}),
    ('page', '≤', 3, {'handbuch.md', 'oncall.md'}),
    ('updated', 'before', '2026-09-01', {'oncall.md'}),
    ('updated', 'after', '2026-10-01T00:00:00Z', {'handbuch.md'}),
]


@pytest.mark.parametrize('field,op,value,expected', CASES)
def test_all_operators_qdrant_and_postfilter(qdrant_app, rows, body, field, op, value, expected):
    condition = {'name': field, 'comparison_operator': op}
    if value is not None:
        condition['value'] = value
    raw = {'conditions': [condition]}
    predicate = Predicate(raw, qdrant_app.config.knowledge['handbook'].metadata_allowlist)
    assert {row['title'] for row in rows if predicate.matches(row['metadata'])} == expected
    body['retrieval_setting']['top_k'] = 100
    body['metadata_condition'] = raw
    assert {r['title'] for r in retrieve(qdrant_app, body)} == expected


def test_coverage():
    assert {op for _, op, _, _ in CASES} == OPS


@pytest.mark.parametrize('operator', ['and', 'or'])
def test_mixed_native_and_fallback(qdrant_app, body, operator):
    body['metadata_condition'] = {'logical_operator': operator, 'conditions': [
        {'name': 'team', 'comparison_operator': 'start with', 'value': 'peo'},
        {'name': 'page', 'comparison_operator': '≤', 'value': 4}]}
    body['retrieval_setting']['top_k'] = 100
    # AND includes handbook and German handbook; OR additionally leave/oncall.
    actual = {r['title'] for r in retrieve(qdrant_app, body)}
    assert actual == ({'handbook.md', 'handbuch.md'} if operator == 'and' else {'handbook.md', 'handbuch.md', 'leave.md', 'oncall.md'})
    pred = Predicate(body['metadata_condition'], qdrant_app.config.knowledge['handbook'].metadata_allowlist)
    assert (pred.native('metadata') is None) == (operator == 'or')


@pytest.mark.parametrize('op,value', [('=', '4'), ('before', 'nonsense'), ('in', [1]), ('contains', 4), ('is', None), ('>', True)])
def test_invalid_values(op, value):
    with pytest.raises(BridgeError):
        Predicate({'conditions': [{'name': 'field', 'comparison_operator': op, 'value': value}]}, ['field'])


def test_types_dates_arrays_missing():
    compare = Predicate.compare
    assert compare(['a', 'b'], 'contains', 'a')
    assert compare(['a'], 'not contains', 'b')
    assert not compare(None, 'is not', 'a')
    assert not compare(True, '=', 1)
    assert not compare('1', '>', 0)
    assert not compare([], 'in', ['a'])
    assert compare(None, 'empty', None)
    assert compare('2026-01-01T00:00:00+02:00', 'before', '2026-01-01')
    assert not compare('invalid', 'after', '2026-01-01')


@pytest.mark.parametrize('field,op,value,expected', CASES)
def test_all_operators_jsonl(json_app, body, field, op, value, expected):
    # A lexical hit for every fixture document makes filtering independently observable.
    body['query'] = 'Employees Vacation Beschäftigte Engineers'
    body['retrieval_setting']['top_k'] = 100
    condition = {'name': field, 'comparison_operator': op}
    if value is not None:
        condition['value'] = value
    body['metadata_condition'] = {'conditions': [condition]}
    assert {r['title'] for r in retrieve(json_app, body)} == expected


def test_large_numeric_range_is_not_downcast():
    raw = {'conditions': [{'name': 'n', 'comparison_operator': '<', 'value': 2**60 + 1}]}
    pred = Predicate(raw, ['n'])
    assert pred.native('metadata') is None
    assert pred.matches({'n': 2**60})


def test_empty_in_and_array_postfilter(qdrant_app, body):
    from qdrant_client import models as m
    from ragwrench.difybridge.embedding import hash_embed
    backend = qdrant_app.backends.items['handbook']
    backend.client.upsert('handbook', points=[m.PointStruct(id=99, vector={'text': hash_embed(body['query'])},
        payload={'content': 'array metadata', 'title': 'array', 'metadata': {'lang': ['en', 'de']}})])
    body['metadata_condition'] = {'conditions': [{'name': 'lang', 'comparison_operator': 'is', 'value': 'en'}]}
    assert 'array' not in {r['title'] for r in retrieve(qdrant_app, body)}
    body['metadata_condition']['conditions'][0] = {'name': 'lang', 'comparison_operator': 'in', 'value': []}
    assert retrieve(qdrant_app, body) == []


@pytest.mark.parametrize('op,value', [('in', ['a']*257), ('is', 'a'*4097), ('in', ['a'*4097])])
def test_filter_value_size_boundary(op, value):
    with pytest.raises(BridgeError) as exc:
        Predicate({'conditions': [{'name': 'n', 'comparison_operator': op, 'value': value}]}, ['n'])
    assert exc.value.status == 400
