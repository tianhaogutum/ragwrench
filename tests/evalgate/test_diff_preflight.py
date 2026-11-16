"""Standalone audit contract; identical malformed inputs must never match."""
import copy
import json
from collections.abc import Mapping
from types import MappingProxyType

import pytest

from ragwrench.evalgate.cli import main
from ragwrench.evalgate.compare import compare_many, compare_many_saved
from ragwrench.evalgate.formats import InputError
from ragwrench.evalgate.metrics import parse_metrics
from ragwrench.evalgate.report import render
from ragwrench.evalgate.saved import saved_evaluation


def audit(*, saved=False, candidates=('a.v1', 'z'), metrics='hit@1,recall@1', **options):
    q = {'q1': {'d': 1}, 'q2': {'d': 1}}
    b = {qid: {'d': 1.0} for qid in q}
    c = {'q2': {'d': 1.0}}
    m = parse_metrics(metrics)
    args = dict(correction='holm', method='randomization', samples=100, preflight_only=True,
                thresholds={'hit@1': 1}, candidate_thresholds={'a.v1': {'hit@1': None}},
                max_family_pairs=10, max_family_resamples=10000)
    args.update(options)
    if saved:
        return compare_many_saved(saved_evaluation(q, b, m),
                                  {n: saved_evaluation(q, c, m) for n in candidates}, **args)
    return compare_many(q, b, {n: c for n in candidates}, m, **args)


def diff(expected, actual):
    from ragwrench.evalgate.preflight import diff_preflight
    return diff_preflight(expected, actual)


def forbid_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('standalone diff read retrieval inputs, scored or ran inference/correction')
    for module, names in [
        ('ragwrench.evalgate.cli', ('read_qrels', 'read_run', 'read_evaluation')),
        ('ragwrench.evalgate.formats', ('read_qrels', 'read_run')),
        ('ragwrench.evalgate.saved', ('read_evaluation', 'evaluate', 'saved_evaluation', 'bind_evaluation')),
        ('ragwrench.evalgate.compare', ('evaluate', 'paired_test', 'adjust_p_values', 'bind_evaluation')),
        ('ragwrench.evalgate.metrics', ('evaluate',)),
        ('ragwrench.evalgate.statistics', ('paired_test', 'adjust_p_values')),
    ]:
        for name in names:
            monkeypatch.setattr(f'{module}.{name}', forbidden)


def ordered(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: ordered(v) for k, v in reversed(list(value.items()))})
    return value


@pytest.mark.parametrize('method', ['bootstrap', 'randomization', 'ttest'])
@pytest.mark.parametrize('limited', [False, True])
def test_equivalence_is_standalone_and_nonmutating(method, limited, monkeypatch):
    opts = dict(method=method)
    if not limited:
        opts.update(max_family_pairs=None, max_family_resamples=None)
    expected, actual = audit(**opts), audit(saved=True, **opts)
    actual['policy']['thresholds']['hit@1'] = 1.0
    actual['policy']['effective_thresholds']['z']['hit@1'] = 1.0
    before = copy.deepcopy((expected, actual))
    forbid_work(monkeypatch)
    assert diff(ordered(expected), ordered(actual)) == {
        'schema_version': 1, 'command': 'diff-preflight', 'matches': True, 'changes': []}
    assert (expected, actual) == before


def change(path, expected, actual, ep=True, ap=True):
    return dict(path=path, expected_present=ep, actual_present=ap, expected=expected, actual=actual)


def test_exact_paths_values_presence_and_order(monkeypatch):
    expected = audit()
    actual = audit(candidates=('a.v1', 'new'), metrics='hit@1,map@2', seed=43, confidence=.9,
                   thresholds={'hit@1': .8, 'map@2': .2}, candidate_thresholds={'new': {'hit@1': None}},
                   max_family_pairs=11, max_family_resamples=None)
    actual['input_bindings']['dataset_sha256'] = '0' * 64
    # Define the diff independently: recurse objects, atomic arrays, whole added/removed entries.
    def reference(a, b, path=''):
        if isinstance(a, dict) and isinstance(b, dict):
            rows = []
            for k in sorted(a.keys() | b.keys()):
                p = f'{path}/{k}'
                if k not in a or k not in b:
                    rows.append(change(p, a.get(k), b.get(k), k in a, k in b))
                else:
                    rows.extend(reference(a[k], b[k], p))
            return rows
        return [] if a == b else [change(path, a, b)]
    rows = reference(expected, actual)
    assert change('/policy/candidate_thresholds/a.v1', {'hit@1': None}, None, True, False) in rows
    assert change('/policy/candidate_thresholds/new', None, {'hit@1': None}, False, True) in rows
    assert change('/policy/thresholds/map@2', None, .2, False, True) in rows
    assert any(r['path'] == '/input_bindings/candidates/z' and not r['actual_present'] for r in rows)
    forbid_work(monkeypatch)
    result = diff(expected, actual)
    assert result == {'schema_version': 1, 'command': 'diff-preflight', 'matches': False, 'changes': rows}
    assert [r['path'] for r in rows] == sorted(r['path'] for r in rows)
    assert render(result, 'json') == render(diff(ordered(expected), ordered(actual)), 'json')


def test_present_null_is_distinct_from_missing_and_numbers_are_exact(monkeypatch):
    expected = audit(candidate_thresholds={})
    actual = audit(candidate_thresholds={'a.v1': {'hit@1': None}})
    forbid_work(monkeypatch)
    rows = diff(expected, actual)['changes']
    assert change('/policy/candidate_thresholds/a.v1', None, {'hit@1': None}, False, True) in rows
    expected = copy.deepcopy(actual)
    actual['policy']['thresholds']['hit@1'] = 1.0000000000000002
    actual['policy']['effective_thresholds']['z']['hit@1'] = 1.0000000000000002
    assert len(diff(expected, actual)['changes']) == 2


# Every fixed object requires all and only its fields. Dynamic map subsets are separately tested.
FIXED_PATHS = [(), ('settings',), ('limits',), ('policy',), ('input_bindings',),
               ('input_bindings', 'evaluator'), ('input_bindings', 'parameters'),
               ('input_bindings', 'baseline'), ('input_bindings', 'candidates', 'a.v1'),
               ('input_bindings', 'candidates', 'z')]
BASE = audit()
MISSING = []
for p in FIXED_PATHS:
    obj = BASE
    for part in p:
        obj = obj[part]
    MISSING.extend((p, key) for key in obj)


def at(value, path):
    for part in path:
        value = value[part]
    return value


@pytest.mark.parametrize('path', FIXED_PATHS)
@pytest.mark.parametrize('kind', ['unknown', 'not-object'])
def test_fixed_object_shapes(path, kind, monkeypatch):
    bad = copy.deepcopy(BASE)
    if kind == 'unknown':
        at(bad, path)['unknown'] = 1
    elif path:
        at(bad, path[:-1])[path[-1]] = []
    else:
        bad = []
    forbid_work(monkeypatch)
    with pytest.raises(InputError, match='expected.*audit'):
        diff(bad, bad)


@pytest.mark.parametrize('path,key', MISSING)
def test_missing_fixed_fields(path, key, monkeypatch):
    bad = copy.deepcopy(BASE)
    del at(bad, path)[key]
    forbid_work(monkeypatch)
    with pytest.raises(InputError, match='expected.*audit'):
        diff(bad, bad)


# Explicit construction keeps parameter IDs readable for failed negative boundaries.
BAD_VALUES = [(('schema_version',), v) for v in (True, 1.0, 2, None)]
BAD_VALUES += [
    (('command',), 'compare'), (('mode',), 'compare'), (('query_count',), 1), (('query_count',), 2.0),
    (('query_count',), True), (('family_size',), 3), (('family_size',), 4.0),
    (('estimated_family_resamples',), 801), (('estimated_family_resamples',), 800.0),
]
for key, values in {
    'candidate_names': [[], ['z', 'a.v1'], ['a.v1', 'a.v1'], ['a/v1', 'z'], ['a~v1', 'z'], ['a.v1', 2], ('a.v1', 'z')],
    'metric_names': [[], ['recall@1', 'hit@1'], ['hit@01', 'recall@1'], [' hit@1', 'recall@1'], ['hit@1', 'hit@1'], ['hit@1000001']],
}.items():
    BAD_VALUES += [((key,), v) for v in values]
for key, values in {'seed': [True, 42.0, '42'], 'samples': [True, 100.0, 99, 1000001],
                    'confidence': [True, 0, 1, float('nan'), float('inf'), '0.95'],
                    'test': ['unknown', []]}.items():
    BAD_VALUES += [(('settings', key), v) for v in values]
for key in ('max_family_pairs', 'max_family_resamples'):
    BAD_VALUES += [(('limits', key), v) for v in (True, 10.0, 0, -1, 1, '10')]
BAD_VALUES += [
    (('input_bindings', 'schema_version'), 2), (('input_bindings', 'schema_version'), True),
    (('input_bindings', 'evaluator', 'version'), 'future'), (('input_bindings', 'evaluator', 'name'), 'other'),
    (('input_bindings', 'parameters', 'gain'), 'unknown'),
    (('input_bindings', 'parameters', 'metrics'), ['hit@1']),
    (('input_bindings', 'parameters', 'tie_rule'), 'unknown'),
    (('input_bindings', 'parameters', 'query_policy'), 'unknown'),
    (('input_bindings', 'candidates'), {}), (('input_bindings', 'candidates'), {'unknown': {}}),
    (('policy', 'correction'), []), (('policy', 'alpha'), True), (('policy', 'alpha'), 0),
    (('policy', 'significance_affects_gate'), 0), (('policy', 'gate'), 'unknown'),
    (('policy', 'family_size'), 3), (('policy', 'max_family_pairs'), 11),
    (('policy', 'estimated_family_resamples'), 0), (('policy', 'effective_thresholds'), {}),
    (('policy', 'candidate_thresholds'), {'unknown': {}}),
    (('policy', 'candidate_thresholds'), {'z': {'hit@1': False}}),
    (('policy', 'candidate_thresholds'), {'z': {'map@1': None}}),
    (('policy', 'thresholds'), {'hit@1': None}), (('policy', 'thresholds'), {'hit@1': False}),
    (('policy', 'thresholds'), {'hit@1': float('inf')}), (('policy', 'thresholds'), {'hit@1': -1}),
    (('policy', 'thresholds'), {'map@1': 1}),
    (('policy', 'effective_thresholds', 'a.v1'), {'hit@1': 1}),
]
for path in [('input_bindings', 'dataset_sha256'), ('input_bindings', 'baseline', 'run_sha256'),
             ('input_bindings', 'candidates', 'z', 'artifact_sha256')]:
    BAD_VALUES += [(path, v) for v in ('A' * 64, 'a' * 63, 'g' * 64, None, 1)]


@pytest.mark.parametrize('path,value', BAD_VALUES)
@pytest.mark.parametrize('side', ['both', 'actual'])
def test_independent_validation(path, value, side, monkeypatch):
    bad = copy.deepcopy(BASE)
    at(bad, path[:-1])[path[-1]] = value
    forbid_work(monkeypatch)
    with pytest.raises(InputError, match=('expected' if side == 'both' else 'actual') + '.*audit'):
        diff(bad if side == 'both' else BASE, bad)


def test_per_pair_limit_and_ttest_zero_work(monkeypatch):
    bad = audit(max_family_resamples=None)
    bad['query_count'] = 500001
    bad['estimated_family_resamples'] = 500001 * 100 * 4
    good = audit(method='ttest', max_family_resamples=1)
    good['query_count'] = 10**9
    forbid_work(monkeypatch)
    with pytest.raises(InputError, match='50000000'):
        diff(bad, bad)
    assert diff(good, good)['matches']
    good['estimated_family_resamples'] = 1
    with pytest.raises(InputError):
        diff(good, good)


@pytest.fixture
def files(tmp_path):
    expected, actual = tmp_path / 'expected.json', tmp_path / 'actual.json'
    expected.write_text(render(audit(), 'json'))
    actual.write_text(render(audit(saved=True), 'json'))
    return expected, actual


def cli_args(files, *extra):
    return ['diff-preflight', '--expected', str(files[0]), '--actual', str(files[1]), *map(str, extra)]


def test_cli_match_difference_and_deterministic_json(files, capsys, monkeypatch):
    forbid_work(monkeypatch)
    assert main(cli_args(files)) == 0
    matched = capsys.readouterr()
    assert matched.err == '' and json.loads(matched.out)['matches']
    value = json.loads(files[1].read_text())
    value['settings']['seed'] = 43
    files[1].write_text(json.dumps(value))
    assert main(cli_args(files)) == 1
    first = capsys.readouterr()
    assert first.err == ''
    assert json.loads(first.out)['changes'] == [change('/settings/seed', 42, 43)]
    files[1].write_text(json.dumps(dict(reversed(list(value.items()))), indent=4))
    assert main(cli_args(files)) == 1
    assert capsys.readouterr().out == first.out
    output = files[0].parent / 'diff.json'
    assert main(cli_args(files, '--output', output)) == 1
    assert capsys.readouterr().out == '' and output.read_text() == first.out
    assert main(cli_args(files, '--output', output)) == 2
    assert output.read_text() == first.out
    capsys.readouterr()
    assert main(cli_args(files, '--output', output, '--force')) == 1
    assert output.read_text() == first.out


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('bad', ['null', '{}', '{', '{"x":1,"x":2}', 'NaN', 'encoding', 'missing', 'late'])
def test_cli_invalid_no_partial_diff_preserves_output(files, side, bad, capsys, monkeypatch):
    if bad == 'encoding':
        files[side].write_bytes(b'\xff')
    elif bad == 'missing':
        files[side].unlink()
    elif bad == 'late':
        value = json.loads(files[side].read_text())
        value['input_bindings']['candidates']['z']['run_sha256'] = 'broken'
        files[side].write_text(json.dumps(value))
    else:
        files[side].write_text(bad)
    output = files[0].parent / 'protected.json'
    output.write_bytes(b'sentinel\n')
    forbid_work(monkeypatch)
    assert main(cli_args(files, '--output', output, '--force')) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and ('expected' if side == 0 else 'actual') in captured.err
    assert output.read_bytes() == b'sentinel\n'
    assert not list(output.parent.glob('.retgate-*'))


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('alias', ['direct', 'symlink', 'relative'])
def test_cli_protects_both_audit_inputs(files, side, alias, monkeypatch, capsys):
    path = files[side]
    before = [p.read_bytes() for p in files]
    if alias == 'symlink':
        path = path.with_name('alias.json')
        path.symlink_to(files[side])
    elif alias == 'relative':
        monkeypatch.chdir(path.parent)
        path = path.name
    forbid_work(monkeypatch)
    assert main(cli_args(files, '--output', path, '--force')) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'overwrite an input' in captured.err
    assert [p.read_bytes() for p in files] == before


@pytest.mark.parametrize('method', ['bootstrap', 'randomization', 'ttest'])
def test_added_removed_candidates_metrics_counts_work_and_policy(method, monkeypatch):
    expected = audit(candidates=('a.v1', 'removed', 'z'), metrics='hit@1,precision@1,recall@1',
                     thresholds={'hit@1': 1, 'precision@1': .5}, method=method)
    actual = audit(candidates=('a.v1', 'added'), metrics='hit@1,map@2',
                   thresholds={'hit@1': 1, 'map@2': .5}, method=method)
    forbid_work(monkeypatch)
    rows = {r['path']: r for r in diff(expected, actual)['changes']}
    assert rows['/family_size'] == change('/family_size', 9, 4)
    if method != 'ttest':
        assert rows['/estimated_family_resamples'] == change('/estimated_family_resamples', 1800, 800)
    else:
        assert '/estimated_family_resamples' not in rows
    assert rows['/policy/thresholds/precision@1'] == change('/policy/thresholds/precision@1', .5, None, True, False)
    assert rows['/policy/thresholds/map@2'] == change('/policy/thresholds/map@2', None, .5, False, True)
    assert rows['/input_bindings/candidates/added']['actual_present']
    assert not rows['/input_bindings/candidates/removed']['actual_present']
    assert not rows['/policy/effective_thresholds/z']['actual_present']
    # Returned subtree ownership: editing a diff never mutates an audit.
    before = copy.deepcopy(actual)
    rows['/input_bindings/candidates/added']['actual']['run_sha256'] = 'changed'
    assert actual == before


def test_validator_public_contract_and_optional_empty_maps(monkeypatch):
    from ragwrench.evalgate.preflight import validate_preflight
    expected = audit(candidate_thresholds={'a.v1': {}})
    actual = audit(candidate_thresholds={})
    forbid_work(monkeypatch)
    assert validate_preflight(ordered(expected)) is None
    assert diff(expected, actual)['changes'] == [
        change('/policy/candidate_thresholds/a.v1', {}, None, True, False)]
    for bad in (None, False, {}, [], 'audit'):
        with pytest.raises(InputError, match='invalid audit'):
            validate_preflight(bad)
    expected['input_bindings']['schema_version'] = 1.0
    with pytest.raises(InputError, match='schema_version'):
        validate_preflight(expected)


def test_cli_write_failure_is_atomic(files, monkeypatch, capsys):
    import ragwrench.evalgate.cli as cli
    output = files[0].parent / 'diff.json'
    output.write_bytes(b'sentinel\n')
    forbid_work(monkeypatch)
    def fail_replace(*args):
        raise OSError('simulated atomic replacement failure')
    monkeypatch.setattr(cli.os, 'replace', fail_replace)
    assert main(cli_args(files, '--output', output, '--force')) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'replacement failure' in captured.err
    assert output.read_bytes() == b'sentinel\n'
    assert not list(output.parent.glob('.retgate-*'))
