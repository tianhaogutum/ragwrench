"""Expected bound audits guard inference; contract recorded before runtime edits."""
import copy
import json
from collections import Counter
from collections.abc import Mapping
from types import MappingProxyType

import pytest

from ragwrench.evalgate.cli import main
from ragwrench.evalgate.compare import compare_many, compare_many_saved
from ragwrench.evalgate.formats import InputError
from ragwrench.evalgate.report import render
from ragwrench.evalgate.saved import saved_evaluation
from .test_preflight import api, argv, family, forbid_inference


@pytest.mark.parametrize('saved', [False, True])
@pytest.mark.parametrize('audit_saved', [False, True])
@pytest.mark.parametrize('method', ['bootstrap', 'randomization', 'ttest'])
@pytest.mark.parametrize('drop', [.49, .5])
def test_expected_match_preserves_reports(family, saved, audit_saved, method, drop):
    options = dict(method=method, thresholds={'hit@1': drop}, max_family_pairs=4, max_family_resamples=800)
    audit = api(family, audit_saved, preflight_only=True, **options)
    before = copy.deepcopy(audit)
    expected = api(family, saved, **options)
    actual = api(family, saved, expected_preflight=audit, **options)
    assert actual == expected
    assert actual['passed'] == (drop == .5)
    for fmt in ('text', 'json', 'markdown', 'junit', 'csv'):
        assert render(actual, fmt) == render(expected, fmt)
    assert api(family, saved, expected_preflight=audit, preflight_only=True, **options) == api(
        family, saved, preflight_only=True, **options)
    assert audit == before


# Exercise every leaf and every object boundary, including the full resolved policy.
def paths(value, prefix=()):
    yield prefix, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from paths(child, (*prefix, key))


def audit_template():
    q = {'q1': {'d': 1}, 'q2': {'d': 1}}
    b = {qid: {'d': 1.0} for qid in q}
    from ragwrench.evalgate.metrics import parse_metrics
    return compare_many(q, b, {'strict': {'q2': {'d': 1.0}}, 'relaxed': {'q2': {'d': 1.0}}},
                        parse_metrics('recall@1,hit@1'), correction='holm', method='bootstrap', samples=100,
                        thresholds={'hit@1': .49}, candidate_thresholds={'relaxed': {'hit@1': None}},
                        preflight_only=True, max_family_pairs=4, max_family_resamples=800)


AUDIT = audit_template()
MUTATIONS = []
for path, value in paths(AUDIT):
    if isinstance(value, dict):
        MUTATIONS.extend([(path, 'unknown'), (path, 'missing')])
    elif path != ('command',):
        MUTATIONS.append((path, 'value'))


@pytest.mark.parametrize('saved', [False, True])
@pytest.mark.parametrize('path,kind', MUTATIONS, ids=lambda v: '.'.join(v) if isinstance(v, tuple) else v)
def test_every_audit_field_guards_inference(family, saved, path, kind, monkeypatch):
    audit = copy.deepcopy(AUDIT)
    parent = audit
    for key in path[:-1]:
        parent = parent[key]
    value = parent[path[-1]] if path else audit
    if kind == 'unknown':
        value['unknown'] = None
    elif kind == 'null':
        path.write_text('null')
    elif kind == 'encoding':
        path.write_bytes(b'\xff')
    elif kind == 'missing':
        if value:
            value.pop(next(iter(value)))
        else:
            parent.pop(path[-1])
    else:
        parent[path[-1]] = (not value if type(value) is bool else value + 1 if type(value) in (int, float)
                            else list(reversed(value)) if isinstance(value, list) and len(value) > 1
                            else 'changed')
    forbid_inference(monkeypatch)
    with pytest.raises(InputError, match='expected preflight'):
        api(family, saved, expected_preflight=audit, max_family_pairs=4, max_family_resamples=800)


@pytest.mark.parametrize('saved', [False, True])
@pytest.mark.parametrize('bad', [False, [], 'audit', 1, {}, {'schema_version': 1}])
def test_unbound_or_nonobject_audit(family, saved, bad, monkeypatch):
    forbid_inference(monkeypatch)
    with pytest.raises(InputError, match='expected preflight'):
        api(family, saved, expected_preflight=bad)


@pytest.mark.parametrize('path,value', [
    (('schema_version',), True), (('schema_version',), 1.0), (('input_bindings', 'schema_version'), True),
    (('settings', 'samples'), 100.0), (('settings', 'seed'), 42.0), (('query_count',), 2.0),
    (('limits', 'max_family_pairs'), 4.0), (('policy', 'family_size'), 4.0),
    (('settings', 'confidence'), True), (('settings', 'confidence'), float('nan')),
    (('settings', 'confidence'), float('inf')), (('policy', 'significance_affects_gate'), 0),
    (('policy', 'candidate_thresholds', 'relaxed', 'hit@1'), False),
    (('candidate_names',), ('relaxed', 'strict')), (('command',), 'compare'),
    (('mode',), 'compare'), (('input_bindings',), None)])
def test_audit_types_are_not_coerced(family, path, value, monkeypatch):
    audit = copy.deepcopy(AUDIT)
    parent = audit
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    forbid_inference(monkeypatch)
    with pytest.raises(InputError, match='expected preflight'):
        api(family, False, expected_preflight=audit, max_family_pairs=4, max_family_resamples=800)


@pytest.mark.parametrize('saved', [False, True])
@pytest.mark.parametrize('options', [dict(method='ttest'), dict(seed=43), dict(samples=101), dict(confidence=.9),
    dict(correction='none'), dict(alpha=.1), dict(thresholds={'hit@1': .5}),
    dict(candidate_thresholds={'strict': {'hit@1': None}}), dict(max_family_pairs=5),
    dict(max_family_resamples=801)])
def test_actual_settings_drift(family, saved, options, monkeypatch):
    audit = api(family, saved, preflight_only=True)
    forbid_inference(monkeypatch)
    with pytest.raises(InputError, match='expected preflight'):
        api(family, saved, expected_preflight=audit, **options)


@pytest.mark.parametrize('saved', [False, True])
@pytest.mark.parametrize('changed', ['baseline', 'candidate', 'ignored', 'judgments'])
def test_actual_equal_score_identity_drift(family, saved, changed, monkeypatch):
    audit = api(family, saved, preflight_only=True)
    q, b, c, m, artifacts, files = copy.deepcopy(family)
    if changed == 'judgments':
        q['q1']['d'] = 2
    elif changed == 'ignored':
        c['ignored'] = {'d': 1.0}
    else:
        run = b if changed == 'baseline' else c
        for docs in run.values():
            for d in docs:
                docs[d] += 10
    new_artifacts = [saved_evaluation(q, run, m) for run in (b, c)]
    assert [a['means'] for a in new_artifacts] == [a['means'] for a in artifacts]
    forbid_inference(monkeypatch)
    with pytest.raises(InputError, match='expected preflight.*input_bindings'):
        api((q, b, c, m, new_artifacts, files), saved, expected_preflight=audit)


def reverse(value):
    if isinstance(value, Mapping):
        return {k: reverse(v) for k, v in reversed(list(value.items()))}
    return value


@pytest.mark.parametrize('saved', [False, True])
def test_order_numeric_mapping_and_presentation_equivalence(family, saved):
    audit = api(family, not saved, method='ttest', preflight_only=True, thresholds={'hit@1': 1})
    audit['policy']['thresholds']['hit@1'] = 1.0
    audit['policy']['effective_thresholds']['strict']['hit@1'] = 1.0
    audit = MappingProxyType({**reverse(audit), 'settings': MappingProxyType(audit['settings'])})
    q, b, c, m, artifacts, files = family
    reordered = (reverse(q), reverse(b), reverse(c), list(reversed(m)), [reverse(a) for a in artifacts], files)
    actual = api(reordered, saved, expected_preflight=audit, method='ttest', top=0, thresholds={'hit@1': 1})
    assert actual == api(family, saved, method='ttest', top=0, thresholds={'hit@1': 1})


@pytest.mark.parametrize('saved', [False, True])
def test_no_duplicate_scoring_or_reads(family, saved, monkeypatch, capsys):
    import ragwrench.evalgate.cli as cli
    import ragwrench.evalgate.compare as comparison
    audit = api(family, saved, preflight_only=True)
    audit_path = family[-1][0].parent / 'audit.json'
    audit_path.write_text(json.dumps(audit))
    calls = []
    for name in ('read_qrels', 'read_run', 'read_evaluation'):
        original = getattr(cli, name)
        def counted(path, *args, _original=original, **kwargs):
            calls.append(path)
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(cli, name, counted)
    score_calls = []
    original = comparison.evaluate
    def evaluate(*args, **kwargs):
        score_calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(comparison, 'evaluate', evaluate)
    def forbidden(*args, **kwargs):
        pytest.fail('must reuse payloads, not rescore or reread originals')
    monkeypatch.setattr('ragwrench.evalgate.saved.evaluate', forbidden)
    monkeypatch.setattr('ragwrench.evalgate.formats.read_qrels', forbidden)
    monkeypatch.setattr('ragwrench.evalgate.formats.read_run', forbidden)
    if saved:
        for p in family[-1][:3]:
            p.unlink()
    assert main(argv(family, saved, '--expected-preflight', audit_path, '--format', 'json')) == 1
    assert capsys.readouterr().err == ''
    q, b, c, bs, cs = family[-1]
    assert Counter(calls) == (Counter({bs: 1, cs: 2}) if saved else Counter({q: 1, b: 1, c: 2}))
    assert len(score_calls) == (0 if saved else 3)


@pytest.mark.parametrize('saved', [False, True])
@pytest.mark.parametrize('kind', ['drift', 'malformed', 'unbound', 'duplicate', 'nan', 'null', 'encoding', 'missing', 'directory',
    'audit_output', 'audit_alias', 'late_candidate'])
def test_cli_rejects_before_inference_preserving_outputs(family, saved, kind, monkeypatch, capsys):
    audit = api(family, saved, preflight_only=True)
    folder = family[-1][0].parent
    path = folder / 'audit.json'
    path.write_text(json.dumps(audit))
    output = folder / 'protected.json'
    output.write_text('sentinel\n')
    extra = []
    if kind == 'drift':
        audit['settings']['seed'] += 1
        path.write_text(json.dumps(audit))
    elif kind in ('malformed', 'unbound', 'duplicate', 'nan'):
        path.write_text({'malformed': '{', 'unbound': '{}', 'duplicate': '{"x":1,"x":2}', 'nan': '{"x":NaN}'}[kind])
    elif kind == 'null':
        path.write_text('null')
    elif kind == 'encoding':
        path.write_bytes(b'\xff')
    elif kind == 'missing':
        path.unlink()
    elif kind == 'directory':
        path.unlink()
        path.mkdir()
    elif kind == 'audit_output':
        output = path
    elif kind == 'audit_alias':
        output.unlink()
        output.symlink_to(path)
    elif kind == 'late_candidate':
        bad = folder / 'bad'
        bad.write_text('{}' if saved else 'broken\n')
        extra = ['--candidate', f'z_bad={bad}']
    before = output.read_bytes()
    expected_before = path.read_bytes() if path.is_file() else None
    forbid_inference(monkeypatch)
    assert main(argv(family, saved, '--expected-preflight', path, '--output', output, '--force', *extra)) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'error:' in captured.err
    assert output.read_bytes() == before
    if expected_before is not None:
        assert path.read_bytes() == expected_before


@pytest.mark.parametrize('saved', [False, True])
def test_late_invalid_actual_member_api(family, saved, monkeypatch):
    q, b, _, m, artifacts, _ = family
    audit = api(family, saved, preflight_only=True)
    forbid_inference(monkeypatch)
    with pytest.raises(InputError):
        if saved:
            compare_many_saved(artifacts[0], {'a_valid': artifacts[0], 'z_bad': {}}, correction='holm',
                               expected_preflight=audit)
        else:
            compare_many(q, b, {'a_valid': b, 'z_bad': {}}, m, correction='holm', expected_preflight=audit)


def test_complex_direct_saved_order_equivalence(monkeypatch):
    from ragwrench.evalgate.metrics import parse_metrics
    q = {'q2': {'z': 2, 'a': 1}, 'q1': {'a': 1, 'z': 0}, 'missing': {'a': 1}}
    b = {'q1': {'a': 4, 'z': 4.0}, 'q2': {'z': 4, 'a': 3}, 'ignored': {'z': 1}}
    c = {'q2': {'a': 4.0, 'z': 4}}
    candidates = {'z': c, 'a': b}
    m = parse_metrics('ndcg@2,hit@1,recall@2')
    options = dict(correction='holm', method='ttest', preflight_only=True)
    audit = compare_many(q, b, candidates, m, **options)
    forbid_inference(monkeypatch)
    assert compare_many(reverse(q), reverse(b), reverse(candidates), list(reversed(m)),
                        expected_preflight=reverse(audit), **options) == audit
    artifacts = {name: saved_evaluation(reverse(q), reverse(run), list(reversed(m)))
                 for name, run in reversed(list(candidates.items()))}
    saved = compare_many_saved(saved_evaluation(q, b, m), artifacts, expected_preflight=audit, **options)
    assert {**saved, 'command': 'compare-many'} == audit


@pytest.mark.parametrize('change', ['names', 'metrics', 'queries', 'gain'])
def test_actual_family_structure_drift(family, change, monkeypatch):
    from ragwrench.evalgate.metrics import parse_metrics
    q, b, c, m, _, _ = copy.deepcopy(family)
    audit = api(family, False, preflight_only=True)
    candidates = {'strict': c, 'relaxed': c}
    options = dict(correction='holm', method='bootstrap', samples=100, thresholds={'hit@1': .49},
                   candidate_thresholds={'relaxed': {'hit@1': None}})
    if change == 'names':
        candidates['renamed'] = candidates.pop('strict')
    elif change == 'metrics':
        m = parse_metrics('hit@1,precision@1')
    elif change == 'queries':
        q['new'] = {'d': 1}
    else:
        options['gain'] = 'exponential'
    forbid_inference(monkeypatch)
    with pytest.raises(InputError, match='expected preflight'):
        compare_many(q, b, candidates, m, expected_preflight=audit, **options)
