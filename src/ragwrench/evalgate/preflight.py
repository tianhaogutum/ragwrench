"""Validate and diff complete bound family audits without retrieval inputs."""
from __future__ import annotations

from collections.abc import Mapping
import math
import re

from . import __version__
from .compare import _family_limits, _family_policy, validate_candidate_names
from .formats import InputError
from .metrics import TIE_RULE, parse_metrics
from .saved import QUERY_POLICY
from .statistics import validate_paired_options


_INTEGER_FIELDS = {'schema_version', 'query_count', 'family_size', 'estimated_family_resamples',
                   'seed', 'samples', 'max_family_pairs', 'max_family_resamples'}


def _require(condition, path, message):
    if not condition:
        raise InputError(f'invalid audit at {path or "/"}: {message}')


def _pointer(path, key):
    return path + '/' + key.replace('~', '~0').replace('/', '~1')


def _object(value, keys, path):
    _require(isinstance(value, Mapping), path, 'must be an object')
    _require(all(isinstance(k, str) for k in value), path, 'object keys must be strings')
    _require(set(value) == set(keys), path, f'fields must be {", ".join(sorted(keys))}')


def _integer(value, path, minimum=None, maximum=None):
    _require(type(value) is int, path, 'must be an integer (not boolean)')
    _require((minimum is None or value >= minimum) and (maximum is None or value <= maximum),
             path, f'integer out of range [{minimum}, {maximum}]')


def _number(value, path, *, probability=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    _require(valid, path, 'must be a finite number (not boolean)')
    _require(0 < value < 1 if probability else value >= 0, path,
             'must be strictly between 0 and 1' if probability else 'must be nonnegative')


def _projection(value, expected, path=''):
    """Check a derived schema/value, preserving integer and boolean types."""
    if isinstance(expected, dict):
        _object(value, expected, path)
        for key, child in expected.items():
            _projection(value[key], child, _pointer(path, key))
    elif isinstance(expected, list):
        _require(isinstance(value, list) and value == expected, path, 'must equal canonical name array')
    elif type(expected) in (int, float):
        if path.rsplit('/', 1)[-1] in _INTEGER_FIELDS:
            _integer(value, path)
        else:
            _number(value, path)
        _require(value == expected, path, f'must equal {expected}')
    else:
        _require(type(value) is type(expected) and value == expected, path, f'must equal {expected!r}')


def _thresholds(value, metrics, path, *, allow_off=False):
    _require(isinstance(value, Mapping), path, 'must be a metric object')
    _require(set(value) <= set(metrics), path, 'keys must be audited metric names')
    for key, number in value.items():
        if not (allow_off and number is None):
            _number(number, _pointer(path, key))


def validate_preflight(audit):
    """Validate one complete current schema-1 bound audit; return None.

    Objects may be Mapping instances; name arrays must be canonical lists.
    This validates structural consistency and digest syntax, not digest provenance:
    the original inputs and full artifact payloads are intentionally unavailable.
    """
    _object(audit, ('schema_version', 'command', 'mode', 'query_count', 'candidate_names', 'metric_names',
                    'family_size', 'estimated_family_resamples', 'limits', 'settings', 'policy', 'input_bindings'), '')
    _projection(audit['schema_version'], 1, '/schema_version')
    _require(isinstance(audit['command'], str) and audit['command'] in ('compare-many', 'compare-many-saved'),
             '/command', 'must be compare-many or compare-many-saved')
    _projection(audit['mode'], 'preflight-only', '/mode')
    _integer(audit['query_count'], '/query_count', 2)
    names, metrics = audit['candidate_names'], audit['metric_names']
    for key, values in (('candidate_names', names), ('metric_names', metrics)):
        _require(isinstance(values, list) and bool(values) and all(isinstance(v, str) for v in values),
                 '/' + key, 'must be a nonempty string array')
        _require(values == sorted(set(values)), '/' + key, 'must be sorted and unique')
    try:
        validate_candidate_names(names)
        parsed = parse_metrics(','.join(metrics))
        _require([str(m) for m in parsed] == metrics, '/metric_names', 'must use canonical metric names')
    except InputError as exc:
        raise InputError(f'invalid audit names: {exc}') from None

    settings, limits, policy = audit['settings'], audit['limits'], audit['policy']
    _object(settings, ('test', 'seed', 'samples', 'confidence'), '/settings')
    _require(isinstance(settings['test'], str) and settings['test'] in ('bootstrap', 'randomization', 'ttest'),
             '/settings/test', 'unknown statistical test')
    _integer(settings['seed'], '/settings/seed')
    _integer(settings['samples'], '/settings/samples', 100, 1_000_000)
    _number(settings['confidence'], '/settings/confidence', probability=True)
    try:
        validate_paired_options(audit['query_count'], settings['test'], samples=settings['samples'],
                                confidence=settings['confidence'])
    except InputError as exc:
        raise InputError(f'invalid audit at /settings: {exc}') from None
    _object(limits, ('max_family_pairs', 'max_family_resamples'), '/limits')
    for key, value in limits.items():
        if value is not None:
            _integer(value, '/limits/' + key, 1)

    policy_keys = {'correction', 'alpha', 'family', 'family_size', 'ci_adjustment', 'gate',
                   'significance_affects_gate', 'thresholds', 'candidate_thresholds',
                   'threshold_precedence', 'effective_thresholds'}
    if limits['max_family_pairs'] is not None:
        policy_keys.add('max_family_pairs')
    if limits['max_family_resamples'] is not None:
        policy_keys.update(('max_family_resamples', 'estimated_family_resamples'))
    _object(policy, policy_keys, '/policy')
    _require(isinstance(policy['correction'], str) and policy['correction'] in ('none', 'bonferroni', 'holm'),
             '/policy/correction', 'unknown correction')
    _number(policy['alpha'], '/policy/alpha', probability=True)
    _thresholds(policy['thresholds'], metrics, '/policy/thresholds')
    overrides = policy['candidate_thresholds']
    _require(isinstance(overrides, Mapping) and set(overrides) <= set(names),
             '/policy/candidate_thresholds', 'must map audited candidate names to metric objects')
    for key, values in overrides.items():
        _thresholds(values, metrics, _pointer('/policy/candidate_thresholds', key), allow_off=True)
    derived = _family_policy(names, parsed, policy['correction'], policy['alpha'],
                             policy['thresholds'], overrides, 0)
    try:
        derived = _family_limits(audit['query_count'], derived, settings['test'], settings['samples'],
                                 limits['max_family_resamples'], limits['max_family_pairs'])
    except InputError as exc:
        raise InputError(f'invalid audit at /limits: {exc}') from None
    _projection(policy, derived, '/policy')
    _projection(audit['family_size'], len(names) * len(metrics), '/family_size')
    work = 0 if settings['test'] == 'ttest' else audit['query_count'] * settings['samples'] * len(names) * len(metrics)
    _projection(audit['estimated_family_resamples'], work, '/estimated_family_resamples')

    bindings = audit['input_bindings']
    _object(bindings, ('schema_version', 'evaluator', 'dataset_sha256', 'parameters', 'baseline', 'candidates'),
            '/input_bindings')
    _projection(bindings['schema_version'], 1, '/input_bindings/schema_version')
    _projection(bindings['evaluator'], {'name': 'retgate', 'version': __version__}, '/input_bindings/evaluator')
    parameters = bindings['parameters']
    _object(parameters, ('metrics', 'gain', 'tie_rule', 'query_policy'), '/input_bindings/parameters')
    _require(isinstance(parameters['gain'], str) and parameters['gain'] in ('linear', 'exponential'),
             '/input_bindings/parameters/gain', 'unknown gain')
    _projection(parameters, {'metrics': metrics, 'gain': parameters['gain'], 'tie_rule': TIE_RULE,
                             'query_policy': QUERY_POLICY}, '/input_bindings/parameters')
    _object(bindings['candidates'], names, '/input_bindings/candidates')
    digests = [(bindings['dataset_sha256'], '/input_bindings/dataset_sha256')]
    for path, binding in [('/input_bindings/baseline', bindings['baseline']), *(
            (_pointer('/input_bindings/candidates', name), bindings['candidates'][name]) for name in names)]:
        _object(binding, ('artifact_sha256', 'run_sha256'), path)
        digests.extend((value, _pointer(path, key)) for key, value in binding.items())
    for value, path in digests:
        _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
                 path, 'must be a lowercase SHA-256 hex digest')


def _json_value(value):
    """Own output subtrees, with stable key ordering even for arbitrary Mappings."""
    if isinstance(value, Mapping):
        return {k: _json_value(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_json_value(v) for v in value]
    return value


def diff_preflight(expected, actual):
    """Return sorted JSON Pointer changes; independently reject either bad audit.

    Lists are atomic. Added/removed object keys carry their whole subtree.
    Presence flags distinguish absent entries from explicit null opt-outs.
    Only the two root family commands normalize; numeric values compare exactly.
    """
    for label, value in (('expected', expected), ('actual', actual)):
        try:
            validate_preflight(value)
        except (InputError, RecursionError) as exc:
            raise InputError(f'{label} audit: {exc}') from None
    changes = []

    def walk(old, new, path=''):
        if path == '/command':
            return
        if isinstance(old, Mapping) and isinstance(new, Mapping):
            for key in sorted(set(old) | set(new)):
                child = _pointer(path, key)
                ep, ap = key in old, key in new
                if ep and ap:
                    walk(old[key], new[key], child)
                else:
                    changes.append({'path': child, 'expected_present': ep, 'actual_present': ap,
                                    'expected': _json_value(old[key]) if ep else None,
                                    'actual': _json_value(new[key]) if ap else None})
        elif old != new:
            changes.append({'path': path, 'expected_present': True, 'actual_present': True,
                            'expected': _json_value(old), 'actual': _json_value(new)})

    walk(expected, actual)
    changes.sort(key=lambda row: row['path'])
    return {'schema_version': 1, 'command': 'diff-preflight', 'matches': not changes, 'changes': changes}
