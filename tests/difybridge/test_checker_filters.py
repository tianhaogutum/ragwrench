"""Filtered checker input, baseline assertions and local-error contracts."""
import json

import pytest

from ragwrench.difybridge import check
from ragwrench.difybridge.cli import main
from ragwrench.difybridge.errors import BridgeError, ConfigError
from .conftest import running
from .test_embedding_checker import fake_endpoint


FILTER = {'logical_operator': 'and', 'conditions': [
    {'name': 'lang', 'comparison_operator': 'is', 'value': 'en'}]}
ONE_HIT = {'conditions': [{'name': 'page', 'comparison_operator': '=', 'value': 7}]}
NO_HITS = {'conditions': [{'name': 'lang', 'comparison_operator': 'is', 'value': 'absent-filter-marker'}]}


@pytest.mark.parametrize('backend_fixture', ['json_app', 'qdrant_app'])
def test_filtered_baseline_on_both_backends(backend_fixture, request):
    with running(request.getfixturevalue(backend_fixture)) as endpoint:
        report = check.check_endpoint(endpoint, 'test-key', 'handbook', 'vacation policy',
            metadata_condition=ONE_HIT, top_k=2, score_threshold=0.2,
            expect_count=1, expect_titles=['leave.md'])
    # Mutated top_k / high-threshold probes must not inherit baseline assertions.
    assert report['passed'], report
    assert len(report['checks']) == 13


def forwarding_endpoint(app, seen, *, ignore_filter=False, corrupt_record=False):
    def reply(path, headers, raw):
        assert path == '/retrieval'
        try:
            body = json.loads(raw)
            seen.append(body)
            if ignore_filter and isinstance(body, dict):
                body.pop('metadata_condition', None)
                raw = json.dumps(body).encode()
        except ValueError:
            pass
        try:
            body = app.retrieve(headers.get('Authorization'), raw)
            if corrupt_record and body['records']:
                body['records'][0] = None
            return 200, {}, body
        except BridgeError as exc:
            return exc.status, {}, exc.body()
    return fake_endpoint(reply)


def test_cli_sends_exact_baseline_and_preserves_filter(json_app, monkeypatch, capsys):
    monkeypatch.setenv('FILTER_CHECK_KEY', 'test-key')
    seen = []
    with forwarding_endpoint(json_app, seen) as endpoint:
        code = main(['check', '--endpoint', endpoint, '--knowledge-id', 'handbook',
            '--query', 'vacation policy', '--key-env', 'FILTER_CHECK_KEY',
            '--metadata-condition', json.dumps(FILTER), '--top-k', '2', '--score-threshold', '0.2',
            '--expect-count', '2', '--expect-title', 'leave.md', '--expect-title', 'handbook.md'])
    output = capsys.readouterr()
    assert code == 0 and not output.err
    assert json.loads(output.out)['passed']
    assert seen[0] == {'knowledge_id': 'handbook', 'query': 'vacation policy',
        'retrieval_setting': {'top_k': 2, 'score_threshold': 0.2}, 'metadata_condition': FILTER}
    assert seen[1]['retrieval_setting'] == {'top_k': 1, 'score_threshold': 0.0}
    assert seen[2]['retrieval_setting'] == {'top_k': 3, 'score_threshold': 0.9}
    assert seen[3]['retrieval_setting'] == {'top_k': 3, 'score_threshold': 1.0}
    assert all(b['metadata_condition'] == FILTER for b in seen[:7])
    assert seen[-1]['metadata_condition']['conditions'][0]['comparison_operator'] == 'exec'
    assert all(secret not in output.out for secret in ['test-key', 'vacation policy', 'handbook.md', 'leave.md'])


def test_checker_catches_endpoint_ignoring_filter(json_app):
    # Baseline would have two hits without this filter; ignoring it must fail.
    with forwarding_endpoint(json_app, [], ignore_filter=True) as endpoint:
        report = check.check_endpoint(endpoint, 'test-key', 'handbook', 'vacation policy',
            metadata_condition=ONE_HIT, expect_count=1)
    assert not report['passed']
    assert 'baseline record count does not match --expect-count' in report['checks'][0]['errors']


@pytest.mark.parametrize('options,passed', [
    ({'metadata_condition': NO_HITS}, False),
    ({'metadata_condition': NO_HITS, 'expect_count': 0}, True),
    ({'metadata_condition': NO_HITS, 'allow_empty': True}, True),
    ({'metadata_condition': NO_HITS, 'allow_empty': True, 'expect_count': 1}, False),
    ({'metadata_condition': NO_HITS, 'allow_empty': True, 'expect_titles': ['secret-title-marker']}, False),
    ({'metadata_condition': ONE_HIT, 'expect_count': 0}, False),
    ({'metadata_condition': ONE_HIT, 'expect_titles': ['LEAVE.md']}, False),
    ({'metadata_condition': ONE_HIT, 'expect_titles': ['leave.md', 'leave.md']}, True),
    ({'expect_count': 2, 'expect_titles': ['handbook.md', 'leave.md']}, True),
])
def test_baseline_assertion_semantics(json_endpoint, options, passed):
    report = check.check_endpoint(json_endpoint, 'test-key', 'handbook', 'vacation policy', **options)
    assert report['passed'] is passed, report
    assert report['checks'][0]['passed'] is passed
    text = json.dumps(report)
    assert all(s not in text for s in ['test-key', 'secret-title-marker', 'absent-filter-marker', 'vacation policy'])


def test_malformed_records_fail_without_echo_or_crash(json_app):
    with forwarding_endpoint(json_app, [], corrupt_record=True) as endpoint:
        report = check.check_endpoint(endpoint, 'test-key', 'handbook', 'vacation policy',
            metadata_condition=FILTER, expect_count=2, expect_titles=['secret-title-marker'])
    assert not report['passed']
    assert 'record must be an object' in report['checks'][0]['errors']
    assert 'baseline is missing an --expect-title' in report['checks'][0]['errors']
    assert 'secret-title-marker' not in json.dumps(report)


def test_endpoint_allowlist_failure_is_conformance_failure(json_endpoint):
    report = check.check_endpoint(json_endpoint, 'test-key', 'handbook', 'vacation policy',
        metadata_condition={'conditions': [{'name': 'private', 'comparison_operator': 'is', 'value': 'secret-filter-marker'}]})
    assert not report['passed']
    assert 'valid retrieval must return HTTP 200' in report['checks'][0]['errors']
    assert 'secret-filter-marker' not in json.dumps(report)


@pytest.mark.parametrize('flags,passed', [
    (['--expect-count', '1'], False),
    (['--expect-title', 'secret-title-marker'], False),
    (['--expect-count', '0', '--metadata-condition', json.dumps(NO_HITS)], True),
    (['--expect-count', '0', '--allow-empty'], False),
])
def test_cli_assertion_exit_codes_and_redaction(json_endpoint, monkeypatch, capsys, flags, passed):
    monkeypatch.setenv('BRIDGE_API_KEY', 'test-key')
    code = main(['check', '--endpoint', json_endpoint, '--knowledge-id', 'handbook',
        '--query', 'vacation policy', *flags])
    output = capsys.readouterr()
    assert code == (0 if passed else 1) and not output.err
    report = json.loads(output.out)
    assert report['passed'] is passed
    assert all(s not in output.out for s in ['secret-title-marker', 'test-key', 'vacation policy', 'absent-filter-marker'])
    if passed:
        assert any('Empty baseline expected' in s for s in report['limitations'])


@pytest.mark.parametrize('raw', [
    'null', '[]', '{}', '{', '{"conditions":[],"conditions":[]}',
    '{"conditions":[{"name":"lang","comparison_operator":"exec","value":"secret-filter-marker"}]}',
    '{"conditions":[{"name":"page","comparison_operator":"=","value":true}]}',
    '{"conditions":[{"name":"page","comparison_operator":"=","value":NaN}]}',
    '{"conditions":[{"name":"page","comparison_operator":"=","value":1e999}]}',
    '{"conditions":[{"name":"lang","comparison_operator":"in","value":[1]}]}',
    '{"conditions":[{"name":"updated","comparison_operator":"before","value":"secret-filter-marker"}]}',
    '{"logical_operator":[],"conditions":[]}',
    '{"conditions":[{"conditions":[]}]}',
    json.dumps({'conditions': [ONE_HIT['conditions'][0]] * 65}),
    '[' * 1500 + ']' * 1500,
])
def test_invalid_json_filter_is_local_error(raw, monkeypatch, capsys):
    monkeypatch.setenv('BRIDGE_API_KEY', 'secret-key-marker')
    seen = []
    monkeypatch.setattr(check, 'request', lambda *a, **k: seen.append(a))
    assert main(['check', '--endpoint', 'http://127.0.0.1:1', '--knowledge-id', 'handbook',
        '--query', 'secret-query-marker', '--metadata-condition', raw]) == 2
    output = capsys.readouterr()
    assert output.out == '' and 'Configuration' in output.err
    assert all(s not in output.err for s in ['secret-filter-marker', 'secret-query-marker', 'secret-key-marker'])
    assert not seen


@pytest.mark.parametrize('options', [
    {'top_k': 0}, {'top_k': 101}, {'top_k': True},
    {'score_threshold': -0.1}, {'score_threshold': 1.1}, {'score_threshold': float('nan')},
    {'expect_count': -1}, {'expect_count': True}, {'expect_count': 4},
    {'expect_count': 0, 'expect_titles': ['x']}, {'top_k': 1, 'expect_titles': ['x', 'y']},
    {'expect_titles': 'x'}, {'expect_titles': [None]},
    {'query': ' '}, {'query': 'x' * 16385}, {'knowledge_id': ''},
    # Each string is within filter limits, but the complete request exceeds 1 MiB.
    {'metadata_condition': {'conditions': [{'name': 'lang', 'comparison_operator': 'in',
        'value': ['x' * 4096] * 256}] * 2}},
])
def test_invalid_options_do_not_send_probes(options, monkeypatch):
    seen = []
    monkeypatch.setattr(check, 'request', lambda *a, **k: seen.append(a))
    args = {'knowledge_id': 'handbook', 'query': 'vacation policy', **options}
    with pytest.raises(ConfigError):
        check.check_endpoint('http://127.0.0.1:1', 'test-key', **args)
    assert not seen


@pytest.mark.parametrize('flags', [
    ['--top-k', '0'], ['--score-threshold', 'nan'], ['--expect-count', '-1'],
    ['--expect-count', '0', '--expect-title', 'secret-title-marker'],
])
def test_cli_invalid_settings_exit_two(flags, monkeypatch, capsys):
    monkeypatch.setenv('BRIDGE_API_KEY', 'test-key')
    seen = []
    monkeypatch.setattr(check, 'request', lambda *a, **k: seen.append(a))
    assert main(['check', '--endpoint', 'http://127.0.0.1:1', '--knowledge-id', 'handbook',
        '--query', 'vacation policy', *flags]) == 2
    output = capsys.readouterr()
    assert not output.out and 'Configuration' in output.err
    assert 'secret-title-marker' not in output.err
    assert not seen
