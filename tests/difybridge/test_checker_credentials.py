"""Checker credential preflight: local failures must never send even auth probes."""
import json

import pytest

from ragwrench.difybridge import check
from ragwrench.difybridge.cli import main
from ragwrench.difybridge.errors import BridgeError, ConfigError
from .test_checker_request_file import BASE, saved
from .test_embedding_checker import fake_endpoint


def arguments(mode, tmp_path):
    if mode == 'file':
        return ['--request-file', str(saved(tmp_path))]
    return ['--knowledge-id', BASE['knowledge_id'], '--query', BASE['query'],
            '--metadata-condition', json.dumps(BASE['metadata_condition']),
            '--top-k', '3', '--score-threshold', '0.2']


def forwarding(app, seen):
    def reply(route, headers, raw):
        seen.append((route, raw))
        try:
            return 200, {}, app.retrieve(headers.get('Authorization'), raw)
        except BridgeError as exc:
            return exc.status, {}, exc.body()
    return reply


@pytest.mark.parametrize('mode', ['inline', 'file'])
def test_newline_key_never_contacts_endpoint(mode, tmp_path, json_app, monkeypatch, capsys):
    monkeypatch.setenv('CHECK_KEY', 'synthetic-private\nkey')
    seen = []
    with fake_endpoint(forwarding(json_app, seen)) as endpoint:
        code = main(['check', '--endpoint', endpoint, '--key-env', 'CHECK_KEY',
                     *arguments(mode, tmp_path)])
    assert (code, len(seen)) == (2, 0)
    output = capsys.readouterr()
    assert not output.out and 'Configuration' in output.err
    assert 'synthetic-private' not in output.err


INVALID_KEYS = [None, '', 'synthetic-private\nkey', 'synthetic-private\rkey',
                'synthetic-private key', 'synthetic-private\tkey', '\x1f',
                '\x7f', 'é', 'x' * 4097]


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('key', INVALID_KEYS, ids=[
    'missing', 'empty', 'lf', 'cr', 'space', 'tab', 'control', 'del', 'unicode', 'oversized'])
def test_invalid_environment_keys_are_local(mode, key, tmp_path, monkeypatch, capsys):
    # A valid default cannot rescue an invalid selected variable.
    monkeypatch.setenv('BRIDGE_API_KEY', 'test-key')
    if key is None:
        monkeypatch.delenv('CHECK_KEY', raising=False)
    else:
        monkeypatch.setenv('CHECK_KEY', key)
    calls = []
    monkeypatch.setattr(check, 'request', lambda *a, **kw: calls.append(a))
    code = main(['check', '--endpoint', 'http://127.0.0.1:1', '--key-env', 'CHECK_KEY',
                 *arguments(mode, tmp_path)])
    assert (code, len(calls)) == (2, 0)
    output = capsys.readouterr()
    assert not output.out and 'Configuration' in output.err
    assert all(s not in output.err for s in ['synthetic-private', 'test-key', 'CHECK_KEY',
                                            str(tmp_path), BASE['query']])


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('name', ['', '1PRIVATE', 'PRIVATE-NAME', 'PRIVATE.NAME',
                                  'PRIVATE NAME', 'PRIVATE\nNAME', 'éPRIVATE', 'PRIVATE=NAME'])
def test_invalid_environment_names_are_local(mode, name, tmp_path, monkeypatch, capsys):
    if name and '=' not in name:
        # Python permits several names that are outside the CLI's contract.
        monkeypatch.setenv(name, 'test-key')
    monkeypatch.setenv('BRIDGE_API_KEY', 'test-key')
    calls = []
    monkeypatch.setattr(check, 'request', lambda *a, **kw: calls.append(a))
    assert main(['check', '--endpoint', 'http://127.0.0.1:1', '--key-env', name,
                 *arguments(mode, tmp_path)]) == 2
    assert not calls
    output = capsys.readouterr()
    assert not output.out and 'Configuration' in output.err
    assert 'PRIVATE' not in output.err and 'test-key' not in output.err


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('key', INVALID_KEYS + ['\x00', True, 123, b'key'], ids=[
    'missing', 'empty', 'lf', 'cr', 'space', 'tab', 'control', 'del', 'unicode',
    'oversized', 'nul', 'bool', 'int', 'bytes'])
def test_library_invalid_keys_never_probe(mode, key, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(check, 'request', lambda *a, **kw: calls.append(a))
    with pytest.raises(ConfigError) as exc:
        if mode == 'file':
            check.check_request_file('http://127.0.0.1:1', key, saved(tmp_path))
        else:
            check.check_endpoint('http://127.0.0.1:1', key, 'handbook', BASE['query'])
    assert not calls and 'synthetic-private' not in str(exc.value)


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('key,expected', [('x', 0), ('!~"\\:token', 0), ('x' * 4096, 0),
                                          ('valid-but-wrong', 1)],
                         ids=['min', 'visible-punctuation', 'max', 'wrong'])
def test_named_key_bounds_and_wrong_key_remain_remote(mode, key, expected, tmp_path,
                                                     json_app, monkeypatch, capsys):
    json_app.api_key = (key if expected == 0 else 'test-key').encode('ascii')
    monkeypatch.setenv('_CHECK_KEY_8', key)
    # Valid wrong named key must not fall back to the correct default either.
    monkeypatch.setenv('BRIDGE_API_KEY', 'test-key')
    seen = []
    with fake_endpoint(forwarding(json_app, seen)) as endpoint:
        code = main(['check', '--endpoint', endpoint, '--key-env', '_CHECK_KEY_8',
                     *arguments(mode, tmp_path), '--expect-count', '2',
                     '--expect-title', 'handbook.md', '--expect-title', 'leave.md'])
    assert (code, len(seen)) == (expected, 13)
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert not output.err and report['passed'] is (expected == 0)
    assert len(report['checks']) == 13
    assert all(route == '/retrieval' for route, _ in seen)
    assert json.loads(seen[0][1]) == BASE
    if mode == 'file':
        assert seen[0][1] == (tmp_path / 'secret-path-marker.json').read_bytes()
    assert 'test-key' not in output.out and 'valid-but-wrong' not in output.out
    assert BASE['query'] not in output.out
