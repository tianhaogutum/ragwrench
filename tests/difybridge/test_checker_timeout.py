"""Public checker timeout preflight and unchanged transport/report contracts."""
import io
import json
from decimal import Decimal
from email.message import Message
from fractions import Fraction
from urllib.request import OpenerDirector
from urllib.response import addinfourl

import pytest

from ragwrench.difybridge import check
from ragwrench.difybridge.cli import main
from ragwrench.difybridge.errors import BridgeError, ConfigError
from .test_checker_credentials import arguments, forwarding
from .test_checker_request_file import BASE, saved
from .test_embedding_checker import fake_endpoint


class PrivateValue:
    def __float__(self):
        raise AssertionError('private-timeout-marker')

    def __str__(self):
        raise AssertionError('private-timeout-marker')


class IntSubclass(int):
    pass


class FloatSubclass(float):
    pass


def invoke(mode, endpoint, tmp_path, **options):
    if mode == 'file':
        return check.check_request_file(endpoint, 'test-key', saved(tmp_path), **options)
    return check.check_endpoint(endpoint, 'test-key', BASE['knowledge_id'], BASE['query'],
                                metadata_condition=BASE['metadata_condition'],
                                score_threshold=0.2, **options)


INVALID = [0, -0.0, -1, -0.25, 121, 120.00000000000001, 10**1000,
           float('nan'), float('inf'), -float('inf'), True, False, None,
           '0.25', 'private-timeout-marker', b'0.25', [], {}, (), object(),
           PrivateValue(), Decimal('0.25'), Fraction(1, 4), 0.25+0j,
           IntSubclass(1), FloatSubclass(0.25)]


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('timeout', INVALID, ids=[
    'zero', 'negative-zero', 'negative-int', 'negative-float', 'over-limit',
    'next-float-above-limit', 'huge-int', 'nan', 'positive-inf', 'negative-inf',
    'true', 'false', 'none', 'numeric-string', 'private-string', 'bytes', 'list',
    'dict', 'tuple', 'object', 'conversion-object', 'decimal', 'fraction',
    'complex', 'int-subclass', 'float-subclass'])
def test_invalid_library_timeout_is_local(mode, timeout, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(check, 'request', lambda *a, **kw: calls.append((a, kw)))
    with pytest.raises(ConfigError) as exc:
        invoke(mode, 'http://127.0.0.1:1', tmp_path, timeout=timeout)
    assert not calls  # Includes missing/wrong/malformed authorization probes.
    assert str(exc.value) == 'Checker timeout must be a finite int or float in (0,120]'
    assert 'private-timeout-marker' not in str(exc.value)


@pytest.mark.parametrize('mode', ['inline', 'file'])
def test_over_limit_sends_no_http(mode, tmp_path, json_app):
    seen = []
    with fake_endpoint(forwarding(json_app, seen)) as endpoint:
        with pytest.raises(ConfigError):
            invoke(mode, endpoint, tmp_path, timeout=121)
    assert not seen


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('options,expected', [({}, 10), ({'timeout': 0.25}, 0.25),
    ({'timeout': 1}, 1), ({'timeout': 120}, 120), ({'timeout': 120.0}, 120.0),
    ({'timeout': 1e-200}, 1e-200)])
def test_valid_timeout_reaches_every_http_open(mode, options, expected, tmp_path, json_app, monkeypatch):
    calls = []
    # Preserve real Request construction and the strict response reader.
    def open_request(self, req, *, timeout):
        calls.append((req, timeout))
        try:
            status, body = 200, json_app.retrieve(req.get_header('Authorization'), req.data)
        except BridgeError as exc:
            status, body = exc.status, exc.body()
        raw = json.dumps(body).encode()
        headers = Message()
        headers['Content-Type'] = 'application/json'
        headers['Content-Length'] = str(len(raw))
        return addinfourl(io.BytesIO(raw), headers, req.full_url, status)
    monkeypatch.setattr(OpenerDirector, 'open', open_request)
    report = invoke(mode, 'http://example.test/base/', tmp_path,
                    expect_count=2, expect_titles=['handbook.md', 'leave.md'], **options)
    assert report['passed'] and len(report['checks']) == len(calls) == 13
    assert all(t == expected and type(t) is type(expected) for _, t in calls)
    assert all(req.full_url == 'http://example.test/base/retrieval' for req, _ in calls)
    assert json.loads(calls[0][0].data) == BASE
    if mode == 'file':
        assert calls[0][0].data == (tmp_path / 'secret-path-marker.json').read_bytes()
    assert calls[0][0].get_header('Authorization') == 'Bearer test-key'
    assert calls[4][0].get_header('Authorization') is None
    assert calls[6][0].get_header('Authorization') == 'Basic invalid'
    assert all(s not in json.dumps(report) for s in
               ['test-key', BASE['query'], 'handbook.md', 'leave.md', str(tmp_path)])


@pytest.mark.parametrize('mode', ['inline', 'file'])
@pytest.mark.parametrize('timeout', ['0', '-1', '121', 'nan', 'inf', '-inf', 'private-timeout-marker'])
def test_cli_invalid_timeout_keeps_exit_two(mode, timeout, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('BRIDGE_API_KEY', 'test-key')
    calls = []
    monkeypatch.setattr(check, 'request', lambda *a, **kw: calls.append(a))
    try:
        code = main(['check', '--endpoint', 'http://127.0.0.1:1',
                     *arguments(mode, tmp_path), '--timeout=' + timeout])
    except SystemExit as exc:
        code = exc.code
    assert code == 2 and not calls
    output = capsys.readouterr()
    assert not output.out and 'Configuration' in output.err
    assert all(s not in output.err for s in ['private-timeout-marker', 'test-key', BASE['query'], str(tmp_path)])
