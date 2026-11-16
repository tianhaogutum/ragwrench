"""ASCII scheme casing only; bare URL grammar and token precedence stay fixed."""
import itertools
import json

import pytest

from ragwrench.citecheck.adapters import adapt
from ragwrench.citecheck.checker import check, fails
from ragwrench.citecheck.markers import parse_markers
from ragwrench.citecheck.prose import mask_markdown
from .test_cli_io import ROOT, cli


SCHEMES = [''.join(chars) for scheme in ['http', 'https']
           for chars in itertools.product(*[(c, c.upper()) for c in scheme])]


@pytest.mark.parametrize('scheme', SCHEMES)
def test_all_ascii_scheme_cases_hide_the_complete_url(scheme):
    url = scheme + '://example.invalid/[9]'
    answer = '说明🙂 ' + url
    assert mask_markdown(answer) == '说明🙂 ' + ' ' * len(url)
    report = check(adapt({'answer': answer, 'sources': []}))
    assert report['findings'] == []
    assert all(not fails([report], gate) for gate in ['info', 'hint', 'warning', 'error', 'none'])
    sentence, = report['sentences']
    assert (sentence['start'], sentence['end'], sentence['text']) == (0, len(answer), answer)
    assert sentence['citation_indices'] == []


@pytest.mark.parametrize('scheme', ['http', 'HTTPS', 'hTtPs'])
@pytest.mark.parametrize('separator', [' ', '\t', '\r', '\n', '\r\n', '\u2003', '\u00a0', '>', '<em>'])
def test_url_boundary_keeps_outside_citation_and_offsets(scheme, separator):
    url = scheme + '://example.invalid/`[8]'
    answer = '说明🙂 ' + url + separator + '[9]`'
    boundary = ' ' * len(separator) if separator == '<em>' else separator
    assert mask_markdown(answer) == '说明🙂 ' + ' ' * len(url) + boundary + '[9]`'
    report = check(adapt({'answer': answer, 'sources': []}))
    finding, = report['findings']
    assert finding['code'] == 'CC001'
    start = answer.index('[9]')
    assert finding['evidence'] == {'marker': '[9]', 'start': start, 'end': start + 3, 'kind': 'dangling'}
    assert fails([report], 'error') and not fails([report], 'none')
    assert [n for s in report['sentences'] for n in s['citation_indices']] == [9]


@pytest.mark.parametrize('body', [
    'ftp://host/[9]', 'ws://host/[9]', 'mailto:host/[9]',
    'htps://host/[9]', 'httpx://host/[9]', 'httpss://host/[9]',
    'һttp://host/[9]', 'hτtp://host/[9]', 'ＨＴＴＰＳ://host/[9]',
    'HTTP//host/[9]', 'HTTPS:/host/[9]', 'HTTPS:host/[9]',
    'HTTP:// [9]', 'HTTPS://>[9]', 'HTTPS://',
])
def test_unsupported_schemes_lookalikes_and_missing_body_stay_prose(body):
    answer = '说明🙂 ' + body
    assert mask_markdown(answer) == answer
    assert [n for m in parse_markers(answer) for n in m.indices] == ([9] if '[9]' in body else [])


@pytest.mark.parametrize('body,expected', [
    # Punctuation/brackets/quotes/backticks/parentheses are still absorbed.
    ('HTTPS://host/[9]),.!?`"quoted"', ' ' * len('HTTPS://host/[9]),.!?`"quoted"')),
    # No new left boundary or URL validation.
    ('xHTTPS:///[9]', 'x' + ' ' * len('HTTPS:///[9]')),
    ('`HTTPS://host/`[9]', ' ' * len('`HTTPS://host/`') + '[9]'),
    ('``HtTp://host/` [8]``[9]', ' ' * len('``HtTp://host/` [8]``') + '[9]'),
    (r'\`HTTPS://host/` [9]`', '\\`' + ' ' * len('HTTPS://host/`') + ' [9]`'),
    (r'\\`HTTPS://host/ [8]`[9]', '\\\\' + ' ' * len('`HTTPS://host/ [8]`') + '[9]'),
    ('<span title="HTTPS://host/`">[9]`', ' ' * len('<span title="HTTPS://host/`">') + '[9]`'),
    ('<!-- HTTPS://host/ > ` [8] -->[9]`', ' ' * len('<!-- HTTPS://host/ > ` [8] -->') + '[9]`'),
    ('<?note HTTPS://host/ > ` [8] ?>[9]`', ' ' * len('<?note HTTPS://host/ > ` [8] ?>') + '[9]`'),
    ('<![CDATA[ HTTPS://host/ > ` [8] ]]>[9]`', ' ' * len('<![CDATA[ HTTPS://host/ > ` [8] ]]>') + '[9]`'),
    ('[archive](HTTPS://host/part(one)`[8])[9]`', ' archive' + ' ' * len('](HTTPS://host/part(one)`[8])') + '[9]`'),
    # Link labels are not nested-bracket aware (same as lowercase http(s)), so the label uses no inner brackets.
    ('[archive HTTPS://host/` 8](part(one))[9]`', ' archive' + ' ' * len(' HTTPS://host/`') + ' 8' + ' ' * len('](part(one))') + '[9]`'),
    ('HTTPS://host/`<em>[9]`', ' ' * len('HTTPS://host/`<em>') + '[9]`'),
    ('HTTPS://host/`<!-- > [8] -->[9]`', ' ' * len('HTTPS://host/`<!-- > [8] -->') + '[9]`'),
    ('<HTTPS://host/[8]>[9]', ' ' * len('<HTTPS://host/[8]>') + '[9]'),
    ('<+ HTTPS://host/[8]>[9]', '<+ ' + ' ' * len('HTTPS://host/[8]') + '>[9]'),
    ('</3 HtTp://host/[8]>[9]', '</3 ' + ' ' * len('HtTp://host/[8]') + '>[9]'),
    # Other alternatives/marker patterns do not gain case-insensitivity.
    ('<![cdata[ > [9] ]]>', ' ' * len('<![cdata[ >') + ' [9] ]]>'),
    ('[id:0]', '[id:0]'),
    ('SOURCE 9', 'SOURCE 9'),
])
def test_case_change_preserves_inline_source_order_and_other_patterns(body, expected):
    assert mask_markdown('说明🙂 ' + body) == '说明🙂 ' + expected
    if body == '[id:0]':
        assert parse_markers(body)[0].error is not None
    if body == 'SOURCE 9':
        assert parse_markers(body) == []


def test_uppercase_url_does_not_contaminate_source_matching_or_gates():
    answer = 'Archive "stable signal" HtTpS://host/[9]999kg"absent"` [4] manual'
    report = check(adapt({'answer': answer, 'sources': [
        {'idx': 4, 'text': 'Archive stable signal manual'}]}))
    assert report['findings'] == []
    assert all(not fails([report], gate) for gate in ['info', 'hint', 'warning', 'error', 'none'])
    sentence, = report['sentences']
    assert sentence['citation_indices'] == [4] and sentence['cited_coverage'] == 1
    assert sentence['text'] == answer


def test_exact_uppercase_url_cli_contract():
    answer = '说明🙂 HTTPS://example.invalid/[9]'
    result = cli('check', '-', '--report', 'json', stdin=json.dumps({'answer': answer, 'sources': []}))
    assert result.returncode == 0 and not result.stderr
    output = json.loads(result.stdout)
    assert output['fail_on'] == 'error' and not output['gate_failed']
    record, = output['records']
    assert record['findings'] == []
    sentence, = record['sentences']
    assert (sentence['start'], sentence['end'], sentence['text']) == (0, 31, answer)
    assert sentence['citation_indices'] == []


def test_url_scheme_cli_fixture():
    fixture = ROOT / 'examples/cite-check/url_schemes.jsonl'
    result = cli('check', fixture, '--jsonl', '--report', 'json')
    assert result.returncode == 1 and not result.stderr
    output = json.loads(result.stdout)
    assert output['gate_failed'] and output['record_count'] == 7
    answers = [json.loads(line)['answer'] for line in fixture.read_text().splitlines()]
    for i, (answer, record) in enumerate(zip(answers, output['records'], strict=True)):
        if i in [2, 3, 4]:
            finding, = record['findings']
            start = answer.index('[9]')
            assert finding['code'] == 'CC001'
            assert finding['evidence'] == {'marker': '[9]', 'start': start, 'end': start + 3, 'kind': 'dangling'}
        else:
            assert record['findings'] == []
