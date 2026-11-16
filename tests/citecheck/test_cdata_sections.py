"""Same-line CDATA terminators, opaque bodies, fallback and source-order contracts."""
import json

import pytest

from ragwrench.citecheck.adapters import adapt
from ragwrench.citecheck.checker import check, fails
from ragwrench.citecheck.prose import mask_markdown
from .test_cli_io import ROOT, cli


@pytest.mark.parametrize('cdata', [
    '<![CDATA[ > [9] ]]>', '<![CDATA[ [9] ]]>', '<![CDATA[]]>', '<![CDATA[> [9] ]]>',
    '<![CDATA[ > "unclosed [9] ]]>', "<![CDATA[ > 'unclosed [9] ]]>",
    '<![CDATA[ > ` [9] ]]>', '<![CDATA[ > `` [9] ]]>',
    '<![CDATA[ > [9] ? > more ]]>', '<![CDATA[ > <![CDATA[nested [9] ]]>',
    '<![CDATA[+ > [bad] 999kg "absent quotation" ]]>',
])
def test_complete_cdata_hides_body_without_body_validation(cdata):
    answer = '说明🙂 ' + cdata
    assert mask_markdown(answer) == '说明🙂 ' + ' ' * len(cdata)
    report = check(adapt({'answer': answer, 'sources': []}))
    assert report['findings'] == []
    assert all(not fails([report], level) for level in ['info', 'hint', 'warning', 'error', 'none'])
    sentence, = report['sentences']
    assert (sentence['start'], sentence['end'], sentence['text']) == (0, len(answer), answer)
    assert sentence['citation_indices'] == []


@pytest.mark.parametrize('cdata', [
    '<![CDATA[]]>', '<![CDATA[ > [8] ]]>', '<![CDATA[ > "` [8] ]]>', '<![CDATA[ > `` [8] ]]>',
    '<![CDATA[ <![CDATA[nested > [8] ]]>', '<![CDATA[ > [8] ]]> <![CDATA[next > [7] ]]>',
])
def test_cdata_backticks_do_not_hide_outside_citation(cdata):
    answer = '说明🙂 ' + cdata + '[9]`'
    assert mask_markdown(answer) == '说明🙂 ' + ' ' * len(cdata) + '[9]`'
    report = check(adapt({'answer': answer, 'sources': []}))
    sentence, = report['sentences']
    assert (sentence['start'], sentence['end'], sentence['text']) == (0, len(answer), answer)
    assert sentence['citation_indices'] == [9]
    finding, = report['findings']
    assert finding['code'] == 'CC001' and fails([report], 'error')
    assert not fails([report], 'none')
    assert finding['evidence'] == {'marker': '[9]', 'start': 4 + len(cdata),
                                   'end': 7 + len(cdata), 'kind': 'dangling'}


@pytest.mark.parametrize('body,expected', [
    # First ]]> wins, even in quotes or after another opener. No nesting.
    ('<![CDATA[]]>[9]', ' ' * len('<![CDATA[]]>') + '[9]'),
    ('<![CDATA[ " ]]>[9]" ]]>', ' ' * len('<![CDATA[ " ]]>') + '[9]" ]]>'),
    ('<![CDATA[ ` ]]>[9]` ]]>', ' ' * len('<![CDATA[ ` ]]>') + '[9]` ]]>'),
    ('<![CDATA[ <![CDATA[nested ]]>[9] ]]>', ' ' * len('<![CDATA[ <![CDATA[nested ]]>') + '[9] ]]>'),
    ('<![CDATA[]]]>[9]', ' ' * len('<![CDATA[]]]>') + '[9]'),
    ('<![CDATA[]]>]', ' ' * len('<![CDATA[]]>') + ']'),
    # Unsupported candidates retain the generic !-tag fallback.
    ('<![CDATA[ [9]', '<![CDATA[ [9]'),
    ('<![CDATA[ [8] > [9]', ' ' * len('<![CDATA[ [8] >') + ' [9]'),
    ('<![CDATA[> [9]', ' ' * len('<![CDATA[>') + ' [9]'),
    ('<![CDATA[ " > [9]', '<![CDATA[ " > [9]'),
    ('<![CDATA[ >\n[9] ]]>', ' ' * len('<![CDATA[ >') + '\n[9] ]]>'),
    ('<![CDATA[\n[9] ]]>', '<![CDATA[\n[9] ]]>'),
    ('<![CDATA[ >\r\n[9] ]]>', ' ' * len('<![CDATA[ >') + '\r\n[9] ]]>'),
    ('<![CDATA[ >\r[9] ]]>', ' ' * len('<![CDATA[ >') + '\r[9] ]]>'),
    # Legacy generic tags exclude LF only; preserve the CR-only mask.
    ('<![CDATA[\r[9] ]]>', ' ' * len('<![CDATA[\r[9] ]]>')),
    ('<![CDATA[ >\n说明🙂 <![CDATA[ > [8] ]]>[9]',
     ' ' * len('<![CDATA[ >') + '\n说明🙂 ' + ' ' * len('<![CDATA[ > [8] ]]>') + '[9]'),
    ('<![CDATA[ > [8] ]]>`code ]]>`<![CDATA[ > [7] ]]>[9]',
     ' ' * len('<![CDATA[ > [8] ]]>`code ]]>`<![CDATA[ > [7] ]]>') + '[9]'),
    ('<![CDATA[ >' * 1000 + ' [9]', ' ' * (len('<![CDATA[ >') * 1000) + ' [9]'),
    ('<![CDATA[ >' * 1000 + '[8] ]]>[9]', ' ' * (len('<![CDATA[ >') * 1000 + len('[8] ]]>')) + '[9]'),
    ('<![cdata[ > [9] ]]>', ' ' * len('<![cdata[ >') + ' [9] ]]>'),
    ('<![CDATA > [9] ]]>', ' ' * len('<![CDATA >') + ' [9] ]]>'),
    ('<! [CDATA[ > [9] ]]>', ' ' * len('<! [CDATA[ >') + ' [9] ]]>'),
    ('<![CDATA[ [8] ] > [9]', ' ' * len('<![CDATA[ [8] ] >') + ' [9]'),
    ('<?note <![CDATA[ > [8] ?>[9]', ' ' * len('<?note <![CDATA[ > [8] ?>') + '[9]'),
    ('<![CDATA[ <?note > [8] ]]>[9]', ' ' * len('<![CDATA[ <?note > [8] ]]>') + '[9]'),
    ('<! > [9]', ' ' * len('<! >') + ' [9]'),
    ('<!-- > [8] -->[9]', ' ' * len('<!-- > [8] -->') + '[9]'),
    ('<!-- <![CDATA[ > [8] -->[9]', ' ' * len('<!-- <![CDATA[ > [8] -->') + '[9]'),
    ('<![CDATA[ <!-- > [8] ]]>[9]', ' ' * len('<![CDATA[ <!-- > [8] ]]>') + '[9]'),
    ('<span note="<![CDATA[ > [8] ]]>">[9]', ' ' * len('<span note="<![CDATA[ > [8] ]]>">') + '[9]'),
])
def test_first_terminator_and_fallback_contracts(body, expected):
    assert mask_markdown('说明🙂 ' + body) == '说明🙂 ' + expected


@pytest.mark.parametrize('body,expected', [
    ('`<![CDATA[ > [8] ]]>`[9]', ' ' * len('`<![CDATA[ > [8] ]]>`') + '[9]'),
    ('``<![CDATA[ > ` [8] ]]>``[9]', ' ' * len('``<![CDATA[ > ` [8] ]]>``') + '[9]'),
    ('`<![CDATA[ > ` [9] ]]>', ' ' * len('`<![CDATA[ > `') + ' [9] ]]>'),
    (r'\`<![CDATA[ > ` [8] ]]>[9]`', '\\`' + ' ' * len('<![CDATA[ > ` [8] ]]>') + '[9]`'),
    (r'\\`<![CDATA[ > [8] ]]>`[9]', '\\\\' + ' ' * len('`<![CDATA[ > [8] ]]>`') + '[9]'),
    ('<![CDATA[ > ` [8] ]]>`[9]`', ' ' * len('<![CDATA[ > ` [8] ]]>`[9]`')),
    ('[archive <![CDATA[ > ` [8] ]]>](part(one))[9]`',
     ' archive ' + ' ' * len('<![CDATA[ > ` [8] ]]>](part(one))') + '[9]`'),
    ('[archive](part(<![CDATA[ > ` [8] ]]>))[9]`',
     ' archive' + ' ' * len('](part(<![CDATA[ > ` [8] ]]>))') + '[9]`'),
    ('https://example.invalid/`<![CDATA[ > ` [8] ]]>[9]`',
     ' ' * len('https://example.invalid/`<![CDATA[ > ` [8] ]]>') + '[9]`'),
    ('<https://example.invalid/[8]> <![CDATA[ > ` [7] ]]>[9]`',
     ' ' * len('<https://example.invalid/[8]> <![CDATA[ > ` [7] ]]>') + '[9]`'),
])
def test_cdata_preserves_source_order_code_link_url_rules(body, expected):
    assert mask_markdown('说明🙂 ' + body) == '说明🙂 ' + expected


def test_cdata_contents_do_not_contaminate_source_matching_or_gates():
    answer = 'Archive "stable signal" <![CDATA[ > [9] 999kg "absent quotation" ` ]]> [4] manual'
    report = check(adapt({'answer': answer, 'sources': [
        {'idx': 4, 'text': 'Archive stable signal manual'}]}))
    assert report['findings'] == []
    assert all(not fails([report], level) for level in ['info', 'hint', 'warning', 'error', 'none'])
    sentence, = report['sentences']
    assert sentence['citation_indices'] == [4] and sentence['cited_coverage'] == 1
    assert sentence['text'] == answer


def test_exact_cdata_cli_contract():
    answer = '说明🙂 <![CDATA[ > [9] ]]>'
    result = cli('check', '-', '--report', 'json', stdin=json.dumps({'answer': answer, 'sources': []}))
    assert result.returncode == 0 and not result.stderr
    output = json.loads(result.stdout)
    assert output['fail_on'] == 'error' and not output['gate_failed']
    record, = output['records']
    assert record['findings'] == []
    sentence, = record['sentences']
    assert (sentence['start'], sentence['end'], sentence['text']) == (0, 23, answer)
    assert sentence['citation_indices'] == []


def test_cdata_cli_fixture():
    fixture = ROOT / 'examples/cite-check/cdata_sections.jsonl'
    answers = [json.loads(line)['answer'] for line in fixture.read_text().splitlines()]
    result = cli('check', fixture, '--jsonl', '--report', 'json')
    assert result.returncode == 1 and not result.stderr
    output = json.loads(result.stdout)
    assert output['record_count'] == len(answers) and output['gate_failed']
    for i, (answer, record) in enumerate(zip(answers, output['records'], strict=True)):
        if i in [1, 2, 3, 4]:
            finding, = record['findings']
            assert finding['code'] == 'CC001'
            start = answer.index('[9]')
            assert finding['evidence'] == {'marker': '[9]', 'start': start, 'end': start + 3, 'kind': 'dangling'}
        else:
            assert record['findings'] == []
