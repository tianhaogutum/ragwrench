"""Same-line PI terminators, opaque bodies, fallback and source-order contracts."""
import json

import pytest

from ragwrench.citecheck.adapters import adapt
from ragwrench.citecheck.checker import check, fails
from ragwrench.citecheck.prose import mask_markdown
from .test_cli_io import ROOT, cli


@pytest.mark.parametrize('pi', [
    '<?note > [9] ?>', '<?note [9] ?>', '<??>', '<? > [9] ?>',
    '<?note > "unclosed [9] ?>', "<?note > 'unclosed [9] ?>",
    '<?note > ` [9] ?>', '<?note > `` [9] ?>',
    '<?note > [9] ? > more ?>', '<?note > <?nested [9] ?>',
    '<?+ > [bad] 999kg "absent quotation" ?>',
])
def test_complete_pi_hides_body_without_target_validation(pi):
    answer = '说明🙂 ' + pi
    assert mask_markdown(answer) == '说明🙂 ' + ' ' * len(pi)
    report = check(adapt({'answer': answer, 'sources': []}))
    assert report['findings'] == []
    assert all(not fails([report], level) for level in ['info', 'hint', 'warning', 'error', 'none'])
    sentence, = report['sentences']
    assert (sentence['start'], sentence['end'], sentence['text']) == (0, len(answer), answer)
    assert sentence['citation_indices'] == []


@pytest.mark.parametrize('pi', [
    '<??>', '<?note > [8] ?>', '<?note > "` [8] ?>', '<?note > `` [8] ?>',
    '<?note <?nested > [8] ?>', '<?note > [8] ?> <?next > [7] ?>',
])
def test_pi_backticks_do_not_hide_outside_citation(pi):
    answer = '说明🙂 ' + pi + '[9]`'
    assert mask_markdown(answer) == '说明🙂 ' + ' ' * len(pi) + '[9]`'
    report = check(adapt({'answer': answer, 'sources': []}))
    sentence, = report['sentences']
    assert (sentence['start'], sentence['end'], sentence['text']) == (0, len(answer), answer)
    assert sentence['citation_indices'] == [9]
    finding, = report['findings']
    assert finding['code'] == 'CC001' and fails([report], 'error')
    assert not fails([report], 'none')
    assert finding['evidence'] == {'marker': '[9]', 'start': 4 + len(pi),
                                   'end': 7 + len(pi), 'kind': 'dangling'}


@pytest.mark.parametrize('body,expected', [
    # First ?> wins, even in quotes or after another opener. No nesting.
    ('<??>[9]', ' ' * len('<??>') + '[9]'),
    ('<?note " ?>[9]" ?>', ' ' * len('<?note " ?>') + '[9]" ?>'),
    ('<?note ` ?>[9]` ?>', ' ' * len('<?note ` ?>') + '[9]` ?>'),
    ('<?note <?nested ?>[9] ?>', ' ' * len('<?note <?nested ?>') + '[9] ?>'),
    # Unsupported candidates retain the generic ?-tag fallback.
    ('<?note [9]', '<?note [9]'),
    ('<?note [8] > [9]', ' ' * len('<?note [8] >') + ' [9]'),
    ('<?> [9]', ' ' * len('<?>') + ' [9]'),
    ('<?note " > [9]', '<?note " > [9]'),
    ('<?note >\n[9] ?>', ' ' * len('<?note >') + '\n[9] ?>'),
    ('<?note\n[9] ?>', '<?note\n[9] ?>'),
    ('<?note >\r\n[9] ?>', ' ' * len('<?note >') + '\r\n[9] ?>'),
    ('<?note >\r[9] ?>', ' ' * len('<?note >') + '\r[9] ?>'),
    # Legacy generic tags exclude LF only; preserve the CR-only mask.
    ('<?note\r[9] ?>', ' ' * len('<?note\r[9] ?>')),
    ('<?note >\n说明🙂 <?note > [8] ?>[9]',
     ' ' * len('<?note >') + '\n说明🙂 ' + ' ' * len('<?note > [8] ?>') + '[9]'),
    ('<?note > [8] ?>`code ?>`<?note > [7] ?>[9]',
     ' ' * len('<?note > [8] ?>`code ?>`<?note > [7] ?>') + '[9]'),
    ('<?note >' * 1000 + ' [9]', ' ' * (len('<?note >') * 1000) + ' [9]'),
    ('<?note >' * 1000 + '[8] ?>[9]', ' ' * (len('<?note >') * 1000 + len('[8] ?>')) + '[9]'),
    ('<! > [9] ?>', ' ' * len('<! >') + ' [9] ?>'),
    ('<!-- > [8] -->[9]', ' ' * len('<!-- > [8] -->') + '[9]'),
    ('<!-- <?note > [8] -->[9]', ' ' * len('<!-- <?note > [8] -->') + '[9]'),
    ('<?note <!-- > [8] ?>[9]', ' ' * len('<?note <!-- > [8] ?>') + '[9]'),
    ('<span note="<?note > [8] ?>">[9]', ' ' * len('<span note="<?note > [8] ?>">') + '[9]'),
])
def test_first_terminator_and_fallback_contracts(body, expected):
    assert mask_markdown('说明🙂 ' + body) == '说明🙂 ' + expected


@pytest.mark.parametrize('body,expected', [
    ('`<?note > [8] ?>`[9]', ' ' * len('`<?note > [8] ?>`') + '[9]'),
    ('``<?note > ` [8] ?>``[9]', ' ' * len('``<?note > ` [8] ?>``') + '[9]'),
    ('`<?note > ` [9] ?>', ' ' * len('`<?note > `') + ' [9] ?>'),
    (r'\`<?note > ` [8] ?>[9]`', '\\`' + ' ' * len('<?note > ` [8] ?>') + '[9]`'),
    (r'\\`<?note > [8] ?>`[9]', '\\\\' + ' ' * len('`<?note > [8] ?>`') + '[9]'),
    ('<?note > ` [8] ?>`[9]`', ' ' * len('<?note > ` [8] ?>`[9]`')),
    ('[archive <?note > ` [8] ?>](part(one))[9]`',
     ' archive ' + ' ' * len('<?note > ` [8] ?>](part(one))') + '[9]`'),
    ('[archive](part(<?note > ` [8] ?>))[9]`',
     ' archive' + ' ' * len('](part(<?note > ` [8] ?>))') + '[9]`'),
    ('https://example.invalid/`<?note > ` [8] ?>[9]`',
     ' ' * len('https://example.invalid/`<?note > ` [8] ?>') + '[9]`'),
    ('<https://example.invalid/[8]> <?note > ` [7] ?>[9]`',
     ' ' * len('<https://example.invalid/[8]> <?note > ` [7] ?>') + '[9]`'),
])
def test_pi_preserves_source_order_code_link_url_rules(body, expected):
    assert mask_markdown('说明🙂 ' + body) == '说明🙂 ' + expected


def test_pi_contents_do_not_contaminate_source_matching_or_gates():
    answer = 'Archive "stable signal" <?note > [9] 999kg "absent quotation" ` ?> [4] manual'
    report = check(adapt({'answer': answer, 'sources': [
        {'idx': 4, 'text': 'Archive stable signal manual'}]}))
    assert report['findings'] == []
    assert all(not fails([report], level) for level in ['info', 'hint', 'warning', 'error', 'none'])
    sentence, = report['sentences']
    assert sentence['citation_indices'] == [4] and sentence['cited_coverage'] == 1
    assert sentence['text'] == answer


def test_exact_pi_cli_contract():
    answer = '说明🙂 <?note > [9] ?>'
    result = cli('check', '-', '--report', 'json', stdin=json.dumps({'answer': answer, 'sources': []}))
    assert result.returncode == 0 and not result.stderr
    output = json.loads(result.stdout)
    assert output['fail_on'] == 'error' and not output['gate_failed']
    record, = output['records']
    assert record['findings'] == []
    sentence, = record['sentences']
    assert (sentence['start'], sentence['end'], sentence['text']) == (0, 19, answer)
    assert sentence['citation_indices'] == []


def test_pi_cli_fixture():
    fixture = ROOT / 'examples/cite-check/processing_instructions.jsonl'
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
