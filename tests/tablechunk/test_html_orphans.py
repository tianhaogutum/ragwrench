"""Orphan row/cell openings fail ownership before prose or attribute handling."""
import io

import pytest

from ragwrench.tablechunk import TableChunkError, split_document, stream_document
from ragwrench.tablechunk.html import parse_html
from .test_cli_benchmark import cli

VALID = '<table><tr><td>earlier</td></tr></table>'


class TinyReader(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 1))


def message(tag):
    parent = 'table/thead/tbody/tfoot' if tag == 'tr' else 'tr'
    return f'<{tag}> must be a direct child of {parent}'


def orphan(tag):
    if tag == 'tr':
        return '<tr><td rowspan="bogus">x</td></tr>'
    attrs = 'colspan="0" colspan="1" headers="missing"' if tag == 'td' else 'rowspan="bogus" id=""'
    return f'<{tag} {attrs}>x</{tag}>'


CONTEXTS = [
    ('', ''),
    ('<div>', '</div>'),
    *[(f'<{group}>', f'</{group}>') for group in ('thead', 'tbody', 'tfoot')],
    ('<h2>Résumé</h2>\n<p>北</p>\n<div>é\t', '</div>\n' + VALID),
    (VALID + '\n<div>北\n é\t', '</div>'),
]


@pytest.mark.parametrize('tag', ['tr', 'td', 'th'])
@pytest.mark.parametrize('prefix,suffix', CONTEXTS)
@pytest.mark.parametrize('entry', ['buffered', 'streamed', 'parse-offset'])
def test_orphan_structure_reports_opening_before_attributes(tag, prefix, suffix, entry):
    text = prefix + orphan(tag) + suffix
    offset = 3 if entry == 'parse-offset' else 0
    with pytest.raises(TableChunkError) as caught:
        if entry == 'buffered':
            split_document(text, input_format='html')
        elif entry == 'streamed':
            list(stream_document(TinyReader(text.encode()), input_format='html'))
        else:
            parse_html(text, line_offset=offset)
    if prefix in ('<thead>', '<tbody>', '<tfoot>'):
        assert str(caught.value) == f'line {1 + offset}, column 0: row group must be a direct child of table'
    else:
        assert str(caught.value) == (f'line {prefix.count(chr(10)) + 1 + offset}, '
                                     f'column {len(prefix.rsplit(chr(10), 1)[-1])}: {message(tag)}')


@pytest.mark.parametrize('tag', ['tr', 'td', 'th'])
@pytest.mark.parametrize('streamed', [False, True])
def test_self_closing_uppercase_orphan_reports_same_ownership(tag, streamed):
    text = f'<p>北</p>é<{tag.upper()}/>'
    with pytest.raises(TableChunkError) as caught:
        if streamed:
            list(stream_document(TinyReader(text.encode()), input_format='html'))
        else:
            split_document(text, input_format='html')
    assert str(caught.value) == f'line 1, column 9: {message(tag)}'


@pytest.mark.parametrize('tag', ['tr', 'td', 'th'])
def test_embedded_markdown_checks_orphans_on_table_block_closing_line(tag):
    # Only this HTML table fragment is sent to parse_html, with global line offsets.
    prefix = '# Intro\n\n<table>\n<tr><td>earlier</td></tr>\n</table>é\t'
    with pytest.raises(TableChunkError) as caught:
        split_document(prefix + orphan(tag))
    assert str(caught.value) == f'line 5, column 10: {message(tag)}'


@pytest.mark.parametrize('tag', ['tr', 'td', 'th'])
def test_ordinary_markdown_orphan_tags_remain_literal_prose(tag):
    text = orphan(tag)
    expected = split_document(text)
    actual = list(stream_document(TinyReader(text.encode())))
    assert [c.as_dict() for c in actual] == [c.as_dict() for c in expected]
    assert len(expected) == 1 and expected[0].text == text
    # A later physical line falls outside the buffered HTML table block too.
    after = split_document(VALID + '\n' + text)
    assert after[-1].text == text and after[-1].metadata['kind'] == 'prose'


@pytest.mark.parametrize('tag', ['tr', 'td', 'th'])
@pytest.mark.parametrize('late', [False, True])
@pytest.mark.parametrize('streamed', [False, True])
def test_cli_orphan_ownership_is_atomic(tmp_path, tag, late, streamed):
    prefix = (VALID + '\n' if late else '') + '<div>é\t'
    source = tmp_path / 'orphan.html'
    source.write_text(prefix + orphan(tag) + '</div>', encoding='utf-8')
    output = tmp_path / 'out.jsonl'
    output.write_bytes(b'preserved\x00\xff\n')
    for destination in [(), ('-o', output)]:
        result = cli('split', source, *(['--stream'] if streamed else []), *destination)
        assert result.returncode == 2
        assert result.stderr == f'ragwrench table-chunk: line {2 if late else 1}, column 7: {message(tag)}\n'
        assert result.stdout == ''
        assert output.read_bytes() == b'preserved\x00\xff\n'
        assert not list(tmp_path.glob('.tablechunk-*'))


def test_orphan_lookalikes_in_ignored_text_and_escaped_prose_stay_accepted():
    text = ('<h2>Résumé</h2><p>北 &lt;td&gt;literal&lt;/td&gt;</p>'
            '<script><tr><td rowspan="bogus">ignored</td></tr></script>'
            '<style><th colspan="0">ignored</th></style><!-- <td>ignored</td> -->'
            + VALID + '<p>after</p>')
    chunks = split_document(text, input_format='html')
    assert chunks[0].text == 'Résumé\n北 <td>literal</td>'
    assert chunks[-1].text == 'Résumé\nafter'
    assert all('ignored' not in c.text for c in chunks)
    assert chunks[1].metadata['lead_in'] == '北 <td>literal</td>'
    assert chunks[1].metadata['normalization']['row_cell_ids'] == [['r0c0']]
    assert [c.as_dict() for c in stream_document(TinyReader(text.encode()), input_format='html')] == [c.as_dict() for c in chunks]
