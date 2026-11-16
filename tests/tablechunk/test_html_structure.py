"""Immediate outer row/cell ownership, source errors and nested-table isolation."""
import io

import pytest

from ragwrench.tablechunk import TableChunkError, split_document, stream_document
from ragwrench.tablechunk.html import parse_html
from .test_cli_benchmark import cli


MISPLACED = [
    (f'<table>{opening}<div>', 'tr', f'</div>{closing}</table>')
    for opening, closing in [('', ''), ('<thead>', '</thead>'),
                             ('<tbody>', '</tbody>'), ('<tfoot>', '</tfoot>')]
] + [
    ('<table><caption>', 'tr', '</caption></table>'),
    *[(f'<table><tr><{wrapper}>', tag, f'</{wrapper}></tr></table>')
      for wrapper in ('div', 'span') for tag in ('td', 'th')],
    ('<table><tbody>', 'td', '</tbody></table>'),
]


def misplaced(prefix, tag, suffix):
    body = '<td>x</td>' if tag == 'tr' else 'x'
    return f'{prefix}<{tag}>{body}</{tag}>{suffix}'


def ownership_error(tag):
    parent = 'table/thead/tbody/tfoot' if tag == 'tr' else 'tr'
    return f'<{tag}> must be a direct child of {parent}'


@pytest.mark.parametrize('prefix,tag,suffix', MISPLACED)
@pytest.mark.parametrize('entry', ['parse-offset', 'buffered', 'streamed', 'embedded'])
def test_misplaced_outer_structure_reports_tag_and_position(prefix, tag, suffix, entry):
    text = misplaced(prefix, tag, suffix)
    offset = 2 if entry in ('parse-offset', 'embedded') else 0
    with pytest.raises(TableChunkError) as error:
        if entry == 'parse-offset':
            parse_html(text, line_offset=offset)
        elif entry == 'embedded':
            split_document('# Intro\n\n' + text)
        elif entry == 'buffered':
            split_document(text, input_format='html')
        else:
            list(stream_document(io.BytesIO(text.encode()), input_format='html'))
    assert str(error.value) == f'line {1 + offset}, column {len(prefix)}: {ownership_error(tag)}'


@pytest.mark.parametrize('streamed', [False, True])
@pytest.mark.parametrize('prefix,tag,suffix', [MISPLACED[2], MISPLACED[5], MISPLACED[6]])
def test_cli_misplaced_structure_is_atomic(tmp_path, streamed, prefix, tag, suffix):
    path = tmp_path / 'misplaced.html'
    path.write_text('<table><tr><td>earlier</td></tr></table>\n' + misplaced(prefix, tag, suffix))
    output = tmp_path / 'out.jsonl'
    output.write_bytes(b'existing\n')
    for destination in [(), ('-o', output)]:
        result = cli('split', path, *(['--stream'] if streamed else []), *destination)
        assert result.returncode == 2
        assert result.stderr == f'ragwrench table-chunk: line 2, column {len(prefix)}: {ownership_error(tag)}\n'
        assert result.stdout == ''
        assert output.read_bytes() == b'existing\n'
        assert not list(tmp_path.glob('.tablechunk-*'))


@pytest.mark.parametrize('group', [None, 'thead', 'tbody', 'tfoot'])
@pytest.mark.parametrize('span', [0, 2])
def test_direct_children_keep_inline_nested_text_and_span_accounting(group, span):
    rows = (f'<tr><td rowspan="{span}"><strong>before</strong>'
            '<table><tbody><tr><td>inner</td></tr></tbody></table>'
            '<em>after</em></td><th><span>a</span></th></tr><tr><td>b</td></tr>')
    text = '<table>' + (f'<{group}>{rows}</{group}>' if group else rows) + '</table>'
    table = parse_html(text)[0]
    grid = [['before inner after', 'a'], ['before inner after', 'b']]
    assert (table.headers if group == 'thead' else table.rows) == grid
    norm = table.normalization
    assert (norm.header_cell_ids if group == 'thead' else norm.row_cell_ids) == [
        ['r0c0', 'r0c1'], ['r0c0', 'r1c0']]
    assert set(norm.source_cells) == {'r0c0', 'r0c1', 'r1c0'}
    source = norm.source_cells['r0c0']
    assert (source['rowspan'], source['resolved_rowspan']) == (span, 2)
    assert table.warnings.count('nested_table_flattened') == 1
    for mode in ('markdown', 'kv', 'html'):
        expected = split_document(text, input_format='html', mode=mode)
        actual = stream_document(io.BytesIO(text.encode()), input_format='html', mode=mode)
        assert [c.as_dict() for c in actual] == [c.as_dict() for c in expected]
