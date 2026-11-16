"""Outer span declarations are unique; nested attributes remain flattened."""
import io
import json

import pytest

from ragwrench.tablechunk import TableChunkError, split_document, stream_document
from ragwrench.tablechunk.html import parse_html
from .test_cli_benchmark import cli


DUPLICATES = [
    ('rowspan="1" rowspan="1"', 'rowspan'),
    ('rowspan="0" rowspan="1"', 'rowspan'),
    ('rowspan="bogus" rowspan="1"', 'rowspan'),
    ('rowspan="1" rowspan="bogus"', 'rowspan'),
    ('ROWSPAN="1" rOwSpAn="1"', 'rowspan'),
    ('rowspan rowspan="1"', 'rowspan'),
    ('rowspan="1" rowspan', 'rowspan'),
    ('colspan="1" colspan="1"', 'colspan'),
    ('colspan="2" colspan="1"', 'colspan'),
    ('colspan="0" colspan="1"', 'colspan'),
    ('colspan="1" colspan="0"', 'colspan'),
    ('COLSPAN="1" cOlSpAn="1"', 'colspan'),
    ('colspan="bogus" colspan="1"', 'colspan'),
    ('rowspan="bogus" colspan="1" colspan="2"', 'colspan'),
]


@pytest.mark.parametrize('attrs,name', DUPLICATES)
@pytest.mark.parametrize('tag', ['td', 'th'])
@pytest.mark.parametrize('prefix', ['<table><tr>', '<table>\n<caption>Early</caption>\n<tr>\t<!--é--> '])
@pytest.mark.parametrize('entry', ['parse-offset', 'buffered', 'streamed', 'embedded'])
def test_duplicate_span_reports_cell_opening(attrs, name, tag, prefix, entry):
    text = f'{prefix}<{tag} {attrs}>x</{tag}></tr></table>'
    offset = 2 if entry in ('parse-offset', 'embedded') else 0
    line = prefix.count('\n') + 1 + offset
    column = len(prefix.rsplit('\n', 1)[-1])
    with pytest.raises(TableChunkError) as error:
        if entry == 'parse-offset':
            parse_html(text, line_offset=offset)
        elif entry == 'buffered':
            split_document(text, input_format='html')
        elif entry == 'embedded':
            split_document('# Intro\n\n' + text)
        else:
            class TinyReader(io.BytesIO):
                def read(self, size=-1):
                    return super().read(min(size, 1))
            list(stream_document(TinyReader(text.encode()), input_format='html'))
    assert str(error.value) == f'line {line}, column {column}: duplicate {name} attribute'


@pytest.mark.parametrize('attrs,name', [DUPLICATES[2], DUPLICATES[9]])
@pytest.mark.parametrize('late', [False, True])
@pytest.mark.parametrize('streamed', [False, True])
def test_cli_duplicate_span_is_atomic(tmp_path, attrs, name, late, streamed):
    path = tmp_path / 'duplicate.html'
    path.write_text(('<table><tr><td>earlier</td></tr></table>\n' if late else '')
                    + f'<table><tr><td {attrs}>x</td></tr></table>')
    output = tmp_path / 'out.jsonl'
    output.write_bytes(b'existing\n')
    for destination in [(), ('-o', output)]:
        result = cli('split', path, *(['--stream'] if streamed else []), *destination)
        assert result.returncode == 2
        assert result.stderr == f'ragwrench table-chunk: line {2 if late else 1}, column 11: duplicate {name} attribute\n'
        assert result.stdout == ''
        assert output.read_bytes() == b'existing\n'
        assert not list(tmp_path.glob('.tablechunk-*'))


@pytest.mark.parametrize('span', [0, 2])
@pytest.mark.parametrize('mode', ['markdown', 'kv', 'html'])
@pytest.mark.parametrize('adapter', ['jsonl', 'langchain', 'llamaindex'])
def test_single_spans_reset_between_cells_tables_and_ignore_nested_attributes(span, mode, adapter):
    text = ('<h2>Sales</h2><p>Lead. Last.</p>\n<table>\n'
            '<tr><th>A</th><th>B</th></tr>\n'
            f'<tr><td ROWSPAN="{span}" colspan="1" class="a" class="b"><em>before</em>'
            '<table><tr><td rowspan="bogus" rowspan="1" colspan="0" colspan="1">detail</td>'
            '<th colspan="bogus" colspan="0">inner</th></tr></table><strong>after</strong>'
            '</td><td rowspan="1">x</td></tr>\n<tr><td>y</td></tr></table>\n'
            '<table><caption>Next</caption><tr><th>C</th></tr>'
            '<tr><td colspan="1">last</td></tr></table>')
    tables = parse_html(text)[2:]
    assert [t.caption for t in tables] == ['', 'Next']
    table = tables[0]
    assert table.rows == [['before detail inner after', 'x'], ['before detail inner after', 'y']]
    assert table.normalization.row_cell_ids == [['r1c0', 'r1c1'], ['r1c0', 'r2c0']]
    assert set(table.normalization.source_cells) == {'r0c0', 'r0c1', 'r1c0', 'r1c1', 'r2c0'}
    source = table.normalization.source_cells['r1c0']
    assert (source['rowspan'], source['resolved_rowspan'], source['colspan']) == (span, 2, 1)
    assert (source['line_start'], source['column_start']) == (4, 4)
    kwargs = dict(input_format='html', mode=mode, source='same.html')
    expected = split_document(text, **kwargs)
    actual = stream_document(io.BytesIO(text.encode()), **kwargs)
    serialize = lambda chunks: ''.join(json.dumps(c.as_dict(adapter), ensure_ascii=False) + '\n' for c in chunks)
    assert serialize(actual) == serialize(expected)
    chunk = next(c for c in expected if c.metadata.get('table_index') == 0)
    assert chunk.metadata['header_path'] == ['Sales']
    assert chunk.metadata['lead_in'] == 'Last.'
    assert chunk.metadata['warnings'] == ['nested_table_flattened']


@pytest.mark.parametrize('attrs,error', [
    ('rowspan="bogus"', 'invalid rowspan'),
    ('rowspan', 'invalid rowspan'),
    ('rowspan="257"', 'rowspan must be 0..256'),
    ('colspan="0"', 'colspan must be 1..256'),
    ('colspan="bogus"', 'invalid colspan'),
    ('colspan="257"', 'colspan must be 1..256'),
])
def test_single_invalid_span_keeps_diagnostic(attrs, error):
    with pytest.raises(TableChunkError) as caught:
        parse_html(f'<table><tr><td {attrs}>x</td></tr></table>')
    assert str(caught.value) == f'line 1, column 11: {error}'


def test_cell_parent_error_precedes_duplicate_attributes():
    with pytest.raises(TableChunkError) as caught:
        parse_html('<table><tr><div><td rowspan="1" rowspan="1">x</td></div></tr></table>')
    assert str(caught.value) == 'line 1, column 16: <td> must be a direct child of tr'
