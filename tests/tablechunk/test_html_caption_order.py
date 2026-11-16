"""Outer captions precede rows/groups; nested labels remain cell content."""
import io
import json

import pytest

from ragwrench.tablechunk import TableChunkError, split_document, stream_document
from ragwrench.tablechunk.html import parse_html
from .test_cli_benchmark import cli


ERROR = '<caption> must precede rows and row groups'
BEFORE_CAPTION = [
    '<tr><td>x</td></tr>',
    '<thead><tr><th>A</th></tr></thead>',
    '<tbody><tr><td>x</td></tr></tbody>',
    '<tfoot><tr><td>x</td></tr></tfoot>',
    '<thead></thead>',
    '<tbody></tbody>',
    '<tfoot></tfoot>',
    '<tbody/>',
]


@pytest.mark.parametrize('before', BEFORE_CAPTION)
@pytest.mark.parametrize('separator,caption', [
    ('', '<caption>Late</caption>'),
    ('\n\t<!--é--> ', '<caption/>'),
])
@pytest.mark.parametrize('entry', ['parse-offset', 'buffered', 'streamed', 'embedded'])
def test_late_caption_reports_opening_position(before, separator, caption, entry):
    prefix = '<table>' + before + separator
    text = prefix + caption + '<tr><td>after</td></tr></table>'
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
    assert str(error.value) == f'line {line}, column {column}: {ERROR}'


@pytest.mark.parametrize('streamed', [False, True])
@pytest.mark.parametrize('before', ['<tr><td>x</td></tr>', '<tbody></tbody>'])
def test_cli_late_caption_is_atomic(tmp_path, streamed, before):
    prefix = '<table>' + before
    path = tmp_path / 'late.html'
    path.write_text('<table><tr><td>earlier</td></tr></table>\n' + prefix
                    + '<caption>Late</caption><tr><td>after</td></tr></table>')
    output = tmp_path / 'out.jsonl'
    output.write_bytes(b'existing\n')
    for destination in [(), ('-o', output)]:
        result = cli('split', path, *(['--stream'] if streamed else []), *destination)
        assert result.returncode == 2
        assert result.stderr == f'ragwrench table-chunk: line 2, column {len(prefix)}: {ERROR}\n'
        assert result.stdout == ''
        assert output.read_bytes() == b'existing\n'
        assert not list(tmp_path.glob('.tablechunk-*'))


@pytest.mark.parametrize('mode', ['markdown', 'kv', 'html'])
@pytest.mark.parametrize('adapter', ['jsonl', 'langchain', 'llamaindex'])
def test_early_caption_reset_and_nested_late_caption_keep_full_records(mode, adapter):
    text = ('<h2>Sales</h2><p>Lead. Last.</p>\n'
            '<table><tr><td>none</td></tr></table>\n'
            '<table><caption/><thead></thead><tbody/><tr><td>empty</td></tr></table>\n'
            '<table>\n<caption><em>Next</em> &amp; label<br>Line</caption>\n'
            '<thead><tr><th>A</th></tr></thead>\n<tbody>\n'
            '<tr><td>before<table><tbody><tr><td>detail</td></tr></tbody>'
            '<caption>inner late</caption><caption>inner duplicate</caption>'
            '</table>after</td></tr>\n</tbody></table>')
    tables = parse_html(text)[2:]
    assert [t.caption for t in tables] == ['', '', 'Next & label\nLine']
    table = tables[-1]
    assert table.rows == [['before detail inner lateinner duplicate after']]
    assert table.warnings == ['nested_table_flattened']
    assert table.normalization.row_cell_ids == [['r1c0']]
    assert set(table.normalization.source_cells) == {'r0c0', 'r1c0'}
    assert table.normalization.source_cells['r1c0']['line_start'] == 8
    kwargs = dict(input_format='html', mode=mode, source='same.html')
    expected = split_document(text, **kwargs)
    actual = stream_document(io.BytesIO(text.encode()), **kwargs)
    serialize = lambda chunks: ''.join(json.dumps(c.as_dict(adapter), ensure_ascii=False) + '\n' for c in chunks)
    assert serialize(actual) == serialize(expected)
    chunk = next(c for c in expected if c.metadata.get('table_index') == 2)
    assert chunk.metadata['caption'] == 'Next & label\nLine'
    assert chunk.metadata['header_path'] == ['Sales']
    assert chunk.metadata['lead_in'] == ''
    first = next(c for c in expected if c.metadata.get('table_index') == 0)
    assert first.metadata['lead_in'] == 'Last.'
