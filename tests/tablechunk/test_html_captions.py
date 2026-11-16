"""Outer caption ownership, exact errors and nested-table isolation."""
import io
import json

import pytest

from ragwrench.tablechunk import TableChunkError, split_document, stream_document
from ragwrench.tablechunk.html import parse_html
from .test_cli_benchmark import cli


MISPLACED_CAPTIONS = [
    ('<table><div>', '</div><tr><td>x</td></tr></table>'),
    ('<table><tbody>', '</tbody><tr><td>x</td></tr></table>'),
    ('<table><tr>', '<td>x</td></tr></table>'),
    ('<table><tr><td>', '</td></tr></table>'),
    ('<table><caption>', '</caption><tr><td>x</td></tr></table>'),
]
ERROR = '<caption> must be a direct child of table'


@pytest.mark.parametrize('prefix,suffix', MISPLACED_CAPTIONS)
@pytest.mark.parametrize('entry', ['parse-offset', 'buffered', 'streamed', 'embedded'])
def test_misplaced_caption_reports_opening_position(prefix, suffix, entry):
    # Tabs and non-ASCII characters count as one source column each.
    text = prefix + '\n\t<!--é--> <caption>label</caption>' + suffix
    offset = 2 if entry in ('parse-offset', 'embedded') else 0
    with pytest.raises(TableChunkError) as error:
        if entry == 'parse-offset':
            parse_html(text, line_offset=offset)
        elif entry == 'embedded':
            split_document('# Intro\n\n' + text)
        elif entry == 'buffered':
            split_document(text, input_format='html')
        else:
            class TinyReader(io.BytesIO):
                def read(self, size=-1):
                    return super().read(min(size, 1))
            list(stream_document(TinyReader(text.encode()), input_format='html'))
    assert str(error.value) == f'line {2 + offset}, column 10: {ERROR}'


@pytest.mark.parametrize('streamed', [False, True])
def test_cli_misplaced_caption_is_atomic(tmp_path, streamed):
    path = tmp_path / 'misplaced.html'
    path.write_text('<table><tr><td>earlier</td></tr></table>\n'
                    '<table><div><caption>label</caption></div><tr><td>x</td></tr></table>')
    output = tmp_path / 'out.jsonl'
    output.write_bytes(b'existing\n')
    for destination in [(), ('-o', output)]:
        result = cli('split', path, *(['--stream'] if streamed else []), *destination)
        assert result.returncode == 2
        assert result.stderr == f'ragwrench table-chunk: line 2, column 12: {ERROR}\n'
        assert result.stdout == ''
        assert output.read_bytes() == b'existing\n'
        assert not list(tmp_path.glob('.tablechunk-*'))


@pytest.mark.parametrize('adapter', ['jsonl', 'langchain', 'llamaindex'])
@pytest.mark.parametrize('mode', ['markdown', 'kv', 'html'])
def test_direct_caption_keeps_context_and_nested_caption_in_cell(mode, adapter):
    text = ('<h2>Sales</h2><p>Lead. Last.</p>\n<table>\n'
            '<caption><em>Outer</em> &amp; label<br>Line</caption>\n'
            '<tr><th>A</th></tr>\n<tr><td>before<table><div><caption>inner</caption>'
            '<tr><td>detail</td></tr></div></table>after</td></tr>\n</table>')
    table = parse_html(text)[-1]
    assert table.caption == 'Outer & label\nLine'
    assert table.rows == [['beforeinner detail after']]
    assert table.normalization.row_cell_ids == [['r1c0']]
    assert set(table.normalization.source_cells) == {'r0c0', 'r1c0'}
    assert table.normalization.source_cells['r1c0']['line_start'] == 5
    assert table.warnings == ['nested_table_flattened']
    kwargs = dict(input_format='html', mode=mode, source='same.html')
    expected = split_document(text, **kwargs)
    actual = stream_document(io.BytesIO(text.encode()), **kwargs)
    serialize = lambda chunks: ''.join(json.dumps(c.as_dict(adapter), ensure_ascii=False) + '\n' for c in chunks)
    assert serialize(actual) == serialize(expected)
    chunk = next(c for c in expected if c.metadata['kind'] == 'table')
    assert chunk.metadata['caption'] == 'Outer & label\nLine'
    assert chunk.metadata['lead_in'] == 'Last.'
    assert chunk.metadata['header_path'] == ['Sales']
