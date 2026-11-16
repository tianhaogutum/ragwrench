"""One outer caption per table, independent of text and nested captions."""
import io
import json

import pytest

from ragwrench.tablechunk import TableChunkError, split_document, stream_document
from ragwrench.tablechunk.html import parse_html
from .test_cli_benchmark import cli


ERROR = 'duplicate outer <caption>'


@pytest.mark.parametrize('first,between,second', [
    ('<caption>First</caption>', '', '<caption>Second</caption>'),
    ('<caption></caption>', '', '<caption>Second</caption>'),
    ('<caption> \t </caption>', '', '<caption></caption>'),
    ('<caption/>', '', '<caption>Second</caption>'),
    ('<caption>First</caption>', '<tr><td>before</td></tr>', '<caption/>'),
])
@pytest.mark.parametrize('separator', ['', '\n\t<!--é--> '])
@pytest.mark.parametrize('entry', ['parse-offset', 'buffered', 'streamed', 'embedded'])
def test_duplicate_caption_reports_second_opening(first, between, second, separator, entry):
    prefix = '<table>' + first + between + separator
    text = prefix + second + '<tr><td>x</td></tr></table>'
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


@pytest.mark.parametrize('first', ['First', ''])
@pytest.mark.parametrize('streamed', [False, True])
def test_cli_duplicate_caption_is_atomic(tmp_path, first, streamed):
    prefix = f'<table><caption>{first}</caption>'
    path = tmp_path / 'duplicate.html'
    path.write_text('<table><tr><td>earlier</td></tr></table>\n' + prefix
                    + '<caption>Second</caption><tr><td>x</td></tr></table>')
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
def test_caption_count_resets_per_table_and_ignores_nested_captions(mode, adapter):
    text = ('<h2>Sales</h2><p>Lead. Last.</p>\n<table>\n'
            '<caption><em>Outer</em> &amp; label<br>Line</caption>\n'
            '<tr><th>A</th></tr>\n<tr><td>before<table>'
            '<caption>inner one</caption><caption>inner two</caption>'
            '<tr><td>detail</td></tr></table>after</td></tr>\n</table>\n'
            '<table><caption/><tr><td>empty</td></tr></table>\n'
            '<table><tr><td>none</td></tr></table>\n'
            '<table><tr><td><table><caption>inner only</caption>'
            '<tr><td>value</td></tr></table></td></tr></table>\n'
            '<table><caption>Next</caption><tr><td>last</td></tr></table>')
    tables = parse_html(text)[2:]
    assert [table.caption for table in tables] == ['Outer & label\nLine', '', '', '', 'Next']
    assert tables[0].rows == [['beforeinner oneinner two detail after']]
    assert tables[3].rows == [['inner only value']]
    norm = tables[0].normalization
    assert norm.row_cell_ids == [['r1c0']]
    assert set(norm.source_cells) == {'r0c0', 'r1c0'}
    assert norm.source_cells['r1c0']['line_start'] == 5
    kwargs = dict(input_format='html', mode=mode, source='same.html')
    expected = split_document(text, **kwargs)
    actual = stream_document(io.BytesIO(text.encode()), **kwargs)
    serialize = lambda chunks: ''.join(json.dumps(c.as_dict(adapter), ensure_ascii=False) + '\n' for c in chunks)
    assert serialize(actual) == serialize(expected)
    chunk = next(c for c in expected if c.metadata['kind'] == 'table')
    assert chunk.metadata['header_path'] == ['Sales']
    assert chunk.metadata['lead_in'] == 'Last.'
    assert chunk.metadata['caption'] == 'Outer & label\nLine'
