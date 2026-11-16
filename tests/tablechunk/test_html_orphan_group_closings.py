"""Orphan row-group closings fail without changing valid table ownership."""
import json

import pytest

from ragwrench.tablechunk import TableChunkError, split_document, stream_document
from ragwrench.tablechunk.html import parse_html
from .test_cli_benchmark import cli
from .test_html_orphans import TinyReader, VALID

MESSAGE = 'unexpected closing row group'
GROUPS = ['thead', 'tbody', 'tfoot']


def closing(tag, shape):
    return {'lower': f'</{tag}>', 'upper': f'</{tag.upper()}>',
            'multiline': f'</{tag.upper()}\n >'}[shape]


CONTEXTS = [
    ('', ''),
    ('<div>', '</div>'),
    ('<p>北</p>\n é\t', ''),
    ('<h2>Résumé</h2>\n<p>北</p>\n<div>é\t', '</div>\n' + VALID),
    (VALID + '\n', ''),
    (VALID + '\n<div>北\n é\t', '</div>'),
    ('<table><caption>Label</caption><tbody><tr><td>x</td></tr></tbody></table>\n', ''),
]


@pytest.mark.parametrize('tag', GROUPS)
@pytest.mark.parametrize('shape', ['lower', 'upper', 'multiline'])
@pytest.mark.parametrize('prefix,suffix', CONTEXTS)
@pytest.mark.parametrize('entry', ['buffered', 'streamed', 'parse-offset'])
def test_orphan_reports_closing_position(tag, shape, prefix, suffix, entry):
    text = prefix + closing(tag, shape) + suffix
    offset = 3 if entry == 'parse-offset' else 0
    with pytest.raises(TableChunkError) as caught:
        if entry == 'buffered':
            split_document(text, input_format='html')
        elif entry == 'streamed':
            list(stream_document(TinyReader(text.encode()), input_format='html'))
        else:
            parse_html(text, line_offset=offset)
    assert str(caught.value) == (f'line {prefix.count(chr(10)) + 1 + offset}, '
                                 f'column {len(prefix.rsplit(chr(10), 1)[-1])}: {MESSAGE}')


@pytest.mark.parametrize('tag', GROUPS)
@pytest.mark.parametrize('shape', ['lower', 'upper', 'multiline'])
def test_embedded_markdown_checks_complete_tags_in_consumed_fragment(tag, shape):
    # A multiline closing is complete here because the table closes later.
    prefix = '# Intro\n\n<table>\n<tr><td>earlier</td></tr>\n</table>é\t<div>'
    suffix = '</div>'
    if shape == 'multiline':
        prefix = '# Intro\n\n<table>\n<tr><td>earlier</td></tr>\n'
        suffix = '</table>'
    with pytest.raises(TableChunkError) as caught:
        split_document(prefix + closing(tag, shape) + suffix)
    message = (f'unclosed or mismatched HTML tag before </{tag}>'
               if shape == 'multiline' else MESSAGE)
    assert str(caught.value) == f'line 5, column {0 if shape == "multiline" else 15}: {message}'


@pytest.mark.parametrize('tag', GROUPS)
@pytest.mark.parametrize('shape', ['lower', 'upper'])
def test_markdown_tags_outside_html_fragment_remain_literal(tag, shape):
    text = closing(tag, shape)
    expected = split_document(text)
    assert len(expected) == 1 and expected[0].text == text
    assert [c.as_dict() for c in stream_document(TinyReader(text.encode()))] == [c.as_dict() for c in expected]
    after = split_document(VALID + '\n' + text)
    assert after[-1].text == text and after[-1].metadata['kind'] == 'prose'
    fenced = '```html\n' + text + '\n```'
    assert split_document(fenced)[0].text == fenced


@pytest.mark.parametrize('tag', GROUPS)
@pytest.mark.parametrize('shape', ['lower', 'upper', 'multiline'])
@pytest.mark.parametrize('late', [False, True])
@pytest.mark.parametrize('streamed', [False, True])
def test_cli_orphan_closing_is_atomic(tmp_path, tag, shape, late, streamed):
    prefix = (VALID + '\n' if late else '') + '<div>é\t'
    source = tmp_path / 'orphan.html'
    source.write_text(prefix + closing(tag, shape) + '</div>', encoding='utf-8')
    output = tmp_path / 'out.jsonl'
    for destination in ['stdout', 'existing', 'absent']:
        output.unlink(missing_ok=True)
        if destination != 'absent':
            output.write_bytes(b'preserved\x00\xff\n')
        result = cli('split', source, *(['--stream'] if streamed else []),
                     *(['-o', output] if destination != 'stdout' else []))
        assert result.returncode == 2
        assert result.stderr == f'ragwrench table-chunk: line {2 if late else 1}, column 7: {MESSAGE}\n'
        assert result.stdout == ''
        if destination == 'absent':
            assert not output.exists()
        else:
            assert output.read_bytes() == b'preserved\x00\xff\n'
        assert not list(tmp_path.glob('.tablechunk-*'))


@pytest.mark.parametrize('tag', GROUPS)
def test_cli_buffered_markdown_fragment_is_atomic(tmp_path, tag):
    source = tmp_path / 'orphan.md'
    source.write_text('# Intro\n\n' + VALID + 'é\t' + closing(tag, 'upper'), encoding='utf-8')
    output = tmp_path / 'out.jsonl'
    output.write_bytes(b'preserved\n')
    for destination in [(), ('-o', output)]:
        result = cli('split', source, *destination)
        assert result.returncode == 2
        assert result.stderr == f'ragwrench table-chunk: line 3, column {len(VALID) + 2}: {MESSAGE}\n'
        assert result.stdout == ''
        assert output.read_bytes() == b'preserved\n'
        assert not list(tmp_path.glob('.tablechunk-*'))


@pytest.mark.parametrize('tag', GROUPS)
@pytest.mark.parametrize('prefix', ['<table>', '<table><tr><td>', '<table><caption>', '<table><div>'])
@pytest.mark.parametrize('entry', ['buffered', 'streamed', 'parse-offset', 'embedded'])
def test_closings_inside_table_preserve_mismatched_tag_error(tag, prefix, entry):
    text = prefix + f'</{tag}></table>'
    offset = 2 if entry in ('parse-offset', 'embedded') else 0
    with pytest.raises(TableChunkError) as caught:
        if entry == 'parse-offset':
            parse_html(text, line_offset=offset)
        elif entry == 'embedded':
            split_document('# Intro\n\n' + text)
        elif entry == 'buffered':
            split_document(text, input_format='html')
        else:
            list(stream_document(TinyReader(text.encode()), input_format='html'))
    assert str(caught.value) == (f'line {1 + offset}, column {len(prefix)}: '
                                 f'unclosed or mismatched HTML tag before </{tag}>')


@pytest.mark.parametrize('entry', ['buffered', 'streamed'])
def test_first_orphan_closing_wins(entry):
    with pytest.raises(TableChunkError, match=f'^line 1, column 0: {MESSAGE}$'):
        text = '</TFOOT></caption></tr><tbody>'
        if entry == 'buffered':
            split_document(text, input_format='html')
        else:
            list(stream_document(TinyReader(text.encode()), input_format='html'))


@pytest.mark.parametrize('mode', ['markdown', 'kv', 'html'])
@pytest.mark.parametrize('adapter', ['jsonl', 'langchain', 'llamaindex'])
def test_ignored_escaped_and_nested_groups_preserve_full_records(mode, adapter):
    text = ('<h2>Résumé</h2><p>北 &lt;/thead&gt;&lt;/tbody&gt;&lt;/tfoot&gt;</p>'
            '<script></thead></TBODY></tfoot></script><style></thead></tbody></TFOOT></style>'
            '<!-- </thead></TBODY></tfoot> -->'
            '<table><caption><em>Label</em> &amp; details<br>Line</caption>'
            '<thead><tr><th id="a">A</th></tr></thead>'
            '<tbody><tr><td headers="a">before<table><caption>inner</caption>'
            '<thead><tr><th>nested header</th></tr></thead>'
            '<tbody><tr><td>nested cell</td></tr></tbody><tfoot/>'
            '</table>after</td></tr></tbody><tfoot/></table>'
            '<table><caption/><tbody/><tr><td>empty</td></tr></table><p>after</p>')
    kwargs = dict(input_format='html', mode=mode)
    expected = split_document(text, **kwargs)
    actual = stream_document(TinyReader(text.encode()), **kwargs)
    serialize = lambda chunks: ''.join(json.dumps(c.as_dict(adapter), ensure_ascii=False) + '\n' for c in chunks)
    assert serialize(actual) == serialize(expected)
    assert expected[0].text == 'Résumé\n北 </thead></tbody></tfoot>'
    assert expected[-1].text == 'Résumé\nafter'
    table = expected[1]
    assert table.metadata['lead_in'] == '北 </thead></tbody></tfoot>'
    assert table.metadata['caption'] == 'Label & details\nLine'
    norm = table.metadata['normalization']
    assert norm['row_cell_ids'] == [['r1c0']]
    assert norm['source_cells']['r1c0']['headers'] == [{'id': 'a', 'cell_id': 'r0c0', 'text': 'A'}]
    assert 'nested_table_flattened' in table.metadata['warnings']
    assert 'beforeinner nested header nested cell after' in ' '.join(table.text.split())
    assert expected[2].metadata['caption'] == ''
