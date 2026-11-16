"""Orphan caption closings fail at their start, without changing table nesting."""
import json

import pytest

from ragwrench.tablechunk import TableChunkError, split_document, stream_document
from ragwrench.tablechunk.html import parse_html
from .test_cli_benchmark import cli
from .test_html_orphans import TinyReader, VALID

ERROR = 'unexpected closing caption'
CLOSINGS = ['</caption>', '</CAPTION>', '</CaPtIoN\n >']
CONTEXTS = [
    ('', ''),
    ('<div>', '</div>'),
    ('<p>北</p>\n é\t', ''),
    ('<h2>Résumé</h2>\n<p>北</p>\n<div>é\t', '</div>\n' + VALID),
    (VALID + '\n', ''),
    (VALID + '\n<div>北\n é\t', '</div>'),
    ('<table><caption>Label</caption><tr><td>x</td></tr></table>\n', ''),
]


@pytest.mark.parametrize('closing', CLOSINGS)
@pytest.mark.parametrize('prefix,suffix', CONTEXTS)
@pytest.mark.parametrize('entry', ['buffered', 'streamed', 'parse-offset'])
def test_orphan_caption_reports_closing_position(closing, prefix, suffix, entry):
    text = prefix + closing + suffix
    offset = 3 if entry == 'parse-offset' else 0
    with pytest.raises(TableChunkError) as caught:
        if entry == 'buffered':
            split_document(text, input_format='html')
        elif entry == 'streamed':
            list(stream_document(TinyReader(text.encode()), input_format='html'))
        else:
            parse_html(text, line_offset=offset)
    assert str(caught.value) == (f'line {prefix.count(chr(10)) + 1 + offset}, '
                                 f'column {len(prefix.rsplit(chr(10), 1)[-1])}: {ERROR}')


@pytest.mark.parametrize('closing', CLOSINGS[:2])
def test_embedded_markdown_checks_closing_caption_on_closing_physical_line(closing):
    prefix = '# Intro\n\n<table>\n<tr><td>earlier</td></tr>\n</table>é\t<div>'
    with pytest.raises(TableChunkError) as caught:
        split_document(prefix + closing + '</div>')
    assert str(caught.value) == f'line 5, column 15: {ERROR}'


@pytest.mark.parametrize('closing', CLOSINGS[:2])
def test_markdown_closing_captions_outside_html_fragment_remain_literal(closing):
    expected = split_document(closing)
    assert len(expected) == 1 and expected[0].text == closing
    assert [c.as_dict() for c in stream_document(TinyReader(closing.encode()))] == [c.as_dict() for c in expected]
    after = split_document(VALID + '\n' + closing)
    assert after[-1].text == closing and after[-1].metadata['kind'] == 'prose'


@pytest.mark.parametrize('closing', CLOSINGS)
@pytest.mark.parametrize('late', [False, True])
@pytest.mark.parametrize('streamed', [False, True])
def test_cli_orphan_closing_caption_is_atomic(tmp_path, closing, late, streamed):
    prefix = (VALID + '\n' if late else '') + '<div>é\t'
    source = tmp_path / 'orphan.html'
    source.write_text(prefix + closing + '</div>', encoding='utf-8')
    output = tmp_path / 'out.jsonl'
    output.write_bytes(b'preserved\x00\xff\n')
    for destination in [(), ('-o', output)]:
        result = cli('split', source, *(['--stream'] if streamed else []), *destination)
        assert result.returncode == 2
        assert result.stderr == f'ragwrench table-chunk: line {2 if late else 1}, column 7: {ERROR}\n'
        assert result.stdout == ''
        assert output.read_bytes() == b'preserved\x00\xff\n'
        assert not list(tmp_path.glob('.tablechunk-*'))


def test_cli_buffered_markdown_fragment_closing_caption_is_atomic(tmp_path):
    source = tmp_path / 'orphan.md'
    source.write_text('# Intro\n\n' + VALID + 'é\t</CAPTION>', encoding='utf-8')
    output = tmp_path / 'out.jsonl'
    output.write_bytes(b'preserved\n')
    for destination in [(), ('-o', output)]:
        result = cli('split', source, *destination)
        assert result.returncode == 2
        assert result.stderr == f'ragwrench table-chunk: line 3, column {len(VALID) + 2}: {ERROR}\n'
        assert result.stdout == ''
        assert output.read_bytes() == b'preserved\n'
        assert not list(tmp_path.glob('.tablechunk-*'))


@pytest.mark.parametrize('prefix', ['<table>', '<table><caption/>', '<table><tr><td>',
                                    '<table><caption><em>', '<table><tbody>'])
@pytest.mark.parametrize('entry', ['buffered', 'streamed', 'parse-offset', 'embedded'])
def test_closing_caption_inside_table_preserves_mismatched_tag_error(prefix, entry):
    text = prefix + '</caption></table>'
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
                                 'unclosed or mismatched HTML tag before </caption>')


@pytest.mark.parametrize('mode', ['markdown', 'kv', 'html'])
@pytest.mark.parametrize('adapter', ['jsonl', 'langchain', 'llamaindex'])
def test_ignored_escaped_and_nested_captions_preserve_full_records(mode, adapter):
    text = ('<h2>Résumé</h2><p>北 &lt;/caption&gt;literal</p>'
            '<script></caption></CAPTION></script>'
            '<style></caption></style><!-- </CAPTION> -->'
            '<table><caption><em>Label</em> &amp; details<br>Line</caption>'
            '<thead><tr><th id="a">A</th></tr></thead>'
            '<tbody><tr><td headers="a">before<table><caption>inner</caption>'
            '<tr><td>nested</td></tr><caption>late</caption>'
            '<caption>duplicate</caption></table>after</td></tr></tbody></table>'
            '<table><caption/><tr><td>empty</td></tr></table><p>after</p>')
    kwargs = dict(input_format='html', mode=mode)
    expected = split_document(text, **kwargs)
    actual = stream_document(TinyReader(text.encode()), **kwargs)
    serialize = lambda chunks: ''.join(json.dumps(c.as_dict(adapter), ensure_ascii=False) + '\n' for c in chunks)
    assert serialize(actual) == serialize(expected)
    assert expected[0].text == 'Résumé\n北 </caption>literal'
    assert expected[-1].text == 'Résumé\nafter'
    table = expected[1]
    assert table.metadata['lead_in'] == '北 </caption>literal'
    assert table.metadata['caption'] == 'Label & details\nLine'
    norm = table.metadata['normalization']
    assert norm['row_cell_ids'] == [['r1c0']]
    assert norm['source_cells']['r1c0']['headers'] == [{'id': 'a', 'cell_id': 'r0c0', 'text': 'A'}]
    assert 'nested_table_flattened' in table.metadata['warnings']
    assert 'beforeinner nested lateduplicate after' in ' '.join(table.text.split())
    assert expected[2].metadata['caption'] == ''
