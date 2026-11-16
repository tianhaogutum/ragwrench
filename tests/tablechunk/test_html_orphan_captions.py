"""Caption openings require an outer table before prose or descendants are handled."""
import json

import pytest

from ragwrench.tablechunk import TableChunkError, split_document, stream_document
from ragwrench.tablechunk.html import parse_html
from .test_cli_benchmark import cli
from .test_html_orphans import TinyReader, VALID

ERROR = '<caption> must be a direct child of table'
CAPTIONS = [
    '<caption>label</caption>',
    '<caption></caption>',
    '<caption/>',
    '<CAPTION>北</CAPTION>',
    '<CAPTION\n title="é"/>',
    '<caption><tr><td colspan="0" colspan="1">x</td></tr></caption>',
]
CONTEXTS = [
    ('', ''),
    ('<div>', '</div>'),
    ('<h2>Résumé</h2>\n<p>北</p>\n<div>é\t', '</div>\n' + VALID),
    (VALID + '\n', ''),
    (VALID + '\n<div>北\n é\t', '</div>'),
]


@pytest.mark.parametrize('caption', CAPTIONS)
@pytest.mark.parametrize('prefix,suffix', CONTEXTS)
@pytest.mark.parametrize('entry', ['buffered', 'streamed', 'parse-offset'])
def test_orphan_caption_reports_opening_before_prose_or_descendants(caption, prefix, suffix, entry):
    text = prefix + caption + suffix
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


@pytest.mark.parametrize('caption', [c for c in CAPTIONS if '\n' not in c])
def test_embedded_markdown_checks_caption_on_closing_physical_line(caption):
    prefix = '# Intro\n\n<table>\n<tr><td>earlier</td></tr>\n</table>é\t<div>'
    with pytest.raises(TableChunkError) as caught:
        split_document(prefix + caption + '</div>')
    assert str(caught.value) == f'line 5, column 15: {ERROR}'


@pytest.mark.parametrize('caption', CAPTIONS[:4])
def test_markdown_captions_outside_html_fragment_remain_literal(caption):
    expected = split_document(caption)
    assert len(expected) == 1 and expected[0].text == caption
    assert [c.as_dict() for c in stream_document(TinyReader(caption.encode()))] == [c.as_dict() for c in expected]
    after = split_document(VALID + '\n' + caption)
    assert after[-1].text == caption and after[-1].metadata['kind'] == 'prose'


@pytest.mark.parametrize('caption', ['<caption>label</caption>', '<CAPTION/>'])
@pytest.mark.parametrize('late', [False, True])
@pytest.mark.parametrize('streamed', [False, True])
def test_cli_orphan_caption_is_atomic(tmp_path, caption, late, streamed):
    prefix = (VALID + '\n' if late else '') + '<div>é\t'
    source = tmp_path / 'orphan.html'
    source.write_text(prefix + caption + '</div>', encoding='utf-8')
    output = tmp_path / 'out.jsonl'
    output.write_bytes(b'preserved\x00\xff\n')
    for destination in [(), ('-o', output)]:
        result = cli('split', source, *(['--stream'] if streamed else []), *destination)
        assert result.returncode == 2
        assert result.stderr == f'ragwrench table-chunk: line {2 if late else 1}, column 7: {ERROR}\n'
        assert result.stdout == ''
        assert output.read_bytes() == b'preserved\x00\xff\n'
        assert not list(tmp_path.glob('.tablechunk-*'))


def test_cli_buffered_markdown_fragment_caption_is_atomic(tmp_path):
    source = tmp_path / 'orphan.md'
    source.write_text('# Intro\n\n' + VALID + 'é\t<caption/>', encoding='utf-8')
    output = tmp_path / 'out.jsonl'
    output.write_bytes(b'preserved\n')
    for destination in [(), ('-o', output)]:
        result = cli('split', source, *destination)
        assert result.returncode == 2
        assert result.stderr == f'ragwrench table-chunk: line 3, column {len(VALID) + 2}: {ERROR}\n'
        assert result.stdout == ''
        assert output.read_bytes() == b'preserved\n'
        assert not list(tmp_path.glob('.tablechunk-*'))


@pytest.mark.parametrize('mode', ['markdown', 'kv', 'html'])
@pytest.mark.parametrize('adapter', ['jsonl', 'langchain', 'llamaindex'])
def test_ignored_and_escaped_captions_preserve_text_context_and_accounting(mode, adapter):
    text = ('<h2>Résumé</h2><p>北 &lt;caption&gt;literal&lt;/caption&gt;</p>'
            '<script><caption><tr><td>ignored</td></tr></caption></script>'
            '<style><caption/></style><!-- <caption>ignored</caption> -->'
            '<table><caption><em>Label</em> &amp; details<br>Line</caption>'
            '<thead><tr><th id="a">A</th></tr></thead>'
            '<tbody><tr><td headers="a">value</td></tr></tbody></table><p>after</p>')
    kwargs = dict(input_format='html', mode=mode)
    expected = split_document(text, **kwargs)
    actual = stream_document(TinyReader(text.encode()), **kwargs)
    serialize = lambda chunks: ''.join(json.dumps(c.as_dict(adapter), ensure_ascii=False) + '\n' for c in chunks)
    assert serialize(actual) == serialize(expected)
    assert expected[0].text == 'Résumé\n北 <caption>literal</caption>'
    assert expected[-1].text == 'Résumé\nafter'
    table = expected[1]
    assert table.metadata['lead_in'] == '北 <caption>literal</caption>'
    assert table.metadata['caption'] == 'Label & details\nLine'
    norm = table.metadata['normalization']
    assert norm['row_cell_ids'] == [['r1c0']]
    assert norm['source_cells']['r1c0']['headers'] == [{'id': 'a', 'cell_id': 'r0c0', 'text': 'A'}]
