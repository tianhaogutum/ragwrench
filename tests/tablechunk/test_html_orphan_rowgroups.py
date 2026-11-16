"""Outer row groups require a table even when their contents could become prose."""
import pytest

from ragwrench.tablechunk import TableChunkError, split_document, stream_document
from ragwrench.tablechunk.html import parse_html
from .test_cli_benchmark import cli
from .test_html_orphans import TinyReader, VALID

MESSAGE = 'row group must be a direct child of table'
CONTEXTS = [
    ('', ''),
    ('<div>', '</div>'),
    ('<h2>Résumé</h2>\n<p>北</p>\n<div>é\t', '</div>\n' + VALID),
    (VALID + '\n<div>北\n é\t', '</div>'),
]


def group_markup(tag, form):
    if form == 'self-closing':
        return f'<{tag.upper()}/>'
    content = {'text': 'label', 'empty': '',
               'invalid-descendant': '<tr><td colspan="0" colspan="1">x</td></tr>'}[form]
    return f'<{tag}>{content}</{tag}>'


@pytest.mark.parametrize('tag', ['thead', 'tbody', 'tfoot'])
@pytest.mark.parametrize('form', ['text', 'empty', 'self-closing', 'invalid-descendant'])
@pytest.mark.parametrize('prefix,suffix', CONTEXTS)
@pytest.mark.parametrize('entry', ['buffered', 'streamed', 'parse-offset'])
def test_orphan_group_fails_at_opening_before_prose_or_descendants(tag, form, prefix, suffix, entry):
    text = prefix + group_markup(tag, form) + suffix
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


@pytest.mark.parametrize('tag', ['thead', 'tbody', 'tfoot'])
def test_embedded_markdown_checks_group_on_closing_physical_line(tag):
    prefix = '# Intro\n\n<table>\n<tr><td>earlier</td></tr>\n</table>é\t'
    with pytest.raises(TableChunkError) as caught:
        split_document(prefix + group_markup(tag, 'text'))
    assert str(caught.value) == f'line 5, column 10: {MESSAGE}'


@pytest.mark.parametrize('tag', ['thead', 'tbody', 'tfoot'])
def test_markdown_groups_outside_html_table_fragment_remain_literal(tag):
    text = group_markup(tag, 'text')
    expected = split_document(text)
    assert len(expected) == 1 and expected[0].text == text
    assert [c.as_dict() for c in stream_document(TinyReader(text.encode()))] == [c.as_dict() for c in expected]
    after = split_document(VALID + '\n' + text)
    assert after[-1].text == text and after[-1].metadata['kind'] == 'prose'


@pytest.mark.parametrize('tag', ['thead', 'tbody', 'tfoot'])
@pytest.mark.parametrize('late', [False, True])
@pytest.mark.parametrize('streamed', [False, True])
def test_cli_orphan_group_is_atomic(tmp_path, tag, late, streamed):
    prefix = (VALID + '\n' if late else '') + '<div>é\t'
    source = tmp_path / 'orphan.html'
    source.write_text(prefix + group_markup(tag, 'text') + '</div>', encoding='utf-8')
    output = tmp_path / 'out.jsonl'
    output.write_bytes(b'preserved\x00\xff\n')
    for destination in [(), ('-o', output)]:
        result = cli('split', source, *(['--stream'] if streamed else []), *destination)
        assert result.returncode == 2
        assert result.stderr == f'ragwrench table-chunk: line {2 if late else 1}, column 7: {MESSAGE}\n'
        assert result.stdout == ''
        assert output.read_bytes() == b'preserved\x00\xff\n'
        assert not list(tmp_path.glob('.tablechunk-*'))


@pytest.mark.parametrize('tag', ['thead', 'tbody', 'tfoot'])
def test_ignored_and_escaped_groups_keep_existing_text_and_accounting(tag):
    text = (f'<h2>Résumé</h2><p>北 &lt;{tag}&gt;literal&lt;/{tag}&gt;</p>'
            f'<script>{group_markup(tag, "invalid-descendant")}</script>'
            f'<style>{group_markup(tag, "empty")}</style>'
            f'<!-- {group_markup(tag, "text")} -->' + VALID + '<p>after</p>')
    chunks = split_document(text, input_format='html')
    assert chunks[0].text == f'Résumé\n北 <{tag}>literal</{tag}>'
    assert chunks[-1].text == 'Résumé\nafter'
    assert chunks[1].metadata['lead_in'] == f'北 <{tag}>literal</{tag}>'
    assert chunks[1].metadata['normalization']['row_cell_ids'] == [['r0c0']]
    assert [c.as_dict() for c in stream_document(TinyReader(text.encode()), input_format='html')] == [c.as_dict() for c in chunks]
