"""Explicit td headers resolve only to outer th cells, independently of layout."""
import io
import json
from pathlib import Path

import pytest

from ragwrench.tablechunk import TableChunkError, split_document, stream_document
from ragwrench.tablechunk.html import parse_html
from ragwrench.tablechunk.splitter import render_table
from .test_cli_benchmark import cli

FIXTURE = Path(__file__).parent / 'fixtures' / 'explicit-headers.html'


class TinyReader(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 1))


def records(chunks, adapter='jsonl'):
    return ''.join(json.dumps(c.as_dict(adapter), ensure_ascii=False) + '\n' for c in chunks)


@pytest.mark.parametrize('mode', ['markdown', 'kv', 'html'])
@pytest.mark.parametrize('adapter', ['jsonl', 'langchain', 'llamaindex'])
def test_explicit_associations_across_spans_and_chunks(mode, adapter):
    text = FIXTURE.read_text()
    table = parse_html(text)[1]
    budget = max(len(render_table(table, [row], mode, ['Sales', table.caption])) for row in table.rows)
    kwargs = dict(input_format='html', mode=mode, counter=len, max_tokens=budget, source='same.html')
    chunks = split_document(text, **kwargs)
    assert records(chunks, adapter) == records(stream_document(TinyReader(text.encode()), **kwargs), adapter)
    assert len(chunks) == 3
    expected = [
        {'id': 'q2', 'cell_id': 'r1c1', 'text': 'Q2'},
        {'id': 'total', 'cell_id': 'r0c1', 'text': 'Sales & tax'},
        {'id': 'west', 'cell_id': 'r2c0', 'text': 'West 北'},
    ]
    for chunk in chunks[:2]:
        norm = chunk.metadata['normalization']
        assert norm['source_cells']['r2c1']['headers'] == expected
        assert norm['source_cells']['r2c1']['resolved_rowspan'] == 2
        assert norm['source_cells']['r2c0']['line_start'] == 7
        assert len(chunk.text) <= budget
    norm = chunks[2].metadata['normalization']
    assert norm['row_cell_ids'] == [['r4c0', 'r4c1', 'r4c1']]
    assert norm['source_cells']['r4c1']['headers'] == [
        {'id': 'east', 'cell_id': 'r4c0', 'text': 'East'},
        {'id': 'total', 'cell_id': 'r0c1', 'text': 'Sales & tax'}]
    # Removing declarations leaves the same text, grid, IDs, counts and coordinates
    # (blank attributes preserve physical opening/closing positions).
    import re
    plain = re.sub(r' (?:id|headers)="[^"]*"', lambda m: ' ' * len(m[0]), text)
    baseline = split_document(plain, **kwargs)
    for actual, before in zip(chunks, baseline):
        assert (actual.id, actual.text) == (before.id, before.text)
        actual_meta = json.loads(json.dumps(actual.metadata))
        for source in actual_meta['normalization']['source_cells'].values():
            source.pop('headers', None)
        assert actual_meta == before.metadata


BAD = [
    ('<tr><th id="a">A</th><th id="a">B</th></tr>', '<th id="a">B', 'r0c1', "duplicate header id 'a'"),
    ('<tr><th id="a">A</th><td headers="a a">x</td></tr>', '<td', 'r0c1', "repeated headers reference 'a'"),
    ('<tr><td headers="missing">x</td></tr>', '<td', 'r0c0', "headers target 'missing' is not an outer <th>"),
    ('<tr><td id="a">x</td><td headers="a">y</td></tr>', '<td headers', 'r0c1', "headers target 'a' is not an outer <th>"),
    ('<tr><td headers="inner"><table><tr><th id="inner">N</th></tr></table></td></tr>', '<td', 'r0c0', "headers target 'inner' is not an outer <th>"),
    ('<tr><th id="A">A</th><td headers="a">x</td></tr>', '<td', 'r0c1', "headers target 'a' is not an outer <th>"),
    ('<tr><th id="bad id">A</th></tr>', '<th', 'r0c0', 'header id must be nonempty without ASCII whitespace'),
    ('<tr><th id>A</th></tr>', '<th', 'r0c0', 'header id must be nonempty without ASCII whitespace'),
    ('<tr><th id="a" ID="b">A</th></tr>', '<th', 'r0c0', 'duplicate id attribute'),
    ('<tr><td headers="a" HEADERS="b">x</td></tr>', '<td', 'r0c0', 'duplicate headers attribute'),
]


@pytest.mark.parametrize('body,opening,cell_id,message', BAD)
@pytest.mark.parametrize('entry', ['buffered', 'streamed', 'embedded', 'parse-offset'])
def test_explicit_association_errors_at_source_opening(body, opening, cell_id, message, entry):
    prefix = '<table>\n<tr><td>earlier</td></tr>\n</table>\n<table>\n <!--é--> '
    text = prefix + body + '\n</table>'
    offset = 2 if entry in ('embedded', 'parse-offset') else 0
    position = text.index(opening, len(prefix))
    line = text[:position].count('\n') + 1 + offset
    column = len(text[:position].rsplit('\n', 1)[-1])
    with pytest.raises(TableChunkError) as error:
        if entry == 'buffered':
            split_document(text, input_format='html')
        elif entry == 'streamed':
            list(stream_document(TinyReader(text.encode()), input_format='html'))
        elif entry == 'embedded':
            split_document('# Intro\n\n' + text)
        else:
            parse_html(text, line_offset=offset)
    assert str(error.value) == f'line {line}, column {column}: source cell {cell_id}: {message}'


@pytest.mark.parametrize('body,opening,cell_id,message', BAD[:5])
@pytest.mark.parametrize('streamed', [False, True])
def test_cli_late_association_errors_are_atomic(tmp_path, body, opening, cell_id, message, streamed):
    text = '<table><tr><td>earlier</td></tr></table>\n<table>' + body + '</table>'
    source = tmp_path / 'bad.html'
    source.write_text(text)
    output = tmp_path / 'out.jsonl'
    output.write_bytes(b'preserved\n')
    column = 7 + body.index(opening)
    for destination in [(), ('-o', output)]:
        result = cli('split', source, *(['--stream'] if streamed else []), *destination)
        assert result.returncode == 2
        assert result.stderr == f'ragwrench table-chunk: line 2, column {column}: source cell {cell_id}: {message}\n'
        assert result.stdout == ''
        assert output.read_bytes() == b'preserved\n'
        assert not list(tmp_path.glob('.tablechunk-*'))


@pytest.mark.parametrize('mode', ['markdown', 'kv', 'html'])
def test_forward_body_headers_fragments_and_table_reset(mode):
    value = 'alpha beta gamma ' * 25
    text = ('<table><tr><td headers="later">' + value + '</td></tr>'
            '<tr><th id="later">Body <em>label</em></th></tr></table>'
            '<table><tr><th id="later">New</th></tr><tr><td headers="later">last</td></tr></table>')
    kwargs = dict(input_format='html', mode=mode, max_tokens=130, counter=len)
    chunks = split_document(text, **kwargs)
    assert records(chunks) == records(stream_document(TinyReader(text.encode()), **kwargs))
    fragments = [c for c in chunks if 'fragment' in c.metadata]
    assert fragments
    for chunk in fragments:
        norm = chunk.metadata['normalization']
        assert norm['source_cells']['r0c0']['headers'] == [
            {'id': 'later', 'cell_id': 'r1c0', 'text': 'Body label'}]
        assert norm['source_cells']['r1c0']['tag'] == 'th'  # outside this chunk's grid
        assert len(chunk.text) <= 130
    last = chunks[-1].metadata['normalization']
    assert last['source_cells']['r1c0']['headers'] == [{'id': 'later', 'cell_id': 'r0c0', 'text': 'New'}]


def test_no_cross_table_or_nested_resolution():
    for suffix in [
        '<table><tr><td headers="old">x</td></tr></table>',
        '<table><tr><td><table><tr><th id="old">inner</th></tr></table></td>'
        '<td headers="old">x</td></tr></table>',
    ]:
        with pytest.raises(TableChunkError, match="headers target 'old' is not an outer <th>"):
            split_document('<table><tr><th id="old">Old</th></tr></table>' + suffix, input_format='html')
        with pytest.raises(TableChunkError, match="headers target 'old' is not an outer <th>"):
            list(stream_document(TinyReader(('<table><tr><th id="old">Old</th></tr></table>' + suffix).encode()),
                                 input_format='html'))


def test_unicode_ids_ascii_separators_empty_and_nested_attributes():
    # NBSP is part of an ID, not an HTML ASCII whitespace reference separator.
    text = ('<table><tr><th id="é 北"> A<br>B &amp; C </th><th id="é">same</th></tr>'
            '<tr><td headers="\té 北&#10;é\r\f ">before<table><tr>'
            '<th id="é 北" id="duplicate">inner</th><td headers="missing missing">detail</td>'
            '</tr></table>after</td><td headers=" "></td></tr></table>')
    chunks = split_document(text, input_format='html')
    assert records(chunks) == records(stream_document(TinyReader(text.encode()), input_format='html'))
    cells = chunks[0].metadata['normalization']['source_cells']
    assert cells['r1c0']['headers'] == [
        {'id': 'é 北', 'cell_id': 'r0c0', 'text': 'A\nB & C'},
        {'id': 'é', 'cell_id': 'r0c1', 'text': 'same'}]
    assert cells['r1c1']['headers'] == []


def test_association_metadata_has_a_streaming_working_limit():
    from ragwrench.tablechunk.html_storage import DiskStore
    from ragwrench.tablechunk.html_streaming import _StreamingParser

    storage = DiskStore(1500)
    try:
        parser = _StreamingParser(storage)
        text = '<table><tr><td>x</td></tr>' + ''.join(
            f'<tr><th id="{name}">' + 'x' * 600 + '</th></tr>' for name in ['a', 'b', 'c'])
        text += '<tr><td headers="a b c">v</td></tr></table>'
        with pytest.raises(TableChunkError, match='source cell r4c0: streaming HTML header associations exceed 1 MiB limit'):
            parser.feed(text)
    finally:
        storage.close()


def test_valid_ids_without_data_associations_preserve_output():
    text = '<table><tr><th id="a" scope="row" headers="ignored">A</th></tr><tr><td>x</td></tr></table>'
    plain = text.replace(' id="a" scope="row" headers="ignored"', ' ' * len(' id="a" scope="row" headers="ignored"'))
    assert records(split_document(text, input_format='html')) == records(split_document(plain, input_format='html'))
