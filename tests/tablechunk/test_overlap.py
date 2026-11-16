"""Whole logical rows repeat within a table; new content always makes progress."""
import io
import json
import subprocess
import sys

import pytest

from ragwrench.tablechunk import TableChunkError, split_document, stream_document
from ragwrench.tablechunk.html import parse_html
from ragwrench.tablechunk.markdown import parse_markdown
from ragwrench.tablechunk.splitter import render_table


class OneByte(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 1) if size >= 0 else 1)


def markdown(rows):
    return '| A |\n|---|\n' + ''.join('| ' + row + ' |\n' for row in rows)


def tables(chunks):
    return [c for c in chunks if c.metadata['kind'] == 'table']


def ranges(chunks):
    return [(c.metadata['row_start'], c.metadata['row_end']) for c in tables(chunks)]


def test_requested_example():
    text = markdown([' '.join([f'r{i}'] + ['word'] * 29) for i in range(5)])
    assert ranges(split_document(text, max_tokens=120)) == [(0, 3), (3, 5)]
    chunks = split_document(text, max_tokens=120, overlap_rows=1)
    assert ranges(chunks) == [(0, 3), (2, 5)]
    assert [c.metadata['overlap_rows'] for c in chunks] == [0, 1]
    assert [c.id for c in chunks] == ['t0-r0-3', 't0-r2-5']


@pytest.mark.parametrize('mode', ['markdown', 'kv', 'html'])
@pytest.mark.parametrize('adapter', ['jsonl', 'langchain', 'llamaindex'])
@pytest.mark.parametrize('input_format', ['markdown', 'html'])
def test_modes_adapters_context_and_original_provenance(mode, adapter, input_format):
    if input_format == 'html':
        text = '''<h1>Topic</h1><p>Lead.</p><table><caption>Label</caption>
<thead><tr><th id="a" rowspan="2">A</th><th id="b">B</th></tr>
<tr><th id="c">C</th></tr></thead><tbody>
<tr><td rowspan="3" headers="a c">é</td><td headers="b">1</td></tr>
<tr><td headers="c">2</td></tr>
<tr><td headers="later b">3</td></tr>
<tr><td>x</td><td>4</td></tr>
<tr><th id="later">y</th><td>5</td></tr></tbody></table>'''
        table = next(b for b in parse_html(text) if hasattr(b, 'rows'))
        context = ['Topic', 'Label']
    else:
        text = '# Topic\n\nLead.\n\n' + markdown(['é', 'b', 'c', 'd', 'e'])
        table = next(b for b in parse_markdown(text) if hasattr(b, 'rows'))
        context = ['Topic', 'Lead.']
    budget = len(render_table(table, table.rows[:3], mode, context))
    kwargs = dict(max_tokens=budget, counter=len, mode=mode, input_format=input_format,
                  overlap_rows=1, source='input')
    chunks = split_document(text, **kwargs)
    streamed = list(stream_document(OneByte(text.encode()), **kwargs))
    assert json.dumps([c.as_dict(adapter) for c in streamed], ensure_ascii=False) == json.dumps(
        [c.as_dict(adapter) for c in chunks], ensure_ascii=False)
    tc = tables(chunks)
    assert ranges(tc) == [(0, 3), (2, 5)]
    for chunk, (start, end), overlap in zip(tc, [(0, 3), (2, 5)], [0, 1]):
        m = chunk.metadata
        assert chunk.text == render_table(table, table.rows[start:end], mode, context)
        assert m['row_line_spans'] == [list(p) for p in table.row_lines[start:end]]
        assert m['line_start'] == min(p[0] for p in table.row_lines[start:end])
        assert m['line_end'] == max(p[1] for p in table.row_lines[start:end])
        assert m['overlap_rows'] == overlap and m['requested_overlap_rows'] == 1
        assert m['token_count'] == len(chunk.text) <= budget
        if table.normalization:
            assert m['normalization']['source_cells']['r2c0']['resolved_rowspan'] == 3
            assert m['normalization']['source_cells']['r4c0']['headers'] == [
                {'id': 'later', 'cell_id': 'r6c0', 'text': 'y'},
                {'id': 'b', 'cell_id': 'r0c1', 'text': 'B'}]
            assert m['normalization'] == table.normalization.for_rows(start, end)
    assert all('overlap_rows' not in c.metadata for c in chunks if c.metadata['kind'] == 'prose')


@pytest.mark.parametrize('bad', [-1, 101, True, False, 1.0, '1', None])
@pytest.mark.parametrize('streamed', [False, True])
def test_invalid_policy(bad, streamed):
    with pytest.raises(TableChunkError, match='^overlap_rows must be an integer in 0..100$'):
        if streamed:
            list(stream_document(io.BytesIO(b''), overlap_rows=bad))
        else:
            split_document('', overlap_rows=bad)


@pytest.mark.parametrize('policy', [1, 2, 100])
def test_shrink_oldest_overlap_and_never_emit_overlap_only(policy):
    text = markdown(['a', 'b', 'c', 'd', 'x' * 12, 'z'])
    # Four short rows fit; the long row permits only the newest suffix row.
    table = parse_markdown(text)[0]
    budget = len(render_table(table, table.rows[:4], 'markdown', []))
    chunks = split_document(text, max_tokens=budget, counter=len, overlap_rows=policy)
    assert ranges(chunks) == [(0, 4), (3, 5), (4, 6)]
    assert [c.metadata['overlap_rows'] for c in chunks] == [0, 1, 1]
    assert all(c.metadata['row_end'] - c.metadata['row_start'] > c.metadata['overlap_rows'] for c in chunks)


@pytest.mark.parametrize('policy', [1, 100])
def test_one_row_chunks_and_last_chunk(policy):
    text = markdown(['a', 'b', 'c'])
    table = parse_markdown(text)[0]
    budget = len(render_table(table, [['a']], 'markdown', []))
    chunks = split_document(text, max_tokens=budget, counter=len, overlap_rows=policy)
    assert ranges(chunks) == [(0, 1), (1, 2), (2, 3)]
    assert [c.metadata['overlap_rows'] for c in chunks] == [0, 0, 0]
    assert ranges(split_document(markdown(['a']), max_tokens=budget, counter=len, overlap_rows=policy)) == [(0, 1)]


@pytest.mark.parametrize('mode', ['markdown', 'kv', 'html'])
@pytest.mark.parametrize('input_format', ['markdown', 'html'])
def test_fragments_are_barriers(mode, input_format):
    rows = ['a', 'b', 'x' * 180, 'c', 'd', 'e', 'f']
    text = markdown(rows) if input_format == 'markdown' else (
        '<table><tr><th id="a">A</th></tr>' + ''.join(
            '<tr><td headers="a">' + row + '</td></tr>' for row in rows) + '</table>')
    table = (parse_markdown(text) if input_format == 'markdown' else parse_html(text))[0]
    budget = len(render_table(table, [['a'], ['b']], mode, []))
    baseline = split_document(text, max_tokens=budget, counter=len, mode=mode, input_format=input_format)
    chunks = split_document(text, max_tokens=budget, counter=len, mode=mode, input_format=input_format, overlap_rows=100)
    fragments = [c for c in chunks if 'fragment' in c.metadata]
    expected = [c.as_dict() for c in baseline if 'fragment' in c.metadata]
    actual = []
    for c in fragments:
        d = c.as_dict(); d['metadata'] = dict(d['metadata'])
        assert d['metadata'].pop('overlap_rows') == 0
        assert d['metadata'].pop('requested_overlap_rows') == 100
        actual.append(d)
    assert actual == expected
    assert ranges([c for c in chunks if 'fragment' not in c.metadata]) == [(0, 2), (3, 5), (4, 6), (5, 7)]
    assert list(stream_document(OneByte(text.encode()), max_tokens=budget, counter=len,
                                mode=mode, input_format=input_format, overlap_rows=100)) == chunks


@pytest.mark.parametrize('input_format', ['markdown', 'html'])
def test_table_and_prose_boundaries_and_header_only(input_format):
    if input_format == 'markdown':
        table = markdown(['a a a', 'b b b', 'c c c'])
        text = table + '\nBetween.\n\n' + table + '\n' + markdown([])
    else:
        table = '<table><tr><th>A</th></tr>' + ''.join('<tr><td>'+v+'</td></tr>' for v in ['a a a', 'b b b', 'c c c']) + '</table>'
        text = table + '<p>Between.</p>' + table + '<table><tr><th>A</th></tr></table>'
    chunks = tables(split_document(text, input_format=input_format, max_tokens=22, overlap_rows=1))
    assert [(c.metadata['table_index'], c.metadata['overlap_rows']) for c in chunks] == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
    assert chunks[-1].metadata['row_start'] == chunks[-1].metadata['row_end'] == 0


@pytest.mark.parametrize('streamed', [False, True])
def test_context_too_large_unchanged(streamed):
    text = '# ' + 'word ' * 30 + '\n\n' + markdown(['a'])
    kwargs = dict(max_tokens=12, overlap_rows=1)
    with pytest.raises(TableChunkError, match='table 0: context and headers exceed max_tokens'):
        if streamed:
            list(stream_document(OneByte(text.encode()), **kwargs))
        else:
            split_document(text, **kwargs)


@pytest.mark.parametrize('streamed', [False, True])
@pytest.mark.parametrize('output', [False, True])
@pytest.mark.parametrize('failure', ['orphan', 'budget', 'utf8', 'policy'])
def test_cli_stages_late_errors(tmp_path, streamed, output, failure):
    good = b'<table><tr><th>A</th></tr><tr><td>x</td></tr></table>'
    tails = {'orphan': b'\n<tbody>late</tbody>',
             'budget': b'<table><tr><th>' + b'word ' * 100 + b'</th></tr></table>',
             'utf8': b'\xff', 'policy': b''}
    path = tmp_path / 'input.html'; path.write_bytes(good + tails[failure])
    dest = tmp_path / 'out.jsonl'; dest.write_bytes(b'KEEP\x00')
    args = [sys.executable, '-m', 'ragwrench.tablechunk', 'split', str(path), '--overlap-rows',
            '101' if failure == 'policy' else '1', '--max-tokens', '30']
    if streamed: args += ['--stream']
    if output: args += ['-o', str(dest)]
    p = subprocess.run(args, capture_output=True)
    assert p.returncode == 2 and p.stdout == b''
    assert dest.read_bytes() == b'KEEP\x00'
    assert not list(tmp_path.glob('.tablechunk-*'))
    if failure == 'orphan':
        assert p.stderr == b'ragwrench table-chunk: line 2, column 0: row group must be a direct child of table\n'
    if failure == 'policy':
        assert p.stderr == b'ragwrench table-chunk: overlap_rows must be an integer in 0..100\n'


@pytest.mark.parametrize('policy,expected', [
    (1, [(0, 3), (2, 5)]), (2, [(0, 3), (1, 4), (2, 5)]),
    (100, [(0, 3), (1, 4), (2, 5)])])
def test_multiple_repeated_rows_and_partial_final_group(policy, expected):
    text = markdown(list('abcde'))
    budget = len(render_table(parse_markdown(text)[0], [['a'], ['b'], ['c']], 'markdown', []))
    kwargs = dict(max_tokens=budget, counter=len, overlap_rows=policy)
    chunks = split_document(text, **kwargs)
    assert ranges(chunks) == expected
    assert [c.metadata['overlap_rows'] for c in chunks] == [0] + [min(policy, 2)] * (len(chunks) - 1)
    assert list(stream_document(OneByte(text.encode()), **kwargs)) == chunks
    short = split_document(markdown(list('abcd')), **kwargs)
    assert ranges(short) == expected[:1] + [(3 - min(policy, 2), 4)]
