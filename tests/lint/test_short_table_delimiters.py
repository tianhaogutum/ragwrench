"""Short GFM delimiter contracts within the existing simple pipe-table dialect."""
import pytest

from ragwrench.lint.fix import fix
from ragwrench.lint.lint import lint
from ragwrench.lint.markdown import scan, table_runs, valid_table
from ragwrench.lint.model import Chunk, Config


def records(chunks):
    return [chunk.record() for chunk in chunks]


@pytest.mark.parametrize('prefix', ['', '> ', '> > '])
@pytest.mark.parametrize('outer_pipes', [False, True])
@pytest.mark.parametrize('cell', ['-', ':-', '-:', ':-:', '--', ':--', '--:', ':--:'])
def test_short_delimiters_accept_all_colon_forms(prefix, outer_pipes, cell):
    rows = ['k \\| key | v', f'{cell} | :-:', 'café \\| 雪 | b']
    text = ''.join(prefix + ('| ' + row + ' |' if outer_pipes else row) + '\n' for row in rows)
    assert valid_table(scan(text).lines)
    assert lint([Chunk('table', text)], text, Config(min_chars=0)).issues == []


@pytest.mark.parametrize('prefixes', [('', '', ''), ('> ', '> ', '> '), ('> > ', '>> ', ' > > ')])
@pytest.mark.parametrize('delimiter', ['- | :-:', ':-- | --:'])
@pytest.mark.parametrize('ending', ['\n', '\r\n', '\r'])
@pytest.mark.parametrize('with_source', [False, True])
def test_short_delimiter_fragments_rejoin_exactly(prefixes, delimiter, ending, with_source):
    rows = ['| k | v |', f'| {delimiter} |', '| café \\| 雪 | b |']
    chunks = [Chunk(str(i), prefix + row + ending, {'row': i, 'origin': f'part-{i}'})
              for i, (prefix, row) in enumerate(zip(prefixes, rows))]
    source = ''.join(chunk.text for chunk in chunks)
    config = Config(min_chars=0)
    result = fix(chunks, source if with_source else None, config)
    assert [(i.rule, i.line) for i in result.before.issues] == [('CL002', 1)] * 3
    assert [chunk.text for chunk in result.chunks] == [source]
    assert [change['operation'] for change in result.changes] == ['rejoin_table'] * 2
    assert result.chunks[0].metadata['chunklint']['inputs'] == [
        {'id': chunk.id, 'metadata': chunk.metadata} for chunk in chunks]
    assert result.after.issues == []
    assert result.invariant['input_nonwhitespace_preserved_except_logged_exact_duplicates']
    if with_source:
        assert result.after.stats['source']['coverage'] == 1
        assert all(result.invariant.values())
    second = fix(result.chunks, source if with_source else None, config)
    assert records(second.chunks) == records(result.chunks)
    assert not second.changes


@pytest.mark.parametrize('prefix', ['', '> ', '> > '])
@pytest.mark.parametrize('bad_cell', ['', ':', '::', '- -', ': -', '- :', '--x', '—', '-\\-', '-\\|'])
def test_malformed_short_delimiters_stay_fragments(prefix, bad_cell):
    rows = [prefix + '| k | v |\n', prefix + f'| {bad_cell} | - |\n', prefix + '| a | b |\n']
    source = ''.join(rows)
    assert not valid_table(scan(source).lines)
    assert [(i.rule, i.line) for i in lint(
        [Chunk('table', source)], source, Config(min_chars=0)).issues] == [('CL002', 1)]
    chunks = [Chunk(str(i), row) for i, row in enumerate(rows)]
    result = fix(chunks, source, Config(min_chars=0))
    assert records(result.chunks) == records(chunks) and not result.changes


@pytest.mark.parametrize('prefix', ['', '> ', '> > '])
@pytest.mark.parametrize('header,delimiter', [
    ('| k | v |', '| - |'),
    ('| k | v |', '| - | -- | :-: |'),
    ('| k \\| v |', '| - | -- |'),
])
def test_short_delimiters_still_require_matching_header_cell_counts(prefix, header, delimiter):
    chunks = [Chunk('h', prefix + header + '\n'), Chunk('d', prefix + delimiter + '\n')]
    assert not valid_table(scan(''.join(c.text for c in chunks)).lines)
    result = fix(chunks, config=Config(min_chars=0))
    assert records(result.chunks) == records(chunks) and not result.changes
    assert all(issue.rule == 'CL002' for issue in result.after.issues)


@pytest.mark.parametrize('left,right', [
    ('> | k | v |\n', '> > | - | :-: |\n'),
    ('> > | k | v |\n', '> | -- | :--: |\n'),
    ('| k | v |\n', '> | - | :-: |\n'),
    ('> | k | v |\n> | - | :-: |\n', '| a | b |\n'),
    ('> | k | v |\n> | -- | --: |\n', '> > | a | b |\n'),
])
def test_short_delimiters_do_not_join_across_quote_depth(left, right):
    chunks = [Chunk('h', left), Chunk('r', right)]
    structure = scan(left + right)
    assert len(table_runs(structure)) == 2
    assert not valid_table(structure.lines)
    result = fix(chunks, left + right, Config(min_chars=0))
    assert records(result.chunks) == records(chunks) and not result.changes
    assert any(issue.rule == 'CL002' for issue in result.after.issues)


@pytest.mark.parametrize('prefix', ['', '> ', '> > '])
def test_short_delimiter_table_splitting_preserves_raw_context(prefix):
    header = prefix + '| k | v |\r\n' + prefix + '| - | :--: |\r\n'
    rows = [prefix + f'| key{i} | 雪{i} |\r\n' for i in range(10)]
    source = header + ''.join(rows)
    config = Config(min_chars=0, max_chars=90, disable=['CL006'])
    result = fix([Chunk('table', source, {'origin': 'short'})], source, config)
    assert len(result.chunks) > 1
    assert all(c.text.startswith(header) and len(c.text) <= 90 for c in result.chunks)
    assert ''.join(c.text[len(header):] for c in result.chunks) == ''.join(rows)
    assert all(c.metadata['chunklint']['parent_metadata'] == {'origin': 'short'} for c in result.chunks)
    assert result.after.stats['source']['coverage'] == 1 and not result.after.issues
    second = fix(result.chunks, source, config)
    assert records(second.chunks) == records(result.chunks) and not second.changes
