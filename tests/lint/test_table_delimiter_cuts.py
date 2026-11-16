"""A delimiter cut needs a verified source row, never guessed punctuation."""
import pytest

from ragwrench.lint.fix import fix
from ragwrench.lint.model import Chunk, Config


def records(chunks):
    return [c.record() for c in chunks]


@pytest.mark.parametrize('ending', ['\n', '\r\n', '\r'])
@pytest.mark.parametrize('cell,cut_in_cell', [('-', 0), ('--', 1), ('-----', 2),
                                           (':-', 1), ('-:', 1), (':-:', 2), (':---:', 3)])
@pytest.mark.parametrize('outer_pipes', [True, False])
def test_source_backed_delimiter_row_cut(ending, cell, cut_in_cell, outer_pipes):
    header = '| k | v |' if outer_pipes else 'k | v'
    row = '| ' + cell + ' | -- |' if outer_pipes else cell + ' | --'
    source = ending.join([header, row, '| café \\| 雪 | beta |', ''])
    cut = len(header + ending) + (2 if outer_pipes else 0) + cut_in_cell
    chunks = [Chunk('h', source[:cut], {'part': 'left', 'nested': {'label': '雪'}}),
              Chunk('r', source[cut:], {'part': 'right'})]
    # Preserve verified coordinates and every original metadata object in provenance.
    for chunk, start, end in [(chunks[0], 0, cut), (chunks[1], cut, len(source))]:
        chunk.start_hint, chunk.end_hint = start, end
    original = records(chunks)
    result = fix(chunks, source, Config(min_chars=0))
    assert [c.text for c in result.chunks] == [source]
    assert result.chunks[0].id == 'h'
    assert result.chunks[0].metadata['chunklint']['inputs'] == [
        {'id': c.id, 'metadata': c.metadata} for c in chunks]
    assert [c['operation'] for c in result.changes] == ['rejoin_table']
    assert not result.after.issues
    assert result.after.stats['source']['coverage'] == 1 and all(result.invariant.values())
    assert records(chunks) == original
    second = fix(result.chunks, source, Config(min_chars=0))
    assert records(second.chunks) == records(result.chunks) and not second.changes


def test_exact_prior_report_delimiter_cut():
    chunks = [Chunk('h', '| k | v |\n| -', {'part': 'left'}),
              Chunk('r', '- | -- |\n| a | b |\n', {'part': 'right'})]
    source = '| k | v |\n| -- | -- |\n| a | b |\n'
    result = fix(chunks, source, Config(min_chars=0))
    assert [c.text for c in result.chunks] == [source]
    assert len(result.changes) == 1 and result.changes[0]['operation'] == 'rejoin_table'
    assert not result.after.issues and result.after.stats['source']['coverage'] == 1


@pytest.mark.parametrize('case,left,right,source', [
    ('no-source', '| k | v |\n| -', '- | -- |\n| a | b |\n', None),
    ('mismatched-source', '| k | v |\n| -', '- | -- |\n| a | b |\n',
     '| k | v |\n| --- | -- |\n| a | b |\n'),
    ('malformed-delimiter', '| k | v |\n| -', 'x | -- |\n| a | b |\n',
     '| k | v |\n| -x | -- |\n| a | b |\n'),
    ('mismatched-count', '| k | v | z |\n| -', '- | -- |\n| a | b |\n',
     '| k | v | z |\n| -- | -- |\n| a | b |\n'),
    ('blank-before-delimiter', '| k | v |\n| -', '- | -- |\n| a | b |\n',
     '| k | v |\n\n| -- | -- |\n| a | b |\n'),
    ('blank-inside-delimiter', '| k | v |\n| -', '- | -- |\n| a | b |\n',
     '| k | v |\n| -\n\n- | -- |\n| a | b |\n'),
    ('blank-after-delimiter', '| k | v |\n| -', '- | -- |\n| a | b |\n',
     '| k | v |\n| -- | -- |\n\n| a | b |\n'),
    ('depth-change', '| k | v |\n> | -', '- | -- |\n> | a | b |\n',
     '| k | v |\n> | -- | -- |\n> | a | b |\n'),
    ('fenced-code', '```\n| k | v |\n| -', '- | -- |\n| a | b |\n```\n',
     '```\n| k | v |\n| -- | -- |\n| a | b |\n```\n'),
])
def test_delimiter_cut_requires_one_source_verified_unquoted_table(case, left, right, source):
    chunks = [Chunk('h', left, {'part': 'left'}), Chunk('r', right, {'part': 'right'})]
    config = Config(min_chars=0, disable=['CL001'] if case == 'fenced-code' else [])
    result = fix(chunks, source, config)
    assert records(result.chunks) == records(chunks) and not result.changes
    assert result.after.record() == result.before.record()
    second = fix(result.chunks, source, config)
    assert records(second.chunks) == records(chunks) and not second.changes


def test_delimiter_cut_respects_disabled_rule():
    chunks = [Chunk('h', '| k | v |\n| -'), Chunk('r', '- | -- |\n| a | b |\n')]
    result = fix(chunks, ''.join(c.text for c in chunks), Config(min_chars=0, disable=['CL002']))
    assert records(result.chunks) == records(chunks) and not result.changes
