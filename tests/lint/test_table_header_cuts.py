"""Header tokens may rejoin only when the source proves the whole table."""
import pytest

from ragwrench.lint.fix import fix
from ragwrench.lint.lint import lint
from ragwrench.lint.model import Chunk, Config


def records(chunks):
    return [c.record() for c in chunks]


@pytest.mark.parametrize('ending', ['\n', '\r\n', '\r'])
@pytest.mark.parametrize('header,cut', [
    ('| key | v |', 4),
    ('key | v', 2),
    ('| café 雪 | v |', 6),
    ('| café 雪 | v |', 8),
    ('| key \\| 雪 | v |', 7),  # cut between escape and pipe
    ('| key \\| 雪 | v |', 8),
    ('| key\u2028雪 | v |', 6),  # Unicode separator is literal cell text
    ('| key | value |', 11),
])
def test_source_backed_header_cell_cut(ending, header, cut):
    source = ending.join([header, '| - | :-: |', '| a | b |', ''])
    chunks = [Chunk('h', source[:cut], {'part': 'left', 'nested': {'label': '雪'}}),
              Chunk('r', source[cut:], {'part': 'right'})]
    original = records(chunks)
    assert not lint([Chunk('whole', source)], source, Config(min_chars=0)).issues
    result = fix(chunks, source, Config(min_chars=0))
    assert [c.text for c in result.chunks] == [source]
    assert result.chunks[0].id == 'h'
    assert result.chunks[0].metadata['chunklint']['inputs'] == [
        {'id': c.id, 'metadata': c.metadata} for c in chunks]
    assert [c['operation'] for c in result.changes] == ['rejoin_table']
    assert not result.after.issues and result.after.stats['source']['coverage'] == 1
    assert all(result.invariant.values()) and records(chunks) == original
    second = fix(result.chunks, source, Config(min_chars=0))
    assert records(second.chunks) == records(result.chunks) and not second.changes


def test_exact_prior_report_header_cut():
    chunks = [Chunk('h', '| ke', {'part': 'left'}),
              Chunk('r', 'y | v |\n| -- | -- |\n| a | b |\n', {'part': 'right'})]
    source = '| key | v |\n| -- | -- |\n| a | b |\n'
    result = fix(chunks, source, Config(min_chars=0))
    assert [c.text for c in result.chunks] == [source]
    assert not result.after.issues and result.after.stats['source']['coverage'] == 1
    assert len(result.changes) == 1


@pytest.mark.parametrize('case,left,right,source', [
    ('no-source', '| ke', 'y | v |\n| -- | -- |\n| a | b |\n', None),
    ('missing-source-character', '| k', 'y | v |\n| -- | -- |\n| a | b |\n',
     '| key | v |\n| -- | -- |\n| a | b |\n'),
    ('malformed-delimiter', '| ke', 'y | v |\n| -x | -- |\n| a | b |\n',
     '| key | v |\n| -x | -- |\n| a | b |\n'),
    ('mismatched-count', '| ke', 'y | v | z |\n| -- | -- |\n| a | b |\n',
     '| key | v | z |\n| -- | -- |\n| a | b |\n'),
    ('blank-in-header', '| ke', 'y | v |\n| -- | -- |\n| a | b |\n',
     '| ke\n\ny | v |\n| -- | -- |\n| a | b |\n'),
    ('blank-before-delimiter', '| ke', 'y | v |\n| -- | -- |\n| a | b |\n',
     '| key | v |\n\n| -- | -- |\n| a | b |\n'),
    ('blank-before-body', '| ke', 'y | v |\n| -- | -- |\n| a | b |\n',
     '| key | v |\n| -- | -- |\n\n| a | b |\n'),
    ('newline-in-header', '| ke', 'y | v |\n| -- | -- |\n| a | b |\n',
     '| ke\ny | v |\n| -- | -- |\n| a | b |\n'),
    ('quoted-prefix', '>', ' | key | v |\n> | -- | -- |\n> | a | b |\n',
     '> | key | v |\n> | -- | -- |\n> | a | b |\n'),
    ('depth-change', '| ke', 'y | v |\n> | -- | -- |\n> | a | b |\n',
     '| key | v |\n> | -- | -- |\n> | a | b |\n'),
    ('prose-after-table', '| ke', 'y | v |\n| -- | -- |\nProse.\n',
     '| key | v |\n| -- | -- |\nProse.\n'),
    ('fence-after-header', '| ke', 'y | v |\n```\n| -- | -- |\n```\n',
     '| key | v |\n```\n| -- | -- |\n```\n'),
    ('reverse-spans', '| ke', 'y | v |\n| -- | -- |\n| a | b |\n',
     'y | v |\n| -- | -- |\n| a | b |\n\n| ke'),
    ('overlapping-spans', '| key', 'key | v |\n| -- | -- |\n| a | b |\n',
     '| key | v |\n| -- | -- |\n| a | b |\n'),
    ('fragmented-right', '| ke', 'y | v |\n| -- | -- |\n| a | b |\n',
     '| key | v |\n| -- | -- |\nintervening\n| a | b |\n'),
])
def test_header_cut_requires_one_source_verified_unquoted_table(case, left, right, source):
    chunks = [Chunk('h', left, {'part': 'left'}), Chunk('r', right, {'part': 'right'})]
    config = Config(min_chars=0)
    result = fix(chunks, source, config)
    assert records(result.chunks) == records(chunks) and not result.changes
    assert result.after.record() == result.before.record()
    second = fix(result.chunks, source, config)
    assert records(second.chunks) == records(chunks) and not second.changes


def test_header_cut_retains_verified_offsets_and_source_whitespace():
    table = '| café \\| 雪 | v |\r\n| -- | -- |\r\n| a | b |\r\n'
    preamble = 'Introduction.\r\n\r\n'
    source = preamble + table
    cut = table.index('雪')
    chunks = [Chunk('intro', preamble),
              Chunk('h', table[:cut], {'part': 'left'}, len(preamble), len(preamble) + cut),
              Chunk('r', table[cut:], {'part': 'right'}, len(preamble) + cut, len(source))]
    result = fix(chunks, source, Config(min_chars=0))
    assert [c.text for c in result.chunks] == [preamble, table]
    assert result.chunks[1].start_hint == len(preamble)
    assert result.chunks[1].end_hint == len(source.rstrip())
    assert result.after.stats['source']['coverage'] == 1
    assert not result.after.issues


def test_header_cut_respects_disabled_rule():
    chunks = [Chunk('h', '| ke'), Chunk('r', 'y | v |\n| -- | -- |\n| a | b |\n')]
    result = fix(chunks, ''.join(c.text for c in chunks), Config(min_chars=0, disable=['CL002']))
    assert records(result.chunks) == records(chunks) and not result.changes
