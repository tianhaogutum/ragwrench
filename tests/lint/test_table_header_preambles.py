"""A whole prose preamble may precede a source-verified header token cut."""
import pytest

from ragwrench.lint.fix import fix
from ragwrench.lint.lint import lint
from ragwrench.lint.model import Chunk, Config


def records(chunks):
    return [c.record() for c in chunks]


def assert_recovered(left, right, source):
    chunks = [Chunk('h', left, {'part': 'left'}), Chunk('r', right, {'part': 'right'})]
    original = records(chunks)
    config = Config(min_chars=0)
    assert not lint([Chunk('whole', source)], source, config).issues
    result = fix(chunks, source, config)
    assert len(result.chunks) == 1 and result.chunks[0].text == source
    assert result.chunks[0].id == 'h'
    assert result.chunks[0].metadata['chunklint']['inputs'] == [
        {'id': c.id, 'metadata': c.metadata} for c in chunks]
    assert result.changes == [{'operation': 'rejoin_table', 'chunk_ids': ['h', 'r'], 'output_id': 'h'}]
    assert not result.after.issues and result.after.stats['source']['coverage'] == 1
    assert all(result.invariant.values()) and records(chunks) == original
    second = fix(result.chunks, source, config)
    assert records(second.chunks) == records(result.chunks) and not second.changes


def test_exact_round_11_preamble_header_cut():
    assert_recovered('Introduction.\n\n| ke', 'y | v |\n| -- | -- |\n| a | b |\n',
                     'Introduction.\n\n| key | v |\n| -- | -- |\n| a | b |\n')


@pytest.mark.parametrize('ending', ['\n', '\r\n', '\r'])
@pytest.mark.parametrize('preamble', ['Prose.\n', 'Introduction.\n\n',
                                      '\t Résumé 雪.  \nSecond line.\n \t\n'])
@pytest.mark.parametrize('header,cut', [('| key | v |', 4), ('key | v', 2),
                                       ('| café 雪 | v |', 8), ('| key \\| 雪 | v |', 7)])
def test_header_cut_preserves_whole_raw_prose_preamble(ending, preamble, header, cut):
    preamble = preamble.replace('\n', ending)
    table = ending.join([header, '| -- | -- |', '| a | b |', ''])
    assert_recovered(preamble + table[:cut], table[cut:], preamble + table)


@pytest.mark.parametrize('case,left,right,source', [
    ('no-source', '| ke', 'y | v |\n| -- | -- |\n| a | b |\n', None),
    ('missing-character', '| k', 'y | v |\n| -- | -- |\n| a | b |\n',
     '| key | v |\n| -- | -- |\n| a | b |\n'),
    ('invalid-delimiter', '| ke', 'y | v |\n| -x | -- |\n| a | b |\n', None),
    ('mismatched-delimiter', '| ke', 'y | v | z |\n| -- | -- |\n| a | b |\n', None),
    ('blank-in-header', '| ke', 'y | v |\n| -- | -- |\n| a | b |\n',
     '| ke\n\ny | v |\n| -- | -- |\n| a | b |\n'),
    ('newline-in-header', '| ke', 'y | v |\n| -- | -- |\n| a | b |\n',
     '| ke\ny | v |\n| -- | -- |\n| a | b |\n'),
    ('blank-before-delimiter', '| ke', 'y | v |\n| -- | -- |\n| a | b |\n',
     '| key | v |\n\n| -- | -- |\n| a | b |\n'),
    ('blank-before-body', '| ke', 'y | v |\n| -- | -- |\n| a | b |\n',
     '| key | v |\n| -- | -- |\n\n| a | b |\n'),
    ('quoted-token', '> | ke', 'y | v |\n> | -- | -- |\n> | a | b |\n', None),
    ('quoted-prefix', '>', ' | key | v |\n> | -- | -- |\n> | a | b |\n', None),
    ('depth-change', '| ke', 'y | v |\n> | -- | -- |\n> | a | b |\n', None),
    ('prose-boundary', '| ke', 'y | v |\nProse.\n| -- | -- |\n| a | b |\n', None),
    ('fence-boundary', '| ke', 'y | v |\n```\n| -- | -- |\n```\n', None),
    ('prose-after-table', '| ke', 'y | v |\n| -- | -- |\nProse.\n', None),
    ('overlapping-spans', '| key', 'key | v |\n| -- | -- |\n| a | b |\n',
     '| key | v |\n| -- | -- |\n| a | b |\n'),
    ('fragmented-right', '| ke', 'y | v |\n| -- | -- |\n| a | b |\n',
     '| key | v |\n| -- | -- |\nintervening\n| a | b |\n'),
    ('fenced-preamble', '```\ncode\n```\n| ke', 'y | v |\n| -- | -- |\n| a | b |\n', None),
    ('quoted-preamble', '> Prose.\n| ke', 'y | v |\n| -- | -- |\n| a | b |\n', None),
    ('earlier-table', '| x | z |\n| - | - |\n\n| ke', 'y | v |\n| -- | -- |\n| a | b |\n', None),
    ('delimiter-with-preamble', '| k | v |\n| -', '- | -- |\n| a | b |\n', None),
])
def test_preamble_recovery_remains_bounded(case, left, right, source):
    prefix = 'Introduction.\n\n'
    chunks = [Chunk('h', prefix + left, {'part': 'left'}), Chunk('r', right, {'part': 'right'})]
    source = None if case == 'no-source' else prefix + (source or left + right)
    result = fix(chunks, source, Config(min_chars=0))
    assert records(result.chunks) == records(chunks) and not result.changes
    assert any(i.rule == 'CL002' for i in result.after.issues)


def test_preamble_whitespace_is_not_rewritten():
    left, right = 'Introduction.  \n\n| ke', 'y | v |\n| -- | -- |\n| a | b |\n'
    chunks = [Chunk('h', left), Chunk('r', right)]
    result = fix(chunks, (left + right).replace('.  ', '.'), Config(min_chars=0))
    assert records(result.chunks) == records(chunks) and not result.changes


def test_preamble_header_cut_respects_disabled_rule():
    chunks = [Chunk('h', 'Introduction.\n\n| ke'), Chunk('r', 'y | v |\n| -- | -- |\n| a | b |\n')]
    result = fix(chunks, ''.join(c.text for c in chunks), Config(min_chars=0, disable=['CL002']))
    assert records(result.chunks) == records(chunks) and not result.changes


def test_preamble_header_cut_requires_ordered_spans():
    left, right = 'Introduction.\n\n| ke', 'y | v |\n| -- | -- |\n| a | b |\n'
    chunks = [Chunk('h', left), Chunk('r', right)]
    result = fix(chunks, right + '\n' + left, Config(min_chars=0))
    assert [a.status for a in result.before.alignments] == ['aligned', 'aligned']
    assert records(result.chunks) == records(chunks) and not result.changes


def test_preamble_header_cut_requires_contiguous_left_alignment():
    left, right = 'Introduction.\n\n| ke', 'y | v |\n| -- | -- |\n| a | b |\n'
    chunks = [Chunk('h', left), Chunk('r', right)]
    result = fix(chunks, (left + right).replace('Introduction.', 'Introduction.\nMissing.'),
                 Config(min_chars=0))
    assert result.before.alignments[0].status == 'fragmented'
    assert records(result.chunks) == records(chunks) and not result.changes
