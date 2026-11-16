"""Exact header cuts in one consistently prefixed, single-level quoted table.

Prefix grammar: 0..3 ASCII spaces, '>', then one ASCII space; identical on
all lines. No preamble, prefix cuts, nested/list/fence containers or ambiguous
source occurrences. Payload and raw source must concatenate exactly.
"""
from collections import Counter
from pathlib import Path

import pytest

from ragwrench.lint.fix import fix
from ragwrench.lint.io import load_chunks, read_text
from ragwrench.lint.lint import lint
from ragwrench.lint.model import Chunk, Config

FIXTURE = Path(__file__).parent / 'fixtures' / 'quoted_header_cut'
CONFIG = Config(min_chars=0)


def records(chunks):
    return [c.record() for c in chunks]


def assert_recovered(chunks, source, expected=None):
    original = records(chunks)
    result = fix(chunks, source, CONFIG)
    expected = source if expected is None else expected
    assert [c.text for c in result.chunks] == [expected]
    output = result.chunks[0]
    assert output.id == chunks[0].id
    provenance = []
    for c in chunks:
        provenance.extend(c.metadata.get('chunklint', {}).get('inputs',
                          [{'id': c.id, 'metadata': c.metadata}]))
    assert output.metadata['chunklint']['inputs'] == provenance
    assert result.changes == [{'operation': 'rejoin_table', 'chunk_ids': [c.id for c in chunks],
                               'output_id': output.id}]
    assert all(result.invariant.values()) and records(chunks) == original
    assert Counter(''.join(c.text for c in chunks)) == Counter(output.text)
    assert output.text.encode() == expected.encode()
    assert not result.after.issues and result.after.stats['source']['coverage'] == 1
    second = fix(result.chunks, source, CONFIG)
    assert records(second.chunks) == records(result.chunks) and not second.changes
    return output


def test_exact_quoted_header_fixture_with_prior_ordered_provenance():
    source = read_text(FIXTURE / 'source.md')
    assert not lint([Chunk('whole', source)], source, CONFIG).issues
    assert_recovered(load_chunks(FIXTURE / 'chunks.jsonl'), source)


@pytest.mark.parametrize('ending', ['\n', '\r\n', '\r'])
@pytest.mark.parametrize('prefix', ['> ', ' > ', '   > '])
@pytest.mark.parametrize('header,cut', [
    ('| key | v |', 4), ('key | v', 2), ('| café 雪 | v |', 8),
    ('| key \\| 雪 | v |', 7), ('| key \\| 雪 | v |', 8),
    ('| key\u2028雪 | v |', 6),
])
def test_exact_single_level_header_cuts(ending, prefix, header, cut):
    source = ending.join(prefix + line for line in [header, '| - | :-: |', '| a | b |']) + ending
    cut += len(prefix)
    chunks = [Chunk('h', source[:cut], {'part': 'left', 'nested': {'label': '雪'}}),
              Chunk('r', source[cut:], {'part': 'right'})]
    assert_recovered(chunks, source)


@pytest.mark.parametrize('case,left,right,source', [
    ('no-source', '> | ke', 'y | v |\n> | -- | -- |\n> | a | b |\n', None),
    ('prefix-cut', '>', ' | key | v |\n> | -- | -- |\n> | a | b |\n', ''),
    ('nested', '> > | ke', 'y | v |\n> > | -- | -- |\n> > | a | b |\n', ''),
    ('compact-nested', '>> | ke', 'y | v |\n>> | -- | -- |\n>> | a | b |\n', ''),
    ('mixed-depth', '> | ke', 'y | v |\n> > | -- | -- |\n> | a | b |\n', ''),
    ('missing-prefix', '> | ke', 'y | v |\n| -- | -- |\n> | a | b |\n', ''),
    ('changed-prefix', '> | ke', 'y | v |\n > | -- | -- |\n> | a | b |\n', ''),
    ('no-space', '>| ke', 'y | v |\n>| -- | -- |\n>| a | b |\n', ''),
    ('tab-prefix', '>\t| ke', 'y | v |\n>\t| -- | -- |\n>\t| a | b |\n', ''),
    ('indented-code', '    > | ke', 'y | v |\n    > | -- | -- |\n    > | a | b |\n', ''),
    ('list-header', '> - | ke', 'y | v |\n> | -- | -- |\n> | a | b |\n', ''),
    ('ordered-list', '> 1. | ke', 'y | v |\n> | -- | -- |\n> | a | b |\n', ''),
    ('list-body', '> | ke', 'y | v |\n> | -- | -- |\n> - | a | b |\n', ''),
    ('fence-boundary', '> | ke', 'y | v |\n> ```\n> | -- | -- |\n> ```\n', ''),
    ('prose-preamble', 'Intro.\n> | ke', 'y | v |\n> | -- | -- |\n> | a | b |\n', ''),
    ('quoted-preamble', '> Intro.\n> | ke', 'y | v |\n> | -- | -- |\n> | a | b |\n', ''),
    ('blank-before-delimiter', '> | ke', 'y | v |\n\n> | -- | -- |\n> | a | b |\n', ''),
    ('quoted-blank', '> | ke', 'y | v |\n> \n> | -- | -- |\n> | a | b |\n', ''),
    ('two-tables', '> | ke', 'y | v |\n> | -- | -- |\n\n> | x | z |\n> | - | - |\n', ''),
    ('invalid-delimiter', '> | ke', 'y | v |\n> | -x | -- |\n> | a | b |\n', ''),
    ('count-mismatch', '> | ke', 'y | v | z |\n> | -- | -- |\n> | a | b |\n', ''),
    ('missing-character', '> | k', 'y | v |\n> | -- | -- |\n> | a | b |\n',
     '> | key | v |\n> | -- | -- |\n> | a | b |\n'),
    ('source-blank-in-header', '> | ke', 'y | v |\n> | -- | -- |\n> | a | b |\n',
     '> | ke\n\ny | v |\n> | -- | -- |\n> | a | b |\n'),
    ('source-blank-before-body', '> | ke', 'y | v |\n> | -- | -- |\n> | a | b |\n',
     '> | key | v |\n> | -- | -- |\n\n> | a | b |\n'),
    ('overlapping-spans', '> | key', 'key | v |\n> | -- | -- |\n> | a | b |\n',
     '> | key | v |\n> | -- | -- |\n> | a | b |\n'),
    ('reverse-spans', '> | ke', 'y | v |\n> | -- | -- |\n> | a | b |\n',
     'y | v |\n> | -- | -- |\n> | a | b |\n\n> | ke'),
    ('fragmented-right', '> | ke', 'y | v |\n> | -- | -- |\n> | a | b |\n',
     '> | key | v |\n> | -- | -- |\nMissing.\n> | a | b |\n'),
    ('source-prefix-whitespace', '> | ke', 'y | v |\n> | -- | -- |\n> | a | b |\n',
     '> | key | v |\n > | -- | -- |\n> | a | b |\n'),
])
def test_refuses_uncertain_or_unsupported_quoted_header_cuts(case, left, right, source):
    chunks = [Chunk('h', left, {'part': 'left'}), Chunk('r', right, {'part': 'right'})]
    source = left + right if source == '' else source
    result = fix(chunks, source, CONFIG)
    assert records(result.chunks) == records(chunks) and not result.changes
    second = fix(result.chunks, source, CONFIG)
    assert records(second.chunks) == records(chunks) and not second.changes


@pytest.mark.parametrize('with_hints', [False, True])
def test_repeated_source_is_ambiguous_even_with_verified_hints(with_hints):
    table = '> | key | v |\n> | -- | -- |\n> | a | b |\n'
    chunks = [Chunk('h', table[:6]), Chunk('r', table[6:])]
    if with_hints:
        chunks[0].start_hint, chunks[0].end_hint = 0, 6
        chunks[1].start_hint, chunks[1].end_hint = 6, len(table)
    result = fix(chunks, table + '\n' + table, CONFIG)
    assert [a.candidates for a in result.before.alignments] == [2, 2]
    assert records(result.chunks) == records(chunks) and not result.changes


@pytest.mark.parametrize('before,after', [('```md\n', '```\n'), ('> ```md\n', '> ```\n'),
                                         ('- ', ''), ('prose ', '')])
def test_source_container_cannot_be_hidden_by_chunk_edges(before, after):
    table = '> | key | v |\n> | -- | -- |\n> | a | b |\n'
    chunks = [Chunk('h', table[:6]), Chunk('r', table[6:])]
    result = fix(chunks, before + table + after, CONFIG)
    assert records(result.chunks) == records(chunks) and not result.changes


def test_quote_repair_keeps_offsets_with_separate_surrounding_chunks():
    table = '  > | café \\| 雪 | v |\r\n  > | -- | -- |\r\n  > | a | b |\r\n'
    intro, outro = 'Introduction.\r\n\r\n', '\r\nDone.'
    cut = table.index('雪')
    source = intro + table + outro
    chunks = [Chunk('intro', intro), Chunk('h', table[:cut], {}, len(intro), len(intro) + cut),
              Chunk('r', table[cut:], {}, len(intro) + cut, len(intro + table)), Chunk('outro', outro)]
    result = fix(chunks, source, CONFIG)
    assert [c.text for c in result.chunks] == [intro, table, outro]
    assert result.chunks[1].start_hint == len(intro) + 2
    assert result.chunks[1].end_hint == len(intro + table.rstrip())
    assert len(result.changes) == 1 and not result.after.issues
    second = fix(result.chunks, source, CONFIG)
    assert records(second.chunks) == records(result.chunks) and not second.changes


def test_quote_header_cut_respects_disabled_rule():
    chunks = load_chunks(FIXTURE / 'chunks.jsonl')
    result = fix(chunks, read_text(FIXTURE / 'source.md'), Config(min_chars=0, disable=['CL002']))
    assert records(result.chunks) == records(chunks) and not result.changes
