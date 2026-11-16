"""Exact delimiter cuts in a single consistently prefixed quoted source table."""
from pathlib import Path

import pytest

from ragwrench.lint.fix import fix
from ragwrench.lint.io import load_chunks, read_text
from ragwrench.lint.lint import lint
from ragwrench.lint.model import Chunk, Config
from .test_quoted_header_cuts import assert_recovered, records

FIXTURE = Path(__file__).parent / 'fixtures' / 'quoted_delimiter_cut'
CONFIG = Config(min_chars=0)


def test_exact_requested_quoted_delimiter_fixture():
    source = read_text(FIXTURE / 'source.md')
    assert not lint([Chunk('whole', source)], source, CONFIG).issues
    assert_recovered(load_chunks(FIXTURE / 'chunks.jsonl'), source)


@pytest.mark.parametrize('ending', ['\n', '\r\n', '\r'])
@pytest.mark.parametrize('prefix', ['> ', ' > ', '  > ', '   > '])
@pytest.mark.parametrize('cell,cut_in_cell', [('--', 1), ('---', 2), (':-', 1),
                                           ('-:', 1), (':-:', 2), (':---:', 3)])
@pytest.mark.parametrize('outer_pipes', [False, True])
def test_exact_delimiter_cell_cuts(ending, prefix, cell, cut_in_cell, outer_pipes):
    header = '| café \\| 雪 | v |' if outer_pipes else 'café \\| 雪 | v'
    delimiter = '| ' + cell + ' | - |' if outer_pipes else cell + ' | -'
    source = ending.join(prefix + row for row in [header, delimiter, '| a | 雪 |']) + ending
    cut = len(prefix + header + ending + prefix) + (2 if outer_pipes else 0) + cut_in_cell
    chunks = [Chunk('h', source[:cut], {'part': 'left', 'chunklint': {
                  'operation': 'rejoin_table', 'inputs': [
                      {'id': 'h0', 'metadata': {'label': '雪'}}, {'id': 'h1', 'metadata': {'part': 1}}]}}),
              Chunk('r', source[cut:], {'part': 'right'})]
    chunks[0].start_hint, chunks[0].end_hint = 0, cut
    chunks[1].start_hint, chunks[1].end_hint = cut, len(source)
    assert_recovered(chunks, source)


@pytest.mark.parametrize('ending', ['\n', '\r\n', '\r'])
@pytest.mark.parametrize('cut_in_row', [4, 5, 6, 7, 8, 9])
def test_short_delimiters_and_whitespace_cuts(ending, cut_in_row):
    header, row = '> | k | v |', '> | - | -: |'
    source = ending.join([header, row, '> | a | b |']) + ending
    cut = len(header + ending) + cut_in_row
    assert_recovered([Chunk('h', source[:cut]), Chunk('r', source[cut:])], source)


@pytest.mark.parametrize('case,left,right,source', [
    ('no-source', '> | k | v |\n> | -', '- | -- |\n> | a | b |\n', None),
    ('nested', '> > | k | v |\n> > | -', '- | -- |\n> > | a | b |\n', ''),
    ('compact-nested', '>> | k | v |\n>> | -', '- | -- |\n>> | a | b |\n', ''),
    ('mixed-depth', '> | k | v |\n> > | -', '- | -- |\n> | a | b |\n', ''),
    ('missing-prefix', '> | k | v |\n> | -', '- | -- |\n| a | b |\n', ''),
    ('missing-delimiter-prefix', '> | k | v |\n| -', '- | -- |\n> | a | b |\n', ''),
    ('changed-prefix', '> | k | v |\n > | -', '- | -- |\n> | a | b |\n', ''),
    ('no-space', '>| k | v |\n>| -', '- | -- |\n>| a | b |\n', ''),
    ('tab-prefix', '>\t| k | v |\n>\t| -', '- | -- |\n>\t| a | b |\n', ''),
    ('indented-code', '    > | k | v |\n    > | -', '- | -- |\n    > | a | b |\n', ''),
    ('list-header', '> - | k | v |\n> | -', '- | -- |\n> | a | b |\n', ''),
    ('ordered-list', '> 1. | k | v |\n> | -', '- | -- |\n> | a | b |\n', ''),
    ('list-body', '> | k | v |\n> | -', '- | -- |\n> - | a | b |\n', ''),
    ('fence-boundary', '> | k | v |\n> | -', '- | -- |\n> ```\n> | a | b |\n> ```\n', ''),
    ('unquoted-preamble', 'Intro.\n> | k | v |\n> | -', '- | -- |\n> | a | b |\n', ''),
    ('quoted-preamble', '> Intro.\n> | k | v |\n> | -', '- | -- |\n> | a | b |\n', ''),
    ('blank-before-delimiter', '> | k | v |\n> \n> | -', '- | -- |\n> | a | b |\n', ''),
    ('blank-before-body', '> | k | v |\n> | -', '- | -- |\n\n> | a | b |\n', ''),
    ('quoted-blank', '> | k | v |\n> | -', '- | -- |\n> \n> | a | b |\n', ''),
    ('two-tables', '> | k | v |\n> | -', '- | -- |\n\n> | x | z |\n> | - | - |\n', ''),
    ('invalid-delimiter', '> | k | v |\n> | -', 'x | -- |\n> | a | b |\n', ''),
    ('colon-only', '> | k | v |\n> | :', ': | -- |\n> | a | b |\n', ''),
    ('empty-cell', '> | k | v |\n> | ', ' | -- |\n> | a | b |\n', ''),
    ('internal-space', '> | k | v |\n> | -', ' - | -- |\n> | a | b |\n', ''),
    ('count-mismatch', '> | k | v | z |\n> | -', '- | -- |\n> | a | b |\n', ''),
    ('prefix-cut', '> | k | v |\n>', ' | -- | -- |\n> | a | b |\n', ''),
    ('no-delimiter-payload', '> | k | v |\n> ', '| -- | -- |\n> | a | b |\n', ''),
    ('cut-in-line-ending', '> | k | v |\r', '\n> | -- | -- |\n> | a | b |\n', ''),
    ('cut-at-delimiter-end', '> | k | v |\n> | -- | -- |', '\n> | a | b |\n', ''),
    ('missing-character', '> | k | v |\n> | -', '- | -- |\n> | a | b |\n',
     '> | k | v |\n> | --- | -- |\n> | a | b |\n'),
    ('source-prefix-whitespace', '> | k | v |\n> | -', '- | -- |\n> | a | b |\n',
     '> | k | v |\n > | -- | -- |\n> | a | b |\n'),
    ('source-blank-in-delimiter', '> | k | v |\n> | -', '- | -- |\n> | a | b |\n',
     '> | k | v |\n> | -\n\n- | -- |\n> | a | b |\n'),
    ('source-blank-before-body', '> | k | v |\n> | -', '- | -- |\n> | a | b |\n',
     '> | k | v |\n> | -- | -- |\n\n> | a | b |\n'),
    ('overlapping', '> | k | v |\n> | --', '-- | -- |\n> | a | b |\n',
     '> | k | v |\n> | -- | -- |\n> | a | b |\n'),
    ('reversed', '> | k | v |\n> | -', '- | -- |\n> | a | b |\n',
     '- | -- |\n> | a | b |\n\n> | k | v |\n> | -'),
    ('fragmented', '> | k | v |\n> | -', '- | -- |\n> | a | b |\n',
     '> | k | v |\n> | -- | -- |\nMissing.\n> | a | b |\n'),
])
def test_refuses_uncertain_delimiter_boundaries(case, left, right, source):
    chunks = [Chunk('h', left, {'part': 'left'}), Chunk('r', right, {'part': 'right'})]
    source = left + right if source == '' else source
    result = fix(chunks, source, CONFIG)
    assert records(result.chunks) == records(chunks) and not result.changes
    second = fix(result.chunks, source, CONFIG)
    assert records(second.chunks) == records(chunks) and not second.changes


@pytest.mark.parametrize('with_hints', [False, True])
def test_ambiguous_even_with_verified_hints(with_hints):
    chunks = load_chunks(FIXTURE / 'chunks.jsonl')
    table = read_text(FIXTURE / 'source.md')
    if with_hints:
        cut = len(chunks[0].text)
        chunks[0].start_hint, chunks[0].end_hint = 0, cut
        chunks[1].start_hint, chunks[1].end_hint = cut, len(table)
    result = fix(chunks, table + '\n' + table, CONFIG)
    assert [a.candidates for a in result.before.alignments] == [2, 2]
    assert records(result.chunks) == records(chunks) and not result.changes


@pytest.mark.parametrize('before,after', [('```md\n', '```\n'), ('> ```md\n', '> ```\n'),
                                         ('- ', ''), ('prose ', '')])
def test_source_containers_outside_chunk_edges(before, after):
    chunks = load_chunks(FIXTURE / 'chunks.jsonl')
    table = read_text(FIXTURE / 'source.md')
    result = fix(chunks, before + table + after, CONFIG)
    assert records(result.chunks) == records(chunks) and not result.changes


def test_separate_surrounding_chunks_and_offset_hints():
    table = '  > | café \\| 雪 | v |\r\n  > | :-: | - |\r\n  > | a | b |\r\n'
    intro, outro = 'Introduction.\r\n\r\n', '\r\nDone.'
    cut = table.index(':-:') + 2
    source = intro + table + outro
    chunks = [Chunk('intro', intro), Chunk('h', table[:cut], {}, len(intro), len(intro) + cut),
              Chunk('r', table[cut:], {}, len(intro) + cut, len(intro + table)), Chunk('outro', outro)]
    result = fix(chunks, source, CONFIG)
    assert [c.text for c in result.chunks] == [intro, table, outro]
    assert result.chunks[1].start_hint == len(intro) + 2
    assert result.chunks[1].end_hint == len(intro + table.rstrip())
    assert len(result.changes) == 1 and not result.after.issues
    assert all(result.invariant.values()) and result.after.stats['source']['coverage'] == 1
    second = fix(result.chunks, source, CONFIG)
    assert records(second.chunks) == records(result.chunks) and not second.changes


def test_disabled_rule():
    chunks = load_chunks(FIXTURE / 'chunks.jsonl')
    result = fix(chunks, read_text(FIXTURE / 'source.md'), Config(min_chars=0, disable=['CL002']))
    assert records(result.chunks) == records(chunks) and not result.changes
