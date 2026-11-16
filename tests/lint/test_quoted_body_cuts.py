"""Exact cuts in the first body row of one consistently quoted source table."""
from pathlib import Path

import pytest

from ragwrench.lint.fix import fix
from ragwrench.lint.io import load_chunks, read_text
from ragwrench.lint.lint import lint
from ragwrench.lint.model import Chunk, Config
from .test_quoted_header_cuts import assert_recovered, records

FIXTURE = Path(__file__).parent / 'fixtures' / 'quoted_body_cut'
CONFIG = Config(min_chars=0)


def test_exact_requested_quoted_body_fixture():
    source = read_text(FIXTURE / 'source.md')
    assert not lint([Chunk('whole', source)], source, CONFIG).issues
    chunks = load_chunks(FIXTURE / 'chunks.jsonl')
    assert [(a.status, a.candidates) for a in lint(chunks, source, CONFIG).alignments] == [
        ('aligned', 1), ('aligned', 1)]
    assert_recovered(chunks, source)


@pytest.mark.parametrize('ending', ['\n', '\r\n', '\r'])
@pytest.mark.parametrize('prefix', ['> ', ' > ', '  > ', '   > '])
@pytest.mark.parametrize('delimiter', ['| -- | -- |', '| - | :-: |', '| :- | -: |'])
@pytest.mark.parametrize('body,cut_in_row', [
    ('| alpha | b |', 4), ('café 雪 | b', 4), ('| café 雪 | b |', 8),
    ('| a \\| 雪 | b |', 5), ('| a \\| 雪 | b |', 6), ('| a\u2028雪 | b |', 5),
])
def test_exact_first_body_cell_cuts(ending, prefix, delimiter, body, cut_in_row):
    rows = ['| k | v |', delimiter, body, '| next | row |']
    source = ending.join(prefix + row for row in rows) + ending
    cut = len(ending.join(prefix + row for row in rows[:2]) + ending + prefix) + cut_in_row
    chunks = [Chunk('h', source[:cut], {'part': 'left', 'chunklint': {
                  'operation': 'rejoin_table', 'inputs': [
                      {'id': 'h0', 'metadata': {'label': '雪'}}, {'id': 'h1', 'metadata': {'part': 1}}]}}),
              Chunk('r', source[cut:], {'part': 'right', 'chunklint': {
                  'operation': 'rejoin_table', 'inputs': [
                      {'id': 'r0', 'metadata': {'part': 2}}, {'id': 'r1', 'metadata': {'part': 3}}]}})]
    chunks[0].start_hint, chunks[0].end_hint = 0, cut
    chunks[1].start_hint, chunks[1].end_hint = cut, len(source)
    assert_recovered(chunks, source)


@pytest.mark.parametrize('ending', ['\n', '\r\n', '\r'])
@pytest.mark.parametrize('cut_in_row', [3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
def test_body_whitespace_and_pipe_cuts_retain_raw_bytes(ending, cut_in_row):
    rows = ['> | k | v |', '> | - | -: |', '> |  alpha  | b |\t ']
    source = ending.join(rows) + ending
    cut = len(ending.join(rows[:2]) + ending) + cut_in_row
    assert_recovered([Chunk('h', source[:cut]), Chunk('r', source[cut:])], source)


@pytest.mark.parametrize('case,left,right,source', [
    ('no-source', '> | k | v |\n> | -- | -- |\n> | al', 'pha | b |\n', None),
    ('nested', '> > | k | v |\n> > | -- | -- |\n> > | al', 'pha | b |\n', ''),
    ('compact-nested', '>> | k | v |\n>> | -- | -- |\n>> | al', 'pha | b |\n', ''),
    ('mixed-depth', '> | k | v |\n> > | -- | -- |\n> | al', 'pha | b |\n', ''),
    ('missing-header-prefix', '| k | v |\n> | -- | -- |\n> | al', 'pha | b |\n', ''),
    ('missing-delimiter-prefix', '> | k | v |\n| -- | -- |\n> | al', 'pha | b |\n', ''),
    ('missing-body-prefix', '> | k | v |\n> | -- | -- |\n| al', 'pha | b |\n', ''),
    ('missing-later-prefix', '> | k | v |\n> | -- | -- |\n> | al', 'pha | b |\n| next | row |\n', ''),
    ('changed-prefix', '> | k | v |\n > | -- | -- |\n> | al', 'pha | b |\n', ''),
    ('changed-body-prefix', '> | k | v |\n> | -- | -- |\n > | al', 'pha | b |\n', ''),
    ('changed-later-prefix', '> | k | v |\n> | -- | -- |\n> | al', 'pha | b |\n > | next | row |\n', ''),
    ('no-space', '>| k | v |\n>| -- | -- |\n>| al', 'pha | b |\n', ''),
    ('tab-prefix', '>\t| k | v |\n>\t| -- | -- |\n>\t| al', 'pha | b |\n', ''),
    ('indented-code', '    > | k | v |\n    > | -- | -- |\n    > | al', 'pha | b |\n', ''),
    ('list-header', '> - | k | v |\n> | -- | -- |\n> | al', 'pha | b |\n', ''),
    ('ordered-list', '> 1. | k | v |\n> | -- | -- |\n> | al', 'pha | b |\n', ''),
    ('list-body', '> | k | v |\n> | -- | -- |\n> - | al', 'pha | b |\n', ''),
    ('fence-boundary', '> | k | v |\n> | -- | -- |\n> | al', 'pha | b |\n> ```\n> | next | row |\n> ```\n', ''),
    ('unquoted-preamble', 'Intro.\n> | k | v |\n> | -- | -- |\n> | al', 'pha | b |\n', ''),
    ('quoted-preamble', '> Intro.\n> | k | v |\n> | -- | -- |\n> | al', 'pha | b |\n', ''),
    ('blank-before-delimiter', '> | k | v |\n> \n> | -- | -- |\n> | al', 'pha | b |\n', ''),
    ('blank-before-body', '> | k | v |\n> | -- | -- |\n\n> | al', 'pha | b |\n', ''),
    ('quoted-blank', '> | k | v |\n> | -- | -- |\n> \n> | al', 'pha | b |\n', ''),
    ('two-tables', '> | k | v |\n> | -- | -- |\n> | al', 'pha | b |\n\n> | x | z |\n> | - | - |\n', ''),
    ('invalid-delimiter', '> | k | v |\n> | -x | -- |\n> | al', 'pha | b |\n', ''),
    ('colon-only', '> | k | v |\n> | :: | -- |\n> | al', 'pha | b |\n', ''),
    ('empty-cell', '> | k | v |\n> |  | -- |\n> | al', 'pha | b |\n', ''),
    ('internal-space', '> | k | v |\n> | - - | -- |\n> | al', 'pha | b |\n', ''),
    ('count-mismatch', '> | k | v | z |\n> | -- | -- |\n> | al', 'pha | b |\n', ''),
    ('missing-delimiter', '> | k | v |\n> | al', 'pha | b |\n', ''),
    ('prefix-cut', '> | k | v |\n> | -- | -- |\n>', ' | alpha | b |\n', ''),
    ('no-body-payload', '> | k | v |\n> | -- | -- |\n> ', '| alpha | b |\n', ''),
    ('cut-in-line-ending', '> | k | v |\n> | -- | -- |\r', '\n> | alpha | b |\n', ''),
    ('cut-at-body-end', '> | k | v |\n> | -- | -- |\n> | alpha | b |', '\n> | next | row |\n', ''),
    ('missing-character', '> | k | v |\n> | -- | -- |\n> | al', 'ha | b |\n',
     '> | k | v |\n> | -- | -- |\n> | alpha | b |\n'),
    ('source-prefix-whitespace', '> | k | v |\n> | -- | -- |\n> | al', 'pha | b |\n',
     '> | k | v |\n> | -- | -- |\n > | alpha | b |\n'),
    ('source-blank-in-body', '> | k | v |\n> | -- | -- |\n> | al', 'pha | b |\n',
     '> | k | v |\n> | -- | -- |\n> | al\n\npha | b |\n'),
    ('source-blank-before-body', '> | k | v |\n> | -- | -- |\n> | al', 'pha | b |\n',
     '> | k | v |\n> | -- | -- |\n\n> | alpha | b |\n'),
    ('overlapping', '> | k | v |\n> | -- | -- |\n> | al', 'lpha | b |\n',
     '> | k | v |\n> | -- | -- |\n> | alpha | b |\n'),
    ('reversed', '> | k | v |\n> | -- | -- |\n> | al', 'pha | b |\n',
     'pha | b |\n\n> | k | v |\n> | -- | -- |\n> | al'),
    ('fragmented', '> | k | v |\n> | -- | -- |\n> | al', 'pha | b |\n> | next | row |\n',
     '> | k | v |\n> | -- | -- |\n> | alpha | b |\nMissing.\n> | next | row |\n'),
])
def test_refuses_uncertain_body_boundaries(case, left, right, source):
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


def test_surrounding_chunks_and_offset_hints():
    table = '  > | k | v |\r\n  > | :-: | - |\r\n  > | café \\| 雪 | b |\r\n'
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
    assert all(result.invariant.values()) and result.after.stats['source']['coverage'] == 1
    second = fix(result.chunks, source, CONFIG)
    assert records(second.chunks) == records(result.chunks) and not second.changes


def test_disabled_rule():
    chunks = load_chunks(FIXTURE / 'chunks.jsonl')
    result = fix(chunks, read_text(FIXTURE / 'source.md'), Config(min_chars=0, disable=['CL002']))
    assert records(result.chunks) == records(chunks) and not result.changes
