"""Exact source-backed payload cuts after complete preceding quoted body rows."""
from pathlib import Path

import pytest

from ragwrench.lint.fix import fix
from ragwrench.lint.io import load_chunks, read_text
from ragwrench.lint.model import Chunk, Config
from .test_quoted_header_cuts import assert_recovered, records

FIXTURE = Path(__file__).parent / 'fixtures' / 'quoted_later_body_cut'
CONFIG = Config(min_chars=0)


def test_exact_requested_later_body_fixture():
    assert_recovered(load_chunks(FIXTURE / 'chunks.jsonl'), read_text(FIXTURE / 'source.md'))


def pair(source, cut, hints=False):
    chunks = [Chunk('a', source[:cut], {'part': 'left', 'chunklint': {
                  'operation': 'rejoin_table', 'inputs': [
                      {'id': 'a0', 'metadata': {'label': '雪'}}, {'id': 'a1', 'metadata': {'part': 1}}]}}),
              Chunk('b', source[cut:], {'part': 'right', 'chunklint': {
                  'operation': 'rejoin_table', 'inputs': [
                      {'id': 'b0', 'metadata': {'part': 2}}, {'id': 'b1', 'metadata': {'part': 3}}]}})]
    if hints:
        chunks[0].start_hint, chunks[0].end_hint = 0, cut
        chunks[1].start_hint, chunks[1].end_hint = cut, len(source)
    return chunks


@pytest.mark.parametrize('ending', ['\n', '\r\n', '\r'])
@pytest.mark.parametrize('prefix', ['> ', ' > ', '  > ', '   > '])
@pytest.mark.parametrize('preceding', [1, 2, 7, 31])
@pytest.mark.parametrize('body,offset', [
    ('| alpha | b |', 4), ('café 雪 | b', 4), ('| café 雪 | b |', 8),
    ('| a \\| 雪 | b |', 5), ('| a \\| 雪 | b |', 6), ('| a\u2028雪 | b |', 5),
])
def test_later_body_payloads_without_row_count_allowlist(ending, prefix, preceding, body, offset):
    rows = ['| k | v |', '| - | :-: |']
    rows += [f'| row {i} | preceding {i} |' for i in range(preceding)]
    rows += [body, '| final | row |']
    source = ending.join(prefix + row for row in rows) + ending
    cut = len(ending.join(prefix + row for row in rows[:2 + preceding]) + ending + prefix) + offset
    assert_recovered(pair(source, cut, hints=True), source)


@pytest.mark.parametrize('ending', ['\n', '\r\n', '\r'])
@pytest.mark.parametrize('offset', range(3, 15))
def test_later_whitespace_and_pipe_cuts(ending, offset):
    rows = ['> | k | v |', '> | :- | -: |', '> | first | row |', '> |  alpha  | b |\t ']
    source = ending.join(rows) + ending
    cut = len(ending.join(rows[:-1]) + ending) + offset
    assert_recovered(pair(source, cut), source)


def uncertain_cases():
    table = read_text(FIXTURE / 'source.md')
    cut = table.index('alpha') + 2
    left, right = table[:cut], table[cut:]
    yield 'no-source', left, right, None
    for name, old, new in [
        ('nested', '> ', '> > '), ('compact-nested', '> ', '>> '),
        ('tab-prefix', '> ', '>\t'), ('no-space', '> ', '>'),
        ('indented', '> ', '    > '),
        ('mixed-depth', '> | first', '> > | first'),
        ('mixed-prefix', '> | first', ' > | first'),
        ('missing-prefix', '> | first', '| first'),
        ('later-mixed-prefix', '> | al', ' > | al'),
        ('list-body', '> | first', '> - | first'),
        ('ordered-list', '> | first', '> 1. | first'),
        ('invalid-row', '> | first | row |', '> prose'),
        ('invalid-delimiter', '| -- | -- |', '| -x | -- |'),
        ('count-mismatch', '| k | v |', '| k | v | z |'),
        ('blank-body', '> | first', '\n> | first'),
        ('quoted-blank', '> | first', '> \n> | first'),
        ('fenced-body', '> | first | row |', '> ```\n> | first | row |\n> ```'),
    ]:
        a, b = left.replace(old, new), right.replace(old, new)
        yield name, a, b, a + b
    for name, a, b in [
        ('prefix-only', left[:-5], ' | alpha | b |\n'),
        ('no-payload', left[:-4], '| alpha | b |\n'),
        ('row-end', table[:-1], '\n> | next | row |\n'),
        ('crlf-cut', left[:left.rfind('\n')] + '\r', '\n> | alpha | b |\n'),
    ]:
        yield name, a, b, a + b
    for name, src in [
        ('missing-byte', table.replace('alpha', 'alphxa')),
        ('changed-byte', table.replace('alpha', 'alXha')),
        ('changed-space', table.replace('alpha', 'al pha')),
        ('source-prefix-space', table.replace('> | first', ' > | first')),
        ('fragmented', table.replace('alpha', 'al\nMissing.\npha')),
        ('literal-code', '```md\n' + table + '```\n'),
        ('quoted-code', '> ```md\n' + table + '> ```\n'),
        ('same-line-list', '- ' + table),
        ('source-list', '- item\n' + table),
        ('partial-final-row', table.rstrip('\n') + ' extra\n'),
        ('source-extra-row', table + '> | next | row |\n'),
        ('source-earlier-row', '> | before | row |\n' + table),
    ]:
        yield name, left, right, src
    yield 'ambiguous-pipe-suffix', table[:-3], table[-3:], table
    yield 'overlap', left, 'l' + right, table
    yield 'reversal', left, right, right + '\n' + left
    yield 'two-tables', left, right + '\n> | x | z |\n> | - | - |\n', table + '\n> | x | z |\n> | - | - |\n'


@pytest.mark.parametrize('case,left,right,source', list(uncertain_cases()))
def test_later_body_refusals(case, left, right, source):
    chunks = [Chunk('a', left, {'part': 'left'}), Chunk('b', right, {'part': 'right'})]
    result = fix(chunks, source, CONFIG)
    assert records(result.chunks) == records(chunks) and not result.changes, case
    second = fix(result.chunks, source, CONFIG)
    assert records(second.chunks) == records(chunks) and not second.changes


@pytest.mark.parametrize('hints', [False, True])
def test_later_ambiguity_even_with_verified_hints(hints):
    table = read_text(FIXTURE / 'source.md')
    chunks = pair(table, table.index('alpha') + 2, hints)
    result = fix(chunks, table + '\n' + table, CONFIG)
    assert [a.candidates for a in result.before.alignments] == [2, 2]
    assert records(result.chunks) == records(chunks) and not result.changes


def test_surrounding_records_and_exact_offsets():
    table = '  > | k | v |\r\n  > | - | -: |\r\n  > | first | row |\r\n  > | café \\| 雪 | b |\r\n'
    intro, outro = 'Introduction.\r\n\r\n', '\r\nDone.'
    cut = table.index('雪')
    chunks = [Chunk('intro', intro), *pair(table, cut), Chunk('outro', outro)]
    source = intro + table + outro
    result = fix(chunks, source, CONFIG)
    assert [c.text for c in result.chunks] == [intro, table, outro]
    assert result.chunks[1].start_hint == len(intro) + 2
    assert result.chunks[1].end_hint == len(intro + table.rstrip())
    assert len(result.changes) == 1 and not result.after.issues
    assert all(result.invariant.values()) and result.after.stats['source']['coverage'] == 1
    second = fix(result.chunks, source, CONFIG)
    assert records(second.chunks) == records(result.chunks) and not second.changes


def test_disabled_table_rule():
    chunks = load_chunks(FIXTURE / 'chunks.jsonl')
    result = fix(chunks, read_text(FIXTURE / 'source.md'), Config(min_chars=0, disable=['CL002']))
    assert records(result.chunks) == records(chunks) and not result.changes
