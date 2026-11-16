"""Explicit quoted pipe-table contracts; fixtures need no LangChain at runtime."""
import json
from pathlib import Path

import pytest

from ragwrench.lint.fix import fix
from ragwrench.lint.io import load_chunks, read_text
from ragwrench.lint.lint import lint
from ragwrench.lint.markdown import scan, table_runs, valid_table
from ragwrench.lint.model import Chunk, Config

FIXTURE = Path(__file__).parent / 'fixtures' / 'quoted_table'
HEADER = '> > | Key | Value |\n>> | :--- | ---: |\n'
ROW = ' > > | café | 雪 |\n'


def records(chunks):
    return [chunk.record() for chunk in chunks]


@pytest.mark.parametrize('prefix', ['> ', '>> ', '> > ', '  >   > ', '>\t> '])
@pytest.mark.parametrize('outer_pipes', [False, True])
def test_quoted_header_recognition_and_original_fragment_line(prefix, outer_pipes):
    lines = ['key | value', ':--- | ---:', 'café | 雪']
    text = '\n'.join(prefix + ('| ' + line + ' |' if outer_pipes else line) for line in lines)
    assert valid_table(scan(text).lines)
    assert not lint([Chunk('q', text)], config=Config(min_chars=0)).issues
    fragment = 'Intro.\n' + prefix + '| café | 雪 |'
    assert [(issue.rule, issue.line) for issue in lint(
        [Chunk('q', fragment)], config=Config(min_chars=0)).issues] == [('CL002', 2)]


@pytest.mark.parametrize('with_source', [False, True])
def test_pinned_splitter_fixture_rejoins_exact_raw_table(with_source):
    source = read_text(FIXTURE / 'source.md')
    chunks = load_chunks(FIXTURE / 'chunks.jsonl')
    provenance = json.loads((FIXTURE / 'provenance.json').read_text())
    assert provenance['packages'] == {'langchain-text-splitters': '0.3.11', 'langchain-core': '0.3.79'}
    assert len(chunks) == provenance['chunks'] == 5
    assert ''.join(chunk.text for chunk in chunks) == source
    config = Config(min_chars=0)
    result = fix(chunks, source if with_source else None, config)
    assert [(i.rule, i.line) for i in result.before.issues] == [('CL002', 1)] * 5
    assert [chunk.text for chunk in result.chunks] == [source]
    assert [c['operation'] for c in result.changes] == ['rejoin_table'] * 4
    assert [entry['metadata'] for entry in result.chunks[0].metadata['chunklint']['inputs']] == [
        chunk.metadata for chunk in chunks]
    assert not result.after.issues
    if with_source:
        assert result.after.stats['source']['coverage'] == 1
    second = fix(result.chunks, source if with_source else None, config)
    assert records(second.chunks) == records(result.chunks) and not second.changes


@pytest.mark.parametrize('ending', ['\n', '\r\n', '\r'])
@pytest.mark.parametrize('with_source', [False, True])
@pytest.mark.parametrize('cut', [1, 2])
def test_header_delimiter_and_body_boundaries_preserve_raw_lines(ending, with_source, cut):
    lines = [line + ending for line in (HEADER + ROW).splitlines()]
    chunks = [Chunk('a', ''.join(lines[:cut])), Chunk('b', ''.join(lines[cut:]))]
    source = ''.join(lines)
    result = fix(chunks, source if with_source else None, Config(min_chars=0))
    assert [chunk.text for chunk in result.chunks] == [source]
    assert not result.after.issues


@pytest.mark.parametrize('left,right', [
    (HEADER, '> | shallow | row |\n'),
    (HEADER, '> > > | deeper | row |\n'),
    (HEADER, '| unquoted | row |\n'),
    ('> > | key | value |\n', '> | --- | --- |\n'),
    ('> > | key | value |\n', '> > | --- |\n'),
    (HEADER, '\n' + ROW),
    (HEADER, '> >\n' + ROW),
    (HEADER + '\n', ROW),
    (HEADER + '> >\n', ROW),
    (HEADER, '> > Paragraph.\n' + ROW),
    (HEADER, '> > ```md\n' + ROW + '> > ```\n'),
    (HEADER, HEADER + ROW),
    (ROW, ROW.replace('café', 'other')),
])
def test_boundaries_do_not_guess_a_table_join(left, right):
    chunks = [Chunk('a', left), Chunk('b', right)]
    result = fix(chunks, config=Config(min_chars=0))
    assert records(result.chunks) == records(chunks) and not result.changes


def test_depth_transition_ends_runs_and_cannot_validate_header():
    text = HEADER + '> | shallow | row |\n' + HEADER
    assert table_runs(scan(text)) == [(0, 2), (2, 3), (3, 5)]
    assert not valid_table(['> | key | value |', '> > | --- | --- |'])


def test_source_blank_boundary_does_not_merge_trimmed_fragments():
    chunks = [Chunk('a', HEADER.rstrip()), Chunk('b', ROW.rstrip())]
    source = HEADER + '\n' + ROW
    result = fix(chunks, source, Config(min_chars=0))
    assert records(result.chunks) == records(chunks) and not result.changes
    assert result.after.stats['source']['coverage'] == 1
    assert any(i.rule == 'CL002' for i in result.after.issues)


def test_quoted_fence_is_opaque_and_escaped_pipe_is_not_a_row():
    text = '> > ```md\n' + HEADER + ROW + '> > ```\n'
    assert table_runs(scan(text)) == []
    assert not lint([Chunk('q', text)], config=Config(min_chars=0)).issues
    assert table_runs(scan('> > escaped \\| pipe')) == []


@pytest.mark.parametrize('with_source', [False, True])
def test_oversized_table_repeats_raw_header_and_keeps_each_body_row(with_source):
    rows = [f' >  > | key{i} | 雪{i} |\r\n' for i in range(12)]
    header = HEADER.replace('\n', '\r\n')
    source = header + ''.join(rows)
    config = Config(min_chars=0, max_chars=100, disable=['CL006'])
    result = fix([Chunk('q', source, {'origin': 'fixture'})], source if with_source else None, config)
    assert len(result.chunks) > 1
    assert all(chunk.text.startswith(header) and len(chunk.text) <= 100 for chunk in result.chunks)
    assert ''.join(chunk.text[len(header):] for chunk in result.chunks) == ''.join(rows)
    assert not result.after.issues
    if with_source:
        assert result.after.stats['source']['coverage'] == 1
    assert all(chunk.metadata['chunklint']['parent_metadata'] == {'origin': 'fixture'} for chunk in result.chunks)
    second = fix(result.chunks, source if with_source else None, config)
    assert records(second.chunks) == records(result.chunks) and not second.changes


def test_table_disable_and_indivisible_row():
    chunks = [Chunk('a', HEADER), Chunk('b', ROW)]
    disabled = fix(chunks, config=Config(min_chars=0, disable=['CL002', 'CL005']))
    assert records(disabled.chunks) == records(chunks) and not disabled.changes
    source = HEADER + '> > | ' + 'x' * 200 + ' | value |\n'
    result = fix([Chunk('q', source)], config=Config(min_chars=0, max_chars=70))
    assert result.chunks[0].text == source
    assert [i.rule for i in result.after.issues] == ['CL005']
