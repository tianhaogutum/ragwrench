"""Explicit blockquote fences, with a fixture emitted by a pinned real splitter."""
from collections import Counter
import json
from pathlib import Path

import pytest

from ragwrench.lint.align import normalized
from ragwrench.lint.fix import fix
from ragwrench.lint.io import load_chunks, read_text
from ragwrench.lint.lint import lint
from ragwrench.lint.markdown import scan
from ragwrench.lint.model import Chunk, Config

FIXTURE = Path(__file__).parent / 'fixtures' / 'quoted_fence'


def records(chunks):
    return [chunk.record() for chunk in chunks]


@pytest.mark.parametrize('prefix', ['> ', '>> ', '> > ', '  >   > ', '>\t> '])
@pytest.mark.parametrize('marker', ['```', '~~~~'])
def test_quoted_fence_reports_original_line_and_hides_code(prefix, marker):
    text = f'Quoted example:\n{prefix}{marker}md\n{prefix}| not | table |\n{prefix}## Heading\n{prefix}- item'
    report = lint([Chunk('q', text)], config=Config(min_chars=0))
    assert [(issue.rule, issue.line) for issue in report.issues] == [('CL001', 2)]
    closed = text + f'\n{prefix}{marker}\t'
    assert not lint([Chunk('q', closed)], config=Config(min_chars=0)).issues


def test_real_splitter_fixture_rejoins_exact_source_and_keeps_metadata():
    source = read_text(FIXTURE / 'source.md')
    chunks = load_chunks(FIXTURE / 'chunks.jsonl')
    provenance = json.loads((FIXTURE / 'provenance.json').read_text())
    assert provenance['packages'] == {'langchain-text-splitters': '0.3.11', 'langchain-core': '0.3.79'}
    assert len(chunks) == provenance['chunks'] == 3
    assert ''.join(chunk.text for chunk in chunks) == source
    config = Config(min_chars=0)
    result = fix(chunks, source, config)
    assert any(issue.rule == 'CL001' and issue.chunk_id == chunks[0].id for issue in result.before.issues)
    assert [chunk.text for chunk in result.chunks] == [source]
    assert [change['operation'] for change in result.changes] == ['rejoin_fence', 'rejoin_fence']
    assert [entry['metadata'] for entry in result.chunks[0].metadata['chunklint']['inputs']] == [
        chunk.metadata for chunk in chunks]
    assert result.after.stats['source']['coverage'] == 1
    assert result.after.issues == []
    second = fix(result.chunks, source, config)
    assert not second.changes
    assert records(second.chunks) == records(result.chunks)


@pytest.mark.parametrize('ending', ['\n', '\r\n', '\r'])
@pytest.mark.parametrize('with_source', [False, True])
def test_quoted_fence_repair_preserves_line_endings(ending, with_source):
    left = f'> > ~~~python{ending}> > x = "雪"{ending}'
    right = f'> > y = "café"{ending}> > ~~~{ending}'
    source = left + right
    result = fix([Chunk('a', left), Chunk('b', right)], source if with_source else None,
                 Config(min_chars=0))
    assert [chunk.text for chunk in result.chunks] == [source]
    assert not result.after.issues


@pytest.mark.parametrize('right', [
    '> ```',                         # Shallower quote is a different container.
    '> > > ```',                     # Deeper marker is literal code.
    '> > ``` trailing',              # Text after a closer is invalid.
    '> > ``',                        # Shorter marker does not close.
    '> > ~~~',                       # Wrong marker character.
    'Outside paragraph.\n> > ```',   # Container ended before the apparent closer.
    '\n> > ```',                     # A bare blank also ends a fenced quote.
])
def test_no_matching_quoted_closer_leaves_input_unchanged(right):
    chunks = [Chunk('a', '> > ```python\n> > payload'), Chunk('b', right)]
    config = Config(min_chars=0, disable=['CL002', 'CL003', 'CL004', 'CL008'])
    result = fix(chunks, config=config)
    assert records(result.chunks) == records(chunks)
    assert not result.changes
    assert any(issue.rule == 'CL001' and issue.chunk_id == 'a' for issue in result.after.issues)
    assert any('no matching closing fence' in entry['reason'] for entry in result.skipped)


def test_quote_boundary_implicitly_ends_fence_and_resumes_top_level_scanning():
    text = '> ```md\n> | code | data |\nOutside.\n| actual | fragment |'
    report = lint([Chunk('q', text)], config=Config(min_chars=0))
    assert [(issue.rule, issue.line) for issue in report.issues] == [('CL002', 4)]


def test_nested_marker_is_literal_inside_quoted_code_and_quote_is_literal_in_top_level_code():
    assert scan('> ```\n> > ```\n> still code').unclosed is not None
    assert scan('```\n> ```\nstill code').unclosed is not None
    assert scan('> ```\n> > ```\n> ```').unclosed is None
    assert scan('```\n> ```\n```').unclosed is None
    assert scan('> ```bad`info\n> content').unclosed is None
    assert scan('>     ```\n> indented code').unclosed is None


@pytest.mark.parametrize('with_source', [False, True])
def test_oversized_quoted_fence_split_repeats_raw_context_and_preserves_payload(with_source):
    prefix, suffix = ' > > ~~~~python\r\n', ' > > ~~~~\r\n'
    rows = [f' > > print("unique_{i}_雪")\r\n' for i in range(15)]
    source = prefix + ''.join(rows) + suffix
    config = Config(min_chars=0, max_chars=110, disable=['CL006'])
    result = fix([Chunk('q', source, {'origin': 'fixture'})], source if with_source else None, config)
    assert len(result.chunks) > 1
    assert all(chunk.text.startswith(prefix) and chunk.text.endswith(suffix) for chunk in result.chunks)
    assert all(len(chunk.text) <= 110 for chunk in result.chunks)
    assert all(scan(chunk.text).unclosed is None for chunk in result.chunks)
    assert ''.join(chunk.text[len(prefix):-len(suffix)] for chunk in result.chunks) == ''.join(rows)
    assert not Counter(normalized(source)) - Counter(normalized(''.join(c.text for c in result.chunks)))
    assert not result.after.issues
    if with_source:
        assert result.after.stats['source']['coverage'] == 1
    assert all(chunk.metadata['chunklint']['parent_metadata'] == {'origin': 'fixture'} for chunk in result.chunks)
    second = fix(result.chunks, source if with_source else None, config)
    assert records(second.chunks) == records(result.chunks)
    assert not second.changes


def test_quoted_fence_rule_disable_and_indivisible_line():
    chunks = [Chunk('a', '> ```\n> payload\n'), Chunk('b', '> ```\n')]
    result = fix(chunks, config=Config(min_chars=0, disable=['CL001', 'CL005']))
    assert records(result.chunks) == records(chunks)
    source = '> ```\n> ' + 'x' * 200 + '\n> ```\n'
    result = fix([Chunk('q', source)], config=Config(min_chars=0, max_chars=60))
    assert result.chunks[0].text == source
    assert [issue.rule for issue in result.after.issues] == ['CL005']
