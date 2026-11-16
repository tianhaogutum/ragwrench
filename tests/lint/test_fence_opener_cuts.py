"""Exact source-backed variable-length opener cuts; fence contents are opaque."""
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from ragwrench.lint.fix import _source_fence_opener_join, fix
from ragwrench.lint.io import load_chunks, read_text
from ragwrench.lint.model import Chunk, Config

FIXTURE = Path(__file__).parent / 'fixtures' / 'fence_opener_cut'
CONFIG = Config(min_chars=0)
LONG_MARKERS = [char * n for n in (5, 6, 8, 37) for char in ('`', '~')]
ALL_MARKERS = ['```', '~~~', '````', '~~~~', *LONG_MARKERS]


def recovered(chunks, source):
    original = deepcopy(chunks)
    result = fix(chunks, source, CONFIG)
    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert chunk.id == chunks[0].id
    assert chunk.text.encode() == ''.join(c.text for c in chunks).encode()
    assert chunk.text in source
    inputs = []
    for c in chunks:
        prior = c.metadata.get('chunklint', {}).get('inputs')
        inputs.extend(prior if prior is not None else [{'id': c.id, 'metadata': c.metadata}])
    assert chunk.metadata == {**chunks[0].metadata, 'chunklint': {
        'operation': 'rejoin_fence', 'inputs': inputs}}
    assert result.changes == [{'operation': 'rejoin_fence', 'chunk_ids': [c.id for c in chunks],
                               'output_id': chunks[0].id}]
    assert all(i.rule == 'CL007' for i in result.after.issues)
    if chunk.text == source:
        assert not result.after.issues
    assert all(result.invariant.values())
    assert result.after.stats['source']['covered_chars'] == result.before.stats['source']['covered_chars']
    assert Counter(chunk.text) == Counter(''.join(c.text for c in chunks))
    second = fix(result.chunks, source, CONFIG)
    assert second.chunks == result.chunks and not second.changes
    assert chunks == original
    return result


def test_exact_requested_fence_opener_fixture():
    source = read_text(FIXTURE / 'source.md')
    result = recovered(load_chunks(FIXTURE / 'chunks.jsonl'), source)
    assert result.chunks[0].text.encode() == (FIXTURE / 'source.md').read_bytes()
    assert result.before.stats['source']['coverage'] == result.after.stats['source']['coverage'] == 1
    assert any(i.rule == 'CL001' for i in result.before.issues)


@pytest.mark.parametrize('marker', ALL_MARKERS)
@pytest.mark.parametrize('cut', [1, 2])
@pytest.mark.parametrize('ending', ['\n', '\r\n', '\r'])
@pytest.mark.parametrize('indent', ['', '   '])
def test_opaque_raw_payload_and_preamble(marker, cut, ending, indent):
    preamble = ending + '# Café 雪' + ending + 'Example.' + ending
    info = 'python\t title="雪"  ' if marker[0] == '`' else 'lang `literal`  '
    source = preamble + indent + marker + info + ending + ending.join([
        '| k | v |', '| -- | -- |', '| alpha | b |', '# heading', '- list',
        '> ```literal', marker[:-1], 'print("café 雪")\t ', '']) + indent + marker + ' \t' + ending
    offset = len(preamble + indent) + cut
    recovered([Chunk('a', source[:offset], {'part': 1}),
               Chunk('b', source[offset:], {'part': 2})], source)


@pytest.mark.parametrize('marker', ALL_MARKERS)
def test_subslice_verified_hints_and_prior_provenance(marker):
    text = '# Example\n' + marker + 'python\nunique payload\n' + marker + '\n'
    source = 'Earlier paragraph.\n\n' + text + '\nAfter.\n'
    cut = text.index(marker) + 2
    start = source.index(text)
    left = Chunk('a', text[:cut], {'chunklint': {'operation': 'merge_heading_forward',
                 'inputs': [{'id': 'h', 'metadata': {'part': 0}},
                            {'id': 'p', 'metadata': {'part': 1}}]}}, start, start + cut)
    right = Chunk('b', text[cut:], {'part': 2}, start + cut, start + len(text))
    recovered([left, right], source)


@pytest.mark.parametrize('left,right,source', [
    ('# Example\n``', '`python\nx\n```\n', '# Example\n``python\nx\n```\n'),
    ('# Example\n``', "python\nx\n```\n", '# Example\n```python\nx\n```\n'),
    ('# Example\n``', '`python\nx\n', '# Example\n```python\nx\n'),
    ('# Example\n``', '`python\nx\n~~~\n', '# Example\n```python\nx\n~~~\n'),
    ('# Example\n``', '`python\nx\n``\n', '# Example\n```python\nx\n``\n'),
    ('# Example\n``', '`python\nx\n``` extra\n', '# Example\n```python\nx\n``` extra\n'),
    ('# Example\n``', '`py`thon\nx\n```\n', '# Example\n```py`thon\nx\n```\n'),
    ('# Example\n``', '`python\nx\n```\n', '# Example\n```python\nx \n```\n'),
    ('# Example\n``', '`python\nx\n```\n', '# Example\n\n```python\nx\n```\n'),
    ('> > # Example\n> > ``', '`python\n> > x\n> > ```\n', '> > # Example\n> > ```python\n> > x\n> > ```\n'),
    ('- ``', '`python\n  x\n  ```\n', '- ```python\n  x\n  ```\n'),
    ('- item\n  ``', '`python\n  x\n  ```\n', '- item\n  ```python\n  x\n  ```\n'),
    ('    ``', '`python\n    x\n    ```\n', '    ```python\n    x\n    ```\n'),
    ('# Example\n``', '`python\nx\n```\n', '~~~~\n# Example\n```python\nx\n```\n~~~~\n'),
    ('# Example\n``', '`\nafter\n```\nx\n```\n', '```\n# Example\n```\nafter\n```\nx\n```\n'),
    ('  ``', '`python\n  x\n  ```\n', '- earlier item\n\n  ```python\n  x\n  ```\n'),
])
def test_refuses_unsupported_source_opener_reconstruction(left, right, source):
    chunks = [Chunk('a', left, {'part': 1}), Chunk('b', right, {'part': 2})]
    result = fix(chunks, source, CONFIG)
    assert result.chunks == chunks and not result.changes


@pytest.mark.parametrize('hinted', [False, True])
@pytest.mark.parametrize('repeat', ['left', 'right', 'both'])
def test_ambiguous_matches_refused_even_with_verified_hints(hinted, repeat):
    left, right = '# Example\n``', '`python\nunique\n```\n'
    source = left + right + '\n' + {'left': left, 'right': right, 'both': left + right}[repeat]
    chunks = [Chunk('a', left), Chunk('b', right)]
    if hinted:
        chunks[0].start_hint, chunks[0].end_hint = 0, len(left)
        chunks[1].start_hint, chunks[1].end_hint = len(left), len(left + right)
    result = fix(chunks, source, CONFIG)
    assert result.chunks == chunks and not result.changes


def test_no_source_and_disabled_rule_remain_unchanged():
    chunks = load_chunks(FIXTURE / 'chunks.jsonl')
    source = read_text(FIXTURE / 'source.md')
    for actual_source, config in [(None, CONFIG), (source, Config(min_chars=0, disable=['CL001']))]:
        result = fix(chunks, actual_source, config)
        assert result.chunks == chunks and not result.changes


@pytest.mark.parametrize('marker', ALL_MARKERS)
@pytest.mark.parametrize('cut', [4, 8, 18])
def test_info_and_payload_cuts_keep_existing_complete_opener_path(cut, marker):
    source = marker + 'python\nprint("unique")\n' + marker + '\n'
    chunks = [Chunk('a', source[:cut]), Chunk('b', source[cut:])]
    result = fix(chunks, source, CONFIG)
    assert len(result.chunks) == 1 and result.chunks[0].text == source
    assert [c['operation'] for c in result.changes] == ['rejoin_fence']


@pytest.mark.parametrize('marker', ALL_MARKERS)
@pytest.mark.parametrize('suffix', ['', ' \t\r\n'])
def test_longer_matching_closer_and_eof(marker, suffix):
    source = '# Example\n' + marker + 'lang\r\nopaque\r\n' + marker * 2 + suffix
    cut = source.index(marker) + 1
    recovered([Chunk('a', source[:cut]), Chunk('b', source[cut:])], source)


@pytest.mark.parametrize('marker', ALL_MARKERS)
@pytest.mark.parametrize('cut', [0, 1, 5, 8, 15, 23])
def test_new_path_refuses_indentation_info_and_payload_cuts(cut, marker):
    source = '  ' + marker + 'python\nopaque payload\n  ' + marker + '\n'
    chunks = [Chunk('a', source[:cut]), Chunk('b', source[cut:])]
    assert _source_fence_opener_join(chunks, 0, source) is None


def test_exact_requested_four_character_opener_fixture():
    fixture = FIXTURE.with_name('four_fence_opener_cut')
    source = read_text(fixture / 'source.md')
    result = recovered(load_chunks(fixture / 'chunks.jsonl'), source)
    assert result.chunks[0].text.encode() == (fixture / 'source.md').read_bytes()
    assert result.before.stats['source']['coverage'] == result.after.stats['source']['coverage'] == 1
    assert [(i.rule, i.chunk_id) for i in result.before.issues] == [('CL001', 'b')]


@pytest.mark.parametrize('marker', ['````', '~~~~', *LONG_MARKERS])
@pytest.mark.parametrize('cut', [1, 2])
def test_shorter_marker_keeps_both_sides_opaque(marker, cut):
    source = '\n'.join(['# Example', marker + 'python', '| before | table |',
                        '# heading before', marker[:3], '| after | table |',
                        '# heading after', 'print("unique")', marker, ''])
    offset = source.index(marker) + cut
    result = recovered([Chunk('a', source[:offset]), Chunk('b', source[offset:])], source)
    assert not result.after.issues


@pytest.mark.parametrize('marker', ['````', '~~~~', *LONG_MARKERS])
@pytest.mark.parametrize('cut', [1, 2])
@pytest.mark.parametrize('case', [
    'missing-byte', 'missing-closer', 'wrong-closer', 'short-closer', 'invalid-closer',
    'truncated-closer', 'whitespace', 'quote', 'list', 'literal-code', 'source-closer',
    'reversed', 'overlap', 'no-source', 'disabled',
])
def test_variable_length_recovery_refusals(marker, cut, case):
    source = '# Example\n' + marker + 'python\nunique payload\n' + marker + '\n'
    offset = source.index(marker) + cut
    a, b = source[:offset], source[offset:]
    closer_size = len(marker) + 1
    if case == 'missing-byte': b = b[1:]
    elif case == 'missing-closer': b = b[:-closer_size]; source = a + b
    elif case == 'wrong-closer': b = b[:-closer_size] + (('~' if marker[0] == '`' else '`') * len(marker)) + '\n'; source = a + b
    elif case == 'short-closer': b = b[:-closer_size] + marker[:3] + '\n'; source = a + b
    elif case == 'invalid-closer': b = b[:-closer_size] + marker + ' extra\n'; source = a + b
    elif case == 'truncated-closer': b = b[:-2]
    elif case == 'whitespace': source = source.replace('payload', 'payload ')
    elif case == 'quote':
        source = '> > ' + source.replace('\n', '\n> > ').removesuffix('> > ')
        offset = source.index(marker) + cut; a, b = source[:offset], source[offset:]
    elif case == 'list':
        source = '- item\n  ' + marker + 'python\n  unique payload\n  ' + marker + '\n'
        offset = source.index(marker) + cut; a, b = source[:offset], source[offset:]
    elif case == 'literal-code':
        outer = ('~' if marker[0] == '`' else '`') * (len(marker) + 1)
        source = outer + '\n' + source + outer + '\n'
    elif case == 'source-closer':
        a = '# Example\n' + marker[:cut]
        b = marker[cut:] + '\nafter\n' + marker + '\nunique payload\n' + marker + '\n'
        source = marker + '\n' + a + b
    elif case == 'overlap': b = source[offset - 1:]
    if case == 'no-source': source = None
    chunks = [Chunk('a', a, {'part': 1}), Chunk('b', b, {'part': 2})]
    if case == 'reversed': chunks.reverse()
    config = Config(min_chars=0, disable=['CL001']) if case == 'disabled' else CONFIG
    result = fix(chunks, source, config)
    assert result.chunks == chunks and not result.changes
    second = fix(result.chunks, source, config)
    assert second.chunks == chunks and not second.changes


@pytest.mark.parametrize('marker', ['````', '~~~~', *LONG_MARKERS])
@pytest.mark.parametrize('cut', [1, 2])
@pytest.mark.parametrize('repeat', ['left', 'right', 'both'])
@pytest.mark.parametrize('hinted', [False, True])
def test_variable_length_ambiguity_refused_even_with_hints(marker, cut, repeat, hinted):
    text = '# Example\n' + marker + 'python\nunique payload\n' + marker + '\n'
    offset = text.index(marker) + cut
    a, b = text[:offset], text[offset:]
    source = text + '\n' + {'left': a, 'right': b, 'both': text}[repeat]
    chunks = [Chunk('a', a), Chunk('b', b)]
    if hinted:
        chunks[0].start_hint, chunks[0].end_hint = 0, len(a)
        chunks[1].start_hint, chunks[1].end_hint = len(a), len(text)
    result = fix(chunks, source, CONFIG)
    assert result.chunks == chunks and not result.changes


def test_exact_requested_five_character_opener_fixture():
    fixture = FIXTURE.with_name('long_fence_opener_cut')
    source = read_text(fixture / 'source.md')
    result = recovered(load_chunks(fixture / 'chunks.jsonl'), source)
    assert result.chunks[0].text.encode() == (fixture / 'source.md').read_bytes()
    assert not result.before.issues  # Right suffix independently looks complete.
    assert result.before.stats['source']['coverage'] == result.after.stats['source']['coverage'] == 1


@pytest.mark.parametrize('marker', LONG_MARKERS)
@pytest.mark.parametrize('cut', [1, 2])
def test_every_shorter_marker_stays_literal(marker, cut):
    lines = ['# Example', marker + 'python']
    for length in range(3, len(marker)):
        lines.extend(['| before | table |', '| -- | -- |', '# heading before',
                      marker[:length], '| after | table |', '| -- | -- |', '# heading after'])
    text = '\n'.join([*lines, 'unique payload', marker, ''])
    offset = text.index(marker) + cut
    recovered([Chunk('a', text[:offset]), Chunk('b', text[offset:])], text)


@pytest.mark.parametrize('marker', LONG_MARKERS)
@pytest.mark.parametrize('cut', [0, 3, 4, 5, 10, 60])
def test_long_marker_path_refuses_wrong_cuts(marker, cut):
    text = '  ' + marker + 'python\nunique payload\n  ' + marker + '\n'
    offset = 2 + cut
    assert _source_fence_opener_join([Chunk('a', text[:offset]), Chunk('b', text[offset:])], 0, text) is None


def test_unquoted_preamble_guard_retains_prior_contract():
    # Preserve the original raw-indent refusal rule, including space-then-tab
    # prose, while continuing to refuse quoted preambles on unquoted openers.
    a, b = ' \tIntro.\n# Example\n``', '`python\nunique\n```\n'
    recovered([Chunk('a', a), Chunk('b', b)], a + b)
    a = '> Intro.\n# Example\n``'
    chunks = [Chunk('a', a), Chunk('b', b)]
    assert _source_fence_opener_join(chunks, 0, a + b) is None
    result = fix(chunks, a + b, CONFIG)
    assert result.chunks == chunks and not result.changes
