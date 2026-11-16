"""Exact source-backed single-level quoted fence opener cuts."""
from copy import deepcopy
from pathlib import Path

import pytest

from ragwrench.lint.fix import _source_fence_opener_join, fix
from ragwrench.lint.lint import lint
from ragwrench.lint.model import Chunk, Config
from ragwrench.lint.io import load_chunks, read_text
from .test_fence_opener_cuts import recovered

FIXTURE = Path(__file__).parent / 'fixtures' / 'quoted_fence_opener_cut'
MARKERS = [char * n for n in (3, 4, 5, 8) for char in ('`', '~')]
CONFIG = Config(min_chars=0)


def test_exact_requested_quoted_opener_fixture():
    source = read_text(FIXTURE / 'source.md')
    result = recovered(load_chunks(FIXTURE / 'chunks.jsonl'), source)
    assert result.chunks[0].text.encode() == (FIXTURE / 'source.md').read_bytes()
    assert [(i.rule, i.chunk_id) for i in result.before.issues] == [('CL001', 'b')]
    assert result.before.stats['source']['coverage'] == result.after.stats['source']['coverage'] == 1


def sample(marker='```', cut=2, prefix='> ', ending='\n', indent=''):
    info = 'python\t title="雪"  ' if marker[0] == '`' else 'lang `literal`  '
    rows = ['# Café 雪', '', 'Example.', indent + marker + info,
            '| before | table |', '| -- | -- |', '# heading before', '- literal list']
    for length in range(1, len(marker)):
        rows.extend([marker[:length], '| after | table |', '| -- | -- |', '# heading after'])
    # An additional > inside the active fence is literal payload, not a container.
    rows.extend(['> ```literal', 'print("雪\u2028café")\t ', indent + marker + ' \t'])
    text = ending.join(prefix + row for row in rows) + ending
    offset = text.index(marker) + cut
    return text[:offset], text[offset:], text


@pytest.mark.parametrize('marker', MARKERS)
@pytest.mark.parametrize('cut', [1, 2])
@pytest.mark.parametrize('ending', ['\n', '\r\n', '\r'])
@pytest.mark.parametrize('prefix', ['> ', ' > ', '  > ', '   > '])
@pytest.mark.parametrize('indent', ['', '   '])
def test_consistent_prefix_preserves_opaque_raw_code(marker, cut, ending, prefix, indent):
    a, b, source = sample(marker, cut, prefix, ending, indent)
    assert not lint([Chunk('whole', source)], source, CONFIG).issues
    recovered([Chunk('a', a, {'part': 'left'}), Chunk('b', b, {'part': 'right'})], source)


@pytest.mark.parametrize('marker', MARKERS)
@pytest.mark.parametrize('hinted', [False, True])
def test_source_subslice_prior_provenance_and_longer_closer(marker, hinted):
    a = '> # Example\r\n> ' + marker[:2]
    b = marker[2:] + 'python\r\n> unique\r\n> ' + marker * 2
    source = 'Earlier paragraph.\r\n\r\n' + a + b + '\r\nAfter.\r\n'
    chunks = [Chunk('a', a, {'part': 'left', 'chunklint': {'inputs': [
                  {'id': 'a0', 'metadata': {'label': '雪'}}, {'id': 'a1', 'metadata': {}}]}}),
              Chunk('b', b, {'part': 'right', 'chunklint': {'inputs': [
                  {'id': 'b0', 'metadata': {}}, {'id': 'b1', 'metadata': {'part': 2}}]}})]
    if hinted:
        start = source.index(a)
        chunks[0].start_hint, chunks[0].end_hint = start, start + len(a)
        chunks[1].start_hint, chunks[1].end_hint = start + len(a), start + len(a+b)
    recovered(chunks, source)


def refusal_cases(marker='```', cut=2):
    a = '> # Example\n> ' + marker[:cut]
    b = marker[cut:] + 'python\n> unique payload\n> ' + marker + '\n'
    text = a + b
    closer = '> ' + marker + '\n'
    yield 'no-source', a, b, None
    yield 'missing-marker-byte', a, b[1:], text
    yield 'changed-byte', a, b.replace('unique', 'changed'), text
    yield 'changed-whitespace', a, b, text.replace('payload', 'payload ')
    yield 'missing-prefix-byte', a, b.replace('> unique', 'unique'), text
    yield 'missing-closer', a, b[:-len(closer)], text[:-len(closer)]
    for name, tail in [('short-closer', marker[:-1]), ('wrong-closer', ('~' if marker[0]=='`' else '`') * len(marker)),
                       ('invalid-closer', marker + ' extra')]:
        right = b[:-len(closer)] + '> ' + tail + '\n'
        yield name, a, right, a + right
    yield 'truncated-closer', a, b[:-2], text
    if marker[0] == '`':
        right = b.replace('python', 'py`thon')
        yield 'invalid-info', a, right, a + right
    for name, transform in [
        ('nested', lambda s:s.replace('> ', '> > ')),
        ('compact-nested', lambda s:s.replace('> ', '>> ')),
        ('no-space', lambda s:s.replace('> ', '>')),
        ('tab-prefix', lambda s:s.replace('> ', '>\t')),
        ('indented-code', lambda s:s.replace('> ', '    > ')),
        ('unquoted-preamble', lambda s:s.replace('> # Example', '# Example')),
        ('mixed-preamble', lambda s:s.replace('> # Example', ' > # Example')),
        ('mixed-payload', lambda s:s.replace('> unique', ' > unique')),
        ('mixed-closer', lambda s:s.replace(closer, ' ' + closer)),
        ('missing-payload-prefix', lambda s:s.replace('> unique', 'unique')),
        ('missing-closer-prefix', lambda s:s.replace(closer, marker+'\n')),
        ('list-preamble', lambda s:s.replace('> # Example', '> - item')),
        ('ordered-list-preamble', lambda s:s.replace('> # Example', '> 1. item')),
        ('list-opener', lambda s:s.replace('> '+marker, '> - '+marker, 1)),
        ('nested-opener', lambda s:s.replace('> '+marker, '> > '+marker, 1)),
    ]:
        changed = transform(text); offset = changed.index(marker) + cut
        yield name, changed[:offset], changed[offset:], changed
    outer = ('~' if marker[0] == '`' else '`') * (len(marker) + 1)
    yield 'literal-code', a, b, '> ' + outer + '\n' + text + '> ' + outer + '\n'
    yield 'unquoted-literal-code', a, b, outer + '\n' + text + outer + '\n'
    closer_b = marker[cut:] + '\n> after\n> ' + marker + '\n> unique\n> ' + marker + '\n'
    yield 'source-closer', a, closer_b, '> ' + marker + '\n' + a + closer_b
    yield 'source-list', a, b, '> - earlier\n' + text
    yield 'source-unquoted-list', a, b, '- earlier\n' + text
    yield 'not-line-start', a, b, 'prefix ' + text
    yield 'overlap', a, text[len(a)-1:], text
    yield 'reversal', a, b, b + '\n' + a


@pytest.mark.parametrize('marker', MARKERS)
@pytest.mark.parametrize('cut', [1, 2])
def test_refuses_unsupported_or_unproven_boundaries(marker, cut):
    for case, a, b, source in refusal_cases(marker, cut):
        chunks = [Chunk('a', a, {'part': 'left'}), Chunk('b', b, {'part': 'right'})]
        original = deepcopy(chunks)
        result = fix(chunks, source, CONFIG)
        assert result.chunks == original and not result.changes, case
        second = fix(result.chunks, source, CONFIG)
        assert second.chunks == original and not second.changes, case


@pytest.mark.parametrize('marker', MARKERS)
@pytest.mark.parametrize('repeat', ['left', 'right', 'both'])
@pytest.mark.parametrize('hinted', [False, True])
def test_ambiguity_even_with_verified_hints(marker, repeat, hinted):
    a = '> # Example\n> ' + marker[:2]
    b = marker[2:] + 'python\n> unique\n> ' + marker + '\n'
    source = a+b+'\n'+{'left':a, 'right':b, 'both':a+b}[repeat]
    chunks = [Chunk('a', a), Chunk('b', b)]
    if hinted:
        chunks[0].start_hint, chunks[0].end_hint = 0, len(a)
        chunks[1].start_hint, chunks[1].end_hint = len(a), len(a+b)
    r = fix(chunks, source, CONFIG)
    assert r.chunks == chunks and not r.changes


@pytest.mark.parametrize('marker', MARKERS)
@pytest.mark.parametrize('offset', [0, 1, 2, 5, 13, 16, 25])
def test_wrong_prefix_info_payload_cuts_do_not_use_new_path(marker, offset):
    source = '> ' + marker + 'python\n> unique\n> ' + marker + '\n'
    assert _source_fence_opener_join([Chunk('a', source[:offset]), Chunk('b', source[offset:])], 0, source) is None


def test_disabled_cl001_remains_unchanged():
    chunks = load_chunks(FIXTURE/'chunks.jsonl')
    r = fix(chunks, read_text(FIXTURE/'source.md'), Config(min_chars=0, disable=['CL001']))
    assert r.chunks == chunks and not r.changes
