"""Source recovery may restore row whitespace, but cannot join separate tables."""
import pytest

from ragwrench.lint.fix import fix
from ragwrench.lint.model import Chunk, Config


def records(chunks):
    return [chunk.record() for chunk in chunks]


@pytest.mark.parametrize('ending', ['\n', '\r\n', '\r'])
@pytest.mark.parametrize('delimiter', ['- | :-:', ':-- | --:', '--- | :---:'])
@pytest.mark.parametrize('trimmed', [False, True])
@pytest.mark.parametrize('cut', [1, 2])
def test_source_blank_refuses_unquoted_whole_row_join(ending, delimiter, trimmed, cut):
    lines = ['| k | v |' + ending, '| ' + delimiter + ' |' + ending, '| a | b |' + ending]
    source = ''.join(lines[:cut]) + ending + ''.join(lines[cut:])
    left, right = ''.join(lines[:cut]), ''.join(lines[cut:])
    if trimmed:
        left, right = left.rstrip(), right.rstrip()
    chunks = [Chunk('h', left, {'part': 'header', 'nested': {'label': '雪'}}),
              Chunk('r', right, {'part': 'body'})]
    # Real hints must survive rejection along with text, IDs and metadata.
    chunks[0].start_hint, chunks[0].end_hint = 0, len(left)
    chunks[1].start_hint = source.index(lines[cut].rstrip())
    chunks[1].end_hint = chunks[1].start_hint + len(right)
    original = records(chunks)
    config = Config(min_chars=0)
    result = fix(chunks, source, config)
    assert records(result.chunks) == original
    assert not result.changes
    assert result.skipped == [{
        'operation': 'rejoin_table', 'chunk_ids': ['h', 'r'],
        'reason': 'recovered source does not have a contiguous table boundary'}]
    assert [(i.rule, i.chunk_id, i.line) for i in result.after.issues] == [
        (i.rule, i.chunk_id, i.line) for i in result.before.issues]
    assert ('CL002', 'r', 1) in [(i.rule, i.chunk_id, i.line) for i in result.after.issues]
    assert result.after.stats['source']['coverage'] == 1 and all(result.invariant.values())
    second = fix(result.chunks, source, config)
    assert records(second.chunks) == original and not second.changes


@pytest.mark.parametrize('ending', ['\n', '\r\n', '\r'])
@pytest.mark.parametrize('prefix', ['', '> ', '> > '])
@pytest.mark.parametrize('with_source', [False, True])
def test_contiguous_rows_still_rejoin(ending, prefix, with_source):
    left = prefix + '| k | v |' + ending + prefix + '| - | :-: |' + ending
    right = prefix + '| café \\| 雪 | b |' + ending
    source = left + right
    chunks = [Chunk('h', left, {'part': 'header'}), Chunk('r', right, {'part': 'body'})]
    config = Config(min_chars=0)
    result = fix(chunks, source if with_source else None, config)
    assert [c.text for c in result.chunks] == [source]
    assert [c['operation'] for c in result.changes] == ['rejoin_table']
    assert result.chunks[0].metadata['chunklint']['inputs'] == [
        {'id': c.id, 'metadata': c.metadata} for c in chunks]
    assert not result.after.issues
    if with_source:
        assert result.after.stats['source']['coverage'] == 1
    second = fix(result.chunks, source if with_source else None, config)
    assert records(second.chunks) == records(result.chunks) and not second.changes


@pytest.mark.parametrize('ending', ['\n', '\r\n', '\r'])
@pytest.mark.parametrize('cut_token', ['caf', 'café \\', 'café \\| 雪 | b'])
def test_unquoted_source_backed_cuts_inside_body_rows_still_recover(ending, cut_token):
    header = '| k | v |' + ending + '| - | :-: |' + ending
    row = '| café \\| 雪 | beta |' + ending
    source = header + row
    cut = len(header) + row.index(cut_token) + len(cut_token)
    chunks = [Chunk('h', source[:cut], {'part': 'left'}),
              Chunk('r', source[cut:], {'part': 'right'})]
    config = Config(min_chars=0)
    result = fix(chunks, source, config)
    assert [c.text for c in result.chunks] == [source]
    assert [c['operation'] for c in result.changes] == ['rejoin_table']
    assert not result.after.issues and result.after.stats['source']['coverage'] == 1
    assert all(result.invariant.values())
    second = fix(result.chunks, source, config)
    assert records(second.chunks) == records(result.chunks) and not second.changes


def test_source_blank_rejection_targets_boundary_with_other_table_runs():
    left = '| prior | table |\n| -- | -- |\n\n| k | v |\n| - | :-: |\n'
    right = '| café | 雪 |\n\n| later | table |\n| -- | -- |\n'
    chunks = [Chunk('h', left), Chunk('r', right)]
    result = fix(chunks, left + '\n' + right, Config(min_chars=0))
    assert records(result.chunks) == records(chunks) and not result.changes
    assert result.after.stats['source']['coverage'] == 1
