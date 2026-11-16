from collections import Counter
from pathlib import Path
import random

import pytest

from ragwrench.lint.align import normalized
from ragwrench.lint.fix import fix
from ragwrench.lint.io import load_chunks, read_text
from ragwrench.lint.markdown import scan, valid_table
from ragwrench.lint.model import Chunk, Config

ROOT = Path(__file__).resolve().parents[2]


def records(chunks):
    return [c.record() for c in chunks]


def test_committed_langchain_e2e():
    chunks = load_chunks(ROOT / 'examples/lint/chunks_langchain.jsonl')
    source = read_text(ROOT / 'examples/lint/doc.md')
    result = fix(chunks, source, Config(min_chars=0, max_chars=160))
    assert result.before.stats['defects'] == 6
    assert result.after.stats['defects'] == 0
    assert result.after.stats['source']['coverage'] == 1
    assert result.invariant['source_coverage_preserved']
    assert {'drop_duplicate', 'rejoin_fence', 'rejoin_table', 'merge_heading_forward', 'split_structure'} <= {
        change['operation'] for change in result.changes}
    assert records(fix(result.chunks, source, Config(min_chars=0, max_chars=160)).chunks) == records(result.chunks)


def test_split_fence_reopened_and_closed():
    source = '~~~~python\n' + ''.join(f'print({i})\n' for i in range(30)) + '~~~~\n'
    result = fix([Chunk('code', source)], source, Config(min_chars=0, max_chars=90))
    assert len(result.chunks) > 3
    assert all(scan(c.text).unclosed is None for c in result.chunks)
    assert all(len(c.text) <= 90 for c in result.chunks)
    assert result.after.stats['source']['coverage'] == 1
    assert all(f'print({i})' in ''.join(c.text for c in result.chunks) for i in range(30))
    assert records(fix(result.chunks, source, Config(min_chars=0, max_chars=90)).chunks) == records(result.chunks)


def test_table_rows_repeat_header():
    header = '| Key | Value |\n| --- | --- |\n'
    source = header + ''.join(f'| key{i} | val{i} |\n' for i in range(20))
    result = fix([Chunk('table', source)], source, Config(min_chars=0, max_chars=100))
    assert len(result.chunks) > 2
    assert all(c.text.startswith(header) for c in result.chunks)
    assert all(valid_table(c.text.splitlines()) for c in result.chunks)
    assert all(len(c.text) <= 100 for c in result.chunks)
    assert result.after.stats['source']['missing_chars'] == 0
    assert records(fix(result.chunks, source, Config(min_chars=0, max_chars=100)).chunks) == records(result.chunks)


def test_heading_carried_into_first_table_piece():
    source = 'Previous paragraph.\n\n## Measurements\n| k | v |\n| --- | --- |\n' + ''.join(
        f'| {i} | value{i} |\n' for i in range(30))
    config = Config(min_chars=0, max_chars=100)
    result = fix([Chunk('x', source)], source, config)
    assert not any(i.rule in ('CL003', 'CL005') for i in result.after.issues)
    assert records(fix(result.chunks, source, config).chunks) == records(result.chunks)


def test_repeated_source_occurrence_not_dropped():
    source = 'Repeated paragraph.\n\nRepeated paragraph.\n'
    result = fix([Chunk('a', 'Repeated paragraph.'), Chunk('b', 'Repeated paragraph.')], source)
    assert len(result.chunks) == 2
    assert result.skipped[0]['operation'] == 'drop_duplicate'
    assert result.after.stats['source']['coverage'] == 1


def test_exact_duplicate_dropped_but_near_not_dropped():
    first = 'The long local retrieval paragraph is completely synthetic and reusable.'
    result = fix([Chunk('a', first), Chunk('b', first), Chunk('c', first + '!')])
    assert [c.id for c in result.chunks] == ['a', 'c']
    assert result.changes[0] == {'operation': 'drop_duplicate', 'chunk_ids': ['b'], 'kept': 'a'}


def test_unresolved_long_line_and_missing_header_stay_reported():
    source = '```\n' + 'x' * 300 + '\n```\n'
    result = fix([Chunk('a', source)], source, Config(min_chars=0, max_chars=100))
    assert result.chunks[0].text == source
    assert any(i.rule == 'CL005' for i in result.after.issues)
    result = fix([Chunk('a', '| a | b |')])
    assert any(i.rule == 'CL002' for i in result.after.issues)


def test_no_unsafe_semantic_repairs():
    result = fix([Chunk('a', 'A sentence is'), Chunk('b', 'continued here.')])
    assert len(result.chunks) == 2
    assert any(i.rule == 'CL004' for i in result.after.issues)


def test_disabled_rules_do_not_fix():
    config = Config(disable=['CL001', 'CL003', 'CL005', 'CL006'])
    result = fix([Chunk('a', '```'), Chunk('b', '```'), Chunk('c', '# End')], config=config)
    assert not result.changes


def test_conflicting_metadata_preserved():
    result = fix([Chunk('a', '# Heading', {'secret_key': 'synthetic-a'}),
                  Chunk('b', 'Body.', {'secret_key': 'synthetic-b'})])
    inputs = result.chunks[0].metadata['chunklint']['inputs']
    assert [p['metadata']['secret_key'] for p in inputs] == ['synthetic-a', 'synthetic-b']
    assert 'synthetic-a' not in str(result.record())


def test_generated_ids_do_not_collide():
    source = '```\n' + ''.join(f'line{i}\n' for i in range(30)) + '```\n'
    result = fix([Chunk('a', source), Chunk('a~2', 'Other.')], config=Config(min_chars=0, max_chars=50))
    ids = [c.id for c in result.chunks]
    assert len(set(ids)) == len(ids)
    assert 'a~2' in ids and 'a~3' in ids


@pytest.mark.parametrize('seed', range(40))
def test_property_cut_positions_conserve_text_and_coverage(seed):
    source = '# Title\nA paragraph about offline retrieval.\n\n```py\n' + ''.join(
        f'value_{i} = {i}\n' for i in range(8)) + '```\n\n| k | v |\n| --- | --- |\n| a | b |\n\n## End\nFinished.\n'
    randomizer = random.Random(seed)
    cuts = sorted({0, len(source), *(randomizer.randrange(1, len(source)) for _ in range(5))})
    chunks = [Chunk(f'c{i}', source[a:b]) for i, (a, b) in enumerate(zip(cuts, cuts[1:]))
              if source[a:b].strip()]
    config = Config(min_chars=0, max_chars=120)
    result = fix(chunks, source, config)
    expected = Counter(normalized(source))
    actual = Counter(normalized(''.join(c.text for c in result.chunks)))
    assert not expected - actual
    assert result.after.stats['source']['coverage'] == 1
    assert records(fix(result.chunks, source, config).chunks) == records(result.chunks)


def test_crlf_repair_preserves_payload():
    source = '~~~py\r\nx = 1\r\ny = 2\r\n~~~\r\n'
    result = fix([Chunk('a', '~~~py\r\nx = 1\r\n'), Chunk('b', 'y = 2\r\n~~~\r\n')], source, Config(min_chars=0))
    assert result.chunks[0].text == source
    assert not result.after.issues


def test_no_source_repeated_rows_from_splitting_are_payload():
    header = '| key | value |\n| --- | --- |\n'
    source = header + '| same | data |\n' * 20
    config = Config(min_chars=0, max_chars=65)
    result = fix([Chunk('a', source)], config=config)
    assert sum(c.text.count('| same | data |') for c in result.chunks) == 20
    assert records(fix(result.chunks, config=config).chunks) == records(result.chunks)


def test_source_recovers_break_in_code_token():
    source = '```py\nprint("complete")\n```\n'
    result = fix([Chunk('a', '```py\npri'), Chunk('b', 'nt("complete")\n```\n')], source,
                 Config(min_chars=0))
    assert result.chunks[0].text == source
    assert result.chunks[0].record()['start_char_idx'] == 0
    assert result.after.stats['defects'] == 0


@pytest.mark.parametrize('seed', range(120))
def test_randomized_real_fixture_cuts_and_limits(seed):
    source = read_text(ROOT / 'examples/lint/doc.md')
    rng = random.Random(seed)
    cuts = sorted({0, len(source), *(rng.randrange(1, len(source)) for _ in range(rng.randrange(1, 12)))})
    chunks = [Chunk(str(i), source[a:b]) for i, (a, b) in enumerate(zip(cuts, cuts[1:])) if source[a:b].strip()]
    config = Config(min_chars=0, max_chars=rng.randrange(30, 250))
    result = fix(chunks, source, config)
    assert result.after.stats['source']['coverage'] == 1
    assert records(fix(result.chunks, source, config).chunks) == records(result.chunks)


def test_unclosed_eof_fence_does_not_swallow_following_paragraphs():
    chunks = [Chunk('a', '````py\nunfinished'), Chunk('b', 'A real paragraph.'), Chunk('c', '```')]
    result = fix(chunks, config=Config(min_chars=0))
    assert records(result.chunks) == records(chunks)
    assert any('no matching closing fence' in entry['reason'] for entry in result.skipped)
    assert any(issue.rule == 'CL001' for issue in result.after.issues)
