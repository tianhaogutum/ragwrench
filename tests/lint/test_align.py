from ragwrench.lint.align import align_chunks, coverage
from ragwrench.lint.lint import lint
from ragwrench.lint.model import Chunk, Config


def test_stripped_whitespace_and_unicode_spans():
    source = '# Title\r\n\r\nA\tB\u00a0中文。\r\n'
    chunk = Chunk('c', '# Title\nAB中文。')
    aligned = align_chunks([chunk], source)[0]
    assert aligned.status == 'aligned'
    span = aligned.spans[0]
    assert (span.start_char, span.end_char) == (0, source.index('。') + 1)
    assert (span.start_line, span.end_line) == (1, 3)
    assert coverage(source, [aligned])['coverage'] == 1


def test_overlapping_chunks():
    source = 'Alpha Beta Gamma Delta.'
    aligned = align_chunks([Chunk('a', 'Alpha Beta Gamma'), Chunk('b', 'Beta Gamma Delta.')], source)
    assert [a.status for a in aligned] == ['aligned', 'aligned']
    assert aligned[1].spans[0].start_char == 6
    assert coverage(source, aligned)['missing_chars'] == 0


def test_verified_llamaindex_hints_and_repeated_occurrences():
    source = 'same text\nsame text\n'
    chunks = [Chunk('a', 'same text', start_hint=10, end_hint=19)]
    assert align_chunks(chunks, source)[0].spans[0].start_char == 10
    # Invalid hints are never trusted.
    chunks[0].start_hint, chunks[0].end_hint = 999, 1000
    assert align_chunks(chunks, source)[0].spans[0].start_char == 0
    aligned = align_chunks([Chunk('a', 'same text'), Chunk('b', 'same text')], source)
    assert [a.spans[0].start_char for a in aligned] == [0, 10]
    assert all(a.candidates == 2 for a in aligned)


def test_no_match_reports_rate_instead_of_crash():
    report = lint([Chunk('a', 'totally unrelated')], 'Original source.', Config(min_chars=0))
    assert report.stats['alignment_failure_rate'] == 1
    assert report.stats['source']['coverage'] == 0
    assert {i.details['kind'] for i in report.issues if i.rule == 'CL007'} == {'alignment', 'coverage'}


def test_partial_and_noncontiguous_lines():
    source = 'first line\nsecond line\nthird line\n'
    aligned = align_chunks([Chunk('a', 'first line\nthird line'), Chunk('b', 'second line\nunmatched')], source)
    assert [a.status for a in aligned] == ['fragmented', 'partial']
    assert aligned[1].matched_chars < aligned[1].total_chars
    assert coverage(source, aligned)['coverage'] == 1


def test_coverage_gaps_do_not_include_whitespace_loss():
    source = 'first\n\nmissing\n\nlast'
    result = coverage(source, align_chunks([Chunk('a', 'first'), Chunk('b', 'last')], source))
    assert result['missing_chars'] == 7
    assert result['gaps'] == [{'start_char': 5, 'end_char': 16}]


def test_cr_only_lines():
    aligned = align_chunks([Chunk('a', 'second')], 'first\rsecond\r')[0]
    assert aligned.spans[0].start_line == 2
