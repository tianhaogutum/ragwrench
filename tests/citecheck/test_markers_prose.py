import pytest

from ragwrench.citecheck.markers import mask_markers, parse_markers
from ragwrench.citecheck.prose import mask_markdown, sentences


@pytest.mark.parametrize("marker,indices", [
    ("[1]", (1,)), ("[1,2]", (1, 2)), ("[1, 2,1]", (1, 2)), ("【1】", (1,)),
    ("【1,2】", (1, 2)), ("Source 1", (1,)), ("source 2", (2,)),
    ("##0$$", (1,)), ("##10$$", (11,)), ("[ID:0]", (1,)),
    ("[999999999]", (999999999,)),
])
def test_markers_offsets_and_masking(marker, indices):
    text = f"A claim {marker} ends here."
    found = parse_markers(text)
    assert len(found) == 1
    assert found[0].indices == indices and found[0].error is None
    assert text[found[0].start:found[0].end] == marker
    masked = mask_markers(text, found)
    assert len(masked) == len(text)
    assert marker not in masked


@pytest.mark.parametrize("marker", ["[abc]", "[1,]", "[1;2]", "[1", "【1", "[0]", "[-1]", "[1000000000]", "##x$$", "##0$", "Source x", "[1 2]", "[]", "[1a]", "[ID:-1]"])
def test_malformed_is_observable(marker):
    found = parse_markers(marker)
    assert found and found[0].error and not found[0].indices


def test_group_and_adjacent():
    assert [m.indices for m in parse_markers("[1,2][3]【4】##4$$")] == [(1, 2), (3,), (4,), (5,)]


def test_markdown_mask_preserves_offsets_and_ignores_code_links():
    text = '# Heading [9]\n~~~py\n"[88]"\n~~~\nThe [archive](https://x.invalid/a) uses `f[77]` locally [1].\n    code[66]\n'
    masked = mask_markdown(text)
    assert len(masked) == len(text)
    assert [m.indices for m in parse_markers(masked)] == [(1,)]
    assert masked.index("archive") == text.index("archive")
    assert "archive" in masked


@pytest.mark.parametrize("code", [
    "``use `x` [9]``",
    "`use ``x`` [9]`",
    "``use ```x``` [9]``",
    "````use ```x``` [9]````",
    "``use `x`\n[9]``",
    "``use `x`\r\n[9]``",
    "``use \\`x\\` [9]``",
])
def test_inline_code_matches_whole_equal_length_runs(code):
    prefix, suffix = "说明🙂 syntax is ", " [1]."
    text = prefix + code + suffix
    masked = mask_markdown(text)
    assert masked == prefix + " " * len(code) + suffix
    marker, = parse_markers(masked)
    assert marker.indices == (1,)
    assert (marker.start, marker.end) == (text.index("[1]"), len(text) - 1)


@pytest.mark.parametrize("text", [
    "Syntax ``[9]` is unmatched.",
    "Syntax ` [9] `` is unmatched.",
    "Syntax `` [9] ``` is unmatched.",
    "Syntax ````` [9] is unmatched.",
])
def test_unmatched_runs_remain_prose_without_partial_delimiters(text):
    assert mask_markdown(text) == text
    assert [m.indices for m in parse_markers(mask_markdown(text))] == [(9,)]


def test_unmatched_run_does_not_prevent_later_valid_span():
    text = "Syntax ``` literal [9] then ``use `x` [8]`` [1]."
    masked = mask_markdown(text)
    assert [m.indices for m in parse_markers(masked)] == [(9,), (1,)]
    assert masked[:text.index("then")] == text[:text.index("then")]


@pytest.mark.parametrize("text,indices", [
    (r"Syntax \`literal [9]\` [1].", [(9,), (1,)]),
    (r"Syntax \\\`literal [9]\\\` [1].", [(9,), (1,)]),
    (r"Syntax \\`literal [9]` [1].", [(1,)]),
    (r"Syntax `literal [9]\` [1].", [(1,)]),
    (r"Syntax \``literal [9]`` [1].", [(9,), (1,)]),
    (r"Syntax \``literal [9]` [1].", [(1,)]),
    (r"Syntax \`literal [9] then ``code [8]`` [1].", [(9,), (1,)]),
])
def test_escaped_backtick_openers_remain_prose_and_code_backslashes_are_literal(text, indices):
    masked = mask_markdown(text)
    assert len(masked) == len(text)
    assert [m.indices for m in parse_markers(masked)] == indices


@pytest.mark.parametrize("boundary", [
    "\n\n", "\n  \n", "\n# Heading\n", "\n    indented\n",
    "\n~~~py\n[8]\n~~~\n", "\n```py\n[8]\n```\n",
])
def test_inline_runs_cannot_pair_across_block_boundaries(boundary):
    text = "Before ``[9]" + boundary + "After ``[1]."
    masked = mask_markdown(text)
    assert len(masked) == len(text)
    assert [m.indices for m in parse_markers(masked)] == [(9,), (1,)]


@pytest.mark.parametrize("fence,closer", [("````", "`````"), ("~~~", "~~~~")])
def test_fences_keep_precedence_and_require_matching_closers(fence, closer):
    text = f"{fence}py\n``[9]``\n```\n[8]\n~~~wrong\n[7]\n{closer}\nReal [1]."
    masked = mask_markdown(text)
    assert len(masked) == len(text)
    assert masked.count("\n") == text.count("\n")
    assert [m.indices for m in parse_markers(masked)] == [(1,)]


def test_line_leading_backtick_span_is_not_a_fence():
    text = "```literal [9]``` The rover uses solar panels [99]."
    assert mask_markdown(text) == " " * len("```literal [9]```") + text[len("```literal [9]```"):]
    assert [m.indices for m in parse_markers(mask_markdown(text))] == [(99,)]


@pytest.mark.parametrize("text,expected", [
    ("Costs are 1.25 USD [1]. 保修期12个月【2】。 Is it valid?", [[1], [2], []]),
    ("Warranty is 12 months. [1] Returns are free. [2]", [[1], [2]]),
    ("Warranty is 12 months [1].\nReturns are free [2].", [[1], [2]]),
    ('The manual says "Signal stable. Start now!" [1]. Another fact [2].', [[1], [2]]),
    ('The manual says “Signal stable.” [1] Another fact [2].', [[1], [2]]),
    ("Dr. Mira runs the lab [1].", [[1]]),
])
def test_sentences_and_citation_ownership(text, expected):
    units = sentences(text, parse_markers(text))
    assert [[idx for m in s.markers for idx in m.indices] for s in units] == expected
    assert all(s.start < s.end for s in units)


@pytest.mark.parametrize("text", ["[1]]", "[1】", "[[1]]", "【1]", "]"])
def test_mismatched_and_stray_brackets_reported(text):
    assert any(m.error for m in parse_markers(text))
