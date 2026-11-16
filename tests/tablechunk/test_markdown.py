import pytest
from ragwrench.tablechunk.markdown import parse_markdown, split_cells
from ragwrench.tablechunk.models import Block, Table
from ragwrench.tablechunk import split_document


@pytest.mark.parametrize(("line", "expected"), [
    ("| A | B |", ["A", "B"]),
    (" A | B ", ["A", "B"]),
    (r"| a\|b | c |", ["a|b", "c"]),
    ("| `a|b` | ``x`|y`` |", ["`a|b`", "``x`|y``"]),
    (r"| `a\|b` | c |", ["`a|b`", "c"]),
    ("| unmatched` | b |", ["unmatched`", "b"]),
    (r"| slash\\ | final\|", ["slash\\", "final|"]),
    ("| | |", ["", ""]),
])
def test_cells(line, expected):
    assert split_cells(line) == expected


def test_alignment_ragged_and_provenance():
    table = parse_markdown("A | B\n:---|---:\nx|y|z\nq|r\n")[0]
    assert isinstance(table, Table)
    assert table.headers == [["A", "B", "column_3"]]
    assert table.rows == [["x", "y", "z"], ["q", "r", ""]]
    assert table.warnings == ["ragged_rows_padded"]
    assert table.row_lines == [(3, 3), (4, 4)]
    assert (table.line_start, table.line_end) == (1, 4)


def test_no_delimiter_is_prose():
    blocks = parse_markdown("| A | B |\n| x | y |")
    assert len(blocks) == 1 and isinstance(blocks[0], Block)
    assert split_document(blocks[0].text)[0].metadata["kind"] == "prose"


@pytest.mark.parametrize("marker", ["```", "~~~~"])
def test_fenced_table_is_not_parsed(marker):
    text = f"{marker}markdown\n| A | B |\n|---|---|\n| x | y |\n{marker}"
    chunks = split_document(text)
    assert len(chunks) == 1 and chunks[0].text == text
    assert chunks[0].metadata["kind"] == "prose"


def test_embedded_html_lines():
    text = "# Title\n\n<table>\n<tr><th>A</th></tr>\n<tr><td>x</td></tr>\n</table>\n\nTail"
    blocks = parse_markdown(text)
    table = next(block for block in blocks if isinstance(block, Table))
    assert table.row_lines == [(5, 5)]
    assert table.line_start == 3
    chunks = split_document(text)
    assert chunks[0].metadata["header_path"] == ["Title"]
    assert chunks[-1].text == "Title\nTail"


def test_html_comments_do_not_change_embedded_table_boundary():
    text = "<table>\n<!-- <table> -->\n<tr><th>A</th></tr>\n<tr><td>x</td></tr>\n</table>"
    chunks = split_document(text)
    assert len(chunks) == 1
    assert "x" in chunks[0].text and chunks[0].metadata["row_start"] == 0
