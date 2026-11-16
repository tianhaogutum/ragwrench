from pathlib import Path
import pytest
from ragwrench.tablechunk.html import parse_html
from ragwrench.tablechunk.models import Table, TableChunkError
from ragwrench.tablechunk import split_document

FIXTURE = Path(__file__).parent / "fixtures" / "export.html"


def test_real_html_export_grid():
    blocks = parse_html(FIXTURE.read_text())
    table = next(block for block in blocks if isinstance(block, Table))
    assert table.headers == [["Region", "Capacity", "Capacity"], ["Region", "Used", "Available"]]
    assert table.columns == ["Region", "Capacity / Used", "Capacity / Available"]
    assert table.rows == [["North", "20\nreserved", "80"], ["North", "30", "70"], ["South", "40", "60"]]
    assert table.row_lines == [(12, 12), (12, 13), (14, 14)]
    assert table.caption == "Regional inventory"
    chunks = split_document(FIXTURE.read_text(), input_format="html", max_tokens=120)
    tables = [c for c in chunks if c.metadata["kind"] == "table"]
    assert tables and all("Regional inventory" in c.text for c in tables)
    assert all(c.metadata["header_path"] == ["Capacity report"] for c in tables)
    assert all(c.metadata["token_count"] <= 120 for c in chunks)


def test_nested_table_is_flattened_warned():
    table = parse_html("<table><tr><th>A</th></tr><tr><td>before<table><tr><td>inside</td></tr></table>after</td></tr></table>")[0]
    assert table.rows == [["before inside after"]]
    assert table.warnings == ["nested_table_flattened"]


def test_headerless_and_ragged():
    table = parse_html("<table><tr><td>x</td><td>y</td></tr><tr><td>z</td></tr></table>")[0]
    assert table.headers == [["column_1", "column_2"]]
    assert table.rows == [["x", "y"], ["z", ""]]
    assert table.warnings == ["ragged_rows_padded", "synthetic_headers"]


@pytest.mark.parametrize("text", [
    "<table><tr><td>x</td></tr>",
    "<table><tr><td>x</tr></table>",
    "<table><tr><td>x<td>y</td></tr></table>",
    "<table><tr><td colspan='0'>x</td></tr></table>",
    "<table><tr><td colspan='257'>x</td></tr></table>",
    "<table><tr><td colspan='999999999999999'>x</td></tr></table>",
    "<table><tr><td rowspan='2'>x</td></tr></table>",
    "<table></table>",
    "<table><tbody><tr><td>x</td></tr></table>",
    "<table><tr><td><strong>x</td></tr></table>",
    "<table><tr><td>x</th></tr></table>",
    "<table>stray text<tr><td>x</td></tr></table>",
])
def test_malformed_or_unsupported_html(text):
    with pytest.raises(TableChunkError):
        parse_html(text)


def test_colspan_next_to_rowspan():
    text = "<table><tr><td rowspan='2'>x</td><td>y</td></tr><tr><td colspan='2'>z</td></tr></table>"
    assert parse_html(text)[0].rows == [["x", "y", ""], ["x", "z", "z"]]


def test_overlapping_spans_rejected():
    text = "<table><tr><td>a</td><td rowspan='2'>b</td></tr><tr><td colspan='2'>c</td></tr></table>"
    with pytest.raises(TableChunkError, match="overlapping"):
        parse_html(text)


def test_html_escaped_and_no_script_execution():
    text = "<h2>Values</h2><script>ignored()</script><table><tr><th>Markup</th></tr><tr><td>&lt;script&gt;alert(1)&lt;/script&gt; &amp; ok</td></tr></table>"
    chunks = split_document(text, input_format="html", mode="html", max_tokens=300)
    assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp; ok" in chunks[0].text
    assert "ignored()" not in chunks[0].text


def test_expansion_limit():
    text = "<table>" + "".join("<tr><td colspan='256'>a</td></tr>" for _ in range(391)) + "</table>"
    with pytest.raises(TableChunkError, match="100000"):
        parse_html(text)


def test_multilevel_headers_survive_html_serialisation():
    chunks = split_document(FIXTURE.read_text(), input_format="html", mode="html", max_tokens=250)
    original = next(b for b in parse_html(FIXTURE.read_text()) if isinstance(b, Table))
    tables = [next(b for b in parse_html(c.text) if isinstance(b, Table)) for c in chunks if c.metadata["kind"] == "table"]
    assert all(t.headers == original.headers for t in tables)
    assert [row for t in tables for row in t.rows] == original.rows
