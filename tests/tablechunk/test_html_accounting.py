"""Local synthetic integration contract for loss-accounted HTML normalization."""
from pathlib import Path

import pytest

from ragwrench.tablechunk import split_document
from ragwrench.tablechunk.html import parse_html
from ragwrench.tablechunk.models import Table
from ragwrench.tablechunk.splitter import render_table

FIXTURE = Path(__file__).parent / "fixtures" / "spans-accounting.html"
HEADER_IDS = [["r0c0", "r0c1", "r0c1"], ["r0c0", "r1c0", "r1c1"]]
ROW_IDS = [["r2c0", "r2c1", "r2c2"], ["r2c0", "r3c0", None], ["r4c0", "r4c1", "r4c1"]]


def test_distinct_equal_headers_are_preserved_but_carried_header_is_not_repeated():
    table = next(b for b in parse_html(FIXTURE.read_text()) if isinstance(b, Table))
    assert table.columns == ["Region", "Revenue / Revenue", "Revenue / Q2"]


@pytest.mark.parametrize("mode", ["markdown", "kv", "html"])
def test_expansion_accounting_survives_chunk_boundaries(mode):
    text = FIXTURE.read_text()
    table = next(b for b in parse_html(text) if isinstance(b, Table))
    # Fits every single row, forces separate chunks without cell fragmentation.
    budget = max(len(render_table(table, [row], mode, ["Sales"])) for row in table.rows)
    chunks = split_document(text, input_format="html", mode=mode, counter=len, max_tokens=budget)
    assert len(chunks) == 3
    seen = {}
    for index, chunk in enumerate(chunks):
        assert len(chunk.text) <= budget
        assert chunk.metadata["row_start"] == index
        norm = chunk.metadata["normalization"]
        assert norm["header_cell_ids"] == HEADER_IDS
        assert norm["row_cell_ids"] == [ROW_IDS[index]]
        required = {cell for row in HEADER_IDS + [ROW_IDS[index]] for cell in row if cell is not None}
        assert set(norm["source_cells"]) == required
        assert all(label in chunk.text for label in ("Region", "Revenue", "Q2"))
        seen.update(norm["source_cells"])
    assert len(seen) == 10  # every outer source cell, including the explicit empty td
    west = seen["r2c0"]
    assert (west["row_index"], west["cell_index"], west["rowspan"], west["colspan"]) == (2, 0, 2, 1)
    assert (west["line_start"], west["column_start"], west["line_end"], west["column_end"]) == (8, 4, 8, 24)
    # Positions identify the opening and closing tag starts, including inline tags.
    assert text.splitlines()[7][west["column_start"]:west["column_end"]] == '<td rowspan="2">West'
    assert text.splitlines()[7][west["column_end"]:].startswith('</td>')
    missing = chunks[1].metadata["normalization"]["diagnostics"]
    assert any(d["code"] == "missing_cell" and d["section"] == "rows" and d["row_index"] == 1
               and d["column_index"] == 2 and d["source_row_index"] == 3 for d in missing)
    assert not any(d["code"] == "missing_cell" for d in chunks[0].metadata["normalization"]["diagnostics"])
    nested = chunks[2].metadata["normalization"]["diagnostics"]
    assert any(d["code"] == "nested_table_flattened" and d["cell_id"] == "r4c0" for d in nested)
    assert "East detail" in chunks[2].text


def test_interior_hole_is_diagnosed_even_when_row_width_matches():
    text = '<table><tr><td>a</td><td rowspan="2">b</td></tr><tr></tr></table>'
    chunk = split_document(text, input_format="html")[0]
    norm = chunk.metadata["normalization"]
    assert norm["row_cell_ids"] == [["r0c0", "r0c1"], [None, "r0c1"]]
    assert "ragged_rows_padded" in chunk.metadata["warnings"]
    assert any(d["code"] == "missing_cell" and d["row_index"] == 1 and d["column_index"] == 0
               for d in norm["diagnostics"])
    assert norm["header_cell_ids"] == [[None, None]]  # synthetic labels have no source td/th


def test_fragmented_carried_cell_keeps_origin_and_character_offsets():
    value = "alpha beta gamma " * 6
    text = f'<table><tr><th>Value</th></tr><tr><td rowspan="2">{value}</td></tr><tr></tr></table>'
    chunks = split_document(text, input_format="html", mode="kv", counter=len, max_tokens=60)
    for row in (0, 1):
        fragments = [c for c in chunks if c.metadata["row_start"] == row]
        recovered = ""
        for chunk in fragments:
            norm = chunk.metadata["normalization"]
            assert norm["row_cell_ids"] == [["r1c0"]]
            assert norm["source_cells"]["r1c0"]["row_index"] == 1
            fragment = chunk.metadata["fragment"]
            assert fragment["char_start"] == len(recovered)
            recovered += chunk.text.split("\n\nValue: ", 1)[1]
            assert fragment["char_end"] == len(recovered)
        assert recovered == value.strip()


def test_embedded_html_coordinates_use_document_lines():
    text = '# Intro\n\n<table>\n<tr><td>one\n two</td></tr>\n</table>'
    chunk = split_document(text)[0]
    source = chunk.metadata["normalization"]["source_cells"]["r0c0"]
    assert (source["line_start"], source["column_start"], source["line_end"], source["column_end"]) == (4, 4, 5, 4)


def test_duplicate_column_paths_have_explicit_diagnostic():
    text = '<table><tr><th>A</th><th>A</th></tr><tr><td>x</td><td>y</td></tr></table>'
    chunk = split_document(text, input_format="html")[0]
    assert chunk.metadata["columns"] == ["A", "A"]
    assert "duplicate_column_path" in chunk.metadata["warnings"]
    norm = chunk.metadata["normalization"]
    assert any(d["code"] == "duplicate_column_path" and d["column_indices"] == [0, 1]
               for d in norm["diagnostics"])
    assert norm["header_cell_ids"] == [["r0c0", "r0c1"]]


def test_header_only_table_accounts_for_every_source_cell():
    chunk = split_document('<table><tr><th colspan="2">A</th></tr></table>', input_format="html")[0]
    norm = chunk.metadata["normalization"]
    assert norm["header_cell_ids"] == [["r0c0", "r0c0"]]
    assert norm["row_cell_ids"] == []
    assert set(norm["source_cells"]) == {"r0c0"}


def test_carried_origins_do_not_inherit_unrelated_earlier_rowspans():
    text = ('<table>\n<tr><td rowspan="2">a</td><td>b</td></tr>\n'
            '<tr><td rowspan="2">c</td></tr>\n<tr><td>d</td></tr>\n</table>')
    table = parse_html(text)[0]
    assert table.row_lines == [(2, 2), (2, 3), (3, 4)]
    norm = split_document(text, input_format="html")[0].metadata["normalization"]
    assert norm["source_cells"]["r1c0"]["line_start"] == 3


def test_cli_overlap_reports_offending_cell_and_keeps_existing_output(tmp_path):
    from .test_cli_benchmark import cli

    path = tmp_path / 'overlap.html'
    path.write_text('<table>\n<tr><td>a</td><td rowspan="2">b</td></tr>\n'
                    '<tr><td colspan="2">c</td></tr>\n</table>')
    output = tmp_path / 'chunks.jsonl'
    output.write_text('existing')
    run = cli('split', path, '-o', output)
    assert run.returncode == 2
    assert 'line 3, column 4: source cell r1c0: overlapping rowspan/colspan' in run.stderr
    assert run.stdout == ''
    assert output.read_text() == 'existing'
