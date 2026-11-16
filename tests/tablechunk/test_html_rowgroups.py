"""Row-group span contracts: source declarations, expanded grids and strict failures."""
from pathlib import Path

import pytest

from ragwrench.tablechunk import split_document
from ragwrench.tablechunk.html import parse_html
from ragwrench.tablechunk.models import TableChunkError
from ragwrench.tablechunk.splitter import render_table

FIXTURE = Path(__file__).parent / "fixtures" / "row-groups.html"
HEADERS = [["Region", "Sales", "Sales"], ["Region", "Q1", "Q2"]]
ROWS = [["West", "10", "10"], ["West", "20", "30"],
        ["East", "40", "50"], ["Total", "70", "80"], ["Total", "90", "100"]]
HEADER_IDS = [["r0c0", "r0c1", "r0c1"], ["r0c0", "r1c0", "r1c1"]]
ROW_IDS = [["r2c0", "r2c1", "r2c1"], ["r2c0", "r3c0", "r3c1"],
           ["r4c0", "r4c1", "r4c2"], ["r5c0", "r5c1", "r5c2"],
           ["r5c0", "r6c0", "r6c1"]]


@pytest.mark.parametrize("mode", ["markdown", "kv", "html"])
@pytest.mark.parametrize("embedded", [False, True])
def test_zero_span_groups_keep_source_records_across_chunks(mode, embedded):
    text = FIXTURE.read_text()
    table = parse_html(text)[0]
    assert table.headers == HEADERS
    assert table.columns == ["Region", "Sales / Q1", "Sales / Q2"]
    assert table.rows == ROWS
    assert table.row_lines == [(7, 7), (7, 8), (11, 11), (14, 14), (14, 15)]
    budget = max(len(render_table(table, [row], mode, ["Intro"] if embedded else [])) for row in ROWS)
    prefix = "# Intro\n\n" if embedded else ""
    chunks = split_document(prefix + text, input_format="markdown" if embedded else "html",
                            mode=mode, counter=len, max_tokens=budget)
    chunks = [c for c in chunks if c.metadata["kind"] == "table"]
    assert len(chunks) == len(ROWS)
    for index, chunk in enumerate(chunks):
        assert chunk.metadata["row_start"] == index
        assert chunk.metadata["row_end"] == index + 1
        norm = chunk.metadata["normalization"]
        assert norm["header_cell_ids"] == HEADER_IDS
        assert norm["row_cell_ids"] == [ROW_IDS[index]]
        for cell_id in ("r0c0", "r2c0", "r5c0"):
            if cell_id in norm["source_cells"]:
                record = norm["source_cells"][cell_id]
                assert record["rowspan"] == 0
                assert record["resolved_rowspan"] == 2
        if index < 2:
            source = norm["source_cells"]["r2c0"]
            assert (source["line_start"], source["column_start"], source["line_end"], source["column_end"]) == (
                9 if embedded else 7, 4, 9 if embedded else 7, 24)
        assert not norm["diagnostics"]


@pytest.mark.parametrize("group", ["thead", "tbody", "tfoot", None])
def test_zero_span_from_middle_stops_at_group_end(group):
    rows = '<tr><td>a</td><td>b</td></tr><tr><td rowspan="0">x</td><td>y</td></tr><tr><td>z</td></tr>'
    wrapped = f'<{group}>{rows}</{group}>' if group else rows
    table = parse_html(f'<table>{wrapped}<tbody><tr><td>last</td><td>row</td></tr></tbody></table>')[0]
    grid = (table.headers if group == "thead" else []) + table.rows
    assert grid == [["a", "b"], ["x", "y"], ["x", "z"], ["last", "row"]]
    source = table.normalization.source_cells["r1c0"]
    assert (source["rowspan"], source["resolved_rowspan"]) == (0, 2)


def test_empty_section_separates_implicit_groups_and_table_state_resets():
    text = ('<table><tr><td rowspan="0">first</td></tr><tbody></tbody>'
            '<tr><td rowspan="0">second</td></tr><tr></tr></table>'
            '<table><tr><td rowspan="0">third</td></tr></table>')
    first, second = parse_html(text)
    assert first.rows == [["first"], ["second"], ["second"]]
    assert first.normalization.source_cells["r0c0"]["resolved_rowspan"] == 1
    assert first.normalization.source_cells["r1c0"]["resolved_rowspan"] == 2
    assert second.rows == [["third"]]
    assert second.normalization.source_cells["r0c0"]["resolved_rowspan"] == 1


@pytest.mark.parametrize("left,right", [("thead", "tbody"), ("tbody", "tbody"),
                                        ("tbody", "tfoot"), ("tfoot", "tbody"),
                                        (None, "tbody"), ("tbody", None)])
def test_positive_rowspan_cannot_cross_group_with_source_diagnostic(left, right):
    def wrap(tag, row):
        return f'<{tag}>{row}</{tag}>' if tag else row
    text = '<table>\n' + wrap(left, '<tr><td rowspan="2">x</td></tr>') + '\n' + wrap(
        right, '<tr><td>y</td></tr>') + '\n</table>'
    column = (len(left) + 2 if left else 0) + 4
    with pytest.raises(TableChunkError) as error:
        parse_html(text)
    assert str(error.value) == (f'line 2, column {column}: source cell r0c0: '
                                'rowspan crosses row group boundary')


def test_positive_span_within_group_has_declared_and_resolved_extent():
    table = parse_html('<table><tbody><tr><td rowspan="2">x</td></tr><tr></tr></tbody></table>')[0]
    assert table.rows == [["x"], ["x"]]
    source = table.normalization.source_cells["r0c0"]
    assert (source["rowspan"], source["resolved_rowspan"]) == (2, 2)


@pytest.mark.parametrize("attributes,message", [('rowspan="0" colspan="2"', 'overlapping rowspan/colspan'),
                                                ('rowspan="2"', 'rowspan extends beyond table')])
def test_zero_spans_keep_overlap_and_out_of_table_errors(attributes, message):
    if 'colspan' in attributes:
        text = (f'<table><tbody><tr><td>a</td><td rowspan="0">b</td></tr>'
                f'<tr><td {attributes}>c</td></tr></tbody></table>')
        cell = 'r1c0'
    else:
        text = f'<table><tbody><tr><td {attributes}>x</td></tr></tbody></table>'
        cell = 'r0c0'
    with pytest.raises(TableChunkError, match=f'source cell {cell}: {message}'):
        parse_html(text)


def test_zero_extent_uses_grid_limit_instead_of_positive_span_limit():
    rows = '<tr><td rowspan="0">x</td></tr>' + '<tr></tr>' * 256
    table = parse_html(f'<table><tbody>{rows}</tbody></table>')[0]
    assert table.rows == [["x"]] * 257
    assert table.normalization.source_cells["r0c0"]["resolved_rowspan"] == 257
    rows = '<tr><td rowspan="0" colspan="256">x</td></tr>' + '<tr></tr>' * 390
    with pytest.raises(TableChunkError, match='source cell r0c0: expanded table exceeds 100000 cells'):
        parse_html(f'<table><tbody>{rows}</tbody></table>')


@pytest.mark.parametrize("text", [
    '<table><tbody><tbody><tr><td>x</td></tr></tbody></tbody></table>',
    '<table><tr><td><tbody></tbody>x</td></tr></table>',
])
def test_nested_row_groups_rejected(text):
    with pytest.raises(TableChunkError, match='row group must be a direct child of table'):
        parse_html(text)


def test_cli_cross_group_error_is_atomic(tmp_path):
    from .test_cli_benchmark import cli

    path = tmp_path / 'cross-group.html'
    path.write_text('<table>\n<tbody><tr><td rowspan="2">x</td></tr></tbody>\n'
                    '<tbody><tr><td>y</td></tr></tbody>\n</table>')
    output = tmp_path / 'chunks.jsonl'
    output.write_text('existing')
    run = cli('split', path, '-o', output)
    assert run.returncode == 2
    assert run.stdout == ''
    assert 'line 2, column 11: source cell r0c0: rowspan crosses row group boundary' in run.stderr
    assert output.read_text() == 'existing'


def test_fragmented_zero_span_retains_resolved_extent_and_origin():
    value = 'alpha beta gamma ' * 6
    text = ('<table><thead><tr><th>Value</th></tr></thead><tbody>'
            f'<tr><td rowspan="0">{value}</td></tr><tr></tr></tbody></table>')
    chunks = split_document(text, input_format='html', mode='kv', counter=len, max_tokens=60)
    for row in (0, 1):
        recovered = ''
        fragments = [c for c in chunks if c.metadata['row_start'] == row]
        assert len(fragments) > 1
        for chunk in fragments:
            norm = chunk.metadata['normalization']
            assert norm['row_cell_ids'] == [['r1c0']]
            record = norm['source_cells']['r1c0']
            assert (record['rowspan'], record['resolved_rowspan'], record['row_index']) == (0, 2, 1)
            fragment = chunk.metadata['fragment']
            assert fragment['char_start'] == len(recovered)
            recovered += chunk.text.split('\n\nValue: ', 1)[1]
            assert fragment['char_end'] == len(recovered)
        assert recovered == value.strip()


def test_nested_table_sections_do_not_end_outer_group():
    text = ('<table><tbody><tr><td rowspan="0">before'
            '<table><tbody><tr><td>inner</td></tr></tbody></table>'
            'after</td><td>a</td></tr><tr><td>b</td></tr></tbody></table>')
    table = parse_html(text)[0]
    assert table.rows == [['before inner after', 'a'], ['before inner after', 'b']]
    assert table.normalization.source_cells['r0c0']['resolved_rowspan'] == 2
    assert set(table.normalization.source_cells) == {'r0c0', 'r0c1', 'r1c0'}


@pytest.mark.parametrize('span', ['-1', '257'])
def test_rowspan_declaration_limits_remain_strict(span):
    with pytest.raises(TableChunkError, match='rowspan'):
        parse_html(f'<table><tr><td rowspan="{span}">x</td></tr></table>')
