"""Non-executing HTML parser with bounded span expansion and explicit errors."""
from dataclasses import dataclass
from html.parser import HTMLParser
import json
import re
from .models import Block, HTMLNormalization, Table, TableChunkError

MAX_SPAN = 256
MAX_GRID_CELLS = 100_000
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
ASCII_WHITESPACE = re.compile(r"[\t\n\f\r ]+")


@dataclass
class Cell:
    text: str
    rowspan: int
    colspan: int
    header: bool
    id: str
    source: dict
    html_id: str | None = None
    header_refs: list[str] | None = None


class _Parser(HTMLParser):
    def __init__(self, offset: int, storage=None):
        super().__init__(convert_charrefs=True)
        # HTMLParser itself owns `offset` (the current column).
        self.line_offset = offset
        self.storage = storage
        self.blocks: list[Block | Table] = []
        self.depth = 0
        self.rows: list[tuple[list[Cell], int, int, bool, int]] = []
        self.group_index = -1
        self.group: int | None = None
        self.row: list[Cell] | None = None
        self.cell: Cell | None = None
        self.cell_parts: list[str] = []
        self.caption_parts: list[str] = []
        self.in_caption = False
        self.caption_seen = False
        self.rows_started = False
        self.in_head = False
        self.warnings: list[str] = []
        self.diagnostics: list[dict] = []
        self.parts: list[str] = []
        self.heading = 0
        self.prose_start = 1 + offset
        self.ignored = 0
        self.table_tags: list[str] = []

    def line(self):
        return self.getpos()[0] + self.line_offset

    def fail(self, message: str):
        raise TableChunkError(f"line {self.line()}, column {self.getpos()[1]}: {message}")

    def fail_cell(self, cell: Cell, message: str):
        source = cell.source
        raise TableChunkError(f"line {source['line_start']}, column {source['column_start']}: "
                              f"source cell {cell.id}: {message}")

    def flush(self):
        text = _clean("".join(self.parts))
        if text:
            self.blocks.append(Block(text, self.prose_start, self.line(), "heading" if self.heading else "prose", self.heading))
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.ignored += 1
        if self.ignored:
            return
        if not self.depth:
            if tag == "caption":
                self.fail("<caption> must be a direct child of table")
            if tag in ("thead", "tbody", "tfoot"):
                self.fail("row group must be a direct child of table")
            if tag == "tr":
                self.fail("<tr> must be a direct child of table/thead/tbody/tfoot")
            if tag in ("td", "th"):
                self.fail(f"<{tag}> must be a direct child of tr")
        if (self.depth or tag == "table") and tag not in VOID_TAGS:
            self.table_tags.append(tag)
        if tag == "table":
            if self.depth:
                if "nested_table_flattened" not in self.warnings:
                    self.warnings.append("nested_table_flattened")
                if self.cell is None:
                    self.fail("nested table must be inside a cell")
                self.diagnostics.append({"code": "nested_table_flattened", "cell_id": self.cell.id,
                                         "line": self.line(), "column": self.getpos()[1],
                                         "message": "Nested structure flattened into the outer source cell; inner coordinates are not mapped."})
            else:
                self.flush()
                self.table_start = self.line()
                self.rows = self.sequence("source_rows")
                self.warnings, self.caption_parts = [], []
                self.caption_seen = False
                self.rows_started = False
                self.diagnostics = self.sequence("diagnostics")
                self.group_index, self.group = -1, None
            self.depth += 1
            return
        if self.depth > 1:
            if tag in ("tr", "td", "th", "br", "p"):
                self.cell_parts.append(" ")
            return
        if self.depth:
            if tag in ("thead", "tbody", "tfoot"):
                if self.table_tags[-2] != "table":
                    self.fail("row group must be a direct child of table")
                self.rows_started = True
                self.group_index += 1
                self.group = self.group_index
                self.in_head = tag == "thead"
            elif tag == "caption":
                if self.table_tags[-2] != "table":
                    self.fail("<caption> must be a direct child of table")
                if self.caption_seen:
                    self.fail("duplicate outer <caption>")
                if self.rows_started:
                    self.fail("<caption> must precede rows and row groups")
                self.caption_seen = True
                self.in_caption = True
            elif tag == "tr":
                if self.table_tags[-2] not in ("table", "thead", "tbody", "tfoot"):
                    self.fail("<tr> must be a direct child of table/thead/tbody/tfoot")
                self.rows_started = True
                if self.row is not None:
                    self.fail("unclosed tr tag")
                if self.group is None:
                    # Consecutive direct tr children form one implicit group.
                    self.group_index += 1
                    self.group = self.group_index
                self.row, self.row_start = [], self.line()
            elif tag in ("td", "th"):
                if self.table_tags[-2] != "tr":
                    self.fail(f"<{tag}> must be a direct child of tr")
                if self.row is None or self.cell is not None:
                    self.fail("cell outside row or unclosed td/th tag")
                attributes = {}
                association_duplicates = []
                for name, value in attrs:
                    if name in ("rowspan", "colspan") and name in attributes:
                        self.fail(f"duplicate {name} attribute")
                    if name in attributes and name == ("id" if tag == "th" else "headers"):
                        association_duplicates.append(name)
                    attributes[name] = value
                spans = []
                for name in ("rowspan", "colspan"):
                    raw = attributes.get(name, "1")
                    if raw is None or not re.fullmatch(r"[0-9]{1,3}", raw):
                        self.fail(f"invalid {name}")
                    value = int(raw)
                    minimum = 0 if name == "rowspan" else 1
                    if not minimum <= value <= MAX_SPAN:
                        self.fail(f"{name} must be {minimum}..{MAX_SPAN}")
                    spans.append(value)
                row_index, cell_index = len(self.rows), len(self.row)
                self.cell = Cell("", *spans, tag == "th", f"r{row_index}c{cell_index}",
                                 {"row_index": row_index, "cell_index": cell_index, "tag": tag,
                                  "rowspan": spans[0], "colspan": spans[1],
                                  "line_start": self.line(), "column_start": self.getpos()[1]})
                if association_duplicates:
                    self.fail_cell(self.cell, f"duplicate {association_duplicates[0]} attribute")
                if tag == "th" and "id" in attributes:
                    value = attributes["id"]
                    if not value or ASCII_WHITESPACE.search(value):
                        self.fail_cell(self.cell, "header id must be nonempty without ASCII whitespace")
                    self.cell.html_id = value
                if tag == "td" and "headers" in attributes:
                    self.cell.header_refs = [ref for ref in ASCII_WHITESPACE.split(attributes["headers"] or "") if ref]
                self.cell_parts = []
            elif tag in ("br", "p", "div", "li") and self.cell is not None:
                self.cell_parts.append("\n")
            elif tag == "br" and self.in_caption:
                self.caption_parts.append("\n")
            return
        if tag in ("p", "div", "section", "li") or re.fullmatch(r"h[1-6]", tag):
            self.flush()
            self.prose_start = self.line()
            if re.fullmatch(r"h[1-6]", tag):
                self.heading = int(tag[1])
        elif tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.ignored:
            self.ignored -= 1
            return
        if self.ignored:
            return
        if not self.depth and tag in ("thead", "tbody", "tfoot"):
            self.fail("unexpected closing row group")
        if not self.depth and tag == "caption":
            self.fail("unexpected closing caption")
        if not self.depth and tag == "tr":
            self.fail("unexpected closing row")
        if not self.depth and tag in ("td", "th"):
            self.fail("unexpected closing cell")
        if self.depth and tag not in VOID_TAGS:
            if not self.table_tags or self.table_tags[-1] != tag:
                self.fail(f"unclosed or mismatched HTML tag before </{tag}>")
            self.table_tags.pop()
        if tag == "table":
            if not self.depth:
                self.fail("unexpected closing table")
            self.depth -= 1
            if self.depth:
                self.cell_parts.append(" ")
                return
            if self.cell is not None or self.row is not None or self.in_caption or self.in_head:
                self.fail("unclosed HTML table structural tag")
            self.blocks.append(self.make_table())
            return
        if self.depth > 1:
            if tag in ("tr", "td", "th", "p"):
                self.cell_parts.append(" ")
            return
        if self.depth:
            if tag in ("td", "th"):
                if self.cell is None or self.row is None:
                    self.fail("unexpected closing cell")
                self.cell.text = _clean("".join(self.cell_parts))
                # End coordinates point to the start of the explicit closing tag.
                self.cell.source.update(line_end=self.line(), column_end=self.getpos()[1])
                self.row.append(self.cell)
                self.cell = None
            elif tag == "tr":
                if self.cell is not None or self.row is None:
                    self.fail("unclosed cell or unexpected closing row")
                self.rows.append((self.row, self.row_start, self.line(), self.in_head, self.group))
                self.row = None
            elif tag in ("thead", "tbody", "tfoot"):
                self.in_head = False
                self.group = None
            elif tag == "caption":
                self.in_caption = False
            elif tag in ("p", "div", "li") and self.cell is not None:
                self.cell_parts.append("\n")
            return
        if tag in ("p", "div", "section", "li") or re.fullmatch(r"h[1-6]", tag):
            self.flush()
            if re.fullmatch(r"h[1-6]", tag):
                self.heading = 0

    def handle_data(self, data):
        if self.ignored:
            return
        if self.cell is not None:
            self.cell_parts.append(data)
        elif self.in_caption:
            self.caption_parts.append(data)
        elif not self.depth:
            if not self.parts:
                self.prose_start = self.line()
            self.parts.append(data)
        elif data.strip():
            self.fail("table text must be inside a cell or caption")

    def sequence(self, kind=""):
        return self.storage.sequence(kind) if self.storage else []

    def mapping(self):
        return self.storage.mapping() if self.storage else {}

    def make_table(self):
        if not self.rows:
            self.fail("table contains no rows")
        grid, spans = self.sequence(), {}
        locations, header_flags = self.sequence(), self.sequence()
        source_cells, cell_ids = self.mapping(), self.sequence()
        # Only declared outer th IDs participate. A full table prepass permits
        # forward references and body headers without layout/DOM inference.
        declared_headers = self.mapping()
        for cells, *_ in self.rows:
            for cell in cells:
                if cell.html_id is not None:
                    if declared_headers.get(cell.html_id) is not None:
                        self.fail_cell(cell, f"duplicate header id {cell.html_id!r}")
                    declared_headers[cell.html_id] = {"id": cell.html_id, "cell_id": cell.id, "text": cell.text}
        # Resolve against source-row groups before expanding any cells. Equal tag
        # names (including adjacent tbody elements) are still distinct groups.
        group_ends = self.mapping()
        for index, row in enumerate(self.rows):
            group_ends[row[4]] = index + 1
        for index, (cells, start, end, in_head, group) in enumerate(self.rows):
            occupied = {}
            for col, (cell, until) in list(spans.items()):
                if until > index:
                    occupied[col] = cell
                    start = min(start, cell.source["line_start"])
                else:
                    del spans[col]
            col = 0
            for cell in cells:
                if cell.header_refs is not None:
                    references, seen, size = [], set(), 2
                    for ref in cell.header_refs:
                        if ref in seen:
                            self.fail_cell(cell, f"repeated headers reference {ref!r}")
                        seen.add(ref)
                        header = declared_headers.get(ref)
                        if header is None:
                            self.fail_cell(cell, f"headers target {ref!r} is not an outer <th>")
                        if self.storage:
                            size += len(json.dumps(header, ensure_ascii=False).encode("utf-8")) + 2
                            if size > self.storage.limit:
                                self.fail_cell(cell, "streaming HTML header associations exceed 1 MiB limit")
                        references.append(header)
                    cell.source["headers"] = references
                rowspan = group_ends[group] - index if cell.rowspan == 0 else cell.rowspan
                cell.source["resolved_rowspan"] = rowspan
                while col in occupied:
                    col += 1
                if any(c in occupied for c in range(col, col + cell.colspan)):
                    self.fail_cell(cell, "overlapping rowspan/colspan")
                if index + rowspan > len(self.rows):
                    self.fail_cell(cell, "rowspan extends beyond table")
                if index + rowspan > group_ends[group]:
                    self.fail_cell(cell, "rowspan crosses row group boundary")
                if col + cell.colspan > MAX_GRID_CELLS // max(1, len(self.rows)):
                    self.fail_cell(cell, "expanded table exceeds 100000 cells")
                source_cells[cell.id] = cell.source
                for c in range(col, col + cell.colspan):
                    occupied[c] = cell
                    if rowspan > 1:
                        spans[c] = (cell, index + rowspan)
                col += cell.colspan
            width = max(occupied, default=-1) + 1
            grid.append([occupied[c].text if c in occupied else "" for c in range(width)])
            cell_ids.append([occupied[c].id if c in occupied else None for c in range(width)])
            locations.append((start, end))
            header_flags.append(in_head or bool(cells) and all(c.header for c in cells))
        width = max(map(len, grid))
        if not width or width * len(grid) > MAX_GRID_CELLS:
            self.fail("empty or oversized table grid")
        if any(len(row) != width or None in row for row in cell_ids):
            self.warnings.append("ragged_rows_padded")
        padded_grid, padded_ids = self.sequence(), self.sequence()
        for row, ids in zip(grid, cell_ids):
            padded_grid.append(row + [""] * (width - len(row)))
            padded_ids.append(ids + [None] * (width - len(ids)))
        grid, cell_ids = padded_grid, padded_ids
        header_count = 0
        for flag in header_flags:
            if not flag:
                break
            header_count += 1
        if header_count:
            if self.storage:
                # Headers repeat in every chunk, so cap their aggregate size too.
                size = 0
                for row, ids in zip(grid[:header_count], cell_ids[:header_count]):
                    size += len(str(row).encode("utf-8")) + len(str(ids).encode("utf-8"))
                    if size > self.storage.limit:
                        self.fail("streaming HTML headers exceed 1 MiB limit")
            headers, rows = list(grid[:header_count]), grid[header_count:]
            header_ids = list(cell_ids[:header_count])
        else:
            headers, rows = [[f"column_{i + 1}" for i in range(width)]], grid
            header_ids = [[None] * width]
            self.warnings.append("synthetic_headers")
            self.diagnostics.append({"code": "synthetic_headers",
                                     "message": "No leading header rows; generated column labels have no source cells."})
        for index, ids in enumerate(cell_ids):
            for column, cell_id in enumerate(ids):
                if cell_id is None:
                    self.diagnostics.append({"code": "missing_cell", "section": "headers" if index < header_count else "rows",
                                             "row_index": index if index < header_count else index - header_count,
                                             "column_index": column, "source_row_index": index,
                                             "line_start": self.rows[index][1], "line_end": self.rows[index][2],
                                             "message": "No source cell covers this grid slot; rendered as an empty value."})
        normalization_type = HTMLNormalization
        if self.storage:
            from .html_storage import DiskNormalization
            normalization_type = DiskNormalization
        normalization = normalization_type(source_cells, header_ids, cell_ids[header_count:], self.diagnostics)
        table = Table(headers, rows, locations[header_count:], self.table_start, self.line(),
                      _clean("".join(self.caption_parts)), list(dict.fromkeys(self.warnings)), normalization)
        paths = {}
        for index, path in enumerate(table.columns):
            paths.setdefault(path, []).append(index)
        for path, indices in paths.items():
            if len(indices) > 1:
                if "duplicate_column_path" not in table.warnings:
                    table.warnings.append("duplicate_column_path")
                self.diagnostics.append({"code": "duplicate_column_path", "column_indices": indices,
                                         "path": path, "message": "Column labels are ambiguous; use column indices and header source IDs."})
        return table


def _clean(text):
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def parse_html(text: str, line_offset: int = 0) -> list[Block | Table]:
    parser = _Parser(line_offset)
    parser.feed(text)
    parser.close()
    if parser.depth:
        raise TableChunkError("unclosed HTML table")
    if parser.ignored:
        raise TableChunkError("unclosed script/style tag")
    parser.flush()
    return parser.blocks
