"""GFM pipe tables, with source locations and fenced-code exclusion."""
import re
from html.parser import HTMLParser
from .models import Block, Table, TableChunkError


class _TableBoundary(HTMLParser):
    """Find a block boundary without counting table-like text in comments/scripts."""
    def __init__(self):
        super().__init__()
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.depth += 1

    def handle_endtag(self, tag):
        if tag == "table":
            self.depth -= 1


def split_cells(line: str) -> list[str]:
    """Split unescaped pipes outside matched inline-code backtick runs."""
    line = line.strip()
    cells, current = [], []
    i = 0
    while i < len(line):
        char = line[i]
        if char == "\\" and i + 1 < len(line):
            following = line[i + 1]
            current.append(following if following in "|\\" else "\\" + following)
            i += 2
            continue
        if char == "`":
            end = i
            while end < len(line) and line[end] == "`":
                end += 1
            run = line[i:end]
            close = re.search(r"(?<!`)" + re.escape(run) + r"(?!`)", line[end:])
            if close:
                stop = end + close.end()
                # GFM's table escape for pipes also applies inside code spans.
                current.append(line[i:stop].replace(r"\|", "|"))
                i = stop
                continue
            current.append(run)
            i = end
            continue
        if char == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        i += 1
    cells.append("".join(current).strip())
    if line.startswith("|"):
        cells.pop(0)
    if line.endswith("|") and len(cells) > 1 and cells[-1] == "":
        # An escaped final pipe belongs to its cell, which is then nonempty.
        cells.pop()
    return cells


def _delimiter(line: str) -> bool:
    cells = split_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def parse_markdown(text: str) -> list[Block | Table]:
    from .html import parse_html
    lines = text.splitlines()
    blocks: list[Block | Table] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            start = i
            marker = fence.group(1)
            i += 1
            while i < len(lines):
                closing = re.fullmatch(r" {0,3}" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}\s*", lines[i])
                i += 1
                if closing:
                    break
            blocks.append(Block("\n".join(lines[start:i]), start + 1, i, "code"))
            continue
        heading = re.match(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            blocks.append(Block(heading.group(2), i + 1, i + 1, "heading", len(heading.group(1))))
            i += 1
            continue
        if re.search(r"<table\b", line, re.I):
            # Block HTML only: accepting partial lines would lose surrounding text.
            if not re.match(r"\s*<table\b", line, re.I):
                raise TableChunkError(f"line {i + 1}: HTML table must start on its own line")
            start = i
            boundary = _TableBoundary()
            while i < len(lines):
                boundary.feed(lines[i] + "\n")
                i += 1
                if boundary.depth == 0:
                    break
            if boundary.depth:
                raise TableChunkError(f"line {start + 1}: unclosed HTML table")
            fragment = "\n".join(lines[start:i])
            parsed = parse_html(fragment, line_offset=start)
            blocks.extend(parsed)
            continue
        if i + 1 < len(lines) and _delimiter(lines[i + 1]) and "|" in line:
            start = i
            headers = split_cells(line)
            delimiter_width = len(split_cells(lines[i + 1]))
            rows, locations = [], []
            i += 2
            while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                if re.match(r"^ {0,3}(#{1,6})\s|^\s*<table\b|^ {0,3}(`{3,}|~{3,})", lines[i], re.I):
                    break
                rows.append(split_cells(lines[i]))
                locations.append((i + 1, i + 1))
                i += 1
            width = max([len(headers), delimiter_width] + [len(row) for row in rows])
            warnings = []
            if any(len(row) != width for row in [headers] + rows) or delimiter_width != width:
                warnings.append("ragged_rows_padded")
            headers += [f"column_{col + 1}" for col in range(len(headers), width)]
            rows = [row + [""] * (width - len(row)) for row in rows]
            blocks.append(Table([headers], rows, locations, start + 1, i, warnings=warnings))
            continue
        # Keep prose by paragraph; the splitter handles headings and budgets.
        start = i
        i += 1
        while i < len(lines) and lines[i].strip():
            if re.match(r"^ {0,3}(#{1,6})\s|^ {0,3}(`{3,}|~{3,})|^\s*<table\b", lines[i], re.I):
                break
            if i + 1 < len(lines) and "|" in lines[i] and _delimiter(lines[i + 1]):
                break
            i += 1
        blocks.append(Block("\n".join(lines[start:i]), start + 1, i))
    return blocks
