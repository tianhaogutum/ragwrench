"""Small structural scanner following CommonMark fenced-block rules.

Not a Markdown renderer: HTML, list containers and indented code are outside the
structural model. Explicit blockquote prefixes are modeled for fences and tables.
Fenced code itself is never scanned as a table/list.
"""
import re
from dataclasses import dataclass

FENCE = re.compile(r"^( {0,3})(`{3,}|~{3,})([^\r\n]*)$")
HEADING = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+|$)")
LIST = re.compile(r"^( {0,3})(?:[-+*]|\d{1,9}[.)])[ \t]+")
DELIMITER = re.compile(r"^:?-+:?$")
QUOTE = re.compile(r" {0,3}> ?")


def split_lines(text, keepends=False):
    """CommonMark line endings only; Unicode separators remain literal text."""
    lines = re.findall(r"[^\r\n]*(?:\r\n|\r|\n|$)", text)
    if lines and lines[-1] == "":
        lines.pop()
    return lines if keepends else [line.rstrip("\r\n") for line in lines]


@dataclass
class Fence:
    char: str
    length: int
    line: int
    opener: str
    quote_depth: int = 0

    def continues(self, line):
        depth, _ = quote_content(line, self.quote_depth)
        return depth == self.quote_depth

    def closes(self, line):
        depth, content = quote_content(line, self.quote_depth)
        match = FENCE.match(content)
        return bool(depth == self.quote_depth and match and match[2][0] == self.char
                    and len(match[2]) >= self.length and not match[3].strip(" \t"))


@dataclass
class Scan:
    lines: list[str]
    outside: list[bool]
    unclosed: Fence | None
    # Zero-based, half-open ranges with an explicit matching closer. Reuse these
    # in repair so lint and structural splitting cannot disagree about nesting.
    code_blocks: list[tuple[int, int]]


def quote_content(line, limit=None):
    """Remove explicit quote prefixes for parsing only; raw payload stays intact.

    Strip at most an active fence's depth: additional '>' inside code is literal.
    Tab stops use the original column positions, before container removal.
    """
    content, depth, offset = line.expandtabs(4), 0, 0
    while limit is None or depth < limit:
        match = QUOTE.match(content, offset)
        if not match:
            break
        depth += 1
        offset = match.end()
    return depth, content[offset:]


def scan(text):
    lines = split_lines(text)
    outside, code_blocks = [], []
    active = None
    for number, line in enumerate(lines, 1):
        # A non-prefixed line ends a fenced quote container (there is no lazy
        # continuation inside fenced code). It may itself start a new block.
        if active and not active.continues(line):
            active = None
        if active:
            outside.append(False)
            if active.closes(line):
                code_blocks.append((active.line - 1, number))
                active = None
        else:
            depth, content = quote_content(line)
            match = FENCE.match(content)
            if match and not (match[2][0] == "`" and "`" in match[3]):
                outside.append(False)
                active = Fence(match[2][0], len(match[2]), number, line, depth)
            else:
                outside.append(True)
    return Scan(lines, outside, active, code_blocks)


def cells(line):
    # Ignore escaped pipes; require a real pipe so prose isn't a table.
    if not re.search(r"(?<!\\)\|", line):
        return []
    parts = re.split(r"(?<!\\)\|", line.strip())
    if parts and parts[0] == "":
        parts.pop(0)
    if parts and parts[-1] == "":
        parts.pop()
    return [part.strip() for part in parts]


def delimiter(line):
    parts = cells(line)
    return bool(parts) and all(DELIMITER.fullmatch(part) for part in parts)


def table_runs(structure):
    runs, start, depth = [], None, None
    for i in range(len(structure.lines) + 1):
        row_depth, content = (quote_content(structure.lines[i])
                              if i < len(structure.lines) and structure.outside[i] else (None, ""))
        is_row = bool(cells(content))
        # A deeper/shallow quote starts a different container, even with no blank.
        if start is not None and (not is_row or row_depth != depth):
            runs.append((start, i))
            start = None
        if is_row and start is None:
            start, depth = i, row_depth
    return runs


def valid_table(lines):
    if len(lines) < 2:
        return False
    parsed = [quote_content(line) for line in lines]
    depth, header = parsed[0]
    return (all(row_depth == depth for row_depth, _ in parsed)
            and delimiter(parsed[1][1]) and len(cells(header)) == len(cells(parsed[1][1])))


def trailing_heading(text):
    structure = scan(text)
    nonempty = [i for i, line in enumerate(structure.lines) if line.strip()]
    if nonempty and structure.outside[nonempty[-1]] and HEADING.match(structure.lines[nonempty[-1]]):
        return nonempty[-1] + 1
    return None


def starts_table(text):
    structure = scan(text)
    for i, line in enumerate(structure.lines):
        if line.strip():
            return structure.outside[i] and bool(cells(quote_content(line)[1]))
    return False


def has_table_header(text):
    structure = scan(text)
    return any(valid_table(structure.lines[a:b]) for a, b in table_runs(structure))
