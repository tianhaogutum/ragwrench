"""Disk-backed Markdown and HTML parsing with fixed-size reads and bounded working buffers."""
import codecs
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import asdict
import json
import re
import tempfile
from typing import BinaryIO

from .markdown import _delimiter, split_cells
from .models import Block, Chunk, Table, TableChunkError
from .splitter import _iter_chunks
from .tokens import TokenCounter, count_tokens

MAX_STREAM_BUFFER = 1024 * 1024
# Match str.splitlines(), including CRLF split across two reads.
_LINE_END = re.compile(r'\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]')
_BOUNDARY = re.compile(r'^ {0,3}(#{1,6})\s|^\s*<table\b|^ {0,3}(`{3,}|~{3,})', re.I)


def _lines(source: BinaryIO) -> Iterator[str]:
    decoder = codecs.getincrementaldecoder('utf-8')()
    buffer = ''
    while True:
        data = source.read(65536)
        buffer += decoder.decode(data, final=not data)
        start = 0
        for match in _LINE_END.finditer(buffer):
            if data and match.group() == '\r' and match.end() == len(buffer):
                break
            line = buffer[start:match.start()]
            if len(line.encode('utf-8')) > MAX_STREAM_BUFFER:
                raise TableChunkError('streaming line exceeds 1 MiB limit')
            yield line
            start = match.end()
        buffer = buffer[start:]
        # A held-back CR is a line ending, not part of the source line.
        tail = buffer.removesuffix('\r') if data else buffer
        if len(tail.encode('utf-8')) > MAX_STREAM_BUFFER:
            raise TableChunkError('streaming line exceeds 1 MiB limit')
        if not data:
            if buffer:
                yield buffer
            return


class _Lookahead:
    def __init__(self, lines):
        self.lines = iter(lines)
        self.buffer = deque()
        self.number = 1

    def peek(self, offset=0):
        while len(self.buffer) <= offset:
            line = next(self.lines, None)
            if line is None:
                return None
            self.buffer.append(line)
        return self.buffer[offset]

    def pop(self):
        value = self.peek()
        if value is not None:
            self.buffer.popleft()
            self.number += 1
        return value


class _DiskRows:
    def __init__(self, stream, count, width):
        self.stream, self.count, self.width = stream, count, width

    def __len__(self):
        return self.count

    def __iter__(self):
        self.stream.seek(0)
        for line in self.stream:
            row = json.loads(line)
            yield row + [''] * (self.width - len(row))


class _RowLines(Sequence):
    """GFM rows occupy consecutive single source lines; no per-row index needed."""
    def __init__(self, start, count):
        self.start, self.count = start, count

    def __len__(self):
        return self.count

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [(self.start + i, self.start + i) for i in range(*key.indices(self.count))]
        if key < 0:
            key += self.count
        if not 0 <= key < self.count:
            raise IndexError(key)
        return self.start + key, self.start + key


class _DiskBlocks:
    """Pending prose can span the entire document; spool it rather than list it."""
    def __init__(self, stream):
        self.stream = stream
        self.count = 0

    def __bool__(self):
        return self.count != 0

    def append(self, block):
        self.stream.write(json.dumps(asdict(block), ensure_ascii=False) + '\n')
        self.count += 1

    def __iter__(self):
        self.stream.seek(0)
        for line in self.stream:
            yield Block(**json.loads(line))

    def clear(self):
        self.stream.seek(0)
        self.stream.truncate()
        self.count = 0


def _blocks(source):
    lines = _Lookahead(_lines(source))
    while (line := lines.peek()) is not None:
        if not line.strip():
            lines.pop()
            continue
        start = lines.number
        fence = re.match(r'^ {0,3}(`{3,}|~{3,})', line)
        if fence:
            marker = fence.group(1)
            closing = re.compile(r' {0,3}' + re.escape(marker[0]) + '{' + str(len(marker)) + r',}\s*')
            content = [lines.pop()]
            size = len(content[0].encode('utf-8'))
            while (line := lines.pop()) is not None:
                content.append(line)
                size += len(line.encode('utf-8')) + 1
                if size > MAX_STREAM_BUFFER:
                    raise TableChunkError('streaming code block exceeds 1 MiB limit')
                if closing.fullmatch(line):
                    break
            yield Block('\n'.join(content), start, lines.number - 1, 'code')
            continue
        heading = re.match(r'^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$', line)
        if heading:
            lines.pop()
            yield Block(heading.group(2), start, start, 'heading', len(heading.group(1)))
            continue
        if re.search(r'<table\b', line, re.I):
            raise TableChunkError(f'line {start}: streaming does not support embedded HTML; use buffered mode')
        following = lines.peek(1)
        if following is not None and _delimiter(following) and '|' in line:
            headers = split_cells(lines.pop())
            delimiter_width = len(split_cells(lines.pop()))
            width, narrowest = max(len(headers), delimiter_width), min(len(headers), delimiter_width)
            count = 0
            # A full table prepass is necessary for global width/warnings/end-line parity.
            with tempfile.TemporaryFile(mode='w+t', encoding='utf-8') as rows:
                while (line := lines.peek()) is not None and line.strip() and '|' in line:
                    if _BOUNDARY.match(line):
                        break
                    row = split_cells(lines.pop())
                    width, narrowest = max(width, len(row)), min(narrowest, len(row))
                    rows.write(json.dumps(row, ensure_ascii=False) + '\n')
                    count += 1
                headers += [f'column_{col + 1}' for col in range(len(headers), width)]
                yield Table([headers], _DiskRows(rows, count, width), _RowLines(start + 2, count),
                            start, lines.number - 1,
                            warnings=['ragged_rows_padded'] if narrowest != width else [])
            continue
        content = [lines.pop()]
        size = len(content[0].encode('utf-8'))
        while (line := lines.peek()) is not None and line.strip():
            if _BOUNDARY.match(line):
                break
            following = lines.peek(1)
            if following is not None and '|' in line and _delimiter(following):
                break
            content.append(lines.pop())
            size += len(line.encode('utf-8')) + 1
            if size > MAX_STREAM_BUFFER:
                raise TableChunkError('streaming paragraph exceeds 1 MiB limit')
        yield Block('\n'.join(content), start, lines.number - 1)


def stream_document(stream: BinaryIO, *, max_tokens: int = 256, mode: str = 'markdown',
                    input_format: str = 'markdown', counter: TokenCounter = count_tokens,
                    source: str = '<string>', overlap_rows: int = 0) -> Iterator[Chunk]:
    """Yield chunks from UTF-8 Markdown or HTML without total input/chunk limits.

    Reads at most 64 KiB at a time. Source lines, paragraphs and code blocks are
    limited to 1 MiB UTF-8, rendered working buffers to 1 Mi characters. Tables
    and pending prose use temporary disk storage. HTML keeps its grid/span limits
    and also bounds working records, headers, parser blocks and unfinished tokens.
    Records and bounded whole-row overlap match split_document on accepted inputs.
    Errors may follow yielded chunks; the caller owns the input and must close a
    partially consumed iterator.
    """
    if input_format not in ('markdown', 'html'):
        raise TableChunkError(f'unknown input format: {input_format}')
    if input_format == 'html':
        from .html_streaming import html_blocks
        blocks = html_blocks(stream, MAX_STREAM_BUFFER)
    else:
        blocks = _blocks(stream)
    try:
        with tempfile.TemporaryFile(mode='w+t', encoding='utf-8') as pending:
            yield from _iter_chunks(blocks, max_tokens=max_tokens, mode=mode, counter=counter,
                                    source=source, buffer_limit=MAX_STREAM_BUFFER,
                                    pending=_DiskBlocks(pending), overlap_rows=overlap_rows)
    finally:
        blocks.close()
