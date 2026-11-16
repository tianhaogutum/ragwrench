"""Budgeted packing and serialisation. No models, plugins, or subprocesses."""
import html
import re
from collections.abc import Iterable, Iterator
from .models import Block, Chunk, Table, TableChunkError
from .tokens import TokenCounter, count_tokens

MAX_INPUT_BYTES = 10 * 1024 * 1024
MAX_CHUNKS = 10_000


def _md(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def render_table(table: Table, rows: list[list[str]], mode: str, context: list[str]) -> str:
    prefix = "\n".join(part for part in context if part)
    if mode == "markdown":
        lines = ["| " + " | ".join(_md(c) for c in table.columns) + " |",
                 "| " + " | ".join("---" for _ in table.columns) + " |"]
        lines += ["| " + " | ".join(_md(c) for c in row) + " |" for row in rows]
        body = "\n".join(lines)
    elif mode == "kv":
        body = "Columns: " + "; ".join(table.columns)
        for row in rows:
            body += "\n\n" + "\n".join(f"{col}: {value}" for col, value in zip(table.columns, row))
    elif mode == "html":
        def cell(value):
            return html.escape(value).replace("\n", "<br>")
        head = "".join("<tr>" + "".join(f"<th>{cell(c)}</th>" for c in row) + "</tr>" for row in table.headers)
        body = "<table><thead>" + head + "</thead><tbody>"
        body += "".join("<tr>" + "".join(f"<td>{cell(c)}</td>" for c in row) + "</tr>" for row in rows)
        body += "</tbody></table>"
        prefix = "".join(f"<p>{html.escape(part)}</p>" for part in context if part)
    else:
        raise TableChunkError(f"unknown mode: {mode}")
    return (prefix + "\n" if prefix else "") + body


def _take(text: str, fits) -> tuple[str, str]:
    """Largest fitting prefix, preferring whitespace; fits must be monotonic."""
    # Exponential probing avoids repeatedly scanning half a huge oversized cell.
    lo, hi = 0, min(64, len(text))
    while fits(text[:hi]):
        lo = hi
        if hi == len(text):
            return text, ""
        hi = min(hi * 2, len(text))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(text[:mid]):
            lo = mid
        else:
            hi = mid - 1
    if not lo:
        raise TableChunkError("budget too small for context, headers and one content character")
    if lo < len(text):
        boundary = max(text.rfind(" ", 0, lo), text.rfind("\n", 0, lo))
        if boundary > lo // 2:
            lo = boundary + 1
    part = text[:lo]
    if not fits(part):
        raise TableChunkError("token counter must be monotonic for prefix splitting")
    return part, text[lo:]


def split_document(text: str, *, max_tokens: int = 256, mode: str = "markdown",
                   input_format: str = "markdown", counter: TokenCounter = count_tokens,
                   source: str = "<string>", overlap_rows: int = 0) -> list[Chunk]:
    """Split a document. Row indices are zero-based half-open; lines one-based inclusive.

    Custom counters must return nonnegative integers and be monotonic on prefixes.
    Oversized rows become disjoint cell fragments, explicitly marked in metadata.
    overlap_rows (0..100) repeats a bounded suffix of whole rows within a table;
    oldest overlap rows are dropped to fit at least one new row. Fragments reset it.
    """
    from .markdown import parse_markdown
    from .html import parse_html
    if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
        raise TableChunkError("input exceeds 10 MiB limit")
    if input_format not in ("markdown", "html"):
        raise TableChunkError(f"unknown input format: {input_format}")
    blocks = parse_html(text) if input_format == "html" else parse_markdown(text)
    return list(_iter_chunks(blocks, max_tokens=max_tokens, mode=mode, counter=counter,
                             source=source, chunk_limit=MAX_CHUNKS, overlap_rows=overlap_rows))


def _iter_chunks(blocks: Iterable[Block | Table], *, max_tokens: int, mode: str,
                 counter: TokenCounter, source: str, chunk_limit: int | None = None,
                 buffer_limit: int | None = None, pending=None, overlap_rows: int = 0) -> Iterator[Chunk]:
    """Shared packing engine; callers own block/input/output storage."""
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise TableChunkError("max_tokens must be a positive integer")
    if not isinstance(overlap_rows, int) or isinstance(overlap_rows, bool) or not 0 <= overlap_rows <= 100:
        raise TableChunkError("overlap_rows must be an integer in 0..100")
    if mode not in ("markdown", "kv", "html"):
        raise TableChunkError(f"unknown mode: {mode}")

    def count(value):
        if buffer_limit is not None and len(value) > buffer_limit:
            raise TableChunkError(f"streaming rendered buffer exceeds {buffer_limit}-character limit; reduce budget or block size")
        n = counter(value)
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise TableChunkError("token counter must return a nonnegative integer")
        return n

    emitted = 0
    headings: list[tuple[int, str]] = []
    lead = ""
    table_index = 0
    pending = [] if pending is None else pending
    pending_heading: Block | None = None

    def header_path():
        return [value for _, value in headings]

    def emit(identifier, content, meta):
        nonlocal emitted
        if chunk_limit is not None and emitted >= chunk_limit:
            raise TableChunkError("output exceeds 10000 chunks; increase max_tokens or split input")
        n = count(content)
        if n > max_tokens:
            raise TableChunkError("internal budget invariant failed")
        emitted += 1
        return Chunk(identifier, content, {"source": source, "token_count": n,
                                           "header_path": header_path(), **meta})

    def flush_prose():
        if not pending:
            return
        prefix = " > ".join(header_path())
        prefix = prefix + "\n" if prefix else ""
        current = ""
        starts, ends, kinds = [], [], []

        def output():
            if current:
                yield emit(f"p{emitted}", prefix + current,
                     {"kind": "prose", "line_start": min(starts), "line_end": max(ends),
                      "warnings": ["code_block_split"] if "code" in kinds and split_code else []})

        split_code = False
        for block in pending:
            remaining = block.text
            candidate = current + ("\n\n" if current else "") + remaining
            if count(prefix + candidate) <= max_tokens:
                current = candidate
                starts.append(block.line_start)
                ends.append(block.line_end)
                kinds.append(block.kind)
                continue
            yield from output()
            current, starts, ends, kinds = "", [], [], []
            while remaining:
                part, remaining = _take(remaining, lambda value: count(prefix + value) <= max_tokens)
                current = part
                starts, ends, kinds = [block.line_start], [block.line_end], [block.kind]
                split_code = block.kind == "code" and (bool(remaining) or split_code)
                if remaining:
                    yield from output()
                    current, starts, ends, kinds = "", [], [], []
        yield from output()
        pending.clear()

    def emit_heading(block):
        value = " > ".join(header_path())
        if count(value) > max_tokens:
            raise TableChunkError("heading exceeds max_tokens")
        return emit(f"p{emitted}", value, {"kind": "prose", "line_start": block.line_start,
                                      "line_end": block.line_end, "warnings": []})

    for block in blocks:
        if isinstance(block, Block):
            if block.kind == "heading":
                yield from flush_prose()
                if pending_heading is not None and headings and block.level <= headings[-1][0]:
                    yield emit_heading(pending_heading)
                while headings and headings[-1][0] >= block.level:
                    headings.pop()
                headings.append((block.level, block.text))
                pending_heading = block
                lead = ""
            else:
                pending_heading = None
                pending.append(block)
                if block.kind == "prose":
                    # The final sentence of the preceding paragraph is the lead-in.
                    sentences = re.split(r"(?<=[.!?。！？])\s+", block.text)
                    lead = sentences[-1].strip()
                else:
                    lead = ""
            continue
        yield from flush_prose()
        pending_heading = None
        table = block
        context = [" > ".join(header_path()), table.caption or lead]

        def render(rows):
            return render_table(table, rows, mode, context)

        if count(render([])) > max_tokens:
            raise TableChunkError(f"table {table_index}: context and headers exceed max_tokens")

        def table_emit(start, end, rows, suffix="", extra=None, repeated=0):
            locations = table.row_lines[start:end]
            metadata = {"kind": "table", "table_index": table_index, "row_start": start,
                        "row_end": end, "columns": table.columns, "headers": table.headers,
                        "caption": table.caption, "lead_in": lead, "mode": mode,
                        "line_start": min((a for a, _ in locations), default=table.line_start),
                        "line_end": max((b for _, b in locations), default=table.line_end),
                        "table_line_start": table.line_start, "table_line_end": table.line_end,
                        "row_line_spans": [list(loc) for loc in locations],
                        "warnings": list(table.warnings)}
            if overlap_rows:
                metadata.update(overlap_rows=repeated, requested_overlap_rows=overlap_rows)
            if table.normalization is not None:
                metadata["normalization"] = table.normalization.for_rows(start, end)
            if extra:
                metadata.update(extra)
            return emit(f"t{table_index}-r{start}-{end}{suffix}", render(rows), metadata)

        group, group_start, group_overlap = [], 0, 0
        for index, row in enumerate(table.rows):
            if count(render(group + [row])) <= max_tokens:
                if not group:
                    group_start = index
                group.append(row)
                continue
            # Each group already contains a new row. Carry only its bounded suffix,
            # and reserve room for this new row before considering further rows.
            overlap = []
            if group:
                yield table_emit(group_start, index, group, repeated=group_overlap)
                if overlap_rows:
                    overlap = group[-overlap_rows:]
                group, group_overlap = [], 0
            if count(render([row])) <= max_tokens:
                while overlap and count(render(overlap + [row])) > max_tokens:
                    overlap.pop(0)
                group_overlap = len(overlap)
                group, group_start = overlap + [row], index - group_overlap
                continue
            # Cell-wise fallback never duplicates the row's content. Full headers remain.
            fragment_index = 0
            for col, value in enumerate(row):
                remaining = value
                offset = 0

                def cell_row(part):
                    fragment = [""] * len(row)
                    fragment[col] = part
                    return fragment

                while remaining:
                    part, remaining = _take(remaining, lambda part: count(render([cell_row(part)])) <= max_tokens)
                    yield table_emit(index, index + 1, [cell_row(part)], f"-c{col}-f{fragment_index}",
                               {"warnings": table.warnings + ["oversized_row_cell_split"],
                                "fragment": {"cell_index": col, "index": fragment_index,
                                             "char_start": offset, "char_end": offset + len(part)}})
                    fragment_index += 1
                    offset += len(part)
            if not fragment_index:
                raise TableChunkError(f"table {table_index}: empty row cannot fit budget")
        if group:
            yield table_emit(group_start, len(table.rows), group, repeated=group_overlap)
        if not table.rows:
            yield table_emit(0, 0, [])
        table_index += 1
        lead = ""
    yield from flush_prose()
    # Heading-only sections remain retrievable even at the end of a document.
    if pending_heading is not None:
        yield emit_heading(pending_heading)
