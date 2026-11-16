"""Deterministic conservative edits, checked before the CLI writes any output."""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import re

from .align import align_chunks, covered_positions, normalized
from .lint import lint
from .markdown import FENCE, HEADING, LIST, quote_content, scan, split_lines, table_runs, trailing_heading, valid_table
from .model import Chunk, Config, InputError


@dataclass
class FixResult:
    chunks: list
    changes: list
    skipped: list
    before: object
    after: object
    invariant: dict

    def record(self):
        return {"changes": self.changes, "skipped": self.skipped,
                "invariant": self.invariant, "before": self.before.record(), "after": self.after.record()}


def _characters(chunks):
    return Counter(char for c in chunks for char in c.text if not char.isspace())


def _coverage_preserved(source, baseline, chunks):
    after = covered_positions(source, align_chunks(chunks, source))
    return all(not flag or after[i] or source[i].isspace() for i, flag in enumerate(baseline))


def _merged(left, right, reason):
    metadata = deepcopy(left.metadata)
    # Keep all metadata, including conflicts, in ordered provenance rather than overwriting it.
    def provenance(chunk):
        prior = chunk.metadata.get("chunklint", {})
        if isinstance(prior, dict) and isinstance(prior.get("inputs"), list):
            return prior["inputs"]
        return [{"id": chunk.id, "metadata": deepcopy(chunk.metadata)}]
    metadata["chunklint"] = {"inputs": provenance(left) + provenance(right), "operation": reason}
    return Chunk(left.id, left.text + ("" if left.text.endswith(("\n", "\r")) else "\n") + right.text,
                 metadata)


def _table_join(left, right):
    a, b = scan(left.text), scan(right.text)
    ar, br = table_runs(a), table_runs(b)
    if not ar or not br:
        return False
    start, end = ar[-1]
    # Blank lines terminate a table: merging across a blank would not fix it.
    if end != len(a.lines) or not a.lines[-1].strip():
        return False
    rstart, rend = br[0]
    return (rstart == 0 and not valid_table(b.lines[rstart:rend])
            and valid_table(a.lines[start:end] + b.lines[rstart:rend]))


def _table_boundary_preserved(left, right, merged):
    """Source recovery must leave the joined boundary in one valid table run.

    Quoted joins also require whole rows. Unquoted source-backed token cuts
    can collapse two fragment lines into one row, so do not require their sum.
    """
    a, b = scan(left.text), scan(right.text)
    ar, br = table_runs(a), table_runs(b)
    quoted = bool(quote_content(a.lines[ar[-1][0]])[0])
    structure = scan(merged.text)
    runs = table_runs(structure)
    if len(runs) != len(ar) + len(br) - 1:
        return False
    start, end = runs[len(ar) - 1]
    return (valid_table(structure.lines[start:end])
            and (not quoted or end - start == ar[-1][1] - ar[-1][0] + br[0][1] - br[0][0]))


def _source_merged(left, right, reason, source, a, b):
    if a.status == b.status == "aligned":
        start, end = a.spans[0].start_char, b.spans[0].end_char
        if start < end and normalized(source[start:end]) == normalized(left.text + right.text):
            merged = _merged(left, right, reason)
            prefix = left.text[:len(left.text) - len(left.text.lstrip())]
            suffix = right.text[len(right.text.rstrip()):]
            merged.text = prefix + source[start:end] + suffix
            merged.start_hint, merged.end_hint = start, end
            return merged
    return None


def _source_quoted_token_join(chunks, index, source):
    """An exact header/delimiter/body cut with one raw quote prefix."""
    left, right = chunks[index:index + 2]
    prefix = re.match(r" {0,3}> ", left.text)
    left_lines = split_lines(left.text, keepends=True)
    if prefix is None or not left_lines:
        return None
    prefix = prefix[0]
    row = len(left_lines) - 1
    row_start = sum(map(len, left_lines[:row]))
    text = left.text + right.text
    structure = scan(text)
    if (len(structure.lines) <= row
            or not row_start + len(prefix) < len(left.text) < row_start + len(structure.lines[row])
            or not all(structure.outside)
            or table_runs(structure) != [(0, len(structure.lines))]
            or not valid_table(structure.lines)):
        return None
    for line in structure.lines:
        depth, content = quote_content(line)
        if (not line.startswith(prefix) or depth != 1 or LIST.match(content)
                or content.startswith("    ")):
            return None
    a, b = align_chunks(chunks, source)[index:index + 2]
    if (a.status != "aligned" or b.status != "aligned" or a.candidates != 1 or b.candidates != 1
            or a.spans[0].end_char > b.spans[0].start_char):
        return None
    # Unlike the unquoted fallback, require the entire raw slice to match:
    # whitespace-tolerant alignment must not rewrite or manufacture a quote prefix.
    start = a.spans[0].start_char - (len(prefix) - 2)
    end = start + len(text)
    if start < 0 or source[start:end] != text or (start and source[start - 1] not in "\r\n"):
        return None
    # Both aligned sides must touch the same payload row. In particular, a
    # left record ending in a row prefix has not supplied any of its payload.
    if not (start + row_start + len(prefix) < a.spans[0].end_char
            <= b.spans[0].start_char < start + row_start + len(structure.lines[row])):
        return None
    source_structure = scan(source)
    first = a.spans[0].start_line - 1
    if not all(source_structure.outside[first:first + len(structure.lines)]):
        return None  # A fence outside these chunk edges still owns the table-like text.
    if row >= 3:
        # Later-body recovery needs the whole actual source table, not a
        # header-looking subset or a truncated final row. Keep the earlier
        # header/delimiter/first-body contracts unchanged.
        last = first + len(structure.lines)
        if ((first, last) not in table_runs(source_structure)
                or source_structure.lines[first:last] != structure.lines
                or any(outside and LIST.match(quote_content(line)[1]) for line, outside in
                       zip(source_structure.lines[:first], source_structure.outside[:first]))):
            return None
    merged = _merged(left, right, "rejoin_table")
    merged.text = text
    merged.start_hint, merged.end_hint = a.spans[0].start_char, b.spans[0].end_char
    return merged


def _source_table_token_join(chunks, index, source):
    """Recognize bounded token cuts only from a contiguous source table."""
    left, right = chunks[index:index + 2]
    structure = scan(left.text)
    if structure.lines and quote_content(structure.lines[0])[0]:
        return _source_quoted_token_join(chunks, index, source)
    # An unquoted partial header may follow whole prose lines. Quoted/fenced
    # preambles remain outside this fallback's contract.
    if (not structure.lines or not all(structure.outside)
            or any(quote_content(line)[0] for line in structure.lines)):
        return None
    row = len(structure.lines) - 1
    a, b = align_chunks(chunks, source)[index:index + 2]
    if a.status != "aligned" or b.status != "aligned" or a.spans[0].end_char > b.spans[0].start_char:
        return None
    merged = _source_merged(left, right, "rejoin_table", source, a, b)
    if merged is None:
        return None
    structure = scan(merged.text)
    runs = table_runs(structure)
    if len(runs) != 1:
        return None
    start, end = runs[0]
    # Only header cuts gain preamble support; delimiter cuts still require
    # exactly one whole header and the partial delimiter in the left record.
    if (end != len(structure.lines) or (row != start and not (start == 0 and row == 1))
            or not all(structure.outside)
            or any(quote_content(line)[0] for line in structure.lines)
            or not valid_table(structure.lines[start:end])):
        return None
    raw = split_lines(merged.text, keepends=True)
    if raw[:start] != split_lines(left.text, keepends=True)[:start]:
        return None  # Do not rewrite whitespace or recover missing preamble lines.
    row_start = sum(map(len, raw[:row]))
    row_end = row_start + len(structure.lines[row])
    prefix_size = len(left.text) - len(left.text.lstrip())
    left_end = prefix_size + a.spans[0].end_char - merged.start_hint
    right_start = prefix_size + b.spans[0].start_char - merged.start_hint
    # Both aligned sides must touch the same header/delimiter row, with no line break
    # or other non-whitespace content hidden between them.
    return merged if row_start < left_end <= right_start < row_end else None


def _closer_ahead(fence, chunks):
    for chunk in chunks:
        for line in split_lines(chunk.text):
            if not fence.continues(line):
                return False
            if fence.closes(line):
                return True
    return False


def _source_fence_opener_join(chunks, index, source):
    """Recover only a one/two-marker cut within a real, closed source opener."""
    left, right = chunks[index:index + 2]
    raw = split_lines(left.text, keepends=True)
    if not raw:
        return None
    quoted = re.fullmatch(r"( {0,3}> ) {0,3}(?:`{1,2}|~{1,2})", raw[-1])
    prefix = quoted[1] if quoted else ""
    if not quoted and not re.fullmatch(r" {0,3}(?:`{1,2}|~{1,2})", raw[-1]):
        return None
    row = len(raw) - 1
    # List containers are outside the scanner; do not reinterpret their
    # continuations as fences, or reinterpret nested/indented preambles.
    if quoted:
        preamble = [quote_content(line) for line in raw[:row]]
        if any(depth != 1 or LIST.match(content) or content.startswith("    ")
               for depth, content in preamble):
            return None
    elif any(quote_content(line)[0] or LIST.match(line.expandtabs(4))
             or line.startswith(("    ", "\t")) for line in raw[:row]):
        return None
    text = left.text + right.text
    structure = scan(text)
    # Require one raw prefix throughout; strip only its active depth for code
    # parsing so extra '>' characters within the payload remain literal.
    if quoted and (any(not line.startswith(prefix) for line in structure.lines)
                   or quote_content(structure.lines[row])[0] != 1):
        return None
    opener = FENCE.match(quote_content(structure.lines[row], 1)[1] if quoted else structure.lines[row])
    if (opener is None
            or not any(start == row for start, _ in structure.code_blocks)):
        return None
    a, b = align_chunks(chunks, source)[index:index + 2]
    if (a.status != "aligned" or b.status != "aligned" or a.candidates != 1 or b.candidates != 1
            or a.spans[0].end_char != b.spans[0].start_char):
        return None
    start = a.spans[0].start_char - (len(left.text) - len(left.text.lstrip()))
    end = start + len(text)
    if (start < 0 or source[start:end] != text
            or (start and source[start - 1] not in "\r\n")
            or a.spans[0].end_char != start + len(left.text)):
        return None
    source_structure = scan(source)
    source_raw = split_lines(source, keepends=True)
    source_row = a.spans[0].end_line - 1
    opener_start = sum(map(len, source_raw[:source_row]))
    if opener_start != start + sum(map(len, raw[:row])):
        return None
    if any(outside and LIST.match(quote_content(line)[1] if quoted else line.expandtabs(4)) for line, outside in
           zip(source_structure.lines[:source_row], source_structure.outside[:source_row])):
        return None
    # A local-looking fence can be literal code or a closer in the full source.
    # Require this exact opener and its explicit closer in the recovered slice.
    if not any(first == source_row
               and sum(map(len, source_raw[:last - 1])) + len(source_structure.lines[last - 1]) <= end
               for first, last in source_structure.code_blocks):
        return None
    merged = _merged(left, right, "rejoin_fence")
    merged.text = text
    merged.start_hint, merged.end_hint = a.spans[0].start_char, b.spans[0].end_char
    return merged


def _pack_rows(prefix, rows, suffix, limit, context=""):
    groups, current = [], []
    size = len(context) + len(prefix) + len(suffix)
    for row in rows:
        if current and size + len(row) > limit:
            groups.append(context + prefix + "".join(current) + suffix)
            context = ""
            current, size = [], len(prefix) + len(suffix)
        current.append(row)
        size += len(row)
    if current:
        groups.append(context + prefix + "".join(current) + suffix)
    return groups


def _split_structures(text, limit):
    """Only split complete fenced blocks and complete pipe tables; preserve all raw lines."""
    if len(text) <= limit:
        return [text]
    raw = split_lines(text, keepends=True)
    structure = scan(text)
    ranges = [(start, end, "code") for start, end in structure.code_blocks]
    for start, end in table_runs(structure):
        if valid_table(structure.lines[start:end]):
            ranges.append((start, end, "table"))
    ranges.sort()
    pieces, cursor, changed = [], 0, False
    for start, end, kind in ranges:
        block = "".join(raw[start:end])
        if len(block) <= limit:
            continue
        context = ""
        preamble = "".join(raw[cursor:start])
        if trailing_heading(preamble):
            pre_lines = split_lines(preamble, keepends=True)
            cut = trailing_heading(preamble) - 1
            # Carry a consecutive heading stack and surrounding blank lines forward.
            while cut > 0 and (not pre_lines[cut - 1].strip() or HEADING.match(pre_lines[cut - 1])):
                cut -= 1
            context = "".join(pre_lines[cut:])
            preamble = "".join(pre_lines[:cut])
        if preamble.strip():
            pieces.append(preamble)
        if kind == "code":
            prefix, suffix = raw[start], raw[end - 1]
            # If the original closer has no newline, synthesized inner closers don't need one either.
            split = _pack_rows(prefix, raw[start + 1:end - 1], suffix, limit, context)
        else:
            prefix = "".join(raw[start:start + 2])
            split = _pack_rows(prefix, raw[start + 2:end], "", limit, context)
        if len(split) > 1:
            pieces.extend(split)
            changed = True
        else:
            pieces.append(context + block)
        cursor = end
    if cursor < len(raw):
        pieces.append("".join(raw[cursor:]))
    return [p for p in pieces if p.strip()] if changed else [text]


def fix(chunks, source=None, config=None):
    config = (config or Config()).validate()
    before = lint(chunks, source, config)
    working = deepcopy(chunks)
    changes, skipped = [], []
    expected = _characters(chunks)
    baseline = covered_positions(source, before.alignments) if source is not None else None

    def deduplicate():
        nonlocal working, expected
        if "CL006" in config.disable:
            return
        seen, index = {}, 0
        while index < len(working):
            chunk = working[index]
            if chunk.text in seen:
                prior = seen[chunk.text]
                # Equal pieces produced from repeated rows/code lines are real payload,
                # not duplicate input records. Keep them stable on subsequent runs.
                provenance = chunk.metadata.get("chunklint", {})
                if isinstance(provenance, dict) and provenance.get("operation") == "split_structure":
                    skipped.append({"operation": "drop_duplicate", "chunk_ids": [chunk.id, prior],
                                    "reason": "structural split piece; repeated payload is preserved"})
                    index += 1
                    continue
                candidate = working[:index] + working[index + 1:]
                if source is not None and not _coverage_preserved(source, baseline, candidate):
                    skipped.append({"operation": "drop_duplicate", "chunk_ids": [chunk.id, prior],
                                    "reason": "distinct source occurrence or uncertain alignment; coverage would decrease"})
                    index += 1
                    continue
                expected.subtract(Counter(c for c in chunk.text if not c.isspace()))
                changes.append({"operation": "drop_duplicate", "chunk_ids": [chunk.id], "kept": prior})
                working = candidate
            else:
                seen[chunk.text] = chunk.id
                index += 1

    deduplicate()
    i = 0
    while i + 1 < len(working):
        left, right = working[i:i + 2]
        structure = scan(left.text)
        reason = None
        source_join = None
        if "CL001" not in config.disable and structure.unclosed:
            if not _closer_ahead(structure.unclosed, working[i + 1:]):
                skipped.append({"operation": "rejoin_fence", "chunk_ids": [left.id],
                                "reason": "no matching closing fence in subsequent chunks"})
                i += 1
                continue
            reason = "rejoin_fence"
        elif ("CL001" not in config.disable and source is not None
              and (source_join := _source_fence_opener_join(working, i, source)) is not None):
            reason = "rejoin_fence"
        elif "CL003" not in config.disable and trailing_heading(left.text):
            reason = "merge_heading_forward"
        elif "CL002" not in config.disable:
            if _table_join(left, right):
                reason = "rejoin_table"
            elif source is not None:
                source_join = _source_table_token_join(working, i, source)
                if source_join is not None:
                    reason = "rejoin_table"
        if reason:
            merged = source_join or _merged(left, right, reason)
            if source is not None and source_join is None:
                alignments = align_chunks(working, source)
                a, b = alignments[i:i + 2]
                # Recover table/code tokens without inventing a newline, only
                # after verifying the exact normalized adjacent source content.
                merged = _source_merged(left, right, reason, source, a, b) or merged
            if (reason == "rejoin_table" and source_join is None
                    and not _table_boundary_preserved(left, right, merged)):
                skipped.append({"operation": reason, "chunk_ids": [left.id, right.id],
                                "reason": "recovered source does not have a contiguous table boundary"})
                i += 1
                continue
            candidate = working[:i] + [merged] + working[i + 2:]
            if source is not None and not _coverage_preserved(source, baseline, candidate):
                skipped.append({"operation": reason, "chunk_ids": [left.id, right.id],
                                "reason": "source coverage would decrease"})
                i += 1
                continue
            changes.append({"operation": reason, "chunk_ids": [left.id, right.id], "output_id": merged.id})
            working = candidate
            # Revisit the preceding boundary: joining a header can make a table recognizable.
            i = max(0, i - 1)
        else:
            i += 1
    if "CL005" not in config.disable:
        used = {c.id for c in working}
        result = []
        for chunk in working:
            pieces = _split_structures(chunk.text, config.max_chars)
            if len(pieces) == 1:
                result.append(chunk)
                continue
            split_chunks, serial = [], 2
            for j, text in enumerate(pieces):
                cid = chunk.id
                if j:
                    while f"{chunk.id}~{serial}" in used:
                        serial += 1
                    cid = f"{chunk.id}~{serial}"
                    used.add(cid)
                    serial += 1
                metadata = deepcopy(chunk.metadata)
                metadata["chunklint"] = {"operation": "split_structure", "parent": chunk.id,
                                         "part": j + 1, "parent_metadata": deepcopy(chunk.metadata)}
                split_chunks.append(Chunk(cid, text, metadata))
            result.extend(split_chunks)
            changes.append({"operation": "split_structure", "chunk_ids": [chunk.id],
                            "output_ids": [c.id for c in split_chunks]})
        working = result
    deduplicate()
    actual = _characters(working)
    missing = expected - actual
    if missing:
        raise InputError("repair aborted: non-whitespace character conservation invariant failed")
    if source is not None and not _coverage_preserved(source, baseline, working):
        raise InputError("repair aborted: source coverage invariant failed; no files written")
    after = lint(working, source, config)
    # Remaining warnings are explicit: long single rows/lines, missing headers, trailing headings,
    # uncertain list/prose boundaries and near duplicates are never guessed away.
    return FixResult(working, changes, skipped, before, after,
                     {"input_nonwhitespace_preserved_except_logged_exact_duplicates": True,
                      "source_coverage_preserved": True if source is not None else None,
                      "source_coverage_checked": source is not None})
