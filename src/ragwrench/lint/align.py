"""Whitespace-tolerant, verifiable character alignment; offsets are Unicode, half-open."""
from array import array
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import re

from .model import Alignment, Span
from .markdown import split_lines


def normalized(text):
    return "".join(text.split())


def normalize_map(text):
    positions = array("I", (i for i, char in enumerate(text) if not char.isspace()))
    return "".join(text[i] for i in positions), positions


@dataclass
class SourceIndex:
    text: str

    def __post_init__(self):
        self.normal, self.positions = normalize_map(self.text)
        # splitlines recognizes too many Unicode separators; source lines are LF/CRLF or CR.
        self.line_starts = [0] + [m.end() for m in re.finditer(r"\r\n|\r|\n", self.text)]

    def span(self, start, end):
        left, right = self.positions[start], self.positions[end - 1] + 1
        return Span(left, right, bisect_right(self.line_starts, left),
                    bisect_right(self.line_starts, right - 1))

    def choose(self, needle, cursor, hint=None):
        """Prefer verified hints, then forward occurrences, then overlaps/backtracking.

        Count ambiguity without collecting potentially millions of match positions.
        """
        if not needle:
            return -1, 0
        first = self.normal.find(needle)
        if first < 0:
            return -1, 0
        ambiguous = 2 if self.normal.find(needle, first + 1) >= 0 else 1
        if hint is not None:
            hinted = bisect_left(self.positions, hint)
            if self.normal.startswith(needle, hinted):
                return hinted, ambiguous
        forward = self.normal.find(needle, cursor)
        return (forward if forward >= 0 else first), ambiguous


def align_chunks(chunks, source):
    index = SourceIndex(source)
    results, cursor = [], 0
    for chunk in chunks:
        needle = normalized(chunk.text)
        hint = None
        if (chunk.start_hint is not None and chunk.end_hint <= len(source)
                and normalized(source[chunk.start_hint:chunk.end_hint]) == needle):
            hint = chunk.start_hint
        start, count = index.choose(needle, cursor, hint)
        if start >= 0:
            results.append(Alignment(chunk.id, "aligned", [index.span(start, start + len(needle))],
                                     len(needle), len(needle), count))
            cursor = max(cursor, start + len(needle))
            continue
        # Repeated table headers/reopened fences are noncontiguous source fragments.
        # Every nonblank line must independently match; partial matches stay visible.
        spans, matched, fragment_cursor, ambiguity = [], 0, cursor, 1
        furthest = cursor
        for line in split_lines(chunk.text):
            fragment = normalized(line)
            if not fragment:
                continue
            start, count = index.choose(fragment, fragment_cursor)
            if start >= 0:
                spans.append(index.span(start, start + len(fragment)))
                matched += len(fragment)
                fragment_cursor = start + len(fragment)
                furthest = max(furthest, fragment_cursor)
                ambiguity = max(ambiguity, count)
        status = "fragmented" if matched == len(needle) else "partial" if matched else "unaligned"
        results.append(Alignment(chunk.id, status, spans, matched, len(needle), ambiguity))
        if status == "fragmented":
            cursor = furthest
    return results


def coverage(source, alignments):
    intervals = sorted((s.start_char, s.end_char) for a in alignments for s in a.spans)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    gaps, cursor = [], 0
    for start, end in merged + [(len(source), len(source))]:
        if any(not char.isspace() for char in source[cursor:start]):
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    total = sum(not c.isspace() for c in source)
    missing = sum(sum(not c.isspace() for c in source[a:b]) for a, b in gaps)
    return {"total_chars": total, "covered_chars": total - missing, "missing_chars": missing,
            "coverage": (total - missing) / total if total else 1.0,
            "gaps": [{"start_char": a, "end_char": b} for a, b in gaps]}


def covered_positions(source, alignments):
    """Compact byte mask used by the fix invariant (no per-character Python set)."""
    mask = bytearray(len(source))
    for alignment in alignments:
        for span in alignment.spans:
            mask[span.start_char:span.end_char] = b"\x01" * (span.end_char - span.start_char)
    return mask
