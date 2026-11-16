"""Small bounded Markdown and EN/CJK sentence scanner, preserving offsets."""
from dataclasses import dataclass
import re

from .markers import Marker
from .lexical import QUOTE_PAIRS
from .model import InputError, MAX_LINK_DEPTH, MAX_LINK_DESTINATION, MAX_SENTENCE, MAX_SENTENCES


@dataclass(frozen=True)
class Sentence:
    idx: int
    start: int
    end: int
    text: str
    markers: tuple[Marker, ...]


BACKTICKS = re.compile(r"`+")
# Complete comments/PIs/CDATA end at their first terminator before CR/LF.
# Quotes, backticks and > in their bodies are literal; no body validation.
OPAQUE_BOUNDARIES = {
    "comment": re.compile(r"-->|[\r\n]"),
    "pi": re.compile(r"\?>|[\r\n]"),
    "cdata": re.compile(r"\]\]>|[\r\n]"),
}
# Only an unquoted > closes a single-line tag. Quotes must be balanced;
# HTML attribute quotes are not backslash-escaped. This is not a full parser.
# Openers allow Unicode word characters except decimal digits/underscore,
# Closing / must be followed by the same word-character rule. The ! and ?
# openers retain their declaration/comment/processing-instruction masks. Other
# punctuation/symbols, whitespace and empty closing names start prose. Tag bodies
# after a supported prefix stay permissive.
TAG = r'''<(?=[^\W\d_]|/[^\W\d_]|[!?])(?:[^>'"\n]|"[^"\n]*"|'[^'\n]*')+>'''
GENERIC_TAG = re.compile(TAG)
INLINE = re.compile(r"(?P<comment><!--)|(?P<pi><\?)|(?P<cdata><!\[CDATA\[)|" + TAG + r"|(?P<url>[Hh][Tt][Tt][Pp][Ss]?://[^\s<>]+)|`+|!?\[|\]|\n")


def _link_destination_end(text: str, start: int, end: int) -> int | None:
    """Return the end of a bounded same-line balanced (...) destination.

    The budget and depth include the outer parentheses. Odd backslash runs
    escape parentheses; everything else inside is opaque. Unsupported syntax
    gets no partial mask and falls back to the ordinary inline scanner.
    """
    if start >= end or text[start] != "(":
        return None
    depth = 0
    escaped = False
    for i in range(start, min(end, start + MAX_LINK_DESTINATION)):
        char = text[i]
        if char in "\r\n":
            return None
        if char == "\\":
            escaped = not escaped
            continue
        if not escaped:
            if char == "(":
                depth += 1
                if depth > MAX_LINK_DEPTH:
                    return None
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return i + 1
        escaped = False
    return None


def _inline_masks(text: str, start: int, end: int):
    """Scan comments, PIs, CDATA, tags, URLs, code and links in source order.

    Index the next equal run once so unmatched openers do not repeatedly scan
    the rest of the block. Different-length runs inside a span are literal.
    Tags outside code are opaque, but tag-like text inside code is literal and
    can contain a closer. Likewise, consume link destinations only when their
    closing label bracket is reached outside code; label content stays prose.
    Bare URLs outside code are opaque up to whitespace or an angle bracket.
    Do not pre-mask all tags, links or URLs before pairing code.
    """
    runs = list(BACKTICKS.finditer(text, start, end))
    run_indices = {run.start(): i for i, run in enumerate(runs)}
    next_equal = [None] * len(runs)
    next_shorter = [None] * len(runs)
    following = {}
    for i in range(len(runs) - 1, -1, -1):
        length = runs[i].end() - runs[i].start()
        next_equal[i] = following.get(length)
        # Outside code an odd backslash run escapes just the first backtick.
        # Any remaining backticks can open a span with a shorter delimiter.
        next_shorter[i] = following.get(length - 1)
        following[length] = i
    position = start
    label_start = None
    opaque_lookahead = {}
    while match := INLINE.search(text, position, end):
        position = match.end()
        token = text[match.start()]
        if match.lastgroup in OPAQUE_BOUNDARIES:
            kind = match.lastgroup
            # Reuse the next terminator/newline (or end) until passed. Repeated
            # incomplete openers must not rescan the same remaining suffix.
            boundary, search_end = opaque_lookahead.get(kind, (None, start))
            if match.start() >= search_end:
                boundary = OPAQUE_BOUNDARIES[kind].search(text, position, end)
                search_end = boundary.end() if boundary else end
                opaque_lookahead[kind] = (boundary, search_end)
            if boundary is not None and boundary[0] not in "\r\n":
                position = search_end
                yield match.start(), position
            elif tag := GENERIC_TAG.match(text, match.start(), end):
                # Unsupported candidates retain the existing !/?-tag rule.
                position = tag.end()
                yield match.start(), position
            continue
        if token == "<" or match.lastgroup == "url":
            yield match.start(), match.end()
            continue
        if token in "![":
            if label_start is None:
                label_start = (match.start(), match.end())
            continue
        if token == "\n":
            label_start = None
            continue
        if token == "]":
            if label_start is not None and (link_end := _link_destination_end(text, match.end(), end)) is not None:
                yield label_start
                yield match.start(), link_end
                position = link_end
            label_start = None
            continue
        opening = match.start()
        i = run_indices[opening]
        slash_start = opening
        while slash_start > start and text[slash_start - 1] == "\\":
            slash_start -= 1
        escaped = (opening - slash_start) % 2
        closer = next_shorter[i] if escaped else next_equal[i]
        if closer is not None:
            # A backslash within an open code span cannot escape its closer.
            yield opening + escaped, runs[closer].end()
            position = runs[closer].end()
            if label_start is not None and "\n" in text[opening:position]:
                label_start = None


def mask_markdown(text: str) -> str:
    chars = list(text)

    def mask(start, end, keep_newlines=True):
        for i in range(start, end):
            if not keep_newlines or chars[i] != "\n":
                chars[i] = " "

    offset = 0
    fence = None
    prose_start = None
    prose_blocks = []
    for line in text.splitlines(keepends=True):
        match = re.match(r" {0,3}(`{3,}|~{3,})", line)
        block = True
        if fence:
            mask(offset, offset + len(line))
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not line[match.end():].strip():
                fence = None
        elif match and (match[1][0] == "~" or "`" not in line[match.end():]):
            # A backtick fence's info string cannot contain backticks. A line
            # starting with e.g. ```literal``` is inline code, not a fence.
            fence = match[1]
            mask(offset, offset + len(line))
        elif re.match(r"(?: {4}|\t| {0,3}#{1,6}\s)", line):
            mask(offset, offset + len(line))
        else:
            block = not line.strip()
        if block:
            if prose_start is not None:
                prose_blocks.append((prose_start, offset))
                prose_start = None
        elif prose_start is None:
            prose_start = offset
        offset += len(line)
    if prose_start is not None:
        prose_blocks.append((prose_start, len(text)))
    for start, end in prose_blocks:
        for inline_start, inline_end in _inline_masks(text, start, end):
            # Soft line endings inside inline code must not split a claim.
            # Replace each code point without changing original-text offsets.
            mask(inline_start, inline_end, keep_newlines=False)
    return "".join(chars)


def sentences(text: str, markers: list[Marker]) -> list[Sentence]:
    """Attach citations immediately after terminal punctuation to that sentence.

    Newlines end prose units. Dotted decimals and common abbreviations do not.
    Offsets always refer to the original, unnormalised answer.
    """
    starts = {m.start: m for m in markers}
    result = []
    begin = 0
    i = 0
    quote_close = None

    def emit(end):
        nonlocal begin
        if text[begin:end].strip():
            if end - begin > MAX_SENTENCE:
                raise InputError(f"sentence exceeds {MAX_SENTENCE} characters")
            if len(result) >= MAX_SENTENCES:
                raise InputError(f"answer exceeds {MAX_SENTENCES} sentences")
            result.append(Sentence(len(result) + 1, begin, end, text[begin:end].strip(),
                                   tuple(m for m in markers if begin <= m.start < end)))
        begin = end

    while i < len(text):
        if i in starts:
            i = starts[i].end
            continue
        c = text[i]
        closing_quote_boundary = False
        if quote_close:
            if c == quote_close:
                quote_close = None
                closing_quote_boundary = i > 0 and text[i - 1] in ".。！？!?"
        elif c in QUOTE_PAIRS:
            quote_close = QUOTE_PAIRS[c]
        boundary = (c in "。！？!?\n" and quote_close is None) or closing_quote_boundary
        if c == ".":
            decimal = i > 0 and i + 1 < len(text) and text[i - 1].isdigit() and text[i + 1].isdigit()
            prefix = text[max(begin, i - 6):i + 1].lower()
            abbreviation = re.search(r"(?:\b(?:mr|mrs|dr|ms|prof|vs|etc)|e\.g|i\.e)\.$", prefix)
            initial = i > 0 and text[i - 1].isupper() and (i == 1 or not text[i - 2].isalpha())
            boundary = quote_close is None and not (decimal or abbreviation or initial)
        if boundary:
            end = i + 1
            if c != "\n":
                # Closing quotes and attached postfix markers belong to this
                # sentence; do not jump across a newline into the next claim.
                while end < len(text):
                    if text[end] in " \t\r\"'”’」』＂｣":
                        end += 1
                    elif end in starts:
                        end = starts[end].end
                    elif text[end] in ".。!?！？":
                        end += 1
                    else:
                        break
            emit(end)
            i = end
        else:
            i += 1
    emit(len(text))
    return result
