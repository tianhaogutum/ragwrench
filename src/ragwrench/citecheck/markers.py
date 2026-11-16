"""Citation parsing with offsets; malformed candidates are not valid citations."""
from dataclasses import dataclass
import re

from .model import MAX_INDEX


@dataclass(frozen=True)
class Marker:
    start: int
    end: int
    raw: str
    indices: tuple[int, ...]
    error: str | None = None


# URLs/Markdown link destinations are masked before this parser is called.
SOURCE = re.compile(r"\b[Ss]ource\s+(?:[0-9]+[A-Za-z]*|[A-Za-z](?![A-Za-z]))\b")
IDS = re.compile(r"[0-9]{1,9}(?:\s*,\s*[0-9]{1,9})*")


def parse_markers(text: str) -> list[Marker]:
    markers = []
    i = 0
    while i < len(text):
        start = i
        raw = None
        body = None
        zero_based = False
        if text[i] in "[【":
            closer = "]" if text[i] == "[" else "】"
            end = i + 1
            while end < len(text) and text[end] not in (closer, "\n"):
                end += 1
            if end < len(text) and text[end] == closer:
                end += 1
                raw = text[i:end]
                body = raw[1:-1].strip()
                if body.startswith("ID:") and text[i] == "[":
                    body = body[3:]
                    zero_based = True
            else:
                raw = text[i:end]
        elif text[i] in "]】":
            end = i + 1
            raw = text[i:end]
        elif text.startswith("##", i) and i + 2 < len(text) and not text[i + 2].isspace():
            end = i + 2
            while end < len(text) and not text[end].isspace() and text[end] not in ".。!?！？，,":
                if text.startswith("$$", end):
                    end += 2
                    break
                end += 1
            raw = text[i:end]
            if raw.endswith("$$"):
                body = raw[2:-2]
                zero_based = True
        elif (match := SOURCE.match(text, i)) is not None:
            end = match.end()
            raw = match.group()
            body = raw.split()[-1]
        if raw is None:
            i += 1
            continue
        indices = ()
        error = None
        if body is None or IDS.fullmatch(body) is None:
            error = "malformed or non-numeric citation marker"
        else:
            parts = body.split(",")
            if len(parts) > 64:
                error = "citation group exceeds 64 indices"
            else:
                values = tuple(int(p) + int(zero_based) for p in parts)
                if any(n < 1 or n > MAX_INDEX for n in values):
                    error = "citation index outside supported range"
                else:
                    indices = tuple(dict.fromkeys(values))
        markers.append(Marker(start, end, raw, indices, error))
        i = end
    return markers


def mask_markers(text: str, markers: list[Marker]) -> str:
    result = list(text)
    for marker in markers:
        result[marker.start:marker.end] = " " * (marker.end - marker.start)
    return "".join(result)
