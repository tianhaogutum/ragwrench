"""Transparent lexical coverage and normalized literal quantities."""
from dataclasses import dataclass
from decimal import Decimal
from html.parser import HTMLParser
import re
import unicodedata

from .model import InputError

STOP = frozenset("a an the and or but is are was were be been being to of in on for from with by as at it its this that these those has have had will would can could should may might we you they our their there here than then also only not no do does did".split())
CJK = r"\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af"
TOKEN = re.compile(rf"[{CJK}]+|[^\W\d_]+|[0-9]+(?:\.[0-9]+)?", re.UNICODE)


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def source_text(text: str) -> str:
    # RAGFlow may embed HTML tables. Parse data only: no URLs/resources loaded.
    parser = _HTMLText()
    parser.feed(text)
    return " ".join(parser.parts)


def tokens(text: str) -> frozenset[str]:
    result = set()
    normalized = re.sub(r"\b[0-9]{1,3}(?:,[0-9]{3})+\b", lambda m: m.group().replace(",", ""), normalize(text))
    for match in TOKEN.finditer(normalized):
        word = match.group()
        if re.fullmatch(rf"[{CJK}]+", word):
            # Character bigrams preserve local CJK word order without a model.
            result.update(word[i:i + 2] for i in range(max(1, len(word) - 1)))
        elif word not in STOP:
            # Deliberately no stemming or translation; results stay explainable.
            result.add(UNITS.get(word, word))
    return frozenset(result)


def coverage(claim: frozenset[str], source: frozenset[str]) -> float:
    return len(claim & source) / len(claim) if claim else 0.0


UNIT_GROUPS = {
    "%": ("%", "percent", "percentage"),
    "month": ("month", "months", "个月", "月"),
    "year": ("year", "years", "年"),
    "day": ("day", "days", "天", "日"),
    "hour": ("h", "hour", "hours", "小时"),
    "minute": ("min", "minute", "minutes", "分钟"),
    "second": ("s", "second", "seconds", "秒"),
    "kg": ("kg", "kilogram", "kilograms", "千克", "公斤"),
    "g": ("g", "gram", "grams", "克"),
    "km": ("km", "kilometer", "kilometers", "公里", "千米"),
    "m": ("m", "meter", "meters", "米"),
    "cm": ("cm", "centimeter", "centimeters", "厘米"),
    "mb": ("mb", "megabyte", "megabytes"),
    "gb": ("gb", "gigabyte", "gigabytes"),
    "usd": ("usd", "dollar", "dollars"),
    "eur": ("eur", "euro", "euros"),
    "cny": ("cny", "yuan", "元"),
    "°c": ("°c", "℃"),
}
UNITS = {alias: unit for unit, aliases in UNIT_GROUPS.items() for alias in aliases}
UNIT_PATTERN = "|".join(re.escape(u) + (r"(?![a-z])" if u[-1].isascii() and u[-1].isalpha() else "") for u in sorted(UNITS, key=len, reverse=True))
DATE = re.compile(r"(?<![0-9])([0-9]{4})(?:[-/]([0-9]{1,2})[-/]([0-9]{1,2})|年([0-9]{1,2})月([0-9]{1,2})日)(?![0-9])")
QUANTITY = re.compile(r"(?<![a-z0-9_.])(?:(?P<currency>[$€¥])\s*)?(?P<num>[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)(?:\s*(?P<unit>" + UNIT_PATTERN + r"))?", re.UNICODE)


@dataclass(frozen=True)
class Quantity:
    value: str
    unit: str
    raw: str

    @property
    def key(self):
        return (self.value, self.unit)


def quantities(text: str, limit: int = 20_000) -> list[Quantity]:
    text = normalize(text)
    result = []
    chars = list(text)
    for match in DATE.finditer(text):
        if len(result) >= limit:
            raise InputError(f"text exceeds {limit} literal quantities")
        year, m1, d1, m2, d2 = match.groups()
        value = f"{int(year):04d}-{int(m1 or m2):02d}-{int(d1 or d2):02d}"
        result.append(Quantity(value, "date", match.group()))
        chars[match.start():match.end()] = " " * (match.end() - match.start())
    text = "".join(chars)
    for match in QUANTITY.finditer(text):
        if len(result) >= limit:
            raise InputError(f"text exceeds {limit} literal quantities")
        raw = match.group("num").replace(",", "")
        if len(raw) > 128:
            raise InputError("numeric literal exceeds 128 characters")
        # The input character bounds also cap Decimal parsing work.
        value = format(Decimal(raw), "f")
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        if Decimal(raw) == 0:
            value = "0"
        unit = UNITS.get(match.group("unit"), "")
        if match.group("currency"):
            unit = {"$": "usd", "€": "eur", "¥": "cny"}[match.group("currency")]
        result.append(Quantity(value, unit, match.group().strip()))
    return result


def quantity_supported(claim: Quantity, source: list[Quantity]) -> bool:
    return any(q.value == claim.value and (not claim.unit or q.unit == claim.unit) for q in source)


QUOTE_PAIRS = {'"': '"', '“': '”', '「': '」', '『': '』', '＂': '＂', '｢': '｣'}
# Single quotes remain literal (including apostrophes in contractions).
QUOTE = re.compile("|".join(
    re.escape(opening) + "([^" + re.escape(closing) + r"\n]{2,})" + re.escape(closing)
    for opening, closing in QUOTE_PAIRS.items()
))


def quote_spans(text: str):
    """Content offsets in unnormalized text; delimiters are excluded."""
    for match in QUOTE.finditer(text):
        group = next(i for i, value in enumerate(match.groups(), 1) if value is not None)
        yield match.span(group)


def quotes(text: str) -> list[str]:
    return [text[start:end] for start, end in quote_spans(text)]


# NFKC handles fullwidth punctuation. These explicit additional equivalences
# cover ideographic stops/commas and typographic quotation marks, without
# deleting punctuation or inferring semantic equivalence.
QUOTE_PUNCTUATION = str.maketrans({
    '。': '.', '、': ',', '“': '"', '”': '"', '「': '"', '」': '"',
    '『': '"', '』': '"', '‘': "'", '’': "'",
})


def quote_key(text: str) -> str:
    return re.sub(r"\s+", " ", normalize(text).translate(QUOTE_PUNCTUATION)).strip()


def _decomposed_origins(text: str, form: str):
    """Canonical-order decomposition with each code point's input index.

    Per-character normalization alone misses composition across characters.
    Stable ordering of each nonstarter run also preserves provenance when
    combining marks move across the eventual composed output characters.
    """
    marks = []
    for idx, char in enumerate(text):
        for part in unicodedata.normalize(form, char):
            combining = unicodedata.combining(part)
            if combining:
                marks.append((combining, part, idx))
            else:
                for _, mark, origin in sorted(marks, key=lambda item: item[0]):
                    yield mark, origin
                marks.clear()
                yield part, idx
    for _, mark, origin in sorted(marks, key=lambda item: item[0]):
        yield mark, origin


def quote_key_spans(text: str) -> tuple[str, list[tuple[int, int]]]:
    """The quote key plus covering source-prose spans for its code points.

    NFKD(input) == NFD(NFKC(input)). Align those canonical decompositions to
    recover the contributors to each composed character, including Hangul
    and reordered marks, without repeatedly normalizing growing prefixes.
    Expansions share a source span; collapsed whitespace covers its whole run.
    """
    composed = unicodedata.normalize("NFKC", text)
    starts, ends = [len(text)] * len(composed), [0] * len(composed)
    if text == composed:
        starts, ends = list(range(len(text))), list(range(1, len(text) + 1))
    else:
        for (_, origin), (_, target) in zip(_decomposed_origins(text, "NFKD"),
                                            _decomposed_origins(composed, "NFD")):
            starts[target] = min(starts[target], origin)
            ends[target] = max(ends[target], origin + 1)
    chars, spans = [], []
    for idx, char in enumerate(composed):
        span = (starts[idx], ends[idx])
        for part in char.casefold().translate(QUOTE_PUNCTUATION):
            if part.isspace():
                if not chars:
                    continue
                if chars[-1] == " ":
                    spans[-1] = (min(spans[-1][0], span[0]), max(spans[-1][1], span[1]))
                    continue
                part = " "
            chars.append(part)
            spans.append(span)
    if chars and chars[-1] == " ":
        chars.pop()
        spans.pop()
    return "".join(chars), spans
