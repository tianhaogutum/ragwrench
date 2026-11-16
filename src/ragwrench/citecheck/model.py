"""Validated, bounded input shared by all adapters and the public checker."""
from dataclasses import asdict, dataclass
import math
import re

MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_ANSWER = 100_000
MAX_SOURCES = 256
MAX_SOURCE_TEXT = 100_000
MAX_TOTAL_TEXT = 2_000_000
MAX_SENTENCES = 1000
MAX_SENTENCE = 10_000
MAX_LINK_DESTINATION = 10_000
MAX_LINK_DEPTH = 64
MAX_INDEX = 999_999_999


class InputError(ValueError):
    """An invalid or unsupported export; safe to display without payload contents."""


@dataclass(frozen=True)
class Source:
    idx: int
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Record:
    answer: str
    sources: tuple[Source, ...]


def validate(record: Record) -> Record:
    if not isinstance(record.answer, str) or not record.answer.strip():
        raise InputError("answer must be a non-empty string")
    if re.search(r"[\ud800-\udfff]", record.answer):
        raise InputError("answer contains an unpaired Unicode surrogate")
    if len(record.answer) > MAX_ANSWER:
        raise InputError(f"answer exceeds {MAX_ANSWER} characters")
    if len(record.sources) > MAX_SOURCES:
        raise InputError(f"at most {MAX_SOURCES} sources are supported")
    seen = set()
    total = 0
    for source in record.sources:
        if type(source.idx) is not int or not 1 <= source.idx <= MAX_INDEX:
            raise InputError("source idx must be a positive integer of at most nine digits")
        if source.idx in seen:
            raise InputError(f"duplicate source idx {source.idx}; citation mapping is ambiguous")
        seen.add(source.idx)
        for key in ("id", "title", "text"):
            value = getattr(source, key)
            if not isinstance(value, str):
                raise InputError(f"source {key} must be a string")
            if re.search(r"[\ud800-\udfff]", value):
                raise InputError(f"source {key} contains an unpaired Unicode surrogate")
            if len(value) > MAX_SOURCE_TEXT:
                raise InputError(f"source {key} exceeds {MAX_SOURCE_TEXT} characters")
            total += len(value)
        if not source.text.strip():
            raise InputError("source text must not be empty; export the actual retrieved content")
    if total > MAX_TOTAL_TEXT:
        raise InputError(f"combined source fields exceed {MAX_TOTAL_TEXT} characters")
    return record


@dataclass(frozen=True)
class Config:
    support_threshold: float = 0.5
    misattribution_margin: float = 0.3
    best_source_threshold: float = 0.65
    min_claim_tokens: int = 2
    strict_quotes: bool = False

    def __post_init__(self):
        for name in ("support_threshold", "misattribution_margin", "best_source_threshold"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise InputError(f"{name} must be finite and between 0 and 1")
        if type(self.min_claim_tokens) is not int or not 1 <= self.min_claim_tokens <= 100:
            raise InputError("min_claim_tokens must be an integer between 1 and 100")
        if type(self.strict_quotes) is not bool:
            raise InputError("strict_quotes must be a boolean")


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    sentence: int | None
    message: str
    source_indices: tuple[int, ...] = ()
    evidence: dict | None = None

    def to_dict(self):
        return asdict(self)
